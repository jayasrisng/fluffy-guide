from __future__ import annotations

import queue
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import numpy as np

from .demo import DemoUtterance, load_demo_script
from .memory import utc_now_iso

if TYPE_CHECKING:
    from faster_whisper import WhisperModel


EmitEvent = Callable[[dict[str, Any]], None]


@dataclass
class ListenerConfig:
    demo_tick_seconds: float
    demo_source_path: Path
    whisper_model_size: str
    whisper_device: str
    sample_rate: int
    mic_mode: str
    mic_chunk_seconds: float
    energy_threshold: float
    confidence_threshold: float
    input_device_index: int | None = None


class ListenerAgent:
    def __init__(
        self,
        cfg: ListenerConfig,
        emit_event: EmitEvent | None = None,
    ) -> None:
        self.cfg = cfg
        self.emit_event = emit_event

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._mode: str = "idle"
        self._demo_idx = 0
        self._demo_script: list[DemoUtterance] = []
        self._last_live_text = ""
        self._last_live_chunk_at = 0.0

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def update_live_settings(
        self,
        *,
        input_device_index: int | None,
        sample_rate: int,
        mic_mode: str,
        mic_chunk_seconds: float,
        energy_threshold: float,
        confidence_threshold: float,
    ) -> bool:
        changed = (
            self.cfg.input_device_index != input_device_index
            or self.cfg.sample_rate != sample_rate
            or self.cfg.mic_mode != mic_mode
            or self.cfg.mic_chunk_seconds != mic_chunk_seconds
            or self.cfg.energy_threshold != energy_threshold
            or self.cfg.confidence_threshold != confidence_threshold
        )
        self.cfg.input_device_index = input_device_index
        self.cfg.sample_rate = sample_rate
        self.cfg.mic_mode = mic_mode
        self.cfg.mic_chunk_seconds = mic_chunk_seconds
        self.cfg.energy_threshold = energy_threshold
        self.cfg.confidence_threshold = confidence_threshold
        return changed

    def start(self, mode: str) -> None:
        if self.is_running and self._mode == mode:
            return
        self.stop()

        self._mode = mode
        self._stop_event.clear()

        if mode == "demo":
            self._demo_script = load_demo_script(self.cfg.demo_source_path)
            self._demo_idx = 0
            self._thread = threading.Thread(target=self._run_demo, daemon=True)
            self._thread.start()
            self._emit_status("Listener running in demo mode")
            return

        if mode == "live":
            self._thread = threading.Thread(target=self._run_live, daemon=True)
            self._thread.start()
            self._emit_status("Listener running in live mode")
            return

        self._mode = "idle"
        self._emit_status("Listener idle")

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)
        self._thread = None
        if self._mode != "idle":
            self._emit_status("Listener stopped")
        self._mode = "idle"

    def list_input_devices(self) -> list[dict[str, Any]]:
        try:
            import sounddevice as sd
        except Exception:
            return []

        devices: list[dict[str, Any]] = []
        seen: set[int] = set()
        try:
            queried = sd.query_devices()
        except Exception:
            queried = []

        for idx, dev in enumerate(queried):
            max_in = int(dev.get("max_input_channels", 0))
            if max_in <= 0:
                continue
            if idx in seen:
                continue
            seen.add(idx)
            default_sr = int(dev.get("default_samplerate", self.cfg.sample_rate))
            devices.append(
                {
                    "index": idx,
                    "name": str(dev.get("name", f"Input {idx}")),
                    "default_samplerate": default_sr,
                    "max_input_channels": max_in,
                }
            )

        if not devices:
            try:
                default_input_idx = int(sd.default.device[0])
            except Exception:
                default_input_idx = -1
            if default_input_idx >= 0:
                try:
                    dev = sd.query_devices(default_input_idx)
                    max_in = int(dev.get("max_input_channels", 0))
                    if max_in > 0:
                        devices.append(
                            {
                                "index": default_input_idx,
                                "name": str(dev.get("name", f"Input {default_input_idx}")),
                                "default_samplerate": int(
                                    dev.get("default_samplerate", self.cfg.sample_rate)
                                ),
                                "max_input_channels": max_in,
                            }
                        )
                except Exception:
                    pass

        return devices

    def start_microphone_test(self, duration_seconds: float = 5.0) -> None:
        thread = threading.Thread(
            target=self._run_microphone_test,
            kwargs={"duration_seconds": duration_seconds},
            daemon=True,
        )
        thread.start()

    def _run_microphone_test(self, duration_seconds: float) -> None:
        ok, result = self._test_microphone(duration_seconds=duration_seconds)
        self._emit_event(
            {
                "type": "mic_test_result",
                "timestamp": utc_now_iso(),
                "success": ok,
                "result": result,
            }
        )

    def _test_microphone(self, duration_seconds: float = 5.0) -> tuple[bool, str]:
        try:
            import sounddevice as sd
            import soundfile as sf
            from faster_whisper import WhisperModel
        except Exception as e:  # noqa: BLE001
            return False, f"Microphone test unavailable: {e}"

        sample_rate = self._resolve_sample_rate_for_device(self.cfg.input_device_index)
        frames = int(max(1.0, duration_seconds) * sample_rate)
        try:
            self._emit_status("Testing microphone capture...")
            recording = sd.rec(
                frames,
                samplerate=sample_rate,
                channels=1,
                dtype="float32",
                device=self.cfg.input_device_index,
            )
            sd.wait()
            waveform = recording.reshape(-1, 1)
            rms = self._rms_energy(waveform)
            if rms < self._min_effective_gate():
                return (
                    False,
                    f"Heard almost no signal (rms={rms:.4f}). Increase source volume or move phone closer.",
                )

            model = WhisperModel(
                self.cfg.whisper_model_size,
                device=self.cfg.whisper_device,
                compute_type="int8",
            )
            text = self._transcribe_chunk(
                model,
                self._boost_quiet_waveform(waveform),
                sf,
                sample_rate=sample_rate,
            )
            if not text:
                return False, "No speech recognized from the test sample."
            return True, text
        except Exception as e:  # noqa: BLE001
            return False, f"Microphone test failed: {e}"

    def _run_demo(self) -> None:
        if not self._demo_script:
            self._emit_status("Demo script is empty")
            return

        while not self._stop_event.is_set():
            utterance = self._demo_script[self._demo_idx % len(self._demo_script)]
            self._demo_idx += 1
            if utterance.text:
                self._emit_transcript(utterance.speaker, utterance.text, "demo")
            time.sleep(self.cfg.demo_tick_seconds)

    def _run_live(self) -> None:
        try:
            import sounddevice as sd
            import soundfile as sf
            from faster_whisper import WhisperModel
        except Exception as e:  # noqa: BLE001
            self._emit_status(
                f"Live mode unavailable ({e}). Use demo mode for guaranteed operation."
            )
            return

        devices = self.list_input_devices()
        if not devices:
            self._emit_status("No valid input device found. Switch to demo mode.")
            return

        self._emit_status("Loading faster-whisper model...")
        try:
            model = WhisperModel(
                self.cfg.whisper_model_size,
                device=self.cfg.whisper_device,
                compute_type="int8",
            )
        except Exception as e:  # noqa: BLE001
            self._emit_status(f"Failed to load model: {e}")
            return

        selected = self.cfg.input_device_index
        if selected is None or all(d["index"] != selected for d in devices):
            selected = devices[0]["index"]
            self.cfg.input_device_index = selected
        active_sample_rate = self._resolve_sample_rate_for_device(selected)
        if active_sample_rate != self.cfg.sample_rate:
            self._emit_status(
                f"Using {active_sample_rate} Hz for this input device (requested {self.cfg.sample_rate} Hz)."
            )
        selected_name = next((d["name"] for d in devices if d["index"] == selected), f"device {selected}")
        self._emit_status(f"Listening on {selected_name}...")

        frames_per_chunk = int(active_sample_rate * self.cfg.mic_chunk_seconds)
        audio_queue: queue.Queue[np.ndarray] = queue.Queue()

        def callback(indata: np.ndarray, _frames: int, _time_info: object, status: Any) -> None:
            if status:
                self._emit_status(f"Audio status: {status}")
            audio_queue.put(indata.copy())

        try:
            with sd.InputStream(
                samplerate=active_sample_rate,
                channels=1,
                dtype="float32",
                device=selected,
                callback=callback,
            ):
                acc: list[np.ndarray] = []
                acc_frames = 0
                self._last_live_chunk_at = time.time()
                while not self._stop_event.is_set():
                    try:
                        chunk = audio_queue.get(timeout=0.4)
                    except queue.Empty:
                        if time.time() - self._last_live_chunk_at > 8:
                            self._emit_status("Listening but hearing nothing useful yet")
                        continue

                    acc.append(chunk)
                    acc_frames += chunk.shape[0]
                    if acc_frames < frames_per_chunk:
                        continue

                    waveform = np.concatenate(acc, axis=0)
                    acc = []
                    acc_frames = 0
                    transcript = self._safe_transcribe_live(
                        model,
                        waveform,
                        sf,
                        sample_rate=active_sample_rate,
                    )
                    if transcript:
                        self._last_live_chunk_at = time.time()
                        self._emit_transcript("lecturer", transcript, "live")
        except Exception as e:  # noqa: BLE001
            self._emit_status(f"Live listener error: {e}")

    def _safe_transcribe_live(
        self,
        model: "WhisperModel",
        waveform: np.ndarray,
        sf: Any,
        *,
        sample_rate: int,
    ) -> str:
        rms = self._rms_energy(waveform)
        if rms < self._min_effective_gate():
            return ""

        prepared = self._boost_quiet_waveform(waveform)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            sf.write(tmp.name, prepared, sample_rate)
            segments, _ = model.transcribe(tmp.name, vad_filter=True)
            parts: list[str] = []
            for seg in segments:
                text = seg.text.strip()
                if not text:
                    continue
                avg_logprob = getattr(seg, "avg_logprob", None)
                if avg_logprob is not None and float(avg_logprob) < self.cfg.confidence_threshold:
                    continue
                parts.append(text)

            if not parts:
                segments, _ = model.transcribe(tmp.name, vad_filter=False)
                for seg in segments:
                    text = seg.text.strip()
                    if not text:
                        continue
                    avg_logprob = getattr(seg, "avg_logprob", None)
                    if avg_logprob is not None and float(avg_logprob) < self.cfg.confidence_threshold:
                        continue
                    parts.append(text)

        joined = " ".join(parts).strip()
        if not joined:
            return ""

        normalized = " ".join(joined.lower().split())
        if normalized == self._last_live_text:
            return ""

        self._last_live_text = normalized
        return joined

    @staticmethod
    def _rms_energy(waveform: np.ndarray) -> float:
        arr = waveform.astype(np.float32).reshape(-1)
        if arr.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(arr))))

    def _transcribe_chunk(
        self,
        model: "WhisperModel",
        waveform: np.ndarray,
        sf: Any,
        *,
        sample_rate: int,
    ) -> str:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            sf.write(tmp.name, waveform, sample_rate)
            segments, _ = model.transcribe(tmp.name, vad_filter=True)
            parts = [seg.text.strip() for seg in segments if seg.text.strip()]
            if not parts:
                segments, _ = model.transcribe(tmp.name, vad_filter=False)
                parts = [seg.text.strip() for seg in segments if seg.text.strip()]
        return " ".join(parts).strip()

    def _min_effective_gate(self) -> float:
        if self.cfg.mic_mode == "lecture_demo":
            return max(0.0009, self.cfg.energy_threshold * 0.15)
        return max(0.0015, self.cfg.energy_threshold * 0.25)

    def _boost_quiet_waveform(self, waveform: np.ndarray) -> np.ndarray:
        arr = waveform.astype(np.float32)
        rms = self._rms_energy(arr)
        if rms <= 0:
            return arr
        if self.cfg.mic_mode == "lecture_demo":
            target_rms = 0.05
            max_gain = 16.0
        else:
            target_rms = 0.03
            max_gain = 10.0
        gain = min(max_gain, max(1.0, target_rms / rms))
        if gain == 1.0:
            return arr
        boosted = np.clip(arr * gain, -1.0, 1.0)
        return boosted

    def _resolve_sample_rate_for_device(self, device_index: int | None) -> int:
        try:
            import sounddevice as sd
        except Exception:
            return self.cfg.sample_rate

        if device_index is None:
            return self.cfg.sample_rate

        try:
            sd.check_input_settings(
                device=device_index,
                samplerate=self.cfg.sample_rate,
                channels=1,
                dtype="float32",
            )
            return self.cfg.sample_rate
        except Exception:
            pass

        try:
            dev = sd.query_devices(device_index)
            fallback = int(dev.get("default_samplerate", self.cfg.sample_rate))
            sd.check_input_settings(
                device=device_index,
                samplerate=fallback,
                channels=1,
                dtype="float32",
            )
            return fallback
        except Exception:
            return self.cfg.sample_rate

    def _emit_status(self, message: str) -> None:
        self._emit_event(
            {
                "type": "status",
                "agent": "listener",
                "message": message,
                "timestamp": utc_now_iso(),
            }
        )

    def _emit_transcript(self, speaker: str, text: str, source: str) -> None:
        self._emit_event(
            {
                "type": "transcript_chunk",
                "speaker": speaker,
                "text": text,
                "source": source,
                "timestamp": utc_now_iso(),
            }
        )

    def _emit_event(self, event: dict[str, Any]) -> None:
        if self.emit_event:
            self.emit_event(event)
