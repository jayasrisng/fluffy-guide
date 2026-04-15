from __future__ import annotations

import html
import queue
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

from fluffy_guide.config import CONFIG
from fluffy_guide.guide import GuideAgent, SummaryAgent
from fluffy_guide.listener import ListenerAgent, ListenerConfig
from fluffy_guide.logging_utils import JsonlLogger
from fluffy_guide.memory import RollingMemory
from fluffy_guide.openai_utils import OpenAIHelper


st.set_page_config(page_title=CONFIG.app_title, layout="wide")


def init_session_state() -> None:
    if "status" not in st.session_state:
        st.session_state.status = {
            "listener": "idle",
            "summary": "idle",
            "guide": "idle",
            "openai": "unknown",
        }
    if "transcript" not in st.session_state:
        st.session_state.transcript = []
    if "rolling_summary" not in st.session_state:
        st.session_state.rolling_summary = ""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "logs_path" not in st.session_state:
        st.session_state.logs_path = None
    if "app_mode" not in st.session_state:
        st.session_state.app_mode = "demo"
    if "run_agents" not in st.session_state:
        st.session_state.run_agents = False
    if "pending_status_messages" not in st.session_state:
        st.session_state.pending_status_messages = []
    if "pending_transcript_chunks" not in st.session_state:
        st.session_state.pending_transcript_chunks = []
    if "pending_summary_updates" not in st.session_state:
        st.session_state.pending_summary_updates = []
    if "pending_log_events" not in st.session_state:
        st.session_state.pending_log_events = []

    if "event_queue" not in st.session_state:
        st.session_state.event_queue = queue.Queue()

    if "memory" not in st.session_state:
        st.session_state.memory = RollingMemory(max_chunks=CONFIG.transcript_buffer_size)

    if "llm" not in st.session_state:
        st.session_state.llm = OpenAIHelper(
            api_key=CONFIG.openai_api_key,
            model=CONFIG.openai_model,
            stt_model=CONFIG.openai_stt_model,
        )

    if st.session_state.logs_path is None:
        log_path = CONFIG.log_dir / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        st.session_state.logs_path = str(log_path)
        st.session_state.logger = JsonlLogger(log_path)
    elif "logger" not in st.session_state:
        st.session_state.logger = JsonlLogger(CONFIG.log_dir / Path(st.session_state.logs_path).name)

    if "selected_input_device_index" not in st.session_state:
        st.session_state.selected_input_device_index = None
    if "selected_input_device_name" not in st.session_state:
        st.session_state.selected_input_device_name = "auto"
    if "sample_rate" not in st.session_state:
        st.session_state.sample_rate = CONFIG.sample_rate
    if "mic_mode" not in st.session_state:
        st.session_state.mic_mode = CONFIG.mic_mode
    if "mic_chunk_seconds" not in st.session_state:
        st.session_state.mic_chunk_seconds = CONFIG.mic_chunk_seconds
    if "energy_threshold" not in st.session_state:
        st.session_state.energy_threshold = CONFIG.energy_threshold
    if "confidence_threshold" not in st.session_state:
        st.session_state.confidence_threshold = CONFIG.confidence_threshold
    if "mic_test_result" not in st.session_state:
        st.session_state.mic_test_result = None
    if "mic_test_pending" not in st.session_state:
        st.session_state.mic_test_pending = False
    if "pause_listener_while_voice" not in st.session_state:
        st.session_state.pause_listener_while_voice = True

    if "listener" not in st.session_state:
        listener_cfg = ListenerConfig(
            demo_tick_seconds=CONFIG.demo_tick_seconds,
            demo_source_path=CONFIG.demo_source_path,
            whisper_model_size=CONFIG.whisper_model_size,
            whisper_device=CONFIG.whisper_device,
            sample_rate=CONFIG.sample_rate,
            mic_mode=CONFIG.mic_mode,
            mic_chunk_seconds=CONFIG.mic_chunk_seconds,
            energy_threshold=CONFIG.energy_threshold,
            confidence_threshold=CONFIG.confidence_threshold,
            input_device_index=None,
        )
        st.session_state.listener = ListenerAgent(
            cfg=listener_cfg,
            emit_event=st.session_state.event_queue.put,
        )

    if "summary_agent" not in st.session_state:
        st.session_state.summary_agent = SummaryAgent(
            memory=st.session_state.memory,
            llm=st.session_state.llm,
            update_seconds=CONFIG.summary_update_seconds,
            emit_event=st.session_state.event_queue.put,
        )

    if "guide" not in st.session_state:
        st.session_state.guide = GuideAgent(memory=st.session_state.memory, llm=st.session_state.llm)

    st.session_state.status["openai"] = "found" if CONFIG.openai_api_key else "missing"


def _append_capped(lst: list[dict], item: dict, max_items: int = 200) -> None:
    lst.append(item)
    if len(lst) > max_items:
        del lst[:-max_items]


def drain_worker_events() -> None:
    event_queue: queue.Queue = st.session_state.event_queue
    logger: JsonlLogger = st.session_state.logger

    while True:
        try:
            event = event_queue.get_nowait()
        except queue.Empty:
            break

        evt_type = event.get("type")
        if evt_type == "status":
            agent = str(event.get("agent", "listener"))
            message = str(event.get("message", ""))
            st.session_state.status[agent] = message
            _append_capped(st.session_state.pending_status_messages, event)
            continue

        if evt_type == "transcript_chunk":
            chunk = {
                "timestamp": str(event.get("timestamp", "")),
                "speaker": str(event.get("speaker", "lecturer")),
                "text": str(event.get("text", "")).strip(),
                "source": str(event.get("source", "listener")),
            }
            if chunk["text"]:
                _append_capped(st.session_state.transcript, chunk, CONFIG.transcript_buffer_size)
                st.session_state.memory.add_chunk(
                    speaker=chunk["speaker"],
                    text=chunk["text"],
                    source=chunk["source"],
                )
                _append_capped(st.session_state.pending_transcript_chunks, chunk)
                log_payload = {
                    "mode": st.session_state.app_mode,
                    "speaker": chunk["speaker"],
                    "chunk_timestamp": chunk["timestamp"],
                    "text": chunk["text"],
                    "input_device_name": st.session_state.selected_input_device_name,
                }
                _append_capped(
                    st.session_state.pending_log_events,
                    {"event_type": "transcript_chunk", "payload": log_payload},
                )
                logger.log("transcript_chunk", log_payload)
            continue

        if evt_type == "summary_update":
            summary = str(event.get("text", "")).strip()
            if summary:
                st.session_state.rolling_summary = summary
                st.session_state.memory.set_summary(summary)
                _append_capped(st.session_state.pending_summary_updates, event)
                log_payload = {
                    "mode": st.session_state.app_mode,
                    "summary": summary,
                    "input_device_name": st.session_state.selected_input_device_name,
                }
                _append_capped(
                    st.session_state.pending_log_events,
                    {"event_type": "summary_refresh", "payload": log_payload},
                )
                logger.log("summary_refresh", log_payload)
            continue

        if evt_type == "mic_test_result":
            st.session_state.mic_test_result = {
                "timestamp": str(event.get("timestamp", "")),
                "success": bool(event.get("success", False)),
                "result": str(event.get("result", "")),
            }
            st.session_state.mic_test_pending = False
            logger.log(
                "mic_test",
                {
                    "mode": st.session_state.app_mode,
                    "input_device_name": st.session_state.selected_input_device_name,
                    **st.session_state.mic_test_result,
                },
            )
            continue


def ensure_agents_running() -> None:
    listener: ListenerAgent = st.session_state.listener
    summary_agent: SummaryAgent = st.session_state.summary_agent

    if st.session_state.run_agents:
        listener.start(mode=st.session_state.app_mode)
        summary_agent.start()
    else:
        listener.stop()
        summary_agent.stop()
        st.session_state.status["listener"] = "stopped"
        st.session_state.status["summary"] = "stopped"


def render_status() -> None:
    status = st.session_state.status
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Mode", st.session_state.app_mode.upper())
    c2.metric("Listener", status["listener"])
    c3.metric("Summary Agent", status["summary"])
    c4.metric("Guide Agent", status["guide"])
    c5.metric("OPENAI_API_KEY", status["openai"])

    dotenv_state = "loaded" if CONFIG.dotenv_loaded else "not found"
    st.caption(f".env status: {dotenv_state}. Path: `{CONFIG.dotenv_path or 'none'}`")


def render_transcript() -> None:
    st.subheader("Transcript")
    chunks = st.session_state.transcript
    if not chunks:
        st.info("Waiting for transcript...")
        return

    lines: list[str] = []
    for c in chunks[-120:]:
        ts = c["timestamp"].split("T")[-1][:8] if c["timestamp"] else "--:--:--"
        safe_text = html.escape(c["text"])
        safe_speaker = html.escape(c["speaker"])
        lines.append(f'<div class="transcript-line"><span class="ts">{ts}</span> <strong>{safe_speaker}:</strong> {safe_text}</div>')

    transcript_html = "\n".join(lines)
    st.markdown(
        f"""
        <style>
          .transcript-box {{
            height: 420px;
            overflow-y: auto;
            border: 1px solid rgba(128,128,128,0.35);
            border-radius: 8px;
            padding: 0.6rem 0.7rem;
            background: rgba(248,249,250,0.5);
          }}
          .transcript-line {{
            margin-bottom: 0.3rem;
            line-height: 1.35;
            font-size: 0.95rem;
          }}
          .transcript-line .ts {{
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
            color: #666;
            margin-right: 0.4rem;
          }}
        </style>
        <div id="transcript-box" class="transcript-box">{transcript_html}</div>
        <script>
          const box = window.parent.document.getElementById("transcript-box");
          if (box) {{ box.scrollTop = box.scrollHeight; }}
        </script>
        """,
        unsafe_allow_html=True,
    )


def render_summary() -> None:
    st.subheader("Rolling Summary")
    st.markdown(st.session_state.rolling_summary or "No summary yet.")


def handle_chat() -> None:
    st.subheader("Chat (silent user questions)")
    for m in st.session_state.chat_history:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    st.caption("Voice question uses a separate user channel and is not added to the lecture transcript.")
    st.session_state.pause_listener_while_voice = st.checkbox(
        "Pause listener while recording voice question",
        value=st.session_state.pause_listener_while_voice,
    )

    with st.form("text_question_form", clear_on_submit=True):
        text_query = st.text_input("Type a question")
        text_submit = st.form_submit_button("Submit text question")
    if text_submit and text_query.strip():
        _process_user_question(text_query.strip(), query_source="text")

    voice_audio = st.audio_input("Record a voice question")
    voice_submit = st.button("Submit voice question")
    if voice_submit:
        if voice_audio is None:
            st.warning("No voice recording found. Record a question first.")
            return
        _process_voice_question(voice_audio)


def _process_voice_question(voice_audio: object) -> None:
    llm: OpenAIHelper = st.session_state.llm
    listener: ListenerAgent = st.session_state.listener
    should_resume = False

    try:
        if st.session_state.pause_listener_while_voice and st.session_state.run_agents:
            listener.stop()
            st.session_state.status["listener"] = "Paused for voice question capture/transcription"
            should_resume = True

        if not llm.enabled:
            st.warning("Voice transcription requires OPENAI_API_KEY. Use text input or configure key.")
            return

        audio_bytes = voice_audio.getvalue()  # type: ignore[attr-defined]
        transcript_text = llm.transcribe_audio_bytes(audio_bytes, filename="voice_question.wav")
        if not transcript_text:
            st.warning("Voice transcription failed or was empty. Try again or use text input.")
            return

        user_display = f"[voice] {transcript_text}"
        st.session_state.chat_history.append({"role": "user", "content": user_display})
        with st.chat_message("user"):
            st.markdown(user_display)
        _process_user_question(transcript_text, query_source="voice", voice_transcript=transcript_text)
    except Exception as e:  # noqa: BLE001
        st.warning(f"Voice question failed: {e}")
    finally:
        if should_resume and st.session_state.run_agents:
            listener.start(mode=st.session_state.app_mode)
            st.session_state.status["listener"] = "Listener resumed after voice question"


def _process_user_question(
    user_query: str,
    *,
    query_source: str,
    voice_transcript: str | None = None,
) -> None:
    if query_source != "voice":
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

    start = time.perf_counter()
    answer = st.session_state.guide.answer(user_query)
    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    st.session_state.status["guide"] = f"answered in {latency_ms} ms"
    st.session_state.chat_history.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.logger.log(
        "qa",
        {
            "transcript": st.session_state.memory.get_recent_text(40),
            "summary": st.session_state.memory.get_summary(),
            "query": user_query,
            "query_source": query_source,
            "voice_transcript": voice_transcript or "",
            "response": answer,
            "latency_ms": latency_ms,
            "mode": st.session_state.app_mode,
            "input_device_name": st.session_state.selected_input_device_name,
        },
    )


def render_controls() -> None:
    st.sidebar.header("Controls")
    st.sidebar.info("For quick demos, play lecture audio from a phone speaker near your laptop mic.")

    demo_mode = st.sidebar.toggle("Demo mode", value=(st.session_state.app_mode == "demo"))
    st.session_state.app_mode = "demo" if demo_mode else "live"

    start_col, stop_col = st.sidebar.columns(2)
    start_clicked = start_col.button("Start agents", use_container_width=True)
    stop_clicked = stop_col.button("Stop agents", use_container_width=True)

    if start_clicked:
        st.session_state.run_agents = True
        st.session_state.status["listener"] = "starting..."
        st.session_state.status["summary"] = "starting..."
    if stop_clicked:
        st.session_state.run_agents = False
        st.session_state.status["listener"] = "stopping..."
        st.session_state.status["summary"] = "stopping..."

    st.sidebar.caption(
        f"Agents are currently **{'running' if st.session_state.run_agents else 'stopped'}**."
    )

    listener: ListenerAgent = st.session_state.listener
    devices = listener.list_input_devices()
    if devices:
        label_to_idx = {f"{d['name']} (idx {d['index']})": d["index"] for d in devices}
        labels = list(label_to_idx.keys())
        selected = st.session_state.selected_input_device_index
        default_index = 0
        if selected is not None:
            for i, d in enumerate(devices):
                if d["index"] == selected:
                    default_index = i
                    break
        selected_label = st.sidebar.selectbox(
            "Microphone input",
            options=labels,
            index=default_index,
            disabled=st.session_state.app_mode == "demo",
        )
        st.session_state.selected_input_device_index = label_to_idx[selected_label]
        st.session_state.selected_input_device_name = selected_label
    else:
        st.session_state.selected_input_device_index = None
        st.session_state.selected_input_device_name = "none"
        st.sidebar.error("No valid input device found. LIVE mode will not work.")

    with st.sidebar.expander("Advanced live settings", expanded=False):
        mic_mode_labels = {
            "Voice (normal)": "voice",
            "Lecture demo (speaker playback)": "lecture_demo",
        }
        current_label = next(
            (label for label, value in mic_mode_labels.items() if value == st.session_state.mic_mode),
            "Voice (normal)",
        )
        chosen_label = st.selectbox(
            "Mic mode",
            options=list(mic_mode_labels.keys()),
            index=list(mic_mode_labels.keys()).index(current_label),
            disabled=st.session_state.app_mode == "demo",
        )
        st.session_state.mic_mode = mic_mode_labels[chosen_label]

        st.session_state.sample_rate = st.number_input(
            "Sample rate (Hz)", min_value=8000, max_value=48000, step=1000, value=int(st.session_state.sample_rate)
        )
        st.session_state.mic_chunk_seconds = st.slider(
            "Chunk duration (seconds)", min_value=1.0, max_value=8.0, value=float(st.session_state.mic_chunk_seconds), step=0.5
        )
        st.session_state.energy_threshold = st.slider(
            "Energy threshold", min_value=0.001, max_value=0.05, value=float(st.session_state.energy_threshold), step=0.001
        )
        st.session_state.confidence_threshold = st.slider(
            "Confidence threshold (avg_logprob)",
            min_value=-3.0,
            max_value=0.0,
            value=float(st.session_state.confidence_threshold),
            step=0.1,
        )

    settings_changed = listener.update_live_settings(
        input_device_index=st.session_state.selected_input_device_index,
        sample_rate=int(st.session_state.sample_rate),
        mic_mode=str(st.session_state.mic_mode),
        mic_chunk_seconds=float(st.session_state.mic_chunk_seconds),
        energy_threshold=float(st.session_state.energy_threshold),
        confidence_threshold=float(st.session_state.confidence_threshold),
    )
    if settings_changed:
        st.session_state.logger.log(
            "live_settings",
            {
                "mode": st.session_state.app_mode,
                "input_device_name": st.session_state.selected_input_device_name,
                "input_device_index": st.session_state.selected_input_device_index,
                "sample_rate": st.session_state.sample_rate,
                "mic_mode": st.session_state.mic_mode,
                "chunk_seconds": st.session_state.mic_chunk_seconds,
                "energy_threshold": st.session_state.energy_threshold,
                "confidence_threshold": st.session_state.confidence_threshold,
            },
        )
        if st.session_state.run_agents and st.session_state.app_mode == "live":
            listener.stop()

    if st.sidebar.button("Test microphone for 5 seconds", disabled=st.session_state.app_mode == "demo"):
        listener.start_microphone_test(duration_seconds=5.0)
        st.session_state.mic_test_pending = True
        st.session_state.status["listener"] = "Running 5-second microphone test..."

    mic_result = st.session_state.mic_test_result
    if mic_result:
        if mic_result["success"]:
            st.sidebar.success(f"Mic test transcript: {mic_result['result']}")
        else:
            st.sidebar.warning(mic_result["result"])

    if st.sidebar.button("Refresh summary now"):
        st.session_state.summary_agent.refresh_once()

    st.sidebar.caption(f"Logs: `{st.session_state.logs_path}`")


init_session_state()
drain_worker_events()
st.title(CONFIG.app_title)
render_controls()
ensure_agents_running()
render_status()

left, right = st.columns([2, 1])
with left:
    render_transcript()
with right:
    render_summary()

st.divider()
handle_chat()

if st.session_state.run_agents or st.session_state.mic_test_pending:
    time.sleep(1.0)
    st.rerun()
