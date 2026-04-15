from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import find_dotenv, load_dotenv


DOTENV_PATH = find_dotenv(usecwd=True)
DOTENV_LOADED = load_dotenv(dotenv_path=DOTENV_PATH, override=False)


@dataclass(frozen=True)
class AppConfig:
    app_title: str = "fluffy-guide"
    transcript_buffer_size: int = int(os.getenv("TRANSCRIPT_BUFFER_SIZE", "120"))
    summary_update_seconds: float = float(os.getenv("SUMMARY_UPDATE_SECONDS", "8"))
    demo_tick_seconds: float = float(os.getenv("DEMO_TICK_SECONDS", "1.5"))
    demo_source_path: Path = Path(
        os.getenv("DEMO_SOURCE_PATH", "data/demo_transcript.jsonl")
    )
    log_dir: Path = Path(os.getenv("LOG_DIR", "logs"))
    dotenv_loaded: bool = DOTENV_LOADED
    dotenv_path: str = DOTENV_PATH
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_stt_model: str = os.getenv("OPENAI_STT_MODEL", "gpt-4o-mini-transcribe")
    whisper_model_size: str = os.getenv("WHISPER_MODEL_SIZE", "tiny")
    whisper_device: str = os.getenv("WHISPER_DEVICE", "cpu")
    sample_rate: int = int(os.getenv("SAMPLE_RATE", "16000"))
    mic_mode: str = os.getenv("MIC_MODE", "voice")
    mic_chunk_seconds: float = float(os.getenv("MIC_CHUNK_SECONDS", "3.0"))
    energy_threshold: float = float(os.getenv("ENERGY_THRESHOLD", "0.008"))
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "-1.1"))


CONFIG = AppConfig()
CONFIG.log_dir.mkdir(parents=True, exist_ok=True)
