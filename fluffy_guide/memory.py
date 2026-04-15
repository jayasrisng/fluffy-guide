from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Deque


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class TranscriptChunk:
    timestamp: str
    speaker: str
    text: str
    source: str = "listener"


class RollingMemory:
    def __init__(self, max_chunks: int = 120) -> None:
        self._chunks: Deque[TranscriptChunk] = deque(maxlen=max_chunks)
        self._summary: str = "No summary yet."
        self._lock = threading.Lock()

    def add_chunk(self, speaker: str, text: str, source: str = "listener") -> TranscriptChunk:
        chunk = TranscriptChunk(
            timestamp=utc_now_iso(),
            speaker=speaker,
            text=text.strip(),
            source=source,
        )
        if not chunk.text:
            return chunk
        with self._lock:
            self._chunks.append(chunk)
        return chunk

    def get_chunks(self) -> list[TranscriptChunk]:
        with self._lock:
            return list(self._chunks)

    def get_recent_text(self, n: int = 20) -> str:
        with self._lock:
            chunks = list(self._chunks)[-n:]
        return "\n".join(f"[{c.speaker}] {c.text}" for c in chunks)

    def set_summary(self, summary: str) -> None:
        with self._lock:
            self._summary = summary.strip() or self._summary

    def get_summary(self) -> str:
        with self._lock:
            return self._summary
