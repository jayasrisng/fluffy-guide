from __future__ import annotations

import threading
import time
from typing import Any, Callable

from .memory import RollingMemory, utc_now_iso
from .openai_utils import OpenAIHelper


EmitEvent = Callable[[dict[str, Any]], None]


class SummaryAgent:
    def __init__(
        self,
        memory: RollingMemory,
        llm: OpenAIHelper,
        update_seconds: float,
        emit_event: EmitEvent | None = None,
    ) -> None:
        self.memory = memory
        self.llm = llm
        self.update_seconds = update_seconds
        self.emit_event = emit_event

        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.is_running:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=0.5)
        self._thread = None

    def _run(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.refresh_once()
            except Exception as e:  # noqa: BLE001
                self._emit_status(f"Summary update failed: {e}")
            time.sleep(self.update_seconds)

    def refresh_once(self) -> None:
        recent = self.memory.get_recent_text(40)
        if not recent:
            return

        if not self.llm.enabled:
            lines = [line for line in recent.splitlines()[-5:] if line.strip()]
            summary = f"Recent lecture points: {' | '.join(lines)[:600]}"
            self._emit_summary(summary)
            return

        system = (
            "You are Summary Agent for a real-time lecture copilot. "
            "Write a compact rolling summary with 3 to 5 bullet points. "
            "Focus on factual lecture content and definitions."
        )
        user = f"Recent transcript:\n{recent}\n\nReturn concise bullets only."
        summary = self.llm.complete(system, user, max_output_tokens=220)
        if summary:
            self._emit_status("Summary refreshed")
            self._emit_summary(summary)

    def _emit_status(self, message: str) -> None:
        self._emit_event(
            {
                "type": "status",
                "agent": "summary",
                "message": message,
                "timestamp": utc_now_iso(),
            }
        )

    def _emit_summary(self, summary: str) -> None:
        self._emit_event(
            {
                "type": "summary_update",
                "text": summary,
                "timestamp": utc_now_iso(),
            }
        )

    def _emit_event(self, event: dict[str, Any]) -> None:
        if self.emit_event:
            self.emit_event(event)


class GuideAgent:
    def __init__(self, memory: RollingMemory, llm: OpenAIHelper) -> None:
        self.memory = memory
        self.llm = llm

    def answer(self, user_query: str) -> str:
        recent = self.memory.get_recent_text(30)
        summary = self.memory.get_summary()

        if not recent.strip():
            return "I don't have lecture content yet."

        if not self.llm.enabled:
            return self._fallback_answer(user_query=user_query, recent=recent, summary=summary)

        system = (
            "You are Guide Agent in a lecture copilot. "
            "Answer only the user, never the lecturer. "
            "Use only provided context and be concise, grounded, and direct. "
            "If context is insufficient, say so briefly."
        )
        user = (
            f"Lecture summary:\n{summary}\n\n"
            f"Recent transcript:\n{recent}\n\n"
            f"User question: {user_query}\n\n"
            "Respond in <=120 words."
        )
        answer = self.llm.complete(system, user, max_output_tokens=220)
        if not answer:
            return "I couldn't produce a grounded answer from the current context."
        return answer

    def _fallback_answer(self, user_query: str, recent: str, summary: str) -> str:
        last_lines = "\n".join(recent.splitlines()[-3:])
        return (
            "OpenAI API key not configured, so this is a heuristic answer. "
            f"Question: {user_query}. "
            f"Latest context: {last_lines}. "
            f"Summary: {summary[:220]}"
        )
