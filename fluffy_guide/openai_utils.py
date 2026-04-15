from __future__ import annotations

import io
from typing import Any

from openai import OpenAI


class OpenAIHelper:
    def __init__(self, api_key: str | None, model: str, stt_model: str) -> None:
        self.model = model
        self.stt_model = stt_model
        self._client = OpenAI(api_key=api_key) if api_key else None

    @property
    def enabled(self) -> bool:
        return self._client is not None

    def complete(self, system_prompt: str, user_prompt: str, max_output_tokens: int = 240) -> str:
        if not self._client:
            return ""

        response = self._client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_output_tokens=max_output_tokens,
            temperature=0.2,
        )

        text = _extract_output_text(response)
        return text.strip()

    def transcribe_audio_bytes(self, audio_bytes: bytes, filename: str = "voice_question.wav") -> str:
        if not self._client:
            return ""
        if not audio_bytes:
            return ""

        file_like = io.BytesIO(audio_bytes)
        file_like.name = filename
        transcript = self._client.audio.transcriptions.create(
            model=self.stt_model,
            file=file_like,
        )
        text = getattr(transcript, "text", "")
        return str(text).strip()


def _extract_output_text(response: Any) -> str:
    if hasattr(response, "output_text") and response.output_text:
        return response.output_text

    output = getattr(response, "output", None)
    if not output:
        return ""

    chunks: list[str] = []
    for item in output:
        content = getattr(item, "content", None)
        if not content:
            continue
        for c in content:
            text = getattr(c, "text", None)
            if text:
                chunks.append(text)
    return "\n".join(chunks)
