from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .memory import utc_now_iso


class JsonlLogger:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event_type: str, payload: dict[str, Any]) -> None:
        record = {
            "timestamp": utc_now_iso(),
            "event_type": event_type,
            **payload,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=True) + "\n")
