from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DemoUtterance:
    speaker: str
    text: str


def load_demo_script(path: Path) -> list[DemoUtterance]:
    if not path.exists():
        return []

    utterances: list[DemoUtterance] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            speaker = str(obj.get("speaker", "lecturer"))
            text = str(obj.get("text", "")).strip()
            if text:
                utterances.append(DemoUtterance(speaker=speaker, text=text))
    return utterances
