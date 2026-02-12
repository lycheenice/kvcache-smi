from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any


def _to_obj(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    return value


def render_json(payload: Any) -> str:
    return json.dumps(_to_obj(payload), ensure_ascii=False, indent=2)


def write_output(payload: Any, output: str | None = None) -> str:
    content = render_json(payload)
    if output:
        Path(output).write_text(content, encoding="utf-8")
    return content
