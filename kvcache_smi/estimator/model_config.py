from __future__ import annotations

import json
from pathlib import Path

from kvcache_smi.core.schemas import ModelConfig


def load_model_config(config_path: str) -> ModelConfig:
    data = json.loads(Path(config_path).read_text(encoding="utf-8"))
    return ModelConfig(
        name=data.get("name", Path(config_path).stem),
        num_layers=int(data["num_layers"]),
        hidden_size=int(data["hidden_size"]),
        num_attention_heads=int(data["num_attention_heads"]),
        num_key_value_heads=(
            int(data["num_key_value_heads"])
            if data.get("num_key_value_heads") is not None
            else None
        ),
        vocab_size=data.get("vocab_size"),
        max_position_embeddings=data.get("max_position_embeddings"),
    )
