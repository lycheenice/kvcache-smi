from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class RunMetadata:
    run_id: str
    tool_version: str = "0.1.0"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    device: str = "unknown"
    backend: str = "python"


@dataclass
class AnalysisReport:
    metadata: RunMetadata
    metrics: dict[str, Any]
    summary: dict[str, Any]
    recommendations: list[str]


@dataclass
class ModelConfig:
    name: str
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int | None = None
    vocab_size: int | None = None
    max_position_embeddings: int | None = None

    @property
    def effective_num_kv_heads(self) -> int:
        return self.num_key_value_heads or self.num_attention_heads


@dataclass
class InferenceConfig:
    max_seq_length: int
    batch_size: int
    precision: str
    use_kv_cache: bool = True
    kv_cache_dtype: str | None = None
    enable_offload: bool = False
    offload_device: str = "cpu"


@dataclass
class KVCacheEstimate:
    total_bytes: int
    total_mib: float
    total_gib: float
    per_token_bytes: int
    per_layer_bytes: int
    max_batch_under_budget: int | None = None
    max_seq_length_under_budget: int | None = None


@dataclass
class KVCacheMetrics:
    timestamp: float
    total_memory_bytes: int
    per_layer_memory: dict[int, int]
    num_cached_tokens: int
    peak_memory_bytes: int
    cache_hit_rate: float | None = None
    offloaded_bytes: int = 0
    transfer_bandwidth_gbps: float | None = None
    offload_latency_ms: float | None = None
    mode: str = "balanced"
