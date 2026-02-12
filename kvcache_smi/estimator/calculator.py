from __future__ import annotations

from kvcache_smi.core.schemas import InferenceConfig, KVCacheEstimate, ModelConfig

PRECISION_TO_BYTES = {
    "fp32": 4,
    "fp16": 2,
    "bf16": 2,
    "int8": 1,
}


def estimate_kv_cache_memory(
    model: ModelConfig,
    inference: InferenceConfig,
    memory_budget_bytes: int | None = None,
    overhead_factor: float = 1.0,
) -> KVCacheEstimate:
    if model.hidden_size % model.num_attention_heads != 0:
        raise ValueError("hidden_size must be divisible by num_attention_heads")

    precision_key = (inference.kv_cache_dtype or inference.precision).lower()
    if precision_key not in PRECISION_TO_BYTES:
        raise ValueError(f"Unsupported precision: {precision_key}")

    precision_bytes = PRECISION_TO_BYTES[precision_key]
    head_dim = model.hidden_size // model.num_attention_heads
    kv_hidden_size = model.effective_num_kv_heads * head_dim

    per_token_bytes = int(2 * model.num_layers * kv_hidden_size * precision_bytes * overhead_factor)
    total_bytes = per_token_bytes * inference.max_seq_length * inference.batch_size
    per_layer_bytes = int(
        2
        * kv_hidden_size
        * inference.max_seq_length
        * inference.batch_size
        * precision_bytes
        * overhead_factor
    )

    max_batch = None
    max_seq = None
    if memory_budget_bytes is not None and memory_budget_bytes > 0:
        per_sequence_bytes = per_token_bytes * inference.max_seq_length
        per_token_all_batch_bytes = per_token_bytes * max(inference.batch_size, 1)
        max_batch = memory_budget_bytes // max(per_sequence_bytes, 1)
        max_seq = memory_budget_bytes // max(per_token_all_batch_bytes, 1)

    return KVCacheEstimate(
        total_bytes=total_bytes,
        total_mib=total_bytes / (1024**2),
        total_gib=total_bytes / (1024**3),
        per_token_bytes=per_token_bytes,
        per_layer_bytes=per_layer_bytes,
        max_batch_under_budget=max_batch,
        max_seq_length_under_budget=max_seq,
    )


def build_recommendations(model: ModelConfig, inference: InferenceConfig, estimate: KVCacheEstimate) -> list[str]:
    recs: list[str] = []
    if model.num_key_value_heads is None or model.num_key_value_heads == model.num_attention_heads:
        recs.append("可考虑 GQA/MQA 减少 KV Cache 占用。")
    if inference.precision.lower() in {"fp16", "bf16", "fp32"}:
        recs.append("可考虑 INT8 KV Cache 量化，理论可再降低约 50% 内存。")
    if estimate.total_gib > 8:
        recs.append("KV Cache 超过 8GiB，建议评估 CPU Offload 或缩短上下文长度。")
    return recs
