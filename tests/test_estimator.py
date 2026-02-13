from kvcache_smi.core.schemas import InferenceConfig, ModelConfig
from kvcache_smi.estimator.calculator import estimate_kv_cache_memory


def test_llama2_7b_estimate_close_to_1gib():
    model = ModelConfig(
        name="llama2-7b",
        num_layers=32,
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=32,
    )
    inference = InferenceConfig(max_seq_length=2048, batch_size=1, precision="fp16")

    estimate = estimate_kv_cache_memory(model, inference)

    assert 0.95 <= estimate.total_gib <= 1.05


def test_gqa_reduces_memory():
    mha_model = ModelConfig(
        name="mha",
        num_layers=28,
        hidden_size=3584,
        num_attention_heads=28,
        num_key_value_heads=28,
    )
    gqa_model = ModelConfig(
        name="gqa",
        num_layers=28,
        hidden_size=3584,
        num_attention_heads=28,
        num_key_value_heads=4,
    )
    inference = InferenceConfig(max_seq_length=4096, batch_size=4, precision="fp16")

    mha_est = estimate_kv_cache_memory(mha_model, inference)
    gqa_est = estimate_kv_cache_memory(gqa_model, inference)

    assert gqa_est.total_bytes < mha_est.total_bytes
    assert mha_est.total_bytes // gqa_est.total_bytes == 7
