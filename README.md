# KVCache-SMI

一个用于大模型推理过程中 KV Cache 性能分析和优化的工具，类似于 `nvidia-smi` 对 GPU 的监控体验。

## v0.1 已实现能力

- ✅ **Estimator（静态估算）**
  - 支持 MHA/GQA/MQA（通过 `num_key_value_heads`）
  - 输出 `bytes / MiB / GiB`、`per-token`、`per-layer` 指标
  - 支持给定显存预算反推最大 batch / 最大序列长度
- ✅ **Monitor（基础监控）**
  - 提供 `minimal / balanced / diagnostic` 三种模式
  - 采集当前占用、峰值占用、增长速率、layer-wise 占用
  - v0.1 CLI 使用 mock 输入演示，Python API 可直接集成到真实推理代码
- ✅ **CLI 基础命令**
  - `kvcache-smi estimate`
  - `kvcache-smi monitor`
- ✅ **Python API（最小可用）**

## 快速开始

### 1) 安装（开发模式）

```bash
pip install -e .
```

### 2) 运行 Estimator

```bash
kvcache-smi estimate \
  --config examples/llama2_7b_config.json \
  --max-seq-length 4096 \
  --batch-size 8 \
  --precision fp16 \
  --memory-budget-bytes 25769803776
```

### 3) 运行 Monitor（v0.1 mock 示例）

```bash
kvcache-smi monitor \
  --mode balanced \
  --mock-layer-bytes '{"0": 4096, "1": 8192}' \
  --num-cached-tokens 128
```

## Python API 示例

```python
from kvcache_smi.core.schemas import InferenceConfig, ModelConfig
from kvcache_smi.estimator.calculator import estimate_kv_cache_memory
from kvcache_smi.monitor.collector import KVCacheMonitor

model = ModelConfig(
    name="demo",
    num_layers=32,
    hidden_size=4096,
    num_attention_heads=32,
    num_key_value_heads=32,
)
inference = InferenceConfig(max_seq_length=2048, batch_size=1, precision="fp16")
est = estimate_kv_cache_memory(model, inference)
print(est.total_gib)

monitor = KVCacheMonitor(mode="balanced")
monitor.capture({0: bytearray(1024), 1: bytearray(2048)}, num_cached_tokens=64)
print(monitor.summary())
```

## 项目结构

```text
kvcache_smi/
  core/
    schemas.py
    reporter.py
  estimator/
    model_config.py
    calculator.py
  monitor/
    hooks.py
    collector.py
  cli/
    main.py
tests/
examples/
```

## 文档

- [架构设计文档 v0.1](docs/architecture_v0.1.md)
- [Architecture Design Document v0.1 (English)](docs/architecture_v0.1_en.md)
- [架构评审与编码落地建议](docs/architecture_review_v0.1.md)
- [编码技能索引](docs/coding_skills_index.md)

## 后续规划

- Benchmark 模块（GPU↔CPU / GPU↔GPU / KV access pattern）
- 真正的 Transformer attention hook 自动注入
- HTML 报告与可视化
- vLLM / TensorRT-LLM 集成

## 许可证

Apache License 2.0，见 [LICENSE](LICENSE)。
