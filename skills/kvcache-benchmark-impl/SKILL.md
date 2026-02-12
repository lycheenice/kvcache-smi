---
name: kvcache-benchmark-impl
description: Implement KV transfer and offload benchmarks with scenario presets and actionable optimization recommendations.
---

# KVCache Benchmark Implementation

## 何时使用
- 编写 GPU↔CPU / GPU↔GPU 传输带宽测试
- 评估 offload 对延迟与吞吐影响
- 将底层带宽结果映射到推理场景建议

## 执行步骤
1. 阅读 `references/benchmark-matrix.md`，优先实现 P0 测试矩阵。
2. 为每个测试定义：数据规模、迭代次数、预热策略、同步点。
3. 输出吞吐（GB/s）和时延（ms）分位值（P50/P95）。
4. 基于场景预设生成建议（chunk size、是否使用 pinned memory）。
5. 结果统一写入 schema，并支持 CSV 导出。

## 验收标准
- 包含至少 1 个 offload 场景预设
- 结果可复现（固定随机种子、固定 warmup）
- 报告包含“建议动作”字段
