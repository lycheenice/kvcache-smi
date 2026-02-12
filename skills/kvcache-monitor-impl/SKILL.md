---
name: kvcache-monitor-impl
description: Implement runtime KV cache monitor with PyTorch hooks, layer-wise metrics, and overhead control profiles.
---

# KVCache Monitor Implementation

## 何时使用
- 开发推理时 KV Cache 实时监控
- 构建 attention 层 hook 与分层统计
- 优化采样开销并做精度/性能权衡

## 执行步骤
1. 先看 `references/hook-strategy.md`，选择 hook 注入策略。
2. 先实现 `balanced` 模式（默认），再补充 `minimal/diagnostic`。
3. 采集四类指标：当前占用、峰值占用、增长速率、layer-wise 占用。
4. 使用统一 schema 输出，附带采样间隔与模式元信息。
5. 通过基准脚本测量 monitor 开销，目标开销 <5%。

## 验收标准
- 默认模式可在常见 transformers attention 模块工作
- 监控指标可导出时间序列
- 有开销基准结果（吞吐下降/延迟上升）
