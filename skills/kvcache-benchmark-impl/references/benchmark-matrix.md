# Benchmark Matrix

## P0 必测
1. GPU->CPU（pinned / non-pinned）
2. CPU->GPU（pinned / non-pinned）
3. GPU->GPU（同机，若支持 P2P）

## 参数建议
- 数据大小：16MB, 64MB, 256MB, 1GB
- 预热：10 次
- 正式迭代：100 次
- 输出：mean, p50, p95, std

## 场景预设
- `decode_small_batch`: 小 batch 高频传输，关注时延抖动
- `long_context_offload`: 大块传输，关注持续吞吐

## 报告建议字段
- `recommended_chunk_mb`
- `use_pinned_memory`
- `expected_latency_impact`
