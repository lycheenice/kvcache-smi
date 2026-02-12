# Hook Strategy

## 推荐模式
- `minimal`: 低频采样，仅总量/峰值。
- `balanced`: 中频采样 + layer-wise（默认）。
- `diagnostic`: 高频采样 + 详细事件日志。

## 实现建议
1. 优先在 attention 模块的 forward 前后记录缓存变化。
2. 将 tensor -> bytes 计算集中在一个 utility，避免重复。
3. 对短时间多事件做批量聚合，减少 Python 回调开销。

## 常见问题
- Hook 多次注册：需要去重并提供 unregister。
- DDP/多进程：每进程独立采集，汇总层统一对齐时间戳。
