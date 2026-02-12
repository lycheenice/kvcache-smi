# Project Layout Reference

## 推荐目录
```text
kvcache_smi/
  core/
    schemas.py
    reporter.py
    device.py
  estimator/
    calculator.py
    model_config.py
  monitor/
    hooks.py
    collector.py
  benchmark/
    transfer.py
    scenarios.py
  cli/
    main.py
```

## 统一数据模型（建议）
- `RunMetadata`: run_id, tool_version, timestamp, device, backend
- `MetricRecord`: name, value, unit, tags, step, ts
- `AnalysisReport`: metadata, metrics, summary, recommendations

## CLI 参数约定
- 输入：`--model`, `--config`, `--device`
- 输出：`--output`, `--format json|csv|html`
- 运行：`--interval`, `--duration`, `--warmup`
