# KVCache-SMI 架构评审与编码落地建议（v0.1）

## 结论

当前架构设计方向**总体合理**，尤其是“Estimator / Monitor / Benchmark”三模块拆分、非侵入式集成和分阶段路线图，符合该类性能工具的产品演进规律。

但要进入可编码阶段，仍有 4 个关键缺口需要优先补齐：
1. 缺少统一数据模型（不同模块产出难以对齐）
2. Monitor 观测边界不清（Hook 精度与性能开销尚未定标）
3. Benchmark 与真实推理场景的映射规则不完整
4. “框架无关”目标与实现复杂度之间缺少 MVP 收敛策略

## 架构合理性分析

### 1) 模块边界清晰，职责分离合理
- Estimator 负责静态建模，Monitor 负责运行态观测，Benchmark 负责链路能力测量，三者组合形成“预测-观测-验证”的闭环。
- 该拆分可减少早期复杂度，支持并行开发和渐进增强。

### 2) 技术选型匹配项目阶段
- Python 优先、可选 C++/CUDA 的策略适合早期快速验证。
- 非侵入式集成（hook / 配置）对用户迁移成本低，利于扩散。

### 3) 路线图具备可执行性
- Phase 1 聚焦 Estimator + Monitor 的基础能力，先建立可用 MVP，再逐步引入 Benchmark 与深度集成，节奏合理。

## 主要风险与改进建议

### 风险 A：统一指标与数据结构缺失
**问题**：文档定义了多种输出格式（JSON/CSV/HTML），但未定义统一 schema（如采样点、层级维度、时间戳、设备字段）。

**建议**：
- 先定义 `MetricRecord` 与 `RunMetadata` 两套核心数据结构。
- Estimator/Monitor/Benchmark 都先输出同一 JSON schema，报告层再做格式转换。

### 风险 B：Monitor 准确性与开销目标冲突
**问题**：目标给出“监控开销 <5%”，但 hook 粒度、采样频率、与 `torch.cuda.memory_stats()` 组合策略未细化。

**建议**：
- 明确三档观测模式：`minimal`、`balanced`、`diagnostic`。
- 在 CI 中建立基准测试：分别验证开销、采样丢失率和内存额外占用。

### 风险 C：Benchmark 与业务场景脱节
**问题**：当前 Benchmark 偏“链路带宽基准”，缺少与真实 KV offload 策略的映射（例如 chunk 大小、异步 copy、流水化并发）。

**建议**：
- 引入“场景预设”：`decode_small_batch`、`long_context_offload` 等。
- 输出“可行动建议”而不是仅输出带宽数字（如推荐 chunk 大小区间）。

### 风险 D：框架无关目标过大
**问题**：同时覆盖 PyTorch、Transformers、vLLM、TensorRT-LLM，短期容易分散资源。

**建议**：
- MVP 只锁定 `PyTorch + Transformers`。
- 通过 `adapter` 接口隔离后端差异，为 vLLM/TensorRT-LLM 留扩展点。

## 建议的最小可编码蓝图（MVP）

### 包结构建议
```text
kvcache_smi/
  estimator/
  monitor/
  benchmark/
  core/
    schemas.py
    reporter.py
    device.py
  cli/
    main.py
```

### Phase 1 交付门槛
1. `estimate` 命令可读取 HuggingFace config 并输出统一 JSON
2. `monitor` 命令可对标准 attention 模块采样并输出时间序列
3. 两个命令共享同一 schema 与报告导出层
4. 提供 2 个端到端示例（7B/70B）

## 编码所需技能（Skills）规划

已在仓库中新增以下技能包，用于指导后续编码：
1. `kvcache-cli-api-bootstrap`：项目脚手架、统一 schema、CLI/API 约定
2. `kvcache-estimator-impl`：Estimator 公式实现、配置解析、误差校验
3. `kvcache-monitor-impl`：Monitor hook 策略、分层统计、开销控制
4. `kvcache-benchmark-impl`：带宽测试矩阵、场景映射、报告生成

> 以上技能都采用 “SKILL.md + references/” 的结构，便于按需加载，避免上下文膨胀。

## 里程碑建议（4 周）
- 第 1 周：CLI 骨架 + schema + Estimator MVP
- 第 2 周：Monitor MVP + 开销基准
- 第 3 周：Benchmark MVP + 预设场景
- 第 4 周：统一报告 + 示例与文档

