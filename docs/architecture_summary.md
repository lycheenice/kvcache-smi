# 架构设计文档 v0.1 - 核心要点总结

本文档是对完整架构设计文档的精简总结，便于快速浏览和讨论。

## 核心功能（三大模块）

### 1. Estimator（估算器）
**目的**：静态估算 KV Cache 内存需求

**主要功能**：
- 基于模型配置（层数、隐藏维度、注意力头数等）计算预期内存
- 分析不同推理配置（序列长度、批次大小、精度）的影响
- 给出最大可支持 batch size 和序列长度
- 提供优化建议（GQA、量化、offload 等）

**典型使用场景**：
- 部署前评估硬件需求
- 优化推理配置
- 成本规划

### 2. Monitor（监控器）
**目的**：实时追踪推理过程中的 KV Cache 使用

**主要功能**：
- 通过 PyTorch hooks 追踪 KV Cache 的创建、更新、释放
- 分层统计（每层的 KV Cache 占用）
- 记录峰值内存和内存增长趋势
- 区分 KV Cache 与其他内存使用

**典型使用场景**：
- 诊断内存问题
- 验证估算准确性
- 生产环境监控

### 3. Benchmark（基准测试）
**目的**：测量数据传输带宽和性能

**主要功能**：
- GPU↔CPU 传输带宽测试（PCIe）
- GPU↔GPU 传输带宽测试（NVLink/PCIe）
- 模拟 KV Cache offload 场景
- 评估 offload 对推理延迟的影响

**典型使用场景**：
- 评估 offload 可行性
- 硬件性能对比
- 优化传输策略

## 关键设计决策

### 1. 技术选型
- **Python 为主**：易于集成深度学习生态，快速开发
- **可选 C++/CUDA**：性能关键路径优化
- **轻量级设计**：监控开销 < 5%，额外内存 < 100MB

### 2. 集成策略
- **非侵入式**：通过 hooks 和配置文件集成，不修改模型代码
- **框架无关**：支持 PyTorch、Transformers 以及 vLLM、TensorRT-LLM 等
- **可选依赖**：核心功能只依赖 PyTorch 和 NumPy

### 3. 输出和可视化
- **多种格式**：JSON（机器可读）、CSV（数据分析）、HTML（报告）
- **实时和离线**：支持实时监控和事后分析
- **未来扩展**：Prometheus exporter、Web 仪表板

## 实现路线图

### Phase 1（优先级最高）
1. **Estimator 基础实现**
   - 模型配置解析器
   - 内存计算公式
   - CLI `estimate` 命令

2. **Monitor 基础实现**
   - PyTorch hook 机制
   - 基础内存追踪
   - CLI `monitor` 命令

3. **文档和示例**
   - 使用指南
   - API 文档
   - 示例代码

### Phase 2（中期目标）
1. **Benchmark 模块**
   - 带宽测试工具
   - 延迟分析

2. **增强功能**
   - 分层统计
   - 可视化报告
   - 优化建议引擎

### Phase 3（长期目标）
1. **深度集成**
   - vLLM、TensorRT-LLM、DeepSpeed
   - 专门的优化分析

2. **生产级特性**
   - Prometheus 监控
   - 分布式支持
   - 持久化存储

## 需要讨论的问题

### 1. 功能优先级
- 是否需要调整三大模块的开发顺序？
- 是否有其他紧急需求的功能？
- 是否需要先支持特定的模型或框架？

### 2. 技术细节
- Hook 机制的具体实现方式（forward hook vs register_hook）
- 内存追踪的粒度和准确性权衡
- 是否需要支持多进程/分布式推理场景？

### 3. 用户体验
- CLI 设计是否足够友好？
- Python API 是否符合直觉？
- 需要哪些默认配置和预设模板？

### 4. 性能和准确性
- 监控开销 < 5% 是否合理？
- 估算误差 < 10% 是否可接受？
- 如何处理不同 GPU 架构的差异？

### 5. 扩展性
- 插件系统的必要性？
- 社区贡献的机制？
- 是否需要配置文件格式标准化？

## 现有方案对比

| 工具/方法 | KV Cache 专门监控 | 估算功能 | 带宽测试 | 易用性 |
|----------|------------------|---------|---------|--------|
| nvidia-smi | ❌ | ❌ | ❌ | ✅ |
| PyTorch Profiler | 部分 | ❌ | ❌ | ⭕ |
| vLLM Metrics | ✅ | 部分 | ❌ | ⭕ |
| **KVCache-SMI** | ✅ | ✅ | ✅ | ✅ |

## 快速参考

### 内存计算公式
```
KV Cache Memory = 2 × L × H × T × P × B

L = num_layers（层数）
H = hidden_size（隐藏维度，考虑 GQA 则是 num_kv_heads × head_dim）
T = num_tokens（token 数量）
P = precision_bytes（精度字节数）
B = batch_size（批次大小）
```

### 示例计算
```
LLaMA-7B (32层, 4096维, 2048长度, FP16, batch=1)
= 2 × 32 × 4096 × 2048 × 2 × 1
≈ 1 GB

LLaMA-70B (80层, 8192维, 4096长度, FP16, batch=8)
= 2 × 80 × 8192 × 4096 × 2 × 8
≈ 32 GB
```

### CLI 命令预览
```bash
# 估算
kvcache-smi estimate --model llama-7b --seq-length 2048 --batch-size 4

# 监控
kvcache-smi monitor --output monitor.json --interval 1

# 测试
kvcache-smi benchmark --type gpu-to-cpu --size 1024

# 报告
kvcache-smi report --input monitor.json --format html
```

## 下一步行动

1. **评审本文档**：确认架构设计方向是否正确
2. **优先级讨论**：确定 Phase 1 的具体实现范围
3. **技术细节确认**：解决关键技术问题
4. **开始实现**：按照路线图开发 MVP（最小可行产品）

---

**文档版本**：v0.1  
**创建日期**：2024-01-15  
**用途**：讨论和评审  
**完整文档**：见 `architecture_v0.1.md`
