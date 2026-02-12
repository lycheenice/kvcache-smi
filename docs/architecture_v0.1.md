# KVCache-SMI 架构设计文档 v0.1

## 1. 项目概述

### 1.1 项目目标
KVCache-SMI 是一个用于大模型推理过程中 KV Cache 性能分析和优化的工具，类似于 nvidia-smi 对 GPU 的监控功能。主要功能包括：
- KV Cache 内存用量估算和实时监控
- Offload 带宽估算和性能分析
- KV Transfer 带宽测量
- 为 KV Cache 优化提供数据支持

### 1.2 目标用户
- LLM 应用开发者
- 模型部署工程师
- 性能优化工程师
- 研究人员

## 2. 背景知识

### 2.1 KV Cache 机制
在 Transformer 架构的大语言模型中，自回归生成过程需要重复计算之前所有 token 的 Key 和 Value：
- **问题**：每生成一个新 token，都需要重新计算所有历史 token 的 K/V，计算复杂度为 O(n²)
- **解决方案**：将已计算的 K/V 缓存起来，新 token 只需与缓存的 K/V 进行注意力计算
- **效果**：将计算复杂度降低到 O(n)，显著提升推理速度

### 2.2 KV Cache 内存占用计算
基本公式：
```
memory = 2 × num_layers × hidden_size × num_tokens × precision_bytes × batch_size
```

参数说明：
- `2`：Key 和 Value 各一份
- `num_layers`：模型层数
- `hidden_size`：隐藏层维度
- `num_tokens`：序列长度（已生成的 token 数）
- `precision_bytes`：精度字节数（FP32=4, FP16=2, INT8=1）
- `batch_size`：批次大小

**示例计算（LLaMA-7B）**：
- 32 层
- 4096 隐藏维度
- 2048 上下文长度
- FP16（2 字节）
- batch_size=1
- 内存占用 ≈ 2 × 32 × 4096 × 2048 × 2 × 1 ≈ 1 GB

### 2.3 KV Cache 挑战
1. **内存瓶颈**：长上下文推理时 KV Cache 可能占用数十 GB 内存
2. **内存碎片**：动态批处理导致的内存分配不连续
3. **带宽限制**：Offload 时的数据传输成为性能瓶颈
4. **批处理限制**：KV Cache 内存占用限制了可并行处理的请求数

## 3. 现有方案调研

### 3.1 PagedAttention (vLLM)
- **核心思想**：将 KV Cache 视为虚拟内存页，允许非连续存储
- **优势**：
  - 减少内存碎片，提高内存利用率
  - 支持更大的批处理
  - 可以方便地实现 KV Cache 共享（如 beam search）
- **实现**：将 KV Cache 分成固定大小的块（如 16 tokens），按需分配
- **参考**：vLLM 项目

### 3.2 FlashAttention
- **核心思想**：通过分块计算和 kernel 融合减少 HBM 访问
- **优势**：
  - 降低注意力计算的内存占用
  - 提升计算速度
- **局限**：主要优化计算过程，不直接减少 KV Cache 存储
- **参考**：FlashAttention-2

### 3.3 KV Cache 压缩技术
1. **量化**：
   - INT8/INT4 量化：KIVI, Atom
   - 动态量化：根据重要性选择性量化
   
2. **剪枝**：
   - H2O (Heavy-Hitter Oracle)：保留重要 token
   - StreamingLLM：保留首尾 token
   - Scissorhands：基于注意力分数剪枝

3. **MQA/GQA**：
   - Multi-Query Attention：多个 Query head 共享 K/V
   - Grouped-Query Attention：Query head 分组共享 K/V
   - 可减少 KV Cache 大小 4-8 倍

### 3.4 Offloading 方案
1. **FlexGen**：
   - CPU/GPU/Disk 三级缓存
   - 适合资源受限环境
   
2. **DeepSpeed-Inference**：
   - CPU offloading
   - 张量并行和流水线并行
   
3. **Hugging Face Accelerate**：
   - 自动混合设备放置
   - 透明的 offloading

### 3.5 现有监控工具
1. **nvidia-smi**：GPU 整体内存监控
2. **PyTorch Profiler**：细粒度内存分析
3. **vLLM Metrics**：特定于 vLLM 的监控
4. **DeepSpeed Profiler**：训练和推理性能分析

**局限性**：
- 缺少专门针对 KV Cache 的监控工具
- 现有工具不能直接区分 KV Cache 和其他内存使用
- 缺少 offload 带宽的实时监控和分析

## 4. 系统架构设计

### 4.1 整体架构
```
┌─────────────────────────────────────────────────────────────┐
│                     KVCache-SMI CLI/API                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │ Estimator    │  │  Monitor     │  │   Benchmark      │  │
│  │              │  │              │  │                  │  │
│  │ - 静态估算   │  │ - 实时监控   │  │ - 带宽测试       │  │
│  │ - 配置分析   │  │ - 内存追踪   │  │ - 性能测试       │  │
│  └──────┬───────┘  └──────┬───────┘  └─────────┬────────┘  │
│         │                  │                     │           │
├─────────┴──────────────────┴─────────────────────┴──────────┤
│                       Core Library                           │
│  ┌────────────────────────────────────────────────────────┐ │
│  │  Model Config Parser   │  Memory Calculator  │ Profiler│ │
│  ├────────────────────────────────────────────────────────┤ │
│  │  Device Info Collector │  Transfer Benchmark │ Reporter│ │
│  └────────────────────────────────────────────────────────┘ │
├──────────────────────────────────────────────────────────────┤
│                     Integration Layer                         │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────┐ │
│  │ PyTorch     │  │ Transformers│  │ Custom Backends      │ │
│  │ Hooks       │  │ Integration │  │ (vLLM, TensorRT-LLM) │ │
│  └─────────────┘  └─────────────┘  └──────────────────────┘ │
└──────────────────────────────────────────────────────────────┘
```

### 4.2 核心模块

#### 4.2.1 Estimator（估算器）
**功能**：根据模型配置和推理参数，静态估算 KV Cache 内存需求

**输入**：
- 模型配置（num_layers, hidden_size, num_attention_heads 等）
- 推理参数（max_seq_length, batch_size, precision）
- 硬件配置（GPU 型号，可用内存）

**输出**：
- 预估 KV Cache 内存占用
- 最大可支持的 batch size
- 最大可支持的序列长度
- 建议的优化策略

**核心算法**：
```python
def estimate_kv_cache_memory(config, inference_params):
    """
    估算 KV Cache 内存占用
    """
    num_layers = config.num_layers
    hidden_size = config.hidden_size
    num_heads = config.num_attention_heads
    head_dim = hidden_size // num_heads
    
    # 考虑 GQA/MQA
    num_kv_heads = getattr(config, 'num_key_value_heads', num_heads)
    kv_hidden_size = num_kv_heads * head_dim
    
    seq_length = inference_params.max_seq_length
    batch_size = inference_params.batch_size
    precision_bytes = inference_params.precision_bytes  # FP16=2, INT8=1
    
    # 计算单个序列的 KV Cache 大小
    per_token_kv_size = 2 * num_layers * kv_hidden_size * precision_bytes
    total_kv_cache = per_token_kv_size * seq_length * batch_size
    
    return {
        'total_bytes': total_kv_cache,
        'per_token_bytes': per_token_kv_size,
        'per_layer_bytes': 2 * kv_hidden_size * seq_length * batch_size * precision_bytes
    }
```

#### 4.2.2 Monitor（监控器）
**功能**：在实际推理过程中实时监控 KV Cache 使用情况

**实现方式**：
1. **PyTorch Hook**：
   - 在注意力层注册 forward hook
   - 追踪 KV Cache 的创建、更新和释放
   
2. **内存追踪**：
   - 使用 `torch.cuda.memory_stats()` 追踪 GPU 内存
   - 分离 KV Cache 内存和其他内存

3. **实时指标**：
   - 当前 KV Cache 占用
   - 内存增长速率
   - 缓存命中率（如果有复用）

**数据收集**：
```python
class KVCacheMonitor:
    def __init__(self):
        self.kv_cache_stats = {
            'total_size': 0,
            'layer_wise_size': {},
            'allocation_history': [],
            'peak_memory': 0
        }
    
    def hook_attention_layer(self, module, input, output):
        """注册到注意力层的 hook"""
        if hasattr(module, 'kv_cache'):
            cache_size = self._get_tensor_memory(module.kv_cache)
            self.kv_cache_stats['total_size'] += cache_size
            # 记录详细信息...
```

#### 4.2.3 Benchmark（基准测试）
**功能**：测量不同场景下的 offload 和 transfer 带宽

**测试场景**：
1. **GPU↔CPU Transfer**：
   - 不同数据大小的传输带宽
   - Pinned memory vs Regular memory
   - 单向 vs 双向传输

2. **GPU↔GPU Transfer**：
   - NVLink 带宽（如果可用）
   - PCIe 带宽
   - P2P 传输性能

3. **KV Cache Specific**：
   - 模拟实际 KV Cache 的访问模式
   - 测量 offload 延迟对推理的影响

**带宽测试代码**：
```python
def benchmark_transfer_bandwidth(data_size_mb, device_src, device_dst, num_iterations=100):
    """
    测试设备间传输带宽
    """
    data = torch.randn(data_size_mb * 1024 * 1024 // 4, device=device_src)
    
    # 预热
    for _ in range(10):
        temp = data.to(device_dst)
    
    # 测试
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(num_iterations):
        temp = data.to(device_dst)
        torch.cuda.synchronize()
    
    elapsed = time.time() - start
    bandwidth_gbps = (data_size_mb * num_iterations) / elapsed / 1024
    
    return bandwidth_gbps
```

### 4.3 数据模型

#### 4.3.1 Model Configuration
```python
@dataclass
class ModelConfig:
    """模型配置"""
    name: str
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: Optional[int]  # For GQA/MQA
    vocab_size: int
    max_position_embeddings: int
    
    @classmethod
    def from_pretrained(cls, model_name_or_path):
        """从 HuggingFace 配置加载"""
        config = AutoConfig.from_pretrained(model_name_or_path)
        return cls(
            name=model_name_or_path,
            num_layers=config.num_hidden_layers,
            hidden_size=config.hidden_size,
            # ...
        )
```

#### 4.3.2 Inference Configuration
```python
@dataclass
class InferenceConfig:
    """推理配置"""
    max_seq_length: int
    batch_size: int
    precision: str  # "fp32", "fp16", "bf16", "int8"
    use_kv_cache: bool = True
    kv_cache_dtype: Optional[str] = None
    enable_offload: bool = False
    offload_device: str = "cpu"  # "cpu" or "disk"
```

#### 4.3.3 Monitoring Metrics
```python
@dataclass
class KVCacheMetrics:
    """KV Cache 监控指标"""
    timestamp: float
    total_memory_bytes: int
    per_layer_memory: Dict[int, int]
    num_cached_tokens: int
    peak_memory_bytes: int
    cache_hit_rate: Optional[float] = None
    
    # Offload 相关
    offloaded_bytes: int = 0
    transfer_bandwidth_gbps: Optional[float] = None
    offload_latency_ms: Optional[float] = None
```

### 4.4 接口设计

#### 4.4.1 CLI 接口
```bash
# 估算命令
kvcache-smi estimate \
    --model "meta-llama/Llama-2-7b-hf" \
    --max-seq-length 4096 \
    --batch-size 8 \
    --precision fp16

# 实时监控
kvcache-smi monitor \
    --interval 1 \
    --output monitor.json

# 带宽测试
kvcache-smi benchmark \
    --type transfer \
    --data-size 1024 \
    --src cuda:0 \
    --dst cpu

# 分析报告
kvcache-smi report \
    --input monitor.json \
    --format html
```

#### 4.4.2 Python API
```python
from kvcache_smi import KVCacheEstimator, KVCacheMonitor, TransferBenchmark

# 1. 估算
estimator = KVCacheEstimator.from_pretrained("meta-llama/Llama-2-7b-hf")
estimate = estimator.estimate(
    max_seq_length=4096,
    batch_size=8,
    precision="fp16"
)
print(f"Estimated KV Cache: {estimate.total_gb:.2f} GB")

# 2. 监控
monitor = KVCacheMonitor()
model = AutoModelForCausalLM.from_pretrained("model_name")

# 注册监控 hooks
monitor.register_hooks(model)

# 推理
with monitor.track():
    outputs = model.generate(input_ids, max_length=100)

# 获取统计信息
stats = monitor.get_statistics()
print(f"Peak KV Cache: {stats.peak_memory_gb:.2f} GB")

# 3. 带宽测试
benchmark = TransferBenchmark()
result = benchmark.measure_bandwidth(
    data_size_mb=1024,
    device_src="cuda:0",
    device_dst="cpu"
)
print(f"Transfer bandwidth: {result.bandwidth_gbps:.2f} GB/s")
```

#### 4.4.3 集成接口
```python
# vLLM 集成
from kvcache_smi.integrations import vLLMMonitor

monitor = vLLMMonitor()
# 自动追踪 vLLM 的 KV Cache 管理

# Transformers 集成
from kvcache_smi.integrations import TransformersMonitor

monitor = TransformersMonitor()
model = AutoModelForCausalLM.from_pretrained(
    "model_name",
    kvcache_monitor=monitor
)
```

## 5. 技术选型

### 5.1 编程语言
- **Python**：主要开发语言，易于集成深度学习框架
- **C++/CUDA**（可选）：性能关键部分，如自定义 CUDA kernel

### 5.2 依赖库
- **核心依赖**：
  - PyTorch：深度学习框架
  - Transformers：模型加载和配置
  - NumPy：数值计算
  
- **可选依赖**：
  - vLLM：集成 vLLM 监控
  - TensorRT-LLM：NVIDIA 推理引擎集成
  - psutil：系统资源监控
  - matplotlib/plotly：可视化
  
- **开发依赖**：
  - pytest：单元测试
  - black/ruff：代码格式化
  - mypy：类型检查

### 5.3 输出格式
- **JSON**：机器可读，便于集成
- **CSV**：便于数据分析
- **Markdown/HTML**：人类可读报告
- **Prometheus**（未来）：监控系统集成

## 6. 实现路线图

### Phase 1：核心功能（v0.1-v0.3）
- [x] 架构设计文档
- [ ] 实现 Estimator 模块
  - [ ] 模型配置解析器
  - [ ] KV Cache 内存计算器
  - [ ] 最大 batch size 估算
- [ ] 实现基础 Monitor
  - [ ] PyTorch hook 机制
  - [ ] 基础内存追踪
- [ ] CLI 基础命令
  - [ ] `estimate` 命令
  - [ ] `monitor` 命令
- [ ] 文档和示例

### Phase 2：高级功能（v0.4-v0.6）
- [ ] Benchmark 模块
  - [ ] GPU↔CPU 带宽测试
  - [ ] GPU↔GPU 带宽测试
  - [ ] Offload 延迟测试
- [ ] 增强 Monitor
  - [ ] 分层统计
  - [ ] 内存碎片分析
  - [ ] 缓存命中率
- [ ] 可视化和报告
  - [ ] 实时监控界面
  - [ ] HTML 报告生成
  - [ ] 性能建议

### Phase 3：集成和优化（v0.7-v1.0）
- [ ] 框架集成
  - [ ] vLLM 集成
  - [ ] TensorRT-LLM 集成
  - [ ] DeepSpeed 集成
- [ ] 高级分析
  - [ ] PagedAttention 分析
  - [ ] 量化效果分析
  - [ ] Offload 策略建议
- [ ] 性能优化
  - [ ] 低开销监控
  - [ ] CUDA kernel 优化
- [ ] 生产级特性
  - [ ] Prometheus exporter
  - [ ] 分布式监控
  - [ ] 持久化存储

### Phase 4：生态和扩展（v1.0+）
- [ ] 支持更多模型架构
  - [ ] Mixture-of-Experts (MoE)
  - [ ] State Space Models (Mamba)
- [ ] 云原生支持
  - [ ] Kubernetes operator
  - [ ] 多租户监控
- [ ] AutoML 集成
  - [ ] 自动调优
  - [ ] 配置建议引擎
- [ ] 社区和生态
  - [ ] 插件系统
  - [ ] 社区贡献

## 7. 性能考虑

### 7.1 监控开销
- **目标**：监控开销 < 5% 推理时间
- **策略**：
  - 使用轻量级 hook
  - 采样监控（可配置）
  - 异步数据收集和处理
  - 延迟计算非关键指标

### 7.2 内存开销
- **目标**：额外内存 < 100MB
- **策略**：
  - 增量统计，避免存储完整历史
  - 使用内存映射文件存储大量数据
  - 可配置的历史数据保留策略

### 7.3 准确性
- **目标**：估算误差 < 10%
- **策略**：
  - 考虑内存对齐和填充
  - 考虑框架开销
  - 提供保守估算和激进估算两种模式

## 8. 安全和隐私

### 8.1 数据安全
- 仅收集性能指标，不收集模型权重或推理数据
- 本地运行，不向外部服务器发送数据
- 支持数据脱敏（如模型名称匿名化）

### 8.2 权限管理
- 读取 GPU 信息需要相应权限
- 文件输出遵循用户权限设置
- 支持只读模式（仅监控，不输出文件）

## 9. 测试策略

### 9.1 单元测试
- 每个模块的核心函数
- 边界条件和异常处理
- 目标覆盖率：> 80%

### 9.2 集成测试
- 端到端推理场景
- 不同模型和配置
- 多 GPU 和分布式场景

### 9.3 基准测试
- 监控开销测试
- 不同规模模型的性能
- 与现有工具对比

### 9.4 兼容性测试
- PyTorch 版本兼容性（2.0+）
- CUDA 版本兼容性（11.8+）
- 不同 GPU 型号（V100, A100, H100 等）

## 10. 文档计划

### 10.1 用户文档
- 快速开始指南
- CLI 命令参考
- Python API 文档
- 常见问题和故障排除

### 10.2 开发者文档
- 架构设计文档（本文档）
- 模块设计文档
- 贡献指南
- API 设计规范

### 10.3 示例和教程
- 基础使用示例
- 集成到现有项目
- 性能优化案例
- 最佳实践

## 11. 未来扩展方向

### 11.1 短期（6 个月内）
1. **多模态支持**：
   - Vision-Language 模型的 KV Cache 分析
   - 跨模态 attention 的内存追踪

2. **更多优化策略**：
   - 动态 offload 决策
   - 智能预取
   - 自适应压缩

3. **更好的可视化**：
   - 实时仪表板
   - 交互式分析工具
   - 性能对比视图

### 11.2 长期（1 年以上）
1. **AI 辅助优化**：
   - 基于历史数据的配置推荐
   - 自动调优系统
   - 异常检测和告警

2. **分布式系统支持**：
   - 多节点监控
   - 全局 KV Cache 管理
   - 跨设备优化

3. **硬件异构支持**：
   - NPU/TPU 支持
   - 新型内存技术（CXL, HBM3）
   - 边缘设备优化

## 12. 总结

KVCache-SMI 旨在成为 LLM 推理场景下 KV Cache 的专业分析工具。通过提供准确的估算、实时的监控和全面的基准测试，帮助开发者和工程师：

1. **理解**：深入理解 KV Cache 的内存占用和性能特征
2. **优化**：基于数据驱动的方式优化推理性能
3. **决策**：为硬件选型和配置调整提供依据
4. **监控**：生产环境中实时追踪 KV Cache 使用情况

本文档作为 v0.1 版本，主要完成了需求分析、现有方案调研和整体架构设计。后续将根据实际开发过程和用户反馈持续迭代和完善。

---

## 附录 A：参考文献

1. **PagedAttention**: Kwon et al., "Efficient Memory Management for Large Language Model Serving with PagedAttention", SOSP 2023
2. **FlashAttention**: Dao et al., "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning", 2023
3. **GQA**: Ainslie et al., "GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints", 2023
4. **H2O**: Zhang et al., "H2O: Heavy-Hitter Oracle for Efficient Generative Inference of Large Language Models", 2023
5. **FlexGen**: Sheng et al., "FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU", ICML 2023
6. **vLLM**: https://github.com/vllm-project/vllm
7. **DeepSpeed-Inference**: https://github.com/microsoft/DeepSpeed

## 附录 B：术语表

- **KV Cache**: Key-Value Cache，注意力机制中缓存的 Key 和 Value 张量
- **Offload**: 将数据从 GPU 转移到 CPU 或磁盘以节省 GPU 内存
- **MQA**: Multi-Query Attention，多个 Query 共享同一组 Key 和 Value
- **GQA**: Grouped-Query Attention，Query 分组共享 Key 和 Value
- **PagedAttention**: 将 KV Cache 视为虚拟内存页的注意力机制
- **HBM**: High Bandwidth Memory，高带宽内存
- **NVLink**: NVIDIA 的 GPU 互连技术
- **PCIe**: Peripheral Component Interconnect Express
- **Throughput**: 吞吐量，单位时间内处理的请求数或 token 数
- **Latency**: 延迟，单个请求的响应时间

## 附录 C：示例输出

### C.1 估算输出示例
```
KV Cache Memory Estimation
==========================
Model: meta-llama/Llama-2-7b-hf
Configuration:
  - Layers: 32
  - Hidden Size: 4096
  - Attention Heads: 32
  - KV Heads: 32 (no GQA)

Inference Parameters:
  - Max Sequence Length: 4096
  - Batch Size: 8
  - Precision: FP16

Results:
  - Per-token KV Size: 1.0 MB
  - Total KV Cache: 32.0 GB
  - Model Weights: ~13.5 GB
  - Total GPU Memory: ~45.5 GB

Recommendations:
  ✓ Can fit on A100 (80GB)
  ✓ Can fit on H100 (80GB)
  ✗ Cannot fit on A100 (40GB) with this batch size
  
Suggested Optimizations:
  - Use GQA to reduce KV cache by 4x
  - Enable INT8 KV cache quantization (KIVI)
  - Reduce batch size to 4 for A100 40GB
```

### C.2 监控输出示例
```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "model": "meta-llama/Llama-2-7b-hf",
  "metrics": {
    "total_memory_gb": 28.5,
    "kv_cache_gb": 24.3,
    "model_weights_gb": 13.5,
    "activations_gb": 2.2,
    "num_cached_tokens": 3072,
    "peak_memory_gb": 32.1,
    "memory_utilization": 0.76
  },
  "per_layer_stats": [
    {
      "layer_id": 0,
      "kv_cache_mb": 768,
      "num_tokens": 3072
    }
  ],
  "performance": {
    "tokens_per_second": 125,
    "inference_latency_ms": 45
  }
}
```

### C.3 带宽测试输出示例
```
Transfer Bandwidth Benchmark
============================
Configuration:
  - Data Size: 1024 MB
  - Iterations: 100
  - Warmup: 10 iterations

Results:
  GPU → CPU (Pinned):   25.3 GB/s
  GPU → CPU (Regular):  12.1 GB/s
  CPU → GPU (Pinned):   26.8 GB/s
  CPU → GPU (Regular):  11.9 GB/s
  
PCIe Specification: Gen4 x16 (theoretical: 32 GB/s)
Efficiency: 79.2%

Offload Latency:
  - 1 GB transfer: ~40 ms
  - Estimated impact on 100 token/s generation: 4% slowdown
```

---

**文档版本**：v0.1  
**创建日期**：2024-01-15  
**最后更新**：2024-01-15  
**作者**：KVCache-SMI 开发团队  
**状态**：草案（待审查）
