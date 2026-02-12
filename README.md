# KVCache-SMI

一个用于大模型推理过程中 KV Cache 性能分析和优化的工具，类似于 nvidia-smi 对 GPU 的监控功能。

## 项目概述

KVCache-SMI 旨在提供：
- **KV Cache 内存用量估算**：基于模型配置静态估算 KV Cache 内存需求
- **实时监控**：在推理过程中实时追踪 KV Cache 使用情况
- **带宽测试**：测量 offload 和 KV transfer 的带宽性能
- **优化建议**：为 KV Cache 优化提供数据支持

## 目标用户

- LLM 应用开发者
- 模型部署工程师
- 性能优化工程师
- 研究人员

## 文档

- [架构设计文档 v0.1](docs/architecture_v0.1.md) - 详细的系统架构设计、现有方案调研和实现路线图

## 当前状态

项目目前处于早期设计阶段。架构设计文档 v0.1 已完成，包含：
- 详细的背景知识和 KV Cache 机制说明
- 现有方案（PagedAttention, FlashAttention, 压缩技术等）的全面调研
- 完整的系统架构设计
- 核心模块（Estimator, Monitor, Benchmark）的详细设计
- 清晰的实现路线图

## 功能规划

### Phase 1: 核心功能
- [ ] Estimator 模块 - 静态估算 KV Cache 内存
- [ ] Monitor 模块 - 实时监控 KV Cache 使用
- [ ] CLI 基础命令
- [ ] Python API

### Phase 2: 高级功能
- [ ] Benchmark 模块 - 带宽测试和性能分析
- [ ] 可视化和报告生成
- [ ] 分层统计和内存碎片分析

### Phase 3: 集成和优化
- [ ] vLLM / TensorRT-LLM / DeepSpeed 集成
- [ ] PagedAttention 和量化分析
- [ ] 生产级特性（Prometheus, 分布式监控）

## 许可证

本项目采用 Apache License 2.0 许可证。详见 [LICENSE](LICENSE) 文件。

## 贡献

欢迎贡献！请先阅读架构设计文档了解项目设计理念。

## 联系方式

如有问题或建议，请通过 GitHub Issues 联系我们。