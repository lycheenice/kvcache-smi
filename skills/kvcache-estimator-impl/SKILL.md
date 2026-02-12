---
name: kvcache-estimator-impl
description: Implement KV cache memory estimator, including model config parsing, GQA/MQA handling, and capacity planning outputs.
---

# KVCache Estimator Implementation

## 何时使用
- 实现或重构 KV Cache 静态估算逻辑
- 对接 HuggingFace model config
- 输出容量规划建议（最大 batch / 最大长度）

## 执行步骤
1. 读取 `references/formulas-and-edge-cases.md`，确定统一公式与单位。
2. 实现配置解析：`num_layers`, `hidden_size`, `num_attention_heads`, `num_key_value_heads`。
3. 实现估算器主函数，返回总量、per-token、per-layer 指标。
4. 增加容量反推函数：给定显存预算，求 max batch 或 max seq length。
5. 输出推荐动作：GQA、量化、offload 的收益估计。

## 验收标准
- 支持 MHA/GQA/MQA
- 明确区分 MiB/GiB
- 提供至少 2 个公开模型配置的校验样例
