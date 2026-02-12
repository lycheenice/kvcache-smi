# Formulas and Edge Cases

## 核心公式
`total_bytes = 2 * num_layers * kv_hidden_size * seq_len * batch_size * precision_bytes`

其中：
- `kv_hidden_size = num_kv_heads * head_dim`
- `head_dim = hidden_size / num_attention_heads`

## 注意事项
1. 对 GQA/MQA，必须优先使用 `num_key_value_heads`。
2. 输出同时给出 bytes / MiB / GiB，避免单位歧义。
3. 估算默认不包含 allocator fragmentation，可单独给出开销系数。
4. 当 `hidden_size % num_attention_heads != 0` 时给出输入异常。

## 容量反推
- `max_batch = floor(memory_budget / per_sequence_bytes)`
- `max_seq = floor(memory_budget / per_token_all_batch_bytes)`
