---
name: kvcache-cli-api-bootstrap
description: Bootstrap KVCache-SMI codebase structure, shared schemas, and CLI/API contracts. Use when starting implementation or adding new commands/modules.
---

# KVCache CLI/API Bootstrap

用于在 KVCache-SMI 中建立可持续扩展的编码骨架。

## 何时使用
- 新建项目代码结构（`core/`, `estimator/`, `monitor/`, `benchmark/`, `cli/`）
- 新增命令时需要统一参数风格与输出格式
- 需要让 CLI 与 Python API 共享同一数据模型

## 执行步骤
1. 先阅读 `references/project-layout.md`，按建议建立目录与模块边界。
2. 在 `core/schemas.py` 定义统一输出结构（Estimator/Monitor/Benchmark 共用）。
3. 在 `cli/main.py` 实现命令路由，保证参数命名一致（如 `--output`, `--format`）。
4. 仅允许报告层进行 JSON/CSV/HTML 转换，业务层只产出 schema 对象。
5. 为每个命令补充一个最小可运行示例和一个失败输入示例。

## 最低验收
- CLI 子命令可被 `--help` 发现
- 三模块输出字段命名一致
- API 与 CLI 的核心逻辑复用，不重复实现
