# 公开评测报告

## Synthetic 电商构建

以下报告统一绑定：

- build：`build_37262492b5395c01c7a1`；
- target commit：`558636f4d607d5a0c594a00d370ced8d37f78a05`；
- target dirty：`false`；
- snapshot：`data/eval/ecommerce_evaluation_snapshot.json`。

报告：

1. [开发集纯索引消融](ecommerce_index_development_build_37262492b5395c01c7a1.md)
2. [开发集完整 E2E](ecommerce_e2e_development_build_37262492b5395c01c7a1.md)
3. 首次冻结 holdout：系统候选提交完成后只运行一次并新增独立报告，之后不得覆盖。

数据仅有 16 条开发题和 10 条 holdout，且全部来自 synthetic 语料。P95/P99 是小样本离线观测，不是生产 SLA；来源命中、路由与拒答指标也不是答案正确率或业务收益。

## Upstream 历史报告

目录中 `build_4ae47172a9869d88ce0f` 与 `build_2b26e83243ccb25024e0` 报告继承自通用工程知识 RAG 的 Mini-Nanobot 实验。它们保留用于审计历史，**不属于电商构建，指标不得混用**。原首次 holdout 继续保持不可覆盖。
