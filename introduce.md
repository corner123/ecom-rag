# 电商研发知识助手：求职与面试材料

## 简历定位

**电商研发知识助手｜AI/RAG 项目实践｜2个月**

如果没有真实雇佣或项目制合作关系，放在“项目经历”，不要填写虚构公司。确有真实合作时才可使用“某电商公司｜AI/RAG 实习生”的匿名表述。

### 简历描述

- 面向订单、库存、促销和支付研发知识查询，构建多源 RAG 原型；使用 YAML catalog 只读接入独立 Git 仓库与 synthetic vendor 规范，并保留 commit、dirty、SHA-256、解析器和引用位置等溯源信息。
- 基于 BGE-small-zh-v1.5、FAISS、BM25 与 RRF 实现混合检索，通过确定性路由区分当前实现、设计历史、外部规范和跨来源比较；涉及代码事实时结合 `rg`、AST 与 Git 实时核验。
- 在近邻召回之上实现 hard-anchor、topic scope 和证据充分性检查，对生产指标、业务收益、独立审计及未来承诺返回结构化拒答，避免把相似文档当作充分证据。
- 建立纯索引消融、E2E 开发集和首次冻结 holdout：当前 16 题开发集 Route accuracy、Primary Hit@5、拒答 F1 均为 1.000；同时保留纯索引 Primary Hit@5 仅 0.583 的局限，所有指标均来自 synthetic 小样本离线评测。
- 使用 FastAPI 提供 `/health`、`/retrieve`、`/answer` 只读接口，保留 FAISS/Milvus 双后端契约与故障分类；Milvus 仅有离线合约测试，不声称生产部署或性能提升。

首次 holdout 完成后，第四条应替换成首次报告的真实结果，并保留“10 题 synthetic 小样本、非生产 SLA”的限定。

## STAR

### Situation

订单、库存、优惠和支付系统的设计文档、故障手册、当前源码与 vendor 规范具有不同权威层级。设计文档能解释为什么，却可能落后于代码；规范能说明外部要求，却不能证明系统已经实现。把所有文本直接放入向量库会把相似近邻误当作当前事实。

### Task

在八周项目周期内完成一个可运行、可引用、可拒答、可评测的内部知识助手 PoC。个人负责数据接入、检索路由、实时源码核验、证据门控、只读接口与离线评测，不声称负责完整电商平台或生产部署。

### Action

1. 用 catalog 管理独立 Git worktree 和规范目录，冻结 commit、content hash 与解析元数据。
2. 构建 code/test/design/history/official 五个物理分区，并实现 BM25、Dense、RRF 检索。
3. 对实现题调用 `rg`、AST 和 Git；对规范比较题同时要求 current implementation 与 external normative 证据。
4. 用 Evidence Guard 拒绝无生产遥测、无实验数据、无独立审计和未来承诺的问题。
5. 建立纯索引与 E2E 两套评测，修正 local-directory 公开 URI 与文件路径标签不一致的问题，再冻结 holdout。

### Result

冻结 synthetic build 包含 2 个 source、37 份采集文档和 39 个索引文档/片段。开发集纯索引中 BM25/Hybrid Primary Hit@5 为 0.583，Dense 为 0.417；完整 E2E 借助路由和实时源码核验，在 16 题上 Primary Hit@5、Route accuracy、拒答 F1 均为 1.000。该结果说明实时核验能补足小型代码语料中的 raw-file 召回，但不能外推到生产或真实公司知识库。

## 八周时间线

| 周次 | 工作内容 | 可验证产物 |
| --- | --- | --- |
| 1 | 需求、数据权限、证据角色 | 场景边界与 synthetic 声明 |
| 2 | 多源采集、分块、元数据 | catalog、manifest、hash |
| 3 | FAISS/BM25 与 RRF | 五分区索引、纯检索查询 |
| 4 | Router、`rg`、AST、Git | 实现题与规范题路由测试 |
| 5 | Evidence Guard、拒答、引用 | 机器可读 refusal reason |
| 6 | FastAPI、鉴权、Web | 三个只读接口与演示页 |
| 7 | 开发集、消融、E2E | 独立开发报告 |
| 8 | 冻结 holdout、失败分析、文档 | 首次报告与求职材料 |

## 高频追问

**为什么不只用向量检索？** 代码问题包含类名、字段名和协议标识符，开发消融中 BM25 的 Primary Hit@5 高于 Dense；当前源码还会变化，因此最终事实需要实时代码搜索。

**为什么 E2E 比纯索引高很多？** 纯索引只看 Top-5 chunk，AST symbol card 会与 raw file 竞争。E2E 根据问题类型分配候选并用 live `rg`/AST/Git 补证。两个 suite 的能力边界不同，不能把差值称为单一算法提升。

**为什么需要拒答？** Dense 永远能返回最近邻，RRF 分数也不是正确率。生产 P99、业务转化率、独立安全审计等问题即使召回到相似文档，也没有所需证据。

**Milvus 做到什么程度？** 实现了 build/model/dimension/count/owner 契约、严格模式和受限可用性回退；当前电商报告固定 FAISS，没有真实 Milvus 服务压测。

**数据是真的吗？** 不是。代码、文档和 PayGate 规范都由仓库脚本生成，目的在于演示系统设计与评测方法，不冒充公司资料。

## 禁止表述

- 不写“某公司线上已使用”，除非确有可证明的真实经历；
- 不写 DAU、QPS、节省工时、转化提升、事故降低或生产 SLA；
- 不把 16 题开发集的 1.000 写成泛化准确率；
- 不把 source-level Hit@5 写成答案正确率或 RAGAS Faithfulness；
- 不声称 PayGate 是真实支付机构；
- 不声称 Milvus 已生产部署或带来指标提升。
