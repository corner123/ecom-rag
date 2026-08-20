# 面试项目说明：多源工程知识 RAG

## 一句话定位

面向 Mini-Nanobot 工程资料构建多源知识检索系统：内部设计与历史决策为主，精选官方规范为辅；当前源码事实由 `rg`、AST 和 Git 实时核验，并通过只读 `knowledge.search` 工具供独立的 Mini-Nanobot Agent 调用。

## STAR

### Situation

工程问题混合了不同权威层级：源码能证明当前实现，却不一定记录“为什么这样设计”；ADR 和文档能说明意图，却可能落后于工作区；外部规范能说明标准要求，却不能证明项目已经实现。把所有内容简单丢进向量库，会产生旧代码、错误权威和相似近邻被当成事实的问题。

### Task

实现一个可运行、可引用、可拒答、可评测的工程知识 RAG，同时保持它与 Coding Agent 两个项目独立。系统需要接入 Git、受控网页和异构本地文档，回答设计、历史和官方规范问题；涉及当前实现时必须回到实时源码；证据不足时返回机器可读的拒答原因；本地开发与分布式向量存储使用同一上层检索契约。

### Action

- 使用 YAML catalog 管理 Mini-Nanobot 本地 Git worktree、可选本地文档目录和 MCP、LangChain、LangGraph、JSON Schema、Python、Docker 官方 URL allowlist；采集阶段保存 commit、dirty、内容 hash、抓取时间和版本。
- 实现按“显式 file_type → MIME → 后缀/特殊文件名”路由的文档加载器，把 Markdown、文本、代码、JSON/YAML/TOML、CSV、XLSX 与 PDF 统一成带 parser/page/sheet/table/warning/degraded 元数据的 `DocumentRecord`；PDF 优先适配可选 MinerU Markdown/JSON，失败时显式降级 PyMuPDF。
- 对 Markdown 做结构化分块，对 Python 通过 AST 生成类/方法 symbol card，同时纳入测试、Dockerfile 和有限 Git history；构建可追溯 manifest。
- 将语料物理拆分为 code、test、design、history、official 五类分区；每个分区始终建立 BGE-small-zh-v1.5 FAISS 稠密索引和 BM25 索引，并用 RRF 做混合召回；可选严格镜像到 build-scoped Milvus HNSW/COSINE collection。
- 定义统一 dense-store 接口和 catalog v4：构建支持 `faiss|milvus|both`，查询支持严格 `faiss|milvus` 与受限 `auto`。Milvus artifact 绑定部署唯一 owner namespace，显式清理同时校验 catalog、owner 和 collection 描述；`auto` 仅对明确可用性故障退回 FAISS，鉴权、schema、维度、行数、build/model/owner 不匹配均失败关闭；旧 v3 FAISS catalog 保持可读。
- 用确定性 Router 区分 implementation、design、official、comparison 与 out-of-scope；实现题要求 `rg`/AST/Git 实时证据，比较题同时要求当前实现与官方证据。
- 在近邻检索之上增加 catalog topic scope、hard-anchor、经验数据缺失和独立审计缺失检查，避免把“最相似文档”误当“足够证据”。
- 设计结构化引用 DTO，区分 `current_implementation`、`internal_design`、`internal_history`、`external_normative`，公开接口过滤本机路径、remote 凭据和 HTTP 内部字段。
- 构建严格 v2 评测：纯索引消融关闭 Router/live/Answerer；E2E 检索策略评测运行路由、实时核验和拒答；另冻结 30 条 holdout，包含真实跨语料比较与 OOD 题。
- 将 RAG 作为独立 FastAPI 服务，在 Mini-Nanobot 中仅增加可选的只读 `knowledge.search` HTTP 客户端；未配置服务时不注册，服务离线不会影响其他 Agent 工具。
- 将前端收敛为只读问答工作台；自托管 OFL 授权的 DM Sans，中文使用系统字体回退，不依赖字体 CDN，也不复制品牌专有字体。

### Result

冻结快照 `build_4ae47172a9869d88ce0f` 共包含 339 份文档、962 个 chunk。70 条 answerable 开发题的纯索引消融中，BM25 的 Primary Hit@5 为 0.629，高于 Dense 的 0.400 和 Hybrid RRF 的 0.457；Hybrid 的 Supporting Recall@5 最高，为 0.277。这说明技术语料中的精确标识符使关键词检索非常重要，混合检索并不会自动提升所有指标。

完整 E2E 开发集共 80 题，Primary Hit@5 为 0.886、拒答 F1 为 0.833。冻结后首次运行的 30 题 holdout 上，Route accuracy 为 0.700、Primary Hit@5 为 0.870、拒答 F1 为 0.636；7 个不可答题全部拒答，但 23 个可答题中有 8 个被误拒答。这个结果暴露出系统当前最大短板是证据门控过严，而不是缺少近邻候选。所有数字来自 `data/eval/reports_public/` 中绑定同一 snapshot 的脱敏报告，且 holdout 未用于后续调参。

上述冻结数字来自 FAISS，并且正式 `engineering-eval` 继续显式固定 FAISS；新增 Milvus 后端不会静默改写历史报告。当前开发机没有可连接的真实 Milvus 服务或 Docker 运行时，Milvus 部分目前有状态化 fake-client 合约测试和一个由 `MILVUS_TEST_URI` 门控、自动清理随机测试 collection 的 opt-in round-trip。因而可以说明“双后端与故障边界已实现”，但不能说明“本机已完成真实分布式部署、性能压测或检索指标提升”。

独立 response-eval/v3 已真实运行 DeepSeek 生成与 RAGAS：16 题开发集基线 Faithfulness 0.920、Answer Relevancy 0.703、Context Precision/Recall 0.837/0.697。候选通过 query-aware 支持证据和实时 AST 父类展开，将 Required-claim Recall@5 从 0.854 提升到 0.910、误拒答率从 8.33% 降为 0%，但 paired Faithfulness 下降 0.042，且一个 Judge 指标连接失败，因此候选没有通过预注册门槛，也没有进入私有 holdout。这个结果应表述为“建立了真实评估和失败关闭流程”，不能表述为“RAGAS 四项都获得提升”。

## 面试时应强调的判断

1. RAG 不替代代码搜索。函数定义、调用点和未提交改动由实时工具核验。
2. 向量相似度不是答案置信度，RRF 分数也不是概率；系统另做证据范围和 anchor 检查。
3. 官方规范、内部设计和当前实现是不同证据角色，回答时不能互相冒充。
4. 消融实验必须只改变检索策略；路由、live tool 或 Answerer 混入后，数字不能称为纯索引对比。
5. 自动 source-level 指标不能声称答案完全正确；答案点与忠实度仍需冻结 Judge 或人工抽检。
6. 自动降级必须窄化。网络不可用可以退回本地快照，但鉴权和数据完整性错误必须暴露，不能为了“可用”而掩盖错误。
7. 多格式接入的价值不只是“能读 PDF”：解析器、页码/工作表/表格坐标、warning 与 degraded 状态必须进入统一元数据，才能审计引用质量。

## 简历可写与暂不可写

可以写：

- “抽象 FAISS/Milvus 双稠密后端；v4 catalog 校验远端 build/model/dimension/count，`auto` 仅在 Milvus 可用性故障时退回本地 FAISS，鉴权与完整性错误失败关闭。”
- “实现多格式文档路由与统一 DocumentRecord；PDF 优先接入可选 MinerU 产物，PyMuPDF fallback 显式记录 degraded/warnings。”

但以下内容暂不写成已完成结果：

- Milvus 主从、高可用部署、HNSW 参数调优或性能/召回提升；当前没有真实服务报告；
- “MinerU OCR 已在当前环境运行”：当前实现了可选 CLI 适配与 fallback，但没有在本机安装/验证完整 MinerU 模型栈；
- ColBERT 已带来提升；
- “RAGAS 四项均提升”或“候选已通过最终盲测”；当前只有基线完整，候选因 Faithfulness 退化且 Judge 覆盖不完整而被拒绝；
- 海量网页爬取、生产并发、SLA 或安全认证；
- “最优策略”或没有对照实验支持的提升比例。
