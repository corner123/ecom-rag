# 外贸情报 Agent 全量重构设计

日期：2026-08-29
目标分支：`codex/foreign-trade-agent-overhaul`
基线提交：`b0ea1de0499f5de4dd50dfee68ea02bc1455af9f`

## 1. 决策与范围

本分支将现有 synthetic 电商研发知识 RAG 全量替换为一个 synthetic 外贸情报 Agent。旧代码不在新产品树中复用或归档；Git 历史和 `origin/main` 是唯一历史保留方式。

新系统必须真实打通以下路径：

1. MySQL 保存七张外贸维度/事实表，并由只读 SQL Agent 执行精确查询与聚合。
2. 官网、B2B、行业新闻、社媒、PDF、扫描 PDF 和海关派生贸易画像按来源路由、解析、切分、补充 metadata。
3. Milvus 保存真实 BGE embedding 与标量 metadata，并执行 filter-first ANN。
4. BM25、Dense、Weighted RRF 和 BGE Reranker 构成可消融的混合检索链路。
5. LangGraph 编排 SQL 与 RAG 双通道、融合、验证、生成和逐 Claim 幻觉检查；Redis 保存短期 checkpoint。
6. 构造公开开发集和本地私有 holdout，计算 Retrieval、Fusion、Generation、Business 四层指标，并基于开发集错误分析完成至少一轮有证据的优化。
7. 使用 Docker Compose 启动 MySQL、Milvus、etcd、MinIO、Redis 和 API，完成真实服务 round-trip 与端到端 smoke test。

所有自生成公司、交易、网站、新闻、B2B、社媒、PDF、判断标签和指标必须标记为 `synthetic`。系统证明的是工程实现，不证明真实企业部署、客户数据、生产 SLA 或业务收益。

## 2. 非目标

- 不采集或伪造真实客户、联系人、海关付费数据、LinkedIn 账号数据或企业内部资料。
- 不把 synthetic Lead Precision 描述为销售团队实际认可率。
- 不把开发集调优结果描述为泛化能力。
- 不实现写数据库、外呼、发邮件、自动联系客户或任何有业务副作用的工具。
- 不保留旧电商 API、旧 Mini-Nanobot 兼容入口、占位多模态检索或硬编码 AI 模型领域 Text-to-SQL。
- 不在没有证据时声称 MinerU、Milvus、Redis、LangGraph、BGE 或 LLM Judge 已运行成功。

## 3. 技术栈与目录

运行时固定为 Python 3.12。主包命名为 `trade_agent`，不沿用领域含义混乱的 `rag_core/engineering`。

```text
ecom-rag/
├── README.md
├── pyproject.toml
├── .env.example
├── docker-compose.yml
├── docker/
│   ├── Dockerfile
│   └── healthcheck.py
├── db/
│   ├── migrations/001_schema.sql
│   ├── init/010_users.sh
│   └── seeds/synthetic/
├── demo/trade_intel_seed/
│   ├── website/
│   ├── b2b/
│   ├── news/
│   ├── social/
│   ├── pdf/
│   └── manifests/
├── data/
│   ├── sources/trade_intel_demo.yaml
│   └── eval/trade_intel/
├── trade_agent/
│   ├── api/
│   ├── agents/
│   ├── config/
│   ├── data/
│   ├── db/
│   ├── entities/
│   ├── evaluation/
│   ├── evidence/
│   ├── generation/
│   ├── index/
│   ├── retrieval/
│   └── schemas/
├── scripts/
└── tests/
    ├── unit/
    ├── contract/
    ├── integration/
    ├── milvus/
    ├── e2e/
    └── security/
```

依赖包括 FastAPI、Pydantic v2、SQLAlchemy/PyMySQL、SQLGlot、LangGraph、`langgraph-checkpoint-redis`、`pymilvus==2.6.17`、sentence-transformers、rank-bm25、PyMuPDF、pypdf、Pillow/reportlab、MinerU CLI adapter、NumPy、pytest 和 httpx。

## 4. 基础设施

`docker-compose.yml` 包含以下服务：

| 服务 | 用途 | 持久化 | 就绪条件 |
| --- | --- | --- | --- |
| `mysql` | 七表结构化事实 | `mysql_data` | `mysqladmin ping` 且 migration/seed 完成 |
| `etcd` | Milvus metadata | `etcd_data` | endpoint health |
| `minio` | Milvus object storage | `minio_data` | live health endpoint |
| `milvus` | 向量与标量过滤 | `milvus_data` | `9091/healthz` 与 SDK connect |
| `redis` | LangGraph checkpoint | `redis_data` | `redis-cli ping` |
| `api` | 数据导入、检索和回答 API | 模型 cache volume | `/ready` 验证全部必需依赖 |

Milvus 锁定为稳定的 `milvusdb/milvus:v2.6.22`，Python SDK 锁定为其官方兼容表中的 `pymilvus==2.6.17`；etcd、MinIO 镜像从该版本官方 standalone Compose 原样锁定。Compose 的开发端口只绑定 `127.0.0.1`。密码来自未提交的 `.env`，仓库只提交 `.env.example`。健康检查不等同于完整 readiness；API `/ready` 必须额外检查数据库 schema fingerprint、Milvus collection contract、Redis 和当前 build ID。

Milvus 真实验收必须完成创建隔离 collection、插入 BGE 向量、flush/load、无过滤 ANN、带 `region + hs_code + source_type` 过滤 ANN、进程重连后查询、contract 校验和测试 collection 清理。Fake client 或 skip 不能作为成功证据。

## 5. MySQL 七表与 SQL 边界

数据库名为 `foreign_trade_db`，包含：

1. `countries(id, country_code, country_name, region, created_at)`
2. `companies(id, company_name, normalized_name, country_id, company_type, website, website_domain, registration_id, address, industry, is_synthetic, created_at)`
3. `hs_codes(id, hs_code, description, category, parent_code, created_at)`
4. `products(id, product_name, sku, hs_code_id, category, description, created_at)`
5. `data_sources(id, source_name, source_type, source_url, update_frequency, last_updated_at, is_synthetic, created_at)`
6. `company_products(id, company_id, product_id, relation_type, created_at)`
7. `trade_records(id, raw_record_id, source_id, importer_id, exporter_id, product_id, hs_code_id, import_country_id, export_country_id, trade_date, quantity, unit, trade_amount, currency, created_at)`

约束包括外键、`source_id + raw_record_id` 唯一键、HS/日期、企业/日期、国家+HS+日期组合索引。金额使用 `DECIMAL`，HS Code 使用字符串防止前导零丢失。`trade_count`、`latest_trade_date` 和 `growth_rate` 是聚合派生指标，不作为原始事实字段。

Synthetic seed 的目标规模：至少 8 个国家、60 家公司、12 个 HS Code、30 个产品、800 条跨 18 个月交易记录。数据必须包含可控的增长、下降、休眠和近期活跃模式，以支持 SQL、Lead 和冲突评估。

查询服务使用独立只读账户。Schema Registry 从 `INFORMATION_SCHEMA` 建立真实表/列快照，再补充业务描述、别名、允许 join、可聚合字段和敏感字段策略。Text-to-SQL 采用受限 Query Plan：

```text
自然语言 → Intent/Constraint Extraction → Schema Linking
        → 结构化 QueryPlan → SQLGlot AST → Policy Injection
        → EXPLAIN/预算校验 → Read-only Execute → Structured Evidence
```

SQL Validator 强制单条 `SELECT`/受控 CTE、表列白名单、注册 join、强制时间/tenant policy、禁止 `SELECT *`、强制行数限制、超时和扫描预算。DDL/DML、注释绕过、系统表、动态 SQL、未登记函数、多语句和笛卡尔积全部 fail-closed。

每次执行产生 `SqlEvidence`：`query_id`、规范化 SQL、fingerprint、dataset/schema 版本、注入的过滤条件、聚合粒度、执行时间、行数、result hash、raw record locator。数值 Claim 必须引用它。

## 6. 多类型数据与路由

`file_type` 表示物理载体，`source_type` 表示业务来源。解析器先按显式 `file_type`、MIME、后缀决定物理解析，再按 `source_type` 进入专用标准化与 chunker。无法识别或解析失败的文档进入 quarantine，并在 manifest 记录错误，不静默按纯文本索引。

至少生成并接入下列数据：

| 数据 | file_type | source_type | 切分规则 |
| --- | --- | --- | --- |
| 官网 About/Products/News/Notice | HTML/Markdown | `official_website` | DOM 标题层级和 section；超长 section 再递归切分 |
| B2B 商铺和产品 | JSON/HTML | `b2b` | `企业 + 产品/品类/SKU` 为自然块 |
| 行业新闻 | JSON/HTML/PDF | `industry_news` | `标题 + 语义段落组`，标题注入每块 |
| 社媒/LinkedIn 导出 | JSONL | `social` | 一 Post 一块；长帖保留 parent post 后按段落切 |
| 文字型 PDF 报告 | PDF | `regulator`/`industry_news` | 页码 + 标题/段落；表格独立块 |
| 扫描 PDF | PDF | `regulator` | MinerU 优先，PyMuPDF/OCR 降级，保留页码、block 和置信度 |
| 海关月度画像 | generated JSON | `customs_profile` | 企业+国家+HS+月份一块，保留聚合口径和 raw record 摘要 |

长文本以 token 近似控制在 350–600 tokens、50–80 tokens overlap；自然边界优先于固定长度。最终参数由开发集 Recall@10 与 Context Precision 选择，初值不是已验证最优值。

PDF 的 MinerU 输出优先读取结构化 content-list/middle JSON，标准化为 page/block/table 单元；只有结构化输出不可用时才使用 MinerU Markdown。若 MinerU 不可用，文字 PDF 可以 PyMuPDF 降级；扫描 PDF 缺少可靠 OCR 时进入 quarantine，不能把空文本当作成功。

## 7. Canonical metadata

每个 Chunk 使用 Pydantic schema 校验。必填字段完整率必须为 100%。

| 字段 | 类型 | 可空 | 用途 |
| --- | --- | --- | --- |
| `chunk_id` | str | 否 | 稳定主键 |
| `document_id` | str | 否 | 原文档 ID |
| `entity_id` | str | 是 | 归一企业 ID |
| `company_name` / `normalized_name` | str | 是 | 展示与实体匹配 |
| `country_code` / `region` | str | 是 | 前置过滤 |
| `hs_code` / `product_name` / `sku` | str | 是 | 过滤与强关键词召回 |
| `file_type` | enum | 否 | 物理解析路由 |
| `source_type` | enum | 否 | 来源路由与治理 |
| `source_weight` | float[0,1] | 否 | 排序先验，不是真值 |
| `fact_type` | enum | 是 | 事实冲突分组 |
| `publish_time` | datetime | 是 | 发布时间 |
| `valid_from` / `valid_to` | datetime | 是 | 事实有效期 |
| `ingested_at` | datetime | 否 | 管道时间 |
| `source_url` / `canonical_url` | str | 是 | 引用与去重 |
| `source_locator` | object | 否 | 页/section/post/table/SQL locator |
| `raw_record_id` | str | 是 | 原始贸易事实定位 |
| `aggregation_info` | object | 是 | 画像窗口与聚合口径 |
| `content_hash` / `parent_document_hash` | sha256 | 否 | 版本与去重 |
| `language` | str | 否 | 分词/生成路由 |
| `ocr_confidence` | float | 是 | OCR 质量治理 |
| `is_synthetic` | bool | 否 | 数据真实性边界 |
| `license_scope` | str | 是 | 使用限制 |
| `dedupe_cluster_id` | str | 是 | 转载不重复计票 |

`source_weight` 由版本化的 `source_type + fact_type` profile 计算。例如历史交易金额优先海关，当前停业状态优先近期直接公告，风险事实优先监管/法院。它只影响候选优先级，冲突仲裁还必须考虑时效、一手性、独立性和口径。

## 8. Milvus Collection

Collection 名为 `trade_intel_chunks_<build_id>`，每个构建不可变。字段包括：

- `chunk_id` VARCHAR 主键；
- `dense_vector` FLOAT_VECTOR，维度从实际 embedding 模型探测并写入 contract；
- `text` VARCHAR；
- materialized scalar fields：`document_id`、`entity_id`、`country_code`、`region`、`hs_code`、`source_type`、`source_weight`、`fact_type`、`publish_time_epoch`、`valid_to_epoch`、`file_type`、`is_synthetic`、`canonical_url_hash`、`dedupe_cluster_id`；
- `metadata_json` 保存完整非过滤 metadata。

Dense index 使用 HNSW + COSINE。开发评测至少比较 `M={16,32}`、`efConstruction={128,256}`、`efSearch={64,128,256}`。查询把结构化 `RetrievalFilter` 编译成受控 Milvus expression，禁止 LLM 直接生成表达式。常用过滤字段为 `region`、`country_code`、`hs_code`、`entity_id`、`source_type`、`fact_type`、时间范围和 `is_synthetic`。

## 9. 检索与优化

检索流水线为：

```text
Query normalization / exact token preservation
  → Metadata Filter First
  → BM25 TopN 与 Milvus Dense TopN 并行
  → Weighted RRF
  → BGE Cross-Encoder Reranker
  → 来源多样性与去重
  → Top10 Evidence
```

BM25 tokenizer 必须保留企业法定名称、HS Code、型号、SKU 和字母数字混合 token；Dense 使用真实 BGE embedding。RRF 公式为：

`score(d) = source_prior(d, fact_type) × Σ retriever_weight_i / (k + rank_i(d))`

每条结果记录 BM25 rank/score、Dense rank/score、RRF components、source profile/version、rerank score、filter expression/version 和 degraded component。Reranker 失败不得静默伪装为完整链路。

评估臂固定为 Dense、BM25、Hybrid RRF、WRRF、WRRF+Filter、WRRF+Filter+Reranker。所有臂使用相同 snapshot、chunk、Query 和候选预算。

## 10. 实体归一、去重与多源事实融合

实体归一按：注册号/域名/精确别名 → 规范名称+国家 → 受限模糊匹配 → 人工复核队列。Embedding 只能用于候选生成，不能自动合并同名不同公司或母子公司。

去重使用 canonical URL、source document hash、content hash、转载关系和语义近重复，生成 `dedupe_cluster_id`。同一新闻的转载不能被计为多个独立来源。

冲突分组键至少包含 `entity_id + fact_type + validity window + unit/aggregation grain`。仲裁解释必须记录：

1. Source reliability；
2. Recency；
3. Directness；
4. Cross-source independence/consistency。

系统允许“历史采购活跃”和“当前停业”同时为真。若同时间、同口径的高权威证据冲突，状态为 `conflicted` 并升级/拒答，不允许 LLM 自动平均或覆盖。

## 11. Evidence 与幻觉治理

SQL 与 RAG 统一输出 `Evidence`：

```text
evidence_id, entity_id, fact_type, source_type, source_weight,
content/excerpt, source_url, source_locator, raw_record_id,
publish_time, valid_from, valid_to, confidence,
retrieval_provenance, sql_provenance, conflict_group_id
```

生成前 `EvidenceValidator` 根据 Query 所需 Claim 类型检查实体、地域、时间、货币/单位、来源权威、独立来源数量、新鲜度和冲突状态。证据不足时允许一次 Query Rewrite/Retry；仍不足则结构化拒答。

生成输出不是自由文本，而是 `Answer + atomic Claims + evidence_ids`。生成后 `ClaimHallucinationGuard` 检查：

- 引用 ID 存在且属于 generation context；
- locator/URL/raw record 可追溯；
- 数字、币种、时间窗口和聚合粒度与 SQL Evidence 一致；
- 实体、地域和法规有效期一致；
- Evidence 内容支持 Claim；
- 建议/预测明确标记为分析，不伪装成事实。

失败动作限定为 `DELETE_CLAIM`、`REWRITE_WITH_LIMITATION`、`RETRIEVE_ONCE` 或 `REFUSE`。核心 Claim 缺证据时必须拒答。

## 12. LangGraph 与 Redis

`StateGraph` 节点：

```text
policy_gate → router
  ├─ sql_plan_validate_execute
  └─ rag_retrieve
→ evidence_normalize → entity_resolve → dedup_conflict
→ evidence_validate → answer_draft → claim_guard → finalizer
```

SQL 与 RAG 在无依赖时并行。`TradeIntelState` 包括 request、route plan、filters、SQL/RAG results、evidence、entities、conflicts、claims、retry/LLM/step budget、errors 和 checkpoint ID。

Redis key 使用 `thread_id:run_id`，设置 TTL。Checkpoint 只保存状态、Evidence ID、结果 hash 和错误分类，不保存数据库全量结果或敏感原文。幂等节点带 idempotency key。

限制固定为可配置项：`MAX_GRAPH_STEPS`、`MAX_RETRIES`、`MAX_LLM_CALLS`、每节点 token/timeout/candidate budget。只有 timeout、429 和临时连接错误可重试；policy、schema、SQL AST、证据冲突等确定性错误不可重试。达到阈值进入明确 fallback。

## 13. API 与 CLI

FastAPI：

- `GET /health`：进程存活；
- `GET /ready`：MySQL/Milvus/Redis/build contract 可用；
- `POST /v1/query`：执行完整 Graph；
- `POST /v1/retrieve`：只执行检索并返回 trace；
- `GET /v1/runs/{run_id}`：读取状态；
- `POST /v1/runs/{run_id}/resume`：从 checkpoint 恢复；
- `GET /v1/evidence/{evidence_id}`：返回可公开的 Evidence locator。

请求模型含 question、country/region/HS/time/source filters、top_k、thread_id 和 answer mode。响应含 route、answer、claims、evidence、conflicts、refusal/degraded reasons、build/profile/version 和阶段延迟。

CLI 提供 `bootstrap-demo`、`db migrate/seed`、`ingest`、`index build`、`query`、`eval`、`smoke` 和 `verify-report`。

## 14. 评估闭环

公开开发集和本地私有 holdout 均使用：

```text
Question
Reference Evidence Set
Reference Key Claims
Reference Business Decision
```

Development 至少 36 题，覆盖企业/HS 精确查询、客户筛选、经营状态、新闻事件、竞品、冲突、数值聚合、多源综合和不可答问题。Private holdout 至少 15 题，不与开发集共享同文档版本、chunk hash 或近重复问题。私有文件加入 `.gitignore`，仓库只提交 hash/snapshot 元数据。

确定性指标：

- `Recall@10 = retrieved required evidence / all required evidence`；
- Context Precision = generation context 中 relevant evidence 比例；
- Context Recall = required key claims 被 context 覆盖比例；
- Conflict Accuracy 与 macro-F1；
- synthetic Lead Precision；
- Evidence Coverage = 有有效 Evidence 绑定的事实 Claim / 全部事实 Claim；
- URL/locator validity；
- 分阶段 cold/warm P50/P95 latency。

Faithfulness 使用确定性结构化检查加可选 LLM-as-Judge；Judge 必须记录 provider/model/prompt hash/temperature/coverage/error，并与人工抽检分开报告。没有外部 key 时报告 `judge_not_run`，不能用零分替代。

闭环流程：

1. 冻结 corpus、chunk、build、模型和开发集；
2. 跑 Dense baseline 和全部消融；
3. 按 `missed evidence`、filter false negative、keyword miss、stale source、wrong authority、dedupe/conflict、unsupported claim、reranker regression 分桶；
4. 只根据 development 改 query normalization、chunk、filter、权重、candidate 和 reranker；
5. 生成新 build ID 并重跑 development，保存 paired delta；
6. 候选冻结后只运行一次 private holdout；
7. 报告失败只能说明候选未通过，不允许针对同一 holdout 调参。

历史报告不可覆盖。每个 run 目录包含 report JSON/Markdown、manifest、config、per-query records 和 checksums。

## 15. 错误处理与安全

统一错误码至少包括：`policy_denied`、`schema_not_registered`、`schema_version_mismatch`、`sql_ast_rejected`、`sql_cost_limit_exceeded`、`sql_timeout`、`source_unavailable`、`source_stale`、`entity_ambiguous`、`evidence_insufficient`、`evidence_conflict`、`claim_unsupported`、`reranker_unavailable`、`checkpoint_unavailable`、`llm_budget_exceeded` 和 `step_limit_exceeded`。

法规、贸易金额、企业状态和高意向判断的核心证据缺失、过期、冲突或越权时 fail-closed。禁止 broad exception 后返回空结果并继续生成。日志不得包含密码、token、全量 SQL result、私有 holdout 标签或未脱敏原文。

## 16. 测试与验收

测试层级：

- Unit：schema、route、chunk 黄金样本、metadata、SQL AST、RRF、实体、去重、冲突、指标、Claim Guard；
- Contract：API、manifest、Milvus collection、checkpoint、snapshot、immutable report；
- Integration：synthetic corpus → MySQL/profile → ingest → Milvus/BM25 → retrieve；
- Milvus：真实 Compose SDK round-trip，不允许 fake/skip；
- E2E：SQL-only、RAG-only、SQL+RAG、冲突、证据不足、恢复；
- Security：SELECT-only、只读账户、secret/path scan、private holdout 不入 Git/image。

最终验收证据：

1. `docker compose up -d --wait` 全部健康；
2. 七表 DDL、外键、唯一键、索引及只读权限实测；
3. 每个 `source_type × file_type` fixture 成功解析或按预期 quarantine；
4. 每类 chunk 黄金样本和 metadata 100% 校验；
5. Milvus 实际 insert/search/filter/reconnect/cleanup；
6. BM25/Dense/WRRF/Reranker 全部有逐 Query trace；
7. LangGraph SQL+RAG trace、Redis 恢复、限制与 fallback；
8. Evidence Validator 和篡改 Claim 的拒绝测试；
9. Development baseline、优化后 paired delta、一次性 private holdout 报告；
10. 全测试通过、secret/host-path 扫描通过；
11. README 只引用新外贸实测结果，并保留 synthetic 与非生产限定；
12. 分支 diff 不包含旧电商运行路径，`origin/main` 不变。

## 17. 实施顺序

1. 删除旧产品树，建立 Python 包、配置、Compose、DDL 和 synthetic 数据生成器；
2. 以 TDD 实现 canonical schemas、数据路由、专用 chunkers 和 manifest；
3. 打通 MySQL、贸易画像、BGE embedding、真实 Milvus 与 BM25；
4. 实现 filter、WRRF、reranker、实体/去重/冲突；
5. 实现 Schema Registry、安全 Text-to-SQL 和 SQL Evidence；
6. 实现 Evidence Validator、Claim Guard、LangGraph 与 Redis checkpoint；
7. 实现 API/CLI、development/holdout 数据和评估器；
8. 启动真实 Compose，执行开发集闭环、优化、冻结候选和一次性 holdout；
9. 做逐项完成审计、更新文档、提交并推送工作分支。
