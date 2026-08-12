# 多源工程知识 RAG：运行与数据边界

## 项目边界

`ai-team-knowledge-base` 与 `Mini-Nanobot` 始终是两个独立项目：

- Mini-Nanobot 负责 Agent 执行、工具调用，以及通过 `rg`、AST、Git 对当前工作区进行实时核验；
- 本项目负责索引内部设计文档、历史决策和精选官方规范，完成检索、引用、拒答与评估；
- `knowledge.search` 只通过只读接口调用本项目，不复制 Mini-Nanobot 的执行权限，也不允许 RAG 服务修改代码仓库。

外部规范只能证明“规范要求什么”，不能证明 Mini-Nanobot 已经实现；内部设计文档只能证明设计意图；当前实现事实必须回到当前源码实时核验。

## 数据流

```text
Mini-Nanobot README / docs / ADR ─┐
授权的 PDF/表格/本地文档目录 ───────┼─ source catalog ─ document router ─ ingestion manifest
精选官方规范网页 ─────────────────┘                                      │
                                                                          └─ partitioned index
                                                                               ├─ BM25
                                                                               ├─ FAISS（始终构建）
                                                                               ├─ Milvus（可选严格镜像）
                                                                               └─ RRF hybrid

用户问题 ─ source-intent route ─ 索引召回 ─ 当前源码 rg/AST/Git 核验 ─ 引用 / 拒答
```

真实源码不依赖预构建向量索引来证明。索引中的代码片段最多用于定位候选，最终证据应标记是否完成实时核验。

## 数据源清单

实际源配置位于 `data/sources/catalog.yaml`，当前包含：

- Mini-Nanobot 的公开 README、`docs/`、Python 源码、测试、benchmark 描述和 Dockerfile；
- MCP 2025-06-18 lifecycle 与 tools 规范；
- LangChain Python overview；
- LangGraph persistence 官方文档；
- JSON Schema Draft 2020-12 core；
- Python 3.12 的 asyncio、pathlib、sqlite3 与 subprocess 文档；
- Docker security 与 resource constraints 文档。

`data/sources/catalog.example.yaml` 还展示了可选 `local_directory` source，用于纳入已授权的 Markdown、文本、PDF、CSV、XLSX、JSON、YAML 与 TOML。示例不会自动把任意下载目录加入正式 catalog；必须先确认内容授权、设置目录变量，再显式复制/调整对应条目。

目录扫描必须排除 `.env`、credential、日志、SQLite 运行库、缓存、模型权重和未授权笔记。抓取器必须使用 HTTPS 域名白名单并记录 canonical URL、内容哈希、版本和抓取时间。

## 文档解析与统一契约

`git_repository` 与 `local_directory` 共用 `DocumentLoaderRouter`，文件类型解析优先级固定为：catalog 的显式 `file_type` 覆盖 → 显式或推断 MIME → 文件后缀/`Dockerfile*` 等特殊文件名。解析结果统一为 `DocumentRecord`，并携带：

- `file_type`、`content_format`、`parser_backend`、`parser_version`；
- `page_number`、`sheet_name`、`table_id`、`block_id`（格式能提供时）；
- `parse_warnings` 与 `parse_degraded`，用于区分完整结构解析和降级提取。

普通文本、Markdown、代码、JSON、YAML 与 TOML 使用本地解析器；CSV 使用 Python `csv`，XLSX 使用 `openpyxl`，所有行列会转成 Markdown table，不做静默截断。PDF 的优先链路是：

```text
PDF ─ MinerU CLI（可选） ─ Markdown/JSON adapter ─ DocumentRecord
  └─ MinerU 缺失/超时/可恢复失败 ─ PyMuPDF 页面文本与表格 ─ parse_degraded=true
```

`MINERU_EXECUTABLE` 默认是 `mineru`。仓库没有捆绑 MinerU 模型或运行时；安装并验证其 CLI 后，路由器才会使用它。MinerU 的临时输出位于隔离临时目录，适配完成后自动清理。PyMuPDF 是已声明的轻量 fallback：它能处理有文本层的 PDF，并尝试提取表格，但不是扫描件 OCR；无可提取文字/表格的扫描 PDF 会明确报错。只有 `BackendUnavailable`/`RecoverableParseError` 会触发 fallback，程序错误不会被吞掉。

秘密文件、依赖/缓存目录和目录外符号链接不会进入语料。解析状态最终随 manifest/chunk 进入索引，因此页面引用和离线审计可以看到是否发生降级。

## 本地环境

推荐激活仓库声明的 Conda 环境后运行：

```powershell
conda activate all-in-rag
$env:PYTHONNOUSERSITE = "1"
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:TOKENIZERS_PARALLELISM = "false"
$env:MINI_NANOBOT_REPO = (Resolve-Path ..\Mini-Nanobot).Path
$env:ENGINEERING_INDEX_BUILD_BACKEND = "faiss"
$env:ENGINEERING_VECTOR_BACKEND = "faiss"
python -m pip check
```

模型下载阶段需要联网；本地 Embedding 的离线测试可另外设置 `$env:HF_HUB_OFFLINE = "1"`。这个变量只影响 Hugging Face，不会阻止 DeepSeek API 请求。完全离线运行还必须设置 `ENGINEERING_GENERATION_PROVIDER=deterministic`。密钥只写入未跟踪的 `.env` 或进程环境变量，`.env.example` 只能保留占位符。

## 推荐运行顺序

1. 检查 `data/sources/catalog.yaml` 的 Mini-Nanobot 路径、官方 URL 白名单和版本；
2. 只读采集各来源，生成包含内容哈希与增删改 diff 的 manifest；
3. 从同一 manifest 构建 `internal/code`、`internal/test`、`internal/design`、`internal/history`、`official/official` 物理分区；默认 `--backend faiss`，只有 Milvus 已准备好时才用 `--backend both`；
4. 查看健康检查，确认 build ID、分区数、文档数以及 live-code 核验根目录；
5. 先执行 `/retrieve` 或本地 retrieve 命令观察路由、证据角色和引用，再执行答案生成；
6. 分别运行 70 条 answerable 的纯索引消融与 80 条开发集 E2E 策略评测；二者不能混为同一报告；
7. 最后启动只读服务，通过根路径 Web 页面人工验收，再由 Mini-Nanobot 的 `knowledge.search` 调用。

若 `sources-sync` 报告旧 manifest schema（例如 1.0）不受当前 1.1 ingestion 支持，这是防止新旧元数据混写的 fail-closed 行为。确认要从当前来源迁移后，给该次同步增加 `--full-rebuild`；它不读取旧 snapshot，只发布全新的当前 schema。后续日常同步去掉该开关，才能继续得到同版本增量 diff。

构建后运行 `python -m scripts.freeze_engineering_eval`，把 index build、Mini commit/worktree hash 和三份题集 hash 写入 `evaluation_snapshot.json`。30 条 holdout 只在系统冻结后运行一次。

具体 CLI/API 命令以项目入口的 `--help` 与 OpenAPI schema 为准；不要让运维文档替代可执行参数校验。

## FAISS / Milvus 双后端

两种后端只替换 hybrid retriever 的稠密召回部分；BM25、分区路由、RRF、Evidence Guard、引用和实时源码核验保持同一实现。FAISS 使用归一化向量的本地 `IndexFlatIP`，适合离线开发与可移植快照。Milvus 每条记录保存稳定 `chunk_id`、向量、文本和完整 metadata JSON，当前索引参数为 HNSW/COSINE（`M=16`、`efConstruction=200`，查询 `ef=128`）。这些是可运行默认值，不是已经通过真实服务对照实验得到的最优参数。

### 构建模式

工程索引 catalog 当前版本是 v4。每次新构建都创建本地 FAISS、BM25、文档元数据和 SHA-256 校验；`--backend both` 或兼容别名 `--backend milvus` 还会为每个物理分区创建不可变、build-scoped 的 Milvus HNSW/COSINE collection。Milvus 任一分区写入或写后校验失败时，新 catalog 不发布，已经创建但未发布的 collection 会尽力删除。

```powershell
# 可移植的默认构建，无需 Milvus 服务
python main.py engineering-build `
  --manifest data/manifests/builds/current.json `
  --index-dir data/indexes/engineering `
  --backend faiss

# 严格双写；要求 Milvus URI、database 和可选 token 已正确配置
$env:ENGINEERING_MILVUS_NAMESPACE = "corner_dev_laptop"
python main.py engineering-build `
  --manifest data/manifests/builds/current.json `
  --index-dir data/indexes/engineering `
  --backend both
```

v4 catalog 记录 embedding model/dimension、`available_backends`、每个分区的 FAISS checksum 和 Milvus collection/owner/build/model/dimension 契约。`ENGINEERING_MILVUS_NAMESPACE` 是 Milvus 构建必填项，必须是当前部署稳定且唯一的所有者标识；它不是 collection 通用前缀，而是共享 database 中防止跨部署误删的安全边界。v3 FAISS-only catalog 仍可读取；它没有 Milvus artifact，不应手改 catalog 冒充双后端，应从原 manifest 重建 v4。

### 查询模式

| `ENGINEERING_VECTOR_BACKEND` / `--backend` | 行为 |
| --- | --- |
| `faiss`（默认） | 严格使用本地 FAISS，不连接 Milvus。 |
| `milvus` | 严格校验并使用所有 Milvus artifact；缺失、离线、鉴权或完整性错误都失败，不自动降级。 |
| `auto` | 有 Milvus artifact 时优先远端；仅连接拒绝、网络不可达、服务不可用或超时等明确可用性错误退回 FAISS；v3/FAISS-only catalog 也会回到 FAISS。 |

`auto` 不会掩盖鉴权/权限、非法参数、schema、行数、维度、build/model metadata 或未知错误；这些情况失败关闭。启动后和请求期状态可在 `GET /health` 的 `index.vector_backend` 查看：`requested_mode`、`primary_backend`、`active_backend`、`available_backends`、`fallback_used`、`degraded`、`reason_code` 和每分区结果。公开健康信息不包含 URI 或 token。

### 旧 collection 清理

新 build 不会自动删除旧 collection，因为上一代服务进程可能仍在读取旧 catalog。先预览由当前 `partitions.json` 判定为不再引用、且名称匹配工程前缀的候选：

```powershell
python main.py engineering-milvus-cleanup `
  --index-dir data/indexes/engineering `
  --namespace corner_dev_laptop
```

确认旧 reader 已排空、dry-run 候选正确后再显式删除：

```powershell
python main.py engineering-milvus-cleanup `
  --index-dir data/indexes/engineering `
  --namespace corner_dev_laptop `
  --execute
```

`--namespace` 必须与当前 v4 catalog 的 owner 完全一致。清理流程与构建共用索引运维锁，会完整加载并校验 catalog、只选择 description 中 owner 匹配的工程 artifact，并在每次删除前重新确认 catalog SHA；鉴权、连接或 describe 异常会中止，而不是显示为空候选。FAISS-only catalog 没有远端 owner：未显式指定 namespace 的 dry-run 只返回“没有可安全判定的远端候选”，`--execute` 则拒绝运行。

### 真实服务验证边界

常规测试通过状态化 fake client 验证 SDK 契约和故障分类，不等于分布式服务验证。当前开发机没有正在监听的 Milvus 服务，也没有 Docker 运行时，因此当前提交只能声称“双后端实现与离线合约测试通过”，不能声称本机已完成真实部署或性能测试。

有隔离测试服务后，使用只供测试的变量运行 opt-in round-trip。测试创建随机 `test_engineering_rag_*` collection，并在 `finally` 中删除；不要把它指向生产 collection 或把 token 写入版本库：

```powershell
$env:MILVUS_TEST_URI = "http://127.0.0.1:19530"
# 可选：$env:MILVUS_TEST_TOKEN / $env:MILVUS_TEST_DATABASE
python -m pytest -q -m milvus tests/test_engineering_vector_backends.py
```

未设置 `MILVUS_TEST_URI` 时测试应显示 skip。要形成可用于面试的“真实 Milvus 已验证”证据，至少保留服务版本、部署方式、测试命令、成功日志和相同语料/Embedding 配置下的延迟与召回对照；不要拿 fake-client 通过或 skip 当成功结果。

### 正式评测保持 FAISS 冻结

`engineering-eval` 的默认 predictor 显式以 `runtime_backend="faiss"` 加载索引，忽略进程中的 `ENGINEERING_VECTOR_BACKEND`。这是为了保证正式消融只改变被声明的检索策略，不把在线 Milvus 可用性、网络延迟或自动降级混入已冻结指标。新增双后端不会重写 `data/eval/reports_public/` 的 first-run holdout 报告；需要比较 FAISS 与真实 Milvus 时，应使用新的独立实验名称、同一 manifest/Embedding/query set，并保留新的配置和报告，不能覆盖历史文件。

## 本地 Web 页面

页面和 API 使用同一 FastAPI 进程，不需要单独的前端构建或第二个端口：

```powershell
$env:ENGINEERING_INDEX_DIR = "data/indexes/engineering"
$env:ENGINEERING_MANIFEST_PATH = "data/manifests/builds/current.json"
$env:MINI_NANOBOT_REPO = (Resolve-Path ..\Mini-Nanobot).Path
$env:ENGINEERING_VECTOR_BACKEND = "faiss"
$env:RAG_API_TOKEN = python -c "import secrets; print(secrets.token_urlsafe(32))"
$env:RAG_UI_PORT = "8000"
python app.py
```

打开 `http://127.0.0.1:8000/`；若启用了鉴权，在查询面板的“连接设置”中输入本次进程使用的 token。收到 401 时该设置会自动展开。页面必须满足以下边界：

- token 仅保留在 JavaScript 内存，不写入 URL 或 Web Storage；
- 所有证据以纯文本节点渲染，不执行检索内容中的 HTML、脚本或命令；
- 只有官方白名单中的 HTTPS 来源会显示为可点击外链；本地源码路径只显示为文本；
- 页面只提交 `query` 和 `top_k`，不能覆盖仓库、索引或服务端路径；
- 正常拒答和候选证据必须与系统故障使用不同状态展示；
- 回答中的 `[E#]` 只映射到本次 API 响应声明的站内引用目标，未知编号保持纯文本；
- 不提供 source sync、index build、evaluation 或其他写操作。

前端拉丁字符使用仓库内自托管的 DM Sans variable font，`web/assets/fonts/OFL-DMSans.txt` 保留 SIL Open Font License；中文字符回退到操作系统的 `PingFang SC`、`Microsoft YaHei UI`/`Microsoft YaHei` 等字体。页面不访问外部字体 CDN，也没有复制字节跳动的专有品牌字体。验收时至少检查 Windows 中文正文、数字/英文指标、代码等宽字体、400/500/600/700/800 字重和窄屏布局。

`python main.py serve --host 127.0.0.1 --port 8000` 与 `python app.py` 启动的是同一应用。内置服务器仍然只允许 loopback；远程访问必须放在带 TLS 和鉴权的反向代理之后。

## DeepSeek 回答生成与隐私边界

DeepSeek 只负责把已经通过证据门控的检索结果归纳为易读答案，不参与查询路由、索引召回、`rg`/AST/Git 实时核验、证据充分性判断或拒答决策。`POST /retrieve` 始终是确定性的，且不会调用 DeepSeek；`POST /answer` 也会先完成同一条确定性检索链路，证据不足时直接拒答，不把问题发送给模型。

使用 `ENGINEERING_GENERATION_PROVIDER` 选择回答模式：

| 值 | 行为 | 适用场景 |
| --- | --- | --- |
| `auto` | 有有效 `DEEPSEEK_API_KEY` 时调用 DeepSeek；未配置时返回 `evidence_only`，不伪造模型摘要 | 已确认外部传输边界的交互演示 |
| `deepseek` | 要求 DeepSeek 配置完整；缺少有效密钥属于启动/首次加载配置错误 | 明确要求在线生成的受控环境 |
| `deterministic`（默认） | 不创建在线客户端，不发送任何问题或证据 | 离线测试、正式评测、敏感语料 |

模型调用失败、超时、返回空内容或使用不存在的 `[E#]` 引用时，Answerer 不改变检索结果，也不会把原始片段拼接成答案；它返回“本次不生成归纳答案”的明确状态，并通过 `generation.status=fallback`（兼容字段 `generation_mode=deterministic_fallback`）和 warning 标明原因。成功的模型回答使用 `generation.status=model`；未启用模型时为 `evidence_only`。不要把 fallback 当作模型生成成功，也不要通过重试绕过 Evidence Guard 的拒答。

`GET /health` 只报告 DeepSeek 客户端是否已配置，不会为探活额外发起模型请求，也不承诺密钥有效。凭据、网络和模型可用性以实际 `/answer` 的 `generation_mode` 为准。

启用 `auto` 或 `deepseek` 是一次外部数据传输决策：用户问题和本次选中的证据片段会发送到 `DEEPSEEK_BASE_URL`，其中可能包含源码、设计文档或历史记录。发送前必须确认这些内容允许交给对应服务处理。未经授权的私有仓库应固定使用 `deterministic`，或改用经过组织批准的兼容服务。服务不会把 API 密钥放入 prompt、响应、页面或结构化日志；运维人员也不应打印环境变量值排查问题。

以下生成参数可以通过进程环境或未跟踪的 `.env` 配置：`DEEPSEEK_API_KEY`、`DEEPSEEK_BASE_URL`、`DEEPSEEK_MODEL`、`ENGINEERING_LLM_TIMEOUT_SECONDS`、`ENGINEERING_LLM_MAX_RETRIES`、`ENGINEERING_LLM_MAX_TOKENS`、`ENGINEERING_LLM_TEMPERATURE` 和 `ENGINEERING_LLM_THINKING_ENABLED`。远程 Base URL 必须使用 HTTPS；只有 loopback 本地开发端点允许 HTTP。不要把真实值写进运行手册、截图、命令历史或 Git。

## 上线前最小检查

- manifest 中每个官方 `canonical_url` 与评测题 `relevant_sources` 一致；
- Mini-Nanobot revision、dirty 状态和索引 build ID 可追溯；
- 对实现类问题，返回结果包含 `current_implementation` 或明确拒答；
- 对官方类问题，引用只来自 allowlist 中的官方域名；
- 对比较类问题，同时具备当前实现与外部规范两侧证据，否则拒答；
- `/retrieve` 不触发在线生成；`/answer` 的 `generation_mode` 与 provider 配置相符，模拟模型故障时能回退且不泄露异常正文；
- 在线生成启用前，已确认用户问题和选中证据允许发送到配置的 DeepSeek 端点；
- API 限制 query 长度和 Top-K，且不能由请求传入任意本地根目录；
- `knowledge.search` 标记为 read-only，网络超时、服务离线、非 2xx、无效 JSON 都返回受控错误；
- `.env`、token、原始抓取快照和本机索引不进入 Git。

## 更新策略

Mini-Nanobot 提交变化后：

1. 读取新 commit 和 dirty 状态，不修改其工作区；
2. 重跑采集并根据 hash 生成 added/modified/deleted diff；
3. 从新的 manifest 全量重建内部索引；当前工程索引发布流程不宣称支持增量更新；
4. 核验 60 条内部题和 30 条 holdout 的路径、AST symbol 与答案点，再通过 freeze 脚本更新 `source_revision`；
5. 跑回归评测并与同配置 baseline 比较。

官方网页变化后：

1. 保留 URL、抓取时间、内容 hash 和 source version；
2. 检查许可与来源条款，不把“可访问”自动理解为“可任意再分发”；
3. 重新核对对应的 20 条官方题；
4. 生成新 manifest 与评测报告，保留旧报告的配置说明。

## 故障定位

| 现象 | 优先检查 |
| --- | --- |
| 设计题只返回源码 | router 意图、`internal/design` 分区和 authority 元数据 |
| 实现题有答案但无实时证据 | `MINI_NANOBOT_REPO`、`rg` 可执行文件、路径边界和 verification terms |
| 官方题 Recall 为 0 | catalog URL 与题集 canonical URL 是否完全一致、抓取是否被重定向或拒绝 |
| 混合检索劣于 BM25 | RRF 权重、候选数、重复 chunk、中文与标识符 tokenization |
| 不可回答题仍生成答案 | evidence sufficiency 与 `refused` 是否由答案层显式输出 |
| `/answer` 仍是证据拼接摘要 | `ENGINEERING_GENERATION_PROVIDER`、密钥是否有效，以及响应中的 `generation_mode` 与 warnings；不要打印密钥 |
| DeepSeek 失败后页面报 500 | provider 异常是否被 Answerer 捕获并进入 `generation.status=fallback`；fallback 应保留 Top-K 证据但不伪造归纳答案 |
| 首次请求明显更慢 | 嵌入或 reranker 冷启动；分开记录冷启动和预热后延迟 |
| `--backend milvus` 构建或启动失败 | 先确认设置了部署唯一的 `ENGINEERING_MILVUS_NAMESPACE`，catalog v4 含同一 owner 的 Milvus artifact，再检查 URI/database/最小权限；严格模式下服务离线也不会回退 |
| `auto` 没有使用 Milvus | 查看 `index.vector_backend.reason_code`；v3/FAISS-only catalog 会报告 artifact 未构建，连接故障会报告 unavailable |
| `auto` 因鉴权或 schema 错误失败 | 这是预期的 fail-closed；修复凭据或重建匹配 artifact，不要扩大降级异常范围 |
| PDF 被标记 degraded | 查看 `parser_backend`、`parse_warnings`；通常是 MinerU 不可用/失败后使用了 PyMuPDF，不代表数据丢失但需要抽检结构 |
| 扫描 PDF 无内容 | PyMuPDF 没有 OCR；安装并验证 MinerU/OCR 能力或提供可搜索文本层，不要把空结果写入索引 |

## 面试叙述边界

可以陈述已运行、已测试且有报告支持的能力；不要把计划能力写成完成能力。尤其不要声称：

- Milvus 或 FAISS 本身提高了语义准确率；
- 当前开发机已完成真实 Milvus 分布式部署、可用性或性能验证；只有 opt-in round-trip 真正在隔离服务上成功并留证后才能这样描述；
- 外部规范检索证明了内部实现符合规范；
- 自动指标等价于人工正确性；
- 教学型 Mini-Nanobot 已达到生产级沙箱、安全认证或 SLA；
- 没有固定配置与实测报告时，某策略是“最优策略”。
