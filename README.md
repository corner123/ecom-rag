# 多源工程知识 RAG

这是一个面向 Mini-Nanobot 工程资料的、可追溯的 RAG 项目。它把三类证据明确分开：

> 项目状态：可本地运行的工程实验系统，已提供 Web 演示、冻结评测记录和公开报告；它不是生产 SLA、安全认证或企业级部署证明。

- 内部项目知识：README、架构文档、ADR、安全说明和 Git 历史；
- 外部官方规范：LangChain overview、MCP、LangGraph persistence、JSON Schema Draft 2020-12、Python 3.12 与 Docker 官方文档；
- 当前源码事实：不把旧向量块当作最终证明，而是通过 `rg`、Python AST 与 Git 对 Mini-Nanobot 当前工作区实时核验。

Mini-Nanobot 和本仓库始终是两个独立项目。Mini-Nanobot 负责 Agent 执行和实时代码搜索；本仓库负责知识采集、索引、检索、引用、拒答和评估。两者只通过只读 HTTP 接口 `POST /retrieve` 集成，Mini-Nanobot 将它注册为可选工具 `knowledge.search`。

## 为什么这样设计

源码中的函数、调用点和未提交修改更适合 `rg`、AST、Git：实时、精确且不需要维护易过期的索引。RAG 更适合回答“为什么这样设计”“历史上做过什么决策”“官方规范如何要求”等非结构化知识问题。

系统按问题类型选择证据：

```text
用户问题
  ├─ 当前实现 ───── internal code/test 候选 + rg/AST/Git 实时核验
  ├─ 设计/历史 ─── internal design/history + 代码作为补充证据
  ├─ 官方规范 ──── allowlist 官方快照
  ├─ 实现与规范比较 ─ 同时要求实时源码证据和官方证据
  └─ 超出范围 ──── 返回稳定 refusal_reason，不根据近邻文本猜测
```

证据角色不会混用：

| 角色 | 能证明什么 | 不能证明什么 |
| --- | --- | --- |
| `current_implementation` | 当前工作区中的实现事实 | 历史设计动机或官方要求 |
| `internal_design` | 架构意图、边界和权衡 | 当前代码一定已实现 |
| `internal_history` | Git 历史与演进 | 当前版本仍保持相同行为 |
| `external_normative` | 外部官方规范的要求 | Mini-Nanobot 已经符合规范 |

## 已实现链路

1. YAML source catalog 声明本地 Git 仓库与官方 URL allowlist；
2. 只读采集器记录 commit、dirty 状态、内容 SHA-256、抓取时间和来源版本；
3. Markdown 结构化分块、Python AST symbol card、测试、Dockerfile 与 Git history 统一进入 build manifest；
4. 内容物理分为 `internal/code`、`internal/test`、`internal/design`、`internal/history`、`official/official`；
5. 每个分区同时建立 BGE-small-zh-v1.5 FAISS 稠密索引和 BM25 索引，使用 RRF 融合；
6. 确定性路由、官方 topic scope、hard-anchor 与经验数据缺失检查共同决定回答或拒答；
7. 返回结构化引用、证据角色、源码行号、symbol、commit/dirty 和官方内容 hash；
8. 提供纯索引消融与完整检索策略两套独立评测，数据集、manifest 和索引 build 强绑定；
9. `/answer` 可在证据充分性检查通过后调用 DeepSeek，把选中的 Top-K 证据归纳为带 `[E#]` 引用的结构化回答；模型未启用或调用失败时明确返回“仅证据/生成失败”状态，原始证据仍可单独展开，但不会伪装成 AI 总结。

可选的 Cross-Encoder reranker 已封装但默认关闭；只有在真实下载模型并完成同配置实验后才应把其结果写进简历。检索、路由、实时核验、证据充分性、拒答和正式评测不依赖在线 LLM；DeepSeek 只作为 `/answer` 的可选证据归纳层。本项目的工程主链路不依赖 Milvus、RAGAS 或 Gradio。

## 快速开始

前置依赖：Git、Conda、Python 3.12、`rg`（ripgrep）。首次同步官方资料和下载 Embedding 模型需要联网；当前主要在 Windows PowerShell + Conda 环境验证。

两个项目保持独立，推荐克隆到同一个父目录：

```powershell
git clone https://github.com/corner123/Mini-Nanobot.git
git clone https://github.com/corner123/AI-TEAM-KNOWLEDGE-BASE.git
Set-Location AI-TEAM-KNOWLEDGE-BASE

conda env create -f environment.yml
conda activate all-in-rag
Copy-Item .env.example .env

# 如果两个仓库不是同级目录，请换成 Mini-Nanobot 的实际路径
$env:MINI_NANOBOT_REPO = (Resolve-Path ..\Mini-Nanobot).Path
python -m pip check
```

已有 `all-in-rag` 环境时使用下面的命令更新：

```powershell
conda env update -n all-in-rag -f environment.yml --prune
conda activate all-in-rag
```

新克隆的仓库不包含原始网页快照、manifest 或本地 FAISS 索引，必须先完成“构建最终快照”，再启动 Web。密钥只放在未跟踪的 `.env` 或进程环境变量中；`.env.example` 只包含占位符。DeepSeek 仅用于可选的在线答案归纳，基础检索可以完全离线运行。

模型已经缓存时可设置 `$env:HF_HUB_OFFLINE = "1"` 做可复现的本地 Embedding 运行。该变量只影响 Hugging Face，**不会**阻止 `/answer` 访问 DeepSeek API；完全离线时还必须设置 `$env:ENGINEERING_GENERATION_PROVIDER = "deterministic"`。

### 主要环境变量

| 变量 | 必需性 | 默认/用途 |
| --- | --- | --- |
| `MINI_NANOBOT_REPO` | 构建与实时核验必需 | Mini-Nanobot 本地仓库；示例按同级目录配置 |
| `ENGINEERING_INDEX_DIR` | 推荐 | `data/indexes/engineering` |
| `ENGINEERING_MANIFEST_PATH` | 推荐 | `data/manifests/builds/current.json` |
| `ENGINEERING_GENERATION_PROVIDER` | 可选 | 默认 `deterministic`；确认外部传输边界后可选 `deepseek` 或 `auto` |
| `DEEPSEEK_API_KEY` | 仅在线回答必需 | 只供 `/answer` 使用，不得提交到 Git |
| `DEEPSEEK_BASE_URL` / `DEEPSEEK_MODEL` | 可选 | DeepSeek 兼容端点与模型名 |
| `RAG_API_TOKEN` | 可选 | 本地 API 令牌；留空时 loopback 访问不鉴权 |
| `RAG_UI_PORT` | 可选 | `8000` |

## 构建最终快照

```powershell
# 1. 采集 Mini-Nanobot 和官方白名单页面，写 manifest
python main.py sources-sync `
  --catalog data/sources/catalog.yaml `
  --manifest data/manifests/builds/current.json

# 2. 从同一 manifest 构建带校验和的分区索引
python main.py engineering-build `
  --manifest data/manifests/builds/current.json `
  --index-dir data/indexes/engineering

# 3. 将题集 hash、Mini commit/worktree hash 与 build ID 固化
python -m scripts.freeze_engineering_eval
```

`data/sources/catalog.yaml` 是工程 RAG 实际使用的受控来源清单，其中 LangChain 仅采集 `docs.langchain.com` 的官方 Python overview。`data/raw/docs/` 下的历史下载文件不会自动进入工程索引。修改 catalog 后必须依次重新执行 `sources-sync` 和 `engineering-build`；只重启 `app.py` 不会更新语料。

### 离线建库与在线提问不是同一步

1. `sources-sync` 读取受控来源、解析文档并切分 chunk，结果写入 manifest；它不处理用户问题。
2. `engineering-build` 对这些 chunk 批量计算 Embedding，并把向量索引、BM25 文本索引和来源元数据持久化到 `data/indexes/engineering/`。首次建库或语料变更后执行一次即可，不需要每次提问前重建。
3. `app.py` 启动时只加载已经落盘的索引。每次提问只计算“问题本身”的一个查询向量，再结合 BM25、RRF、Evidence Guard，以及实现类问题需要的 `rg` / AST / Git 实时核验来选出证据。
4. 页面中的“仅检索证据”调用 `/retrieve`，适合观察召回内容，不调用 DeepSeek；“检索并生成回答”直接调用 `/answer`，后端会自动先执行同一套检索和证据检查，再在证据充分时生成回答，因此不需要先点一次“仅检索证据”。

也就是说，正常使用顺序是“语料更新后离线建库一次 → 启动服务 → 可以连续提问”；不是“每问一句就重新向量化全部文档”。

索引使用跨进程 OS 文件锁、staging 校验、备份回滚和每个分区文件的 SHA-256；空 manifest、重复 chunk、来源错配或损坏文件会拒绝发布。锁文件路径保持稳定，进程退出时由操作系统释放锁，避免 PID 探测和空锁文件竞态。

当前本地索引发布不提供无停机并发读取保证：目录替换期间，并发读者可能短暂看不到索引根目录。构建和发布最终快照时应暂停 API 读流量；若需要无停机部署，应在服务外使用不可变版本目录与原子版本指针，不能把当前备份回滚机制描述成生产级热切换。

## 启动 Web 页面、查询与只读 API

推荐直接启动同源的 Web 页面与只读 API：

```powershell
conda activate all-in-rag
$env:PYTHONNOUSERSITE = "1"
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:MINI_NANOBOT_REPO = (Resolve-Path ..\Mini-Nanobot).Path
$env:ENGINEERING_INDEX_DIR = "data/indexes/engineering"
$env:ENGINEERING_MANIFEST_PATH = "data/manifests/builds/current.json"
# 完整 RAG 演示：检索后调用 DeepSeek 归纳证据。
# 需要在未跟踪的 .env 或 Conda 环境中配置有效 DEEPSEEK_API_KEY。
$env:ENGINEERING_GENERATION_PROVIDER = "deepseek"

# 可选：启用本地 API 鉴权；留空则仅依赖 loopback 网络边界
$env:RAG_API_TOKEN = python -c "import secrets; print(secrets.token_urlsafe(32))"
python app.py
```

如果只想离线检查召回结果，不允许把问题和证据发送到外部模型，请把上面的 provider 改为：

```powershell
$env:ENGINEERING_GENERATION_PROVIDER = "deterministic"
```

浏览器打开 `http://127.0.0.1:8000/`；若启用了鉴权，展开查询面板中的“连接设置”，输入同一个 `RAG_API_TOKEN`。令牌只保留在当前页面内存中，不会写入 URL、HTML 或浏览器存储。收到 401 时页面会自动展开并提示该设置。首次健康检查需要加载本地 Embedding 模型，可能比后续请求慢。

页面是 RAG 仓库的只读观察与演示界面，只提供健康检查、证据检索和基于证据的回答；不会提供语料同步、索引构建、评测上传或修改 Mini-Nanobot 的入口。它使用仓库内独立实现的原生 HTML/CSS/JS，由 FastAPI 同源托管，不需要 Node、CDN 或 Gradio。

### 单页问答工作台

本地页面围绕一次完整 RAG 请求收敛为一个紧凑工作台：

- **主路径唯一**：问题输入、模式选择和提交按钮放在同一查询区；“检索并生成回答”会直接执行检索，不要求先点“仅检索证据”。
- **回答与证据分层**：主区域只展示模型综合回答或清晰的未启用/失败/拒答状态；Top-K 原始片段默认折叠，需要核查时再展开。
- **流水线可观察**：紧凑状态条展示路由、证据充分性、生成方式和耗时，并区分检索到的证据、实际送入模型的上下文以及回答实际引用的证据。
- **高级信息按需展开**：Top-K、访问令牌、证据规则、索引 build 与 Mini-Nanobot revision 不再占据起始页的主要空间。
- **边界保持不变**：页面仍只调用 `GET /health`、`POST /retrieve` 和 `POST /answer`，不会上传文档、创建或删除知识库、同步语料、构建索引、修改 Mini-Nanobot，或引入多轮聊天状态。
- **安全与可访问性**：令牌只驻留页面内存；保留同源请求、内容安全策略、引用跳转、键盘操作、移动端布局、焦点恢复和 `aria-live` 通知。

#### 设计参考与归属

页面的信息层级受到 [RAG Web UI](https://github.com/rag-web-ui/rag-web-ui)（Apache-2.0）的视觉启发，但针对本项目的只读工程知识流程独立实现。仓库没有复制上游 logo、图片、静态资源或前端源码，也没有宣称或复用上游的文档上传、多轮聊天、知识库 CRUD、API Key 管理等本项目并不支持的能力。

`ENGINEERING_GENERATION_PROVIDER` 有三种模式：

| 模式 | `/answer` 行为 |
| --- | --- |
| `auto` | 有有效 `DEEPSEEK_API_KEY` 时使用 DeepSeek；未配置时只返回检索证据，不生成 AI 总结。 |
| `deepseek` | 强制启用 DeepSeek；缺少有效密钥会把服务视为配置错误。 |
| `deterministic`（默认） | 永不调用在线模型；明确标记为 `evidence_only` 并只返回结构化检索证据，适合离线运行和可复现测试。 |

`/retrieve` 在三种模式下都不会调用 DeepSeek。路由、实时核验、Evidence Guard 和拒答先确定性执行；只有证据充足的 `/answer` 才会把用户问题和本次选中的证据片段发送给 DeepSeek。模型超时、鉴权失败、服务异常、空响应或返回无效引用时，系统会保留原检索结果，但不会把原文拼接伪装成 AI 总结。

`/answer` 使用 `engineering-answer/v2` 契约分开描述三层证据：

- `retrieved_evidence`：本次检索返回的完整 Top-K，顺序和 `[E#]` 编号不变；
- `generation_context`：经过同页重叠去重、定义类问题的 prose/overview 优先和上下文预算后，被选为模型输入的证据条目子集；响应保留完整 chunk 便于审阅，构造提示词时仍会按单条与总字符预算截断正文；
- `answer_citations`：模型回答文本中真正引用的证据子集。

`generation` 同时返回 `model | evidence_only | fallback | refusal`、是否尝试/成功、provider/model、非敏感失败码以及召回/上下文数量。旧的 `generation_mode`、`generation_provider` 和 `citations` 字段暂时保留兼容。

健康检查中的“DeepSeek 已配置”只表示已创建在线生成客户端，不会主动发送付费请求验证密钥。一次 `/answer` 返回 `generation.status=model`（兼容字段为 `generation_mode=model`）才代表本次模型调用和引用校验均成功；`generation.status=fallback` 表示检索成功，但模型调用或输出校验失败。

这是明确的隐私边界：启用 `auto`/`deepseek` 后，发送内容可能包含 Mini-Nanobot 的源码、内部设计或历史片段。不要对无权发送给第三方的私有语料启用在线生成；这类场景使用 `deterministic`，或把 `DEEPSEEK_BASE_URL` 指向经过授权的兼容服务。密钥不得放入 URL、页面输入、命令历史、日志或版本库。

本地结构化检索：

```powershell
python main.py engineering-query `
  "Mini-Nanobot 为什么使用 SQLite snapshot checkpoint？" `
  --index-dir data/indexes/engineering `
  --mini-repo $env:MINI_NANOBOT_REPO `
  --top-k 5
```

也可以直接使用 CLI 启动同一个页面和 API：

```powershell
$env:ENGINEERING_INDEX_DIR = "data/indexes/engineering"
$env:ENGINEERING_MANIFEST_PATH = "data/manifests/builds/current.json"
python main.py serve --host 127.0.0.1 --port 8000
```

端点：

- `GET /`：本地工程知识检索页面；
- `GET /health`：build freshness、分区和实时核验状态，不泄露本机绝对路径；
- `POST /retrieve`：返回 `engineering-retrieval/v1`、路由、充分性、`refusal_reason`、引用和证据；
- `POST /answer`：自动执行“Top-K 检索 → 证据门控 → 上下文去重 → 可选模型综合 → 引用校验”；证据不足时在调用模型前拒答，模型失败时保留证据并返回明确失败状态。

内置 Uvicorn 入口只允许 loopback 绑定；远程访问必须在其前面配置带 TLS 与鉴权的反向代理。Mini-Nanobot 对任何非 loopback 地址都要求 HTTPS，并拒绝 30x 跳转，防止查询内容或 Authorization 泄漏。

### 演示问题

| 类型 | 示例 | 预期表现 |
| --- | --- | --- |
| 当前实现 | `当前 QueryEngine.submit_message 在哪个文件实现？它的主要处理流程是什么？` | 路由到 implementation，返回 `LIVE VERIFIED` 源码、symbol、行号和引用 |
| 内部设计 | `为什么 Mini-Nanobot 使用手写 ReAct 循环？` | 以 ADR/架构文档说明动机，代码只作补充证据 |
| 官方框架概览 | `什么是 LangChain？` | 只引用受控的 LangChain 官方 Overview；大小写不同仍使用同一路由 |
| 官方规范 | `根据 MCP 2025-06-18 官方规范，tools/list 和 tools/call 是什么？` | 只引用白名单官方规范，并声明不能反向证明项目已实现 |
| 实现与规范比较 | `对比当前 MCPToolAdapter 与 MCP 官方 tools 规范是否一致？` | 同时要求实时实现与官方规范两侧证据，否则拒答 |
| 超出范围 | `根据 Kubernetes 官方规范，PodSecurity admission 如何配置？` | 返回稳定的 out-of-scope 拒答，不用相似近邻猜测 |

## Mini-Nanobot 集成

Mini-Nanobot 仓库中设置：

```powershell
$env:RAG_API_BASE_URL = "http://127.0.0.1:8000"
$env:RAG_API_TOKEN = "<与 RAG 服务相同的随机令牌>"
$env:RAG_API_TIMEOUT_SECONDS = "15"
```

未设置 `RAG_API_BASE_URL` 时，`knowledge.search` 不注册，Mini-Nanobot 的其他工具不受影响。该工具固定只提交 `query` 与 `top_k`，不能指定服务端路径、仓库或索引，也没有任何写权限。

## 评测

三份 v2 数据：

| 数据 | 数量 | 作用 |
| --- | ---: | --- |
| `mini_nanobot_internal.jsonl` | 60 | 开发回归：设计、代码、安全、状态与拒答 |
| `official_engineering_specs.jsonl` | 20 | 开发回归：五类官方规范 |
| `engineering_holdout_v2.jsonl` | 30 | 冻结 holdout：内部、官方、跨语料比较、OOD 与 false-refusal controls |

### 已冻结的本轮结果

本轮快照为 `build_4ae47172a9869d88ce0f`：6 个来源组（1 个 Mini-Nanobot Git worktree、5 组官方资料），339 份文档、962 个 chunk；五个物理分区分别为 code 485、test 59、design 35、history 10、official 373。Embedding 使用 `BAAI/bge-small-zh-v1.5`。

该历史快照明确记录 `Mini-Nanobot dirty=true`，因此公开报告能证明这次本地运行及其数据绑定，但不能仅凭 Mini-Nanobot 远程仓库逐字重建当时的工作区内容。全新克隆应重新执行 source sync、index build 和 freeze，产生新的 build ID；在对应 Mini-Nanobot 变更正式提交前，不把本轮数字描述为跨机器完全可复现。

70 条 answerable 开发题的纯索引消融（Top-5、oracle route、相同候选预算）：

| 策略 | Primary Hit@5 | Supporting Recall@5 | nDCG@5 | MRR | P95 ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| BM25 | 0.629 | 0.213 | 0.488 | 0.472 | 5.53 |
| Dense | 0.400 | 0.227 | 0.350 | 0.301 | 138.99 |
| Hybrid RRF | 0.457 | 0.277 | 0.385 | 0.323 | 159.17 |

BM25 在 Primary Hit、nDCG 和 MRR 上最好；Hybrid 只在 Supporting Recall 上最高。当前语料规模较小，且问题含大量类名、方法名与协议字段，因此不能声称“向量或混合检索必然优于关键词检索”。完整报告见 [index ablation](data/eval/reports_public/index_ablation_build_4ae47172a9869d88ce0f.md)。

端到端策略结果：

| 数据 | Route accuracy | Route macro-F1 | Primary Hit@5 | Primary Recall@5 | Live primary hit@5 | 拒答 F1 | 可答题误拒答率 | P95 ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 开发集 80 题 | 1.000 | 1.000 | 0.886 | 0.886 | 0.714 | 0.833 | 0.057 | 1412.40 |
| 首次冻结 holdout 30 题 | 0.700 | 0.626 | 0.870 | 0.775 | 0.786 | 0.636 | 0.348 | 1425.26 |

开发集路由是显式回归集，1.000 不能外推为泛化能力。holdout 含 23 个可答题和 7 个不可答题：7 个不可答题全部拒答，但 8/23 可答题被误拒答；主要问题是 `unsupported_anchor` 等证据门控过严。报告在第一次运行后原样保留，未据此继续调参：[development E2E](data/eval/reports_public/e2e_development_build_4ae47172a9869d88ce0f.md)、[first-run holdout](data/eval/reports_public/e2e_holdout_first_run_build_4ae47172a9869d88ce0f.md)。

纯索引消融只使用 answerable 开发题，并使用 oracle route，关闭 Router、实时检索和 Answerer；因此报告不计算路由成绩：

```powershell
python main.py engineering-eval `
  --suite index `
  --dataset data/eval/mini_nanobot_internal.jsonl data/eval/official_engineering_specs.jsonl `
  --snapshot data/eval/evaluation_snapshot.json `
  --output data/eval/reports/index_ablation
```

端到端检索策略评测固定 hybrid index，运行 Router、rg/AST/Git、证据充分性与拒答：

```powershell
python main.py engineering-eval `
  --suite e2e `
  --dataset data/eval/mini_nanobot_internal.jsonl data/eval/official_engineering_specs.jsonl `
  --snapshot data/eval/evaluation_snapshot.json `
  --output data/eval/reports/e2e_development
```

holdout 已在设计冻结后运行一次并原样保留报告，随后连同题目公开用于审计，因此不能再作为未来迭代的未见测试集。后续不再据其失败调参；需要最终评测时应先建立新的私有 holdout。当前指标是 source-level Primary Hit/Recall、Supporting Recall、MRR、graded nDCG、精确 symbol recall、路由 macro-F1、拒答 precision/recall/F1、拒答原因和预热后 P50/P95。它们不能冒充 passage entailment、答案正确率、RAGAS faithfulness 或生产 SLA。

正式 `engineering-eval` 不调用 DeepSeek：纯索引 suite 关闭 Answerer，E2E suite 使用确定性 Answerer 评估路由、检索、实时核验和拒答。页面中更流畅的模型回答不属于上述冻结指标，也不能据此声称答案正确率或忠实度得到量化提升。

## 测试

RAG 仓库测试：

```powershell
python -m pytest -q
```

真实 loopback API、token、严格请求 schema、Mini registry/executor、离线降级、重定向防泄漏与冻结产物不变性 smoke：

```powershell
python -m scripts.smoke_http_integration `
  --rag-root . `
  --mini-root $env:MINI_NANOBOT_REPO
```

Mini-Nanobot 是独立仓库，其工具注册与 `knowledge.search` 客户端契约需要在 Mini 仓库单独运行：

```powershell
Push-Location $env:MINI_NANOBOT_REPO
try {
  python -m pytest -q
} finally {
  Pop-Location
}
```

RAG 测试覆盖 source allowlist、秘密过滤、Git snapshot 一致性、路由、BM25/dense/RRF、live rg/AST/Git、引用 DTO、API token、评测 schema、Windows 安全锁、索引校验和和回滚前置条件；Mini 测试独立覆盖工具注册、只读权限、HTTP 契约和错误隔离。

## 目录

```text
data/sources/                     来源目录与白名单
data/manifests/builds/            可复现采集快照（本地运行产物）
data/indexes/engineering/         分区 FAISS/BM25 索引（本地运行产物）
data/eval/                        v2 开发集、holdout 与 snapshot
web/                              同源只读前端（HTML/CSS/JS，无构建步骤）
rag_core/sources/                 Git/official source adapters
rag_core/ingestion/               manifest 与分块流水线
rag_core/engineering/             分区索引、服务、引用、充分性和回答边界
rag_core/retrieval/engineering/   路由、BM25、RRF、rerank、rg/AST/Git
rag_core/evaluation/              纯索引与 E2E 评测
engineering_api.py                FastAPI 只读边界
app.py                            Web 页面与 API 的便捷启动器
scripts/                          smoke、迁移和冻结脚本
tests/                            离线回归测试
```

更详细的运行边界见 [工程 RAG 运维说明](docs/engineering_rag_operations.md)，评测定义见 [工程评测说明](docs/engineering_evaluation.md)。面试叙述可参考 [项目 STAR 与真实性边界](introduce.md)。

## 简历与面试真实性边界

只陈述已经运行且有报告支持的能力。尤其不要声称：

- “海量网络爬虫”：外部资料是少量、白名单、版本化的官方规范；
- “RAG 替代代码搜索”：当前实现事实最终依赖实时 `rg`、AST 与 Git；
- “Milvus、ColBERT、RAGAS 已用于最终结果”：除非对应实验真实运行并保存报告；
- “自动指标证明答案完全正确”：当前正式自动指标主要是来源、符号、路由和拒答；
- “达到生产 SLA 或通过安全认证”：本项目是可复现实验系统，不是生产部署证明。
