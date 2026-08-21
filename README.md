# ecom-rag

面向电商研发与技术支持场景的、证据优先的工程知识 RAG。项目读取一个**独立 Git 仓库**中的订单、库存、优惠和支付资料，区分设计意图、当前源码、测试、历史记录与外部规范；涉及当前实现时使用 `rg`、Python AST 和 Git 实时核验，证据不足时返回稳定拒答原因。

> 本仓库的电商语料和 PayGate 规范完全由脚本生成，均为 synthetic demo，不包含公司内部资料、客户数据、生产遥测或真实支付机构信息。本项目不能证明生产 SLA、业务收益或企业部署。

## 已实现能力

- YAML catalog 管理独立 Git worktree 与本地规范目录，采集 commit、dirty、SHA-256、解析器和来源版本。
- Markdown、代码、JSON/YAML/TOML、CSV、XLSX 与 PDF 统一进入带页码、工作表、表格和降级告警的 `DocumentRecord`。
- BGE-small-zh-v1.5 + FAISS/BM25 + RRF 混合检索；保留严格 Milvus 镜像、契约校验和受限故障回退能力。
- 确定性 Router 区分 implementation、design、official、comparison 和 out-of-scope。
- 当前实现题使用实时 `rg`、AST、Git 证据；官方规范不能单独证明代码已经实现。
- Evidence Guard 检查 hard anchor、official topic scope、经验数据缺失和独立审计缺失。
- FastAPI 提供只读 `GET /health`、`POST /retrieve`、`POST /answer`，请求仍为 `query + top_k`。
- 正式检索评测保持确定性，不调用在线生成模型；DeepSeek/RAGAS 只属于独立 response-eval 流程。

## 架构边界

```text
ecom-rag/                         采集、索引、检索、引用、拒答、评测、Web
../ecommerce-engineering-demo/   synthetic 业务代码与文档，独立 Git 历史
```

RAG 对目标仓库只读，不提供代码修改、文档上传、索引构建或知识库删除 API。新配置使用：

- `KNOWLEDGE_TARGET_REPO`：独立目标仓库路径；
- `ENGINEERING_TARGET_SOURCE_ID`：manifest 中对应的 Git source id；
- `MINI_NANOBOT_REPO` 与 `--mini-repo`：仅保留为上游兼容别名。

## 快速开始

前置：Git、Conda、Python 3.12、`rg`。项目在 Windows PowerShell 与 `all-in-rag` Conda 环境验证。

```powershell
conda activate all-in-rag
$env:PYTHONNOUSERSITE = "1"
$env:PYTHONDONTWRITEBYTECODE = "1"
$env:HF_HUB_OFFLINE = "1"  # 已缓存 embedding 模型时使用

# 1. 生成独立 synthetic 业务仓库；已有目标时脚本拒绝覆盖
python -m scripts.bootstrap_ecommerce_demo

$env:KNOWLEDGE_TARGET_REPO = (Resolve-Path ..\ecommerce-engineering-demo).Path
$env:ENGINEERING_TARGET_SOURCE_ID = "ecommerce_demo"
$env:ENGINEERING_MANIFEST_PATH = "data/manifests/builds/ecommerce_demo.json"
$env:ENGINEERING_INDEX_DIR = "data/indexes/ecommerce_demo"

# 2. 离线采集与建库
python main.py sources-sync --full-rebuild
python main.py engineering-build --backend faiss

# 3. 冻结题集与构建绑定
python -m scripts.freeze_engineering_eval `
  --manifest data/manifests/builds/ecommerce_demo.json `
  --source-id ecommerce_demo `
  --dataset data/eval/ecommerce_development.jsonl data/eval/ecommerce_holdout_v1.jsonl `
  --output data/eval/ecommerce_evaluation_snapshot.json

# 4. 本地查询或启动 Web
python main.py engineering-query "当前 InventoryLedger.reserve_stock 如何处理重复 reservation_id？" `
  --target-repo $env:KNOWLEDGE_TARGET_REPO
python app.py
```

默认 `ENGINEERING_GENERATION_PROVIDER=deterministic`，不会调用外部模型。若启用在线生成，问题和选中证据可能被发送给第三方模型；无权外传的资料必须保持 deterministic 或使用经过授权的服务。

## 电商冻结构建与开发集结果

当前 synthetic 构建：

- build：`build_37262492b5395c01c7a1`；
- 目标代码提交：`558636f4d607d5a0c594a00d370ced8d37f78a05`，`dirty=false`；
- 2 个 source、37 份采集文档、39 个索引文档/片段；
- 五个分区：code 28、design 6、history 1、test 3、official 1。

12 条可答开发题的纯索引消融：

| 策略 | Primary Hit@5 | nDCG@5 | MRR | P95 ms |
| --- | ---: | ---: | ---: | ---: |
| BM25 | 0.583 | 0.378 | 0.333 | 1.28 |
| Dense | 0.417 | 0.278 | 0.190 | 80.59 |
| Hybrid RRF | 0.583 | 0.361 | 0.278 | 110.09 |

16 条开发题的完整 E2E：Route accuracy 1.000、Primary Hit@5 1.000、拒答 F1 1.000、可答题误拒答率 0、P95 约 937 ms。开发题很少且参与了规则修正，不能解释为泛化能力或生产 SLA；正式首次 holdout 报告单独保存，运行后不覆盖。

报告见 [公开评测目录](data/eval/reports_public/README.md)。指标是来源/符号/路由/拒答层面的离线指标，不是答案正确率、业务转化率或线上稳定性证明。

## API

```json
POST /retrieve
{"query":"根据 PayGate 官方规范，event_id 如何用于回调去重？","top_k":5}
```

`/answer` 会先执行同一套检索和 Evidence Guard。响应区分 `retrieved_evidence`、`generation_context` 和 `answer_citations`；模型失败时保留检索证据，但不会把原文拼接伪装成模型答案。

## 测试与发布检查

```powershell
$env:PYTHONNOUSERSITE = "1"
python -m pip check
python -m pytest -q
python -m scripts.smoke_ecommerce_demo
```

Milvus 默认使用状态化 fake client 测试；没有真实服务 round-trip、分布式部署或性能报告时，不得声称生产 Milvus 成果。

## 求职叙述边界

可把本项目写成“电商研发知识助手｜AI/RAG 项目实践｜2个月”，但 synthetic demo 必须明确。没有真实劳动或项目制合作关系时，应放在“项目经历”，不能虚构公司、实习证明、DAU、QPS、节省工时、业务收益或团队规模。完整简历、STAR、八周时间线和追问口径见 [introduce.md](introduce.md)。

仓库继承自个人通用工程知识 RAG 的代码历史；旧 Mini-Nanobot 数据与报告仅是 upstream 历史证据，不属于本电商构建，也不得与上述指标混用。
