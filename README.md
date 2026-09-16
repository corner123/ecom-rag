# 外贸情报智能体（合成演示）

> **证据边界：**本仓库是合成、非生产用途的工程演示。已提交的评测报告采用本地 CPU 哈希词元余弦检索、BM25 和词法重排；不能证明真实 Milvus/BGE 的验收结果，也不能证明生成答案质量、真实买家覆盖、商业决策有效性、生产就绪程度或客户数据上的表现。

项目结合只读 MySQL 查询与基于来源证据的 RAG 检索。它采集合成的网站、B2B、新闻、社交、PDF、扫描版 PDF 和海关衍生记录，建立统一元数据，使用 BM25、稠密检索、加权 RRF、过滤和重排寻找证据，再由 LangGraph 工作流把事实性主张关联到可复核的证据。Redis 保存经过字段限制的检查点状态。

## 已实现的能力

- 七表外贸数据库、确定性的合成种子数据、只读查询账户、Schema Registry、受限查询计划、SQLGlot 策略校验，以及 `SqlEvidence`。
- 来源路由、解析、隔离、分块、溯源清单、实体消歧、去重、冲突处理，以及统一元数据。
- 固定版本的 BGE-M3 向量模型契约、Milvus 集合契约、BM25、先过滤再进行稠密检索、加权 RRF、固定版本的 BGE 重排模型契约，以及检索轨迹。
- LangGraph 对 SQL 和 RAG 的编排、证据校验、逐条主张检查、结构化拒答、重试次数与步骤数及 LLM 调用次数限制、Redis 检查点、FastAPI 接口，以及 CLI 命令。
- 公开开发集评测、保持在 Git 跟踪范围之外的私有留出集流程、哈希冻结、泄漏检查、不可变报告，以及确定性规则指标。

架构与信任边界见[架构说明](docs/architecture.md)，数据结构见[数据契约](docs/data-contract.md)，评测方法见[评测说明](docs/evaluation.md)，运行步骤见[运维文档](docs/operations.md)。逐项需求状态见[完成情况审计](docs/completion-audit.md)。

## 已测得的合成评测结果

公开开发集有 44 条问题。在 `full_rerank` 方案中，上下文精确率从 `0.075231` 提升到 `0.194907`，Recall@10 保持 `0.472222`，配对比较中没有召回率或精确率倒退。被接受的改动是明确按自然月过滤。这些数字来自一次本地 CPU 顺序运行，不能解释为神经检索效果或生产延迟。

首次私有留出集发布的报告仅包含聚合数据：44 条检索案例全部完成，Recall@10 为 `0.5833333333333334`，上下文精确率为 `0.20601851851851852`。由于生成服务不可用，生成环节未运行（`not_run`）；LLM Judge 状态为 `judge_not_run`，忠实度和相关性分数均为空值。规则指标仍是正式评测依据。

- [开发集配对报告](data/eval/trade_intel/reports_public/cycle-d2f1d18c65a549e2a191a989fd734236/report.md)
- [首次留出集聚合报告](data/eval/trade_intel/reports_public/report-7cd261cf288521093c50d1083afec460ea2bb2c624cbea16e46abdfe8d70daf8-local-build-a3137d12491fb54f1a058fefa47cc96e-full_rerank-20260910T214832Z-7bb1aa43/report.md)
- [已冻结的公开留出集快照](data/eval/trade_intel/holdout_snapshot.json)

旧电商数据集和报告保留在独立路径中，仅作历史资料，不参与上述外贸智能体的评测，也不能作为其结果的证据。

## 本地准备

本地测试支持 Python 3.12。创建环境并安装锁定版本的依赖：

```sh
uv sync --frozen
.venv/bin/python -c "import trade_agent"
```

本地 `.venv` 可能不包含 `pip` 模块；仅在容器中执行的依赖检查见[运维文档](docs/operations.md)。

在新目录生成确定性的演示语料：

```sh
.venv/bin/python -m trade_agent.cli bootstrap-demo --output demo/trade_intel_seed --clean
```

使用真实服务的路径需要 MySQL、Milvus、etcd、MinIO、Redis、固定版本的 BGE 模型文件和本地提供的密钥。请按[运维文档中的完整初始化顺序](docs/operations.md#secrets-and-service-startup)启动基础服务、迁移并填充数据库、采集并构建索引，最后启动 API。已提交的本地评测报告不能代替这些运行检查。

## 安全地提供服务凭据

将示例文件复制为不受 Git 跟踪的 `.env`，仅在其中编辑非敏感的主机和模型配置。通过终端隐藏输入将密钥载入当前 shell；不要把 `.env` 当作 shell 代码执行，也不要把密钥值放进命令参数：

```bash
set -eu
cp .env.example .env
if [ -n "${EDITOR:-}" ]; then
  "$EDITOR" .env
fi
export MYSQL__ROOT_PASSWORD="$(python3 -c 'import getpass; print(getpass.getpass("MySQL root 密码："))')"
export MYSQL__MIGRATION_PASSWORD="$(python3 -c 'import getpass; print(getpass.getpass("MySQL 迁移账户密码："))')"
export MYSQL__QUERY_PASSWORD="$(python3 -c 'import getpass; print(getpass.getpass("MySQL 查询账户密码："))')"
export MINIO_ROOT_PASSWORD="$(python3 -c 'import getpass; print(getpass.getpass("MinIO root 密码："))')"
: "${MYSQL__ROOT_PASSWORD:?MYSQL__ROOT_PASSWORD must be non-empty}"
: "${MYSQL__MIGRATION_PASSWORD:?MYSQL__MIGRATION_PASSWORD must be non-empty}"
: "${MYSQL__QUERY_PASSWORD:?MYSQL__QUERY_PASSWORD must be non-empty}"
: "${MINIO_ROOT_PASSWORD:?MINIO_ROOT_PASSWORD must be non-empty}"
export MYSQL__ROOT_PASSWORD MYSQL__MIGRATION_PASSWORD MYSQL__QUERY_PASSWORD MINIO_ROOT_PASSWORD

cleanup_service_secrets() {
  unset MYSQL__ROOT_PASSWORD MYSQL__MIGRATION_PASSWORD MYSQL__QUERY_PASSWORD MINIO_ROOT_PASSWORD
}
trap cleanup_service_secrets EXIT
```

Compose 会读取上述已导出的变量；`-e` 只将变量名传给一次性迁移容器。以下命令只是完整初始化流程中的数据库迁移步骤，应先按[运维文档](docs/operations.md#secrets-and-service-startup)启动基础服务：

```sh
docker compose --env-file .env run --rm --no-deps \
  -e MYSQL__MIGRATION_PASSWORD api \
  python -m trade_agent.db.migrate
```

## 命令行入口

```text
trade-intel bootstrap-demo [generator options]
trade-intel db migrate|seed [options]
trade-intel ingest [ingestion options]
trade-intel index [index options]
trade-intel query QUESTION [--top-k N] [--api-url URL]
trade-intel eval [evaluation options]
trade-intel smoke [foundation|retrieval]
trade-intel verify-report [report path or --latest ROOT --kind ROLE]
```

`QUESTION` 表示要查询的问题，方括号表示可选参数。`eval` 使用合成数据的本地评测适配器，不会自动切换到 Milvus/BGE 或在线生成。

## 仓库审计

无需启动 Docker，即可运行确定性的仓库与报告审计：

```sh
.venv/bin/python -m scripts.verify_repository
```

审计会检查要求提交的证据、公开报告的不可变校验和、CLI 命令委托、未受 Git 跟踪的一次性留出集锁、公开文件中没有私有留出集问题与标签、旧电商运行路径已移除，以及受跟踪交付文件的安全边界。它不会宣称实时基础设施已通过验收。Docker、MySQL、Milvus/BGE、Redis 和完整工作流的运行验收步骤列在[完成情况审计](docs/completion-audit.md)中。
