# 工程知识 RAG 评测说明

## 两套评测必须分开

### 纯索引消融（suite=index）

只回答“BM25、dense、hybrid 的来源级召回有什么差异”。它：

- 仅运行 `answerable=true` 题；
- 使用题集显式标注的 oracle route；
- 关闭真实 Router、`rg`/AST/Git、Evidence Guard 和 Answerer；
- 三种策略共享同一 index build、embedding model、Top-K 与 per-arm candidate budget；
- hybrid 任一支路异常时 fail closed，不发布降级后的“hybrid”成绩；
- oracle route 不计 Routing Accuracy，报告显示 `N/A`。

### E2E 检索策略评测（suite=e2e）

运行完整的 Router → 分区检索 → live rg/AST/Git → Evidence Guard → answer/refusal。它评估来源召回、路由、实时核验和拒答策略。正式 E2E predictor 显式构造确定性 Answerer，不读取 `auto/deepseek` 在线生成配置，也不调用 DeepSeek；因此报告不能称为“LLM 答案正确率”或 “RAGAS faithfulness”。

suite 与 predictor contract 会强校验：index predictor 必须声明 oracle、无 live、无 answer、fail closed；E2E predictor 必须声明真实路由、live 与 answer。标错 suite 会直接终止。

这里的 `answer=true` 表示执行 Evidence Guard 后的确定性 answer/refusal 契约，不表示调用在线模型。Web/API 中可选的 DeepSeek 归纳层不进入上述 v2 检索报告。模型回答使用下述独立的 response-eval/v3 协议评测，两个报告体系不能混写。

### 响应评测（response-eval/v3）

响应评测单独冻结问题、人工复核的 reference answer、原子 reference claims、每项 claim 的可接受证据来源、索引 build、Mini-Nanobot live revision、Top-K、检索 profile、Evidence Guard profile、支持证据选择 profile、生成模型、生成 prompt、Judge 模型、RAGAS 版本与本地评测 Embedding。参考答案和 claim 不会进入检索或生成 prompt，只在 Judge 阶段加入。

指标分三组：

| 层级 | 指标 | 口径 |
| --- | --- | --- |
| 检索 | Hit@K、MRR | 至少一个 required claim 的可接受来源是否命中，以及第一个命中来源的倒数排名 |
| 检索 | Required-claim Recall@K | 每个必需 claim 的任一可接受来源命中即算该 claim 命中 |
| 检索 | logical-source Precision@K | 去掉 symbol/line fragment 后，Top-K 逻辑来源中相关来源的比例 |
| 检索 | source-option recall | 所有备选来源的诊断性覆盖率；不是主完整性指标 |
| RAGAS 上下文 | LLM Context Precision、LLM Context Recall | 评估实际送入模型的 `generation_context`，不是整个候选池 |
| RAGAS 回答 | Faithfulness、Response Relevancy | 回答是否能由上下文支持、是否直接回应问题 |
| 补充回答 | Answer Correctness | 与冻结 reference answer 的语义/事实一致性；不属于本项目宣称的“四核心指标” |
| 安全 | refusal P/R/F1、false-refusal | 不可回答题是否拒答，以及可回答题是否被错误拒答 |
| 工程 | retrieval/generation/total P50/P95 | 冷启动单列；小样本 P99 仅 exploratory，不是 SLA |
| 成本 | prompt/completion/cache token | 只统计 `generation_status.attempted=true` 的真实模型调用 |

RAGAS 使用 `ragas==0.2.15` 的 `Faithfulness`、`ResponseRelevancy`、`LLMContextPrecisionWithReference` 与 `LLMContextRecall`；Judge 显式注入 DeepSeek chat wrapper，Response Relevancy 显式注入本地 `BAAI/bge-small-zh-v1.5` embedding wrapper，不隐式回退 OpenAI。单指标失败记录为 `null + error`，绝不写成 0。Judge 恢复会校验模型、prompt hash、RAGAS、Embedding、timeout/retry 和代码身份，只补失败指标并保留历史错误。

response-eval/v3 开发集有 16 题（12 可回答、4 边界/不可回答），私有 holdout 有 24 题且保持 Git 忽略。只有候选通过预注册开发集门槛后才允许消耗一次私有 holdout；失败候选不能用私有集继续调参。

## v2 数据契约

每行必须包含：

- `expected_route`：`design | implementation | official | comparison | out_of_scope`；
- `primary_sources`：回答成立所需的主证据；
- `supporting_sources`：补充证据；
- `relevant_sources`：必须严格等于前两者按顺序拼接；
- `relevant_symbols`：大小写敏感的真实 AST qualified names；
- `refusal_reason`：可回答题必须为 null，不可回答题必须是固定枚举；
- `label_version=engineering-eval/v2`；
- `dataset_role=development_regression | holdout`。

loader 会拒绝字符串冒充数组、重复来源、非法 route、空字段、answerability 冲突和 v1 隐式回退。

数据文件：

| 文件 | 数量 | 角色 |
| --- | ---: | --- |
| `mini_nanobot_internal.jsonl` | 60 | development regression |
| `official_engineering_specs.jsonl` | 20 | development regression |
| `engineering_holdout_v2.jsonl` | 30 | holdout |

开发集用于修正实现；holdout 只在系统冻结后运行，并保留第一次结果。题目随后公开用于审计，因此不能再作为未来迭代的未见测试集，也不能根据其失败继续调参；新的最终评测需要新的私有 holdout。

## 快照绑定

`scripts.freeze_engineering_eval` 将三者绑定：

1. 当前 ingestion manifest 的 build ID；
2. Mini-Nanobot commit、dirty 状态和被纳入语料的 worktree content hash；
3. 每个 JSONL 的 SHA-256。

结果写入 `data/eval/evaluation_snapshot.json`。正式 runner 会同时检查 snapshot ↔ dataset ↔ index；E2E factory 还会重新只读采集 Mini-Nanobot 并检查 live repo ↔ manifest。任一不一致时拒绝生成正式报告。

## 指标

对 Top-K 结果先按 canonical source 去重；同一文件的多个 chunk 只保留第一次出现。

| 指标 | 定义 |
| --- | --- |
| Primary Hit@K | 是否命中至少一个 primary source |
| Primary Recall@K | 命中的 primary source 比例 |
| Supporting Recall@K | 有 supporting label 的题上，命中比例 |
| Primary MRR | 第一个 primary source 的倒数排名 |
| graded nDCG@K | primary relevance=2，supporting=1，重复 source gain=0 |
| Source precision@K | Top-K 去重来源中，primary/supporting 所占比例 |
| Symbol Recall@K | 大小写敏感 qualified symbol 命中率 |
| Route accuracy / macro-F1 | 仅 E2E 计算 |
| Refusal P/R/F1 | 对不可回答题的显式拒答 |
| Strict reason recall | 拒答且 machine reason 与标注相同的不可回答题比例 |
| False refusal rate | answerable 题被拒答的比例 |
| P50/P95 | 每策略预热一次后的 predictor 调用延迟 |

这些是 source/file-level 指标。同一网页中命中错误段落仍可能得到 file hit，因此不能把 Source Recall 或 Source Precision 写成 passage correctness、citation entailment 或答案正确率。实现题额外报告 symbol/live 命中，缓解单纯文件命中的宽松问题。

## 运行

```powershell
conda activate all-in-rag
$env:MINI_NANOBOT_REPO = (Resolve-Path ..\Mini-Nanobot).Path
$env:ENGINEERING_INDEX_DIR = "data/indexes/engineering"
$env:ENGINEERING_MANIFEST_PATH = "data/manifests/builds/current.json"
$env:HF_HUB_OFFLINE = "1"
$env:ENGINEERING_GENERATION_PROVIDER = "deterministic"

python main.py engineering-eval `
  --suite index `
  --dataset data/eval/mini_nanobot_internal.jsonl data/eval/official_engineering_specs.jsonl `
  --snapshot data/eval/evaluation_snapshot.json `
  --output data/eval/reports/index_ablation

python main.py engineering-eval `
  --suite e2e `
  --dataset data/eval/mini_nanobot_internal.jsonl data/eval/official_engineering_specs.jsonl `
  --snapshot data/eval/evaluation_snapshot.json `
  --output data/eval/reports/e2e_development

python main.py engineering-eval `
  --suite e2e `
  --dataset data/eval/engineering_holdout_v2.jsonl `
  --snapshot data/eval/evaluation_snapshot.json `
  --output data/eval/reports/e2e_holdout_first_run
```

响应基线/候选使用独立命令；以下是开发集示例，真实运行需要 `.env` 中可用的 DeepSeek 凭据：

```powershell
python main.py engineering-response-eval `
  --dataset data/eval/response_development_v3.jsonl `
  --snapshot data/eval/response_development_v3.snapshot.json `
  --manifest data/manifests/builds/current.json `
  --index-dir data/indexes/engineering `
  --mini-repo ..\Mini-Nanobot `
  --output data/eval/reports/response_dev_candidate.json `
  --profile-name candidate `
  --sufficiency-profile split_natural_slash_concepts `
  --support-selection-profile query_aware_diverse
```

若 Judge 只有部分指标失败，保留原产物并写入新文件：

```powershell
python main.py engineering-response-eval `
  --dataset data/eval/response_development_v3.jsonl `
  --snapshot data/eval/response_development_v3.snapshot.json `
  --output data/eval/reports/response_dev_candidate_recovered.json `
  --profile-name candidate `
  --judge-only `
  --retry-from data/eval/reports/response_dev_candidate.json
```

恢复时仍需传入与原实验相同的 index、manifest、Mini repo、Top-K 和两个 profile；任何 Judge 或代码身份差异都会失败关闭。

## 2026-08-14 实际响应实验

当前 build 为 `build_2b26e83243ccb25024e0`。生成模型为 `deepseek-v4-flash`，Judge 为 `deepseek-v4-pro`，两者同属 DeepSeek 系列，因此报告明确保留同家族自评偏差；响应评测 Embedding 为本地 `BAAI/bge-small-zh-v1.5`。

完整基线（equal RRF + legacy Guard/support）结果：

| 指标 | 基线 |
| --- | ---: |
| Hit@5 / MRR | 1.000 / 0.903 |
| Required-claim Recall@5 | 0.854 |
| logical-source Precision@5 | 0.606 |
| Context Precision / Recall | 0.837 / 0.697 |
| Faithfulness / Answer Relevancy | 0.920 / 0.703 |
| Answer Correctness | 0.477 |
| False-refusal / Refusal F1 | 8.33% / 0.889 |

第一次全局 BM25 加权候选导致另一道实现题被错误拒答，提前终止。第二次仅放宽自然语言 slash anchor 的候选恢复了误拒答，但目标题证据只覆盖 1/3，Faithfulness 0.649，未过开发门槛。第三次加入 query-aware 支持证据和实时 AST 父类展开：

| 指标 | 基线 | 候选 | 差异 |
| --- | ---: | ---: | ---: |
| Required-claim Recall@5 | 0.854 | 0.910 | +0.056 |
| logical-source Precision@5 | 0.606 | 0.622 | +0.017 |
| False-refusal | 8.33% | 0% | -8.33 pp |
| Context Precision | 0.837 | 0.782 | paired -0.027 |
| Faithfulness | 0.920 | 0.858 | paired -0.042 |
| Answer Relevancy | 0.703 | 0.854 | paired +0.170（仅 10 个共同有效样本） |
| Answer Correctness | 0.477 | 0.506 | paired +0.016 |

候选有一个 Answer Relevancy Judge 连接失败，Judge score coverage 为 11/12，因此公开聚合报告明确 `publishable=false`。即使忽略该缺失项，Faithfulness 下降仍超过预注册允许的 0.02，候选判定失败，生产默认未切换，私有 holdout 未使用。后续按相同代码重跑时 DeepSeek Judge 返回 HTTP 402 `Insufficient Balance`，因此没有伪造或用 0 补齐分数。脱敏失败报告见 [response development comparison](../data/eval/reports_public/response_development_support_parent_incomplete_build_2b26e83243ccb25024e0.md)。

JSON 报告保留逐题结果、每策略 contract、build/model、dataset hash 和 snapshot；Markdown 只用于快速浏览。冷启动应单独测量，不能混入稳态 P50/P95。正式 runner 本身已经使用确定性 Answerer，上述环境变量是额外的操作防线，避免同一终端随后启动 Web/API 时意外发送评测问题与证据。
