# Synthetic 电商首次 holdout 审计说明

候选提交 `8a0e188107d9ec56898cf072ad8216e44f883eb8` 在冻结 snapshot 和 FAISS build `build_37262492b5395c01c7a1` 上首次运行 `ecommerce_holdout_v1.jsonl`。原始 JSON/Markdown 报告保持不改、不覆盖。

## 可使用结果

- 10 题：7 个可答、3 个不可答；
- Route accuracy / macro-F1：1.000 / 1.000；
- Primary Hit@5 / Recall@5：1.000 / 1.000；
- MRR：0.857；
- 拒答 F1：0.857；
- 可答题误拒答率：0.143；
- 3 个不可答题全部拒答，7 个可答题中 1 个误拒答。

## 失败题

`ECOM-HOLD-006` 询问当前 `verify_signature` 与 PayGate “签名算法”是否一致。系统已召回 `commerce_demo/payment.py` 和 `local-directory:vendor`，但 official topic scope 只登记了 `X-PayGate-Signature`、`HMAC-SHA256` 等词，没有把自然语言“签名算法”视为已覆盖 topic，最终以 `topic_not_indexed` 失败关闭。

该失败说明 Evidence Guard 仍有过严边界。由于它来自首次 holdout，本构建不增加 alias、不重新调参，也不覆盖报告。未来修复必须使用新的开发题、私有 holdout、snapshot、build ID 和并列报告。

## nDCG 异常

原始汇总报告的 nDCG@5 为 1.015，单题最高超过 1，违反归一化指标的理论范围。只读检查显示，实时结果与索引结果可能具有不同内部 source identity，却在公开报告中归一为同一文件路径，导致同一 relevant source 被重复计入 DCG，而 ideal DCG 只计一次。

当前首次报告保留该异常值以维持审计性，但 README、简历和结论均不使用 nDCG。修复评测器属于后续新版本工作，不能回写本次首次报告。
