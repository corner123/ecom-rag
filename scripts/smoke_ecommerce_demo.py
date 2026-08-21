"""Offline end-to-end smoke for the synthetic commerce corpus and HTTP API."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from fastapi.testclient import TestClient

from engineering_api import create_app
from rag_core.engineering import EngineeringRAGService


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-repo",
        default=os.getenv(
            "KNOWLEDGE_TARGET_REPO",
            str(ROOT.parent / "ecommerce-engineering-demo"),
        ),
    )
    parser.add_argument(
        "--manifest",
        default=str(ROOT / "data/manifests/builds/ecommerce_demo.json"),
    )
    parser.add_argument(
        "--index-dir",
        default=str(ROOT / "data/indexes/ecommerce_demo"),
    )
    args = parser.parse_args()

    os.environ["ENGINEERING_GENERATION_PROVIDER"] = "deterministic"
    service = EngineeringRAGService.from_index(
        args.index_dir,
        target_repo=args.target_repo,
        manifest_path=args.manifest,
        runtime_backend="faiss",
    )

    checks = {
        "implementation": (
            "当前 InventoryLedger.reserve_stock 如何处理重复 reservation_id？",
            {"current_implementation"},
        ),
        "official": (
            "根据 PayGate 官方规范，回调消费者应使用哪个字段去重？",
            {"external_normative"},
        ),
        "comparison": (
            "当前 PaymentWebhookHandler 实现与 PayGate 官方规范的 event_id 去重要求是否一致？",
            {"current_implementation", "external_normative"},
        ),
    }
    summary: dict[str, object] = {}
    for name, (query, required_roles) in checks.items():
        outcome = service.retrieve(query, top_k=5)
        roles = {citation.evidence_role for citation in outcome.citations}
        if not outcome.sufficient_evidence or not required_roles.issubset(roles):
            raise AssertionError(
                f"{name} smoke failed: sufficient={outcome.sufficient_evidence}, "
                f"roles={sorted(roles)}, reason={outcome.refusal_reason}"
            )
        summary[name] = {
            "intent": outcome.intent.value,
            "roles": sorted(roles),
            "results": len(outcome.results),
        }

    refused = service.retrieve(
        "生产环境当前支付回调 P99、SLA 和月均故障率分别是多少？",
        top_k=5,
    )
    if refused.sufficient_evidence or refused.refusal_reason != "missing_production_telemetry":
        raise AssertionError(
            "production-telemetry smoke did not fail closed with the expected reason"
        )
    summary["refusal"] = {
        "intent": refused.intent.value,
        "reason": refused.refusal_reason,
    }

    client = TestClient(create_app(service=service, token="smoke-token"))
    if client.get("/health").status_code != 401:
        raise AssertionError("health endpoint must enforce the configured token")
    headers = {"Authorization": "Bearer smoke-token"}
    health = client.get("/health", headers=headers)
    if health.status_code != 200 or health.json().get("status") != "ok":
        raise AssertionError(f"authenticated health failed: {health.text}")
    response = client.post(
        "/retrieve",
        headers=headers,
        json={"query": checks["implementation"][0], "top_k": 5},
    )
    if response.status_code != 200 or response.json().get("schema_version") != "engineering-retrieval/v1":
        raise AssertionError(f"HTTP retrieval contract failed: {response.text}")
    summary["http"] = {
        "health": health.json().get("status"),
        "retrieval_schema": response.json().get("schema_version"),
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
