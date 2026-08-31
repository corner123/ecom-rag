from __future__ import annotations

import pytest


def test_summary_validation_is_fail_closed_for_zero_chunks() -> None:
    from scripts.smoke_foundation import validate_foundation_summary

    summary = {
        "status": "ok",
        "synthetic_only": True,
        "mysql_tables": 7,
        "trade_records": 825,
        "months": 18,
        "corpus": {"sources": 74, "documents": 116, "chunks": 0},
        "metadata_required_completeness": 1.0,
        "quarantine_unexpected": 0,
    }
    with pytest.raises(ValueError, match="chunks"):
        validate_foundation_summary(summary)


def test_summary_validation_accepts_complete_expected_contract() -> None:
    from scripts.smoke_foundation import validate_foundation_summary

    summary = {
        "status": "ok",
        "synthetic_only": True,
        "mysql_tables": 7,
        "trade_records": 825,
        "months": 18,
        "corpus": {"sources": 74, "documents": 116, "chunks": 120},
        "metadata_required_completeness": 1.0,
        "quarantine_unexpected": 0,
    }
    assert validate_foundation_summary(summary) is summary


def test_failure_payload_contains_only_stage_and_exception_class() -> None:
    from scripts.smoke_foundation import failure_payload

    payload = failure_payload("mysql", RuntimeError("password=top-secret /Users/private/app"))
    assert payload == {"error": "RuntimeError", "stage": "mysql", "status": "error"}
    assert "top-secret" not in str(payload)
    assert "/Users" not in str(payload)


def test_summary_validator_rejects_missing_quarantine_field() -> None:
    from scripts.smoke_foundation import validate_foundation_summary

    with pytest.raises(ValueError, match="quarantine_unexpected"):
        validate_foundation_summary({"status": "ok"})
