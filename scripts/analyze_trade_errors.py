"""Analyze Task 5 development runs and select one bounded optimization candidate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trade_agent.data.manifest import canonical_json
from trade_agent.evaluation.error_analysis import ErrorAnalyzer, HoldoutPolicyError, load_evaluation_runs
from trade_agent.evaluation.optimizer import DevelopmentOptimizer, OptimizationObjective


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True,
                        help="Task 5 run directories or parents containing them")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-run-id")
    parser.add_argument("--max-mean-latency-increase-ms", type=float, default=50.0)
    parser.add_argument("--max-refusal-rate-increase", type=float, default=0.0)
    args = parser.parse_args(argv)
    try:
        runs = load_evaluation_runs(args.runs)
        analysis = ErrorAnalyzer().analyze(runs)
        objective = OptimizationObjective(
            baseline_run_id=args.baseline_run_id,
            max_mean_latency_increase_ms=args.max_mean_latency_increase_ms,
            max_refusal_rate_increase=args.max_refusal_rate_increase,
        )
        decision = DevelopmentOptimizer().select(runs, objective)
    except (HoldoutPolicyError, ValueError) as exc:
        parser.error(str(exc))
    payload = {"analysis": analysis.to_dict(), "decision": decision.to_dict()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(canonical_json(payload) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "selected_run_id": decision.selected_run_id,
                      "accepted_change": decision.accepted_change}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
