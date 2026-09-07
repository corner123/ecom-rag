"""Command-line entry point for the trade intelligence agent."""
from __future__ import annotations

import argparse
from importlib import import_module
import json
import sys
from typing import Sequence
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


_EVALUATION_UNAVAILABLE = {
    "code": "evaluation_not_implemented",
    "status": "unavailable",
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="trade-intel", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("bootstrap-demo", help="generate the synthetic demo corpus")
    commands.add_parser("db", help="run database migration or seed commands")
    commands.add_parser("ingest", help="ingest a reviewed source catalog")
    commands.add_parser("index", help="build and publish an immutable index bundle")
    query = commands.add_parser("query", help="submit one guarded API query")
    query.add_argument("question")
    query.add_argument("--top-k", type=int, default=10)
    query.add_argument("--api-url", default="http://127.0.0.1:8000")
    query.add_argument("--idempotency-key")
    commands.add_parser("eval", help="run evaluation when the evaluation plan is installed")
    commands.add_parser("smoke", help="run foundation or retrieval smoke verification")
    commands.add_parser("verify-report", help="verify an evaluation report when available")
    return parser


def _delegate(module_name: str, arguments: Sequence[str]) -> int:
    """Invoke an existing module entry point with an isolated argv."""
    module = import_module(module_name)
    entrypoint = getattr(module, "main", None)
    if not callable(entrypoint):
        raise RuntimeError(f"{module_name} has no callable main")
    previous = sys.argv
    sys.argv = [module_name, *arguments]
    try:
        try:
            result = entrypoint()
        except SystemExit as exit_error:
            return int(exit_error.code or 0)
        return 0 if result is None else int(result)
    finally:
        sys.argv = previous


def _db(arguments: Sequence[str]) -> int:
    if not arguments or arguments[0] not in {"migrate", "seed"}:
        print(
            json.dumps({"code": "db_action_required", "status": "error"}, sort_keys=True),
            file=sys.stderr,
        )
        return 2
    module = {
        "migrate": "trade_agent.db.migrate",
        "seed": "trade_agent.db.seed",
    }[arguments[0]]
    return _delegate(module, arguments[1:])


def _query(args: argparse.Namespace) -> int:
    if not 1 <= args.top_k <= 100:
        print(json.dumps({"code": "invalid_top_k", "status": "error"}), file=sys.stderr)
        return 2
    payload: dict[str, object] = {"question": args.question, "top_k": args.top_k}
    if args.idempotency_key is not None:
        payload["idempotency_key"] = args.idempotency_key
    request = Request(
        args.api_url.rstrip("/") + "/v1/query",
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=10.0) as response:
            body = json.load(response)
    except HTTPError as error:
        try:
            body = json.load(error)
        except Exception:
            body = {"detail": {"code": "api_error"}}
        print(json.dumps(body, ensure_ascii=False, sort_keys=True), file=sys.stderr)
        return 1
    except (OSError, TimeoutError, URLError):
        print(
            json.dumps({"code": "api_unavailable", "status": "error"}, sort_keys=True),
            file=sys.stderr,
        )
        return 1
    print(json.dumps(body, ensure_ascii=False, sort_keys=True))
    return 0


def _smoke(arguments: Sequence[str]) -> int:
    target = arguments[0] if arguments else "foundation"
    remaining = arguments[1:] if arguments else ()
    if target == "foundation":
        return _delegate("scripts.smoke_foundation", remaining)
    if target == "retrieval":
        return _delegate("scripts.smoke_milvus_roundtrip", remaining)
    print(json.dumps({"code": "unknown_smoke", "status": "error"}), file=sys.stderr)
    return 2


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args, remaining = parser.parse_known_args(argv)
    if args.command in {"eval", "verify-report"}:
        if remaining:
            parser.error("unrecognized arguments: " + " ".join(remaining))
        print(json.dumps(_EVALUATION_UNAVAILABLE, sort_keys=True), file=sys.stderr)
        return 2
    if args.command == "bootstrap-demo":
        return _delegate("scripts.bootstrap_trade_intel_demo", remaining)
    if args.command == "db":
        return _db(remaining)
    if args.command == "ingest":
        return _delegate("scripts.ingest_trade_sources", remaining)
    if args.command == "index":
        return _delegate("scripts.build_trade_index", remaining)
    if args.command == "smoke":
        return _smoke(remaining)
    if remaining:
        parser.error("unrecognized arguments: " + " ".join(remaining))
    return _query(args)


if __name__ == "__main__":
    raise SystemExit(main())
