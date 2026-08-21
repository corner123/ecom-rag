"""Create an independent synthetic commerce Git repository for the demo."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "demo" / "ecommerce_seed"
DEFAULT_TARGET = ROOT.parent / "ecommerce-engineering-demo"


def _run_git(target: Path, *args: str, env: dict[str, str] | None = None) -> None:
    subprocess.run(
        ["git", *args],
        cwd=target,
        check=True,
        text=True,
        env=env,
    )


def create_demo_repository(target: str | Path) -> Path:
    destination = Path(target).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(
            f"refusing to overwrite existing demo repository: {destination}"
        )
    if not SEED.is_dir():
        raise FileNotFoundError(f"demo seed is missing: {SEED}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(SEED, destination)
    try:
        _run_git(destination, "init", "-b", "main")
        _run_git(destination, "add", "--all")
        commit_env = dict(os.environ)
        commit_env.update(
            {
                "GIT_AUTHOR_DATE": "2026-08-01T09:00:00+08:00",
                "GIT_COMMITTER_DATE": "2026-08-01T09:00:00+08:00",
            }
        )
        _run_git(
            destination,
            "-c",
            "user.name=ecom-rag demo",
            "-c",
            "user.email=ecom-rag-demo@example.invalid",
            "commit",
            "-m",
            "feat: add synthetic commerce engineering fixture",
            env=commit_env,
        )
    except Exception:
        shutil.rmtree(destination)
        raise
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default=str(DEFAULT_TARGET))
    args = parser.parse_args()
    destination = create_demo_repository(args.target)
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=destination,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    print(f"created synthetic demo repository: {destination}")
    print(f"revision: {revision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
