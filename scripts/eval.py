"""Thin wrapper: `python scripts/eval.py` == `strafrecht-rag eval` (kept for the PRD layout)."""

from strafrecht_rag.cli import app

if __name__ == "__main__":
    app(["eval"])
