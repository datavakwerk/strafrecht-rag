"""Shorten the slow stages in an asciinema recording without touching anything else.

Loading the models takes ~20 s and the local LLM a minute or more per answer on a laptop
CPU. For the README recordings those two stretches are compressed to a few seconds each:
the spinner still runs, the rest of the timeline is untouched. The README says so.

A segment runs from the first frame containing ``start`` (or the beginning, when ``start``
is empty) up to the first later frame containing ``end``.

usage: python scripts/compress_cast.py in.cast out.cast [--max SECONDS]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SEGMENTS = [
    ("", "chunks ·"),  # component loading, until the first stage reports the store size
    ("door de cross-encoder", "tekens ruwe output"),  # LLM generation
    ("door de cross-encoder", "model meldt GEEN_BRON"),
]


def _find(events: list, needle: str, start: int) -> int | None:
    for i in range(start, len(events)):
        if events[i][1] == "o" and needle in events[i][2]:
            return i
    return None


def compress(src: Path, dst: Path, *, max_seconds: float) -> list[tuple[float, float]]:
    lines = src.read_text(encoding="utf-8").splitlines()
    header, events = lines[0], [json.loads(line) for line in lines[1:] if line.strip()]
    factors = [1.0] * len(events)  # per-event time-delta multiplier
    report = []
    for start, end in SEGMENTS:
        a = 0 if not start else _find(events, start, 0)
        if a is None:
            continue
        b = _find(events, end, a + 1)
        if b is None:
            continue
        span = events[b][0] - events[a][0]
        if span <= max_seconds:
            continue
        factor = max_seconds / span
        for i in range(a + 1, b + 1):
            factors[i] = min(factors[i], factor)
        report.append((span, span * factor))
    out, clock, prev = [], events[0][0], events[0][0]
    for i, (t, kind, data) in enumerate(events):
        clock += (t - prev) * factors[i]
        prev = t
        out.append([round(clock, 6), kind, data])
    dst.write_text(
        header + "\n" + "\n".join(json.dumps(e, ensure_ascii=False) for e in out) + "\n",
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--max", type=float, default=3.0, help="seconds a slow stage may take")
    args = ap.parse_args()
    for before, after in compress(args.src, args.dst, max_seconds=args.max):
        print(f"slow stage: {before:.1f}s → {after:.1f}s")
