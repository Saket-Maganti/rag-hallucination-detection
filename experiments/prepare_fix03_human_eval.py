#!/usr/bin/env python3
"""Prepare blinded human-evaluation sheets for Fix 3 metric calibration.

The source JSONL contains automatic metric scores. This script deliberately
omits those scores from the rater-facing files so annotation can be blind.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_SOURCE = Path("data/revision/fix_03/human_eval_template.jsonl")
DEFAULT_OUTDIR = Path("data/revision/fix_03")

RATER_FIELDS = [
    "id",
    "dataset",
    "condition",
    "question",
    "answer",
    "passage_1",
    "passage_2",
    "passage_3",
    "label",
    "notes",
]

ADJUDICATION_FIELDS = [
    "id",
    "dataset",
    "condition",
    "question",
    "answer",
    "passage_1",
    "passage_2",
    "passage_3",
    "rater_a_label",
    "rater_b_label",
    "adjudicated_label",
    "adjudication_notes",
]


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "id" not in row:
                raise ValueError(f"Missing id on line {line_no}")
            rows.append(row)
    return rows


def split_passages(context: str, max_passages: int = 3) -> list[str]:
    passages = [part.strip() for part in context.split("\n\n---\n\n")]
    passages = [part.replace("\n", " ").strip() for part in passages if part.strip()]
    passages = passages[:max_passages]
    while len(passages) < max_passages:
        passages.append("")
    return passages


def rater_row(row: dict) -> dict:
    p1, p2, p3 = split_passages(row.get("context", ""))
    return {
        "id": row.get("id", ""),
        "dataset": row.get("dataset", ""),
        "condition": row.get("condition", ""),
        "question": row.get("question", ""),
        "answer": row.get("answer", ""),
        "passage_1": p1,
        "passage_2": p2,
        "passage_3": p3,
        "label": "",
        "notes": "",
    }


def adjudication_row(row: dict) -> dict:
    base = rater_row(row)
    return {
        "id": base["id"],
        "dataset": base["dataset"],
        "condition": base["condition"],
        "question": base["question"],
        "answer": base["answer"],
        "passage_1": base["passage_1"],
        "passage_2": base["passage_2"],
        "passage_3": base["passage_3"],
        "rater_a_label": "",
        "rater_b_label": "",
        "adjudicated_label": "",
        "adjudication_notes": "",
    }


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    rows = read_jsonl(args.source)
    rater_rows = [rater_row(row) for row in rows]
    adjudication_rows = [adjudication_row(row) for row in rows]

    write_csv(args.outdir / "human_eval_rater_a.csv", rater_rows, RATER_FIELDS)
    write_csv(args.outdir / "human_eval_rater_b.csv", rater_rows, RATER_FIELDS)
    write_csv(
        args.outdir / "human_eval_adjudication_template.csv",
        adjudication_rows,
        ADJUDICATION_FIELDS,
    )

    print(f"Wrote {len(rows)} blinded rows to {args.outdir}")


if __name__ == "__main__":
    main()
