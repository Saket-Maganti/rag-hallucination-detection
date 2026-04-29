#!/usr/bin/env python3
"""Analyze completed Fix 3 two-rater human evaluation labels."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path


DEFAULT_TEMPLATE = Path("data/revision/fix_03/human_eval_template.jsonl")
DEFAULT_RATER_A = Path("data/revision/fix_03/human_eval_rater_a.csv")
DEFAULT_RATER_B = Path("data/revision/fix_03/human_eval_rater_b.csv")
DEFAULT_ADJUDICATED = Path("data/revision/fix_03/human_eval_adjudicated.csv")
DEFAULT_OUTDIR = Path("results/revision/fix_03")

LABEL_VALUES = {
    "unsupported": 0.0,
    "partially_supported": 0.5,
    "supported": 1.0,
}
AUTO_METRICS = ["auto_deberta", "auto_second_nli", "auto_ragas"]
METRIC_NAMES = {
    "auto_deberta": "DeBERTa-v3 NLI",
    "auto_second_nli": "Second NLI",
    "auto_ragas": "RAGAS-style judge",
}


def read_jsonl_by_id(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            rows[str(row["id"])] = row
    return rows


def read_csv_by_id(path: Path) -> dict[str, dict]:
    with path.open(newline="") as f:
        return {str(row["id"]): row for row in csv.DictReader(f)}


def normalize_label(label: str) -> str:
    return label.strip().lower().replace(" ", "_").replace("-", "_")


def require_label(label: str, row_id: str, column: str) -> str:
    label = normalize_label(label)
    if label not in LABEL_VALUES:
        allowed = ", ".join(LABEL_VALUES)
        raise ValueError(f"Row {row_id} has invalid {column}={label!r}; use one of {allowed}")
    return label


def cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    if len(labels_a) != len(labels_b):
        raise ValueError("Label vectors differ in length")
    n = len(labels_a)
    observed = sum(a == b for a, b in zip(labels_a, labels_b)) / n
    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)
    expected = sum((counts_a[label] / n) * (counts_b[label] / n) for label in LABEL_VALUES)
    if math.isclose(1.0, expected):
        return 1.0 if math.isclose(1.0, observed) else float("nan")
    return (observed - expected) / (1.0 - expected)


def average_ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = rank
        i = j
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or not xs:
        raise ValueError("Vectors must be non-empty and equal length")
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return float("nan")
    return num / (den_x * den_y)


def spearman(xs: list[float], ys: list[float]) -> float:
    return pearson(average_ranks(xs), average_ranks(ys))


def kendall_tau_b(xs: list[float], ys: list[float]) -> float:
    """Kendall tau-b with tie correction for small calibration audits."""
    concordant = 0
    discordant = 0
    ties_x = 0
    ties_y = 0
    n = len(xs)
    for i in range(n):
        for j in range(i + 1, n):
            dx = (xs[i] > xs[j]) - (xs[i] < xs[j])
            dy = (ys[i] > ys[j]) - (ys[i] < ys[j])
            if dx == 0 and dy == 0:
                continue
            if dx == 0:
                ties_x += 1
            elif dy == 0:
                ties_y += 1
            elif dx == dy:
                concordant += 1
            else:
                discordant += 1
    denominator = math.sqrt((concordant + discordant + ties_x) * (concordant + discordant + ties_y))
    if denominator == 0:
        return float("nan")
    return (concordant - discordant) / denominator


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, default=DEFAULT_TEMPLATE)
    parser.add_argument("--rater-a", type=Path, default=DEFAULT_RATER_A)
    parser.add_argument("--rater-b", type=Path, default=DEFAULT_RATER_B)
    parser.add_argument("--adjudicated", type=Path, default=DEFAULT_ADJUDICATED)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    args = parser.parse_args()

    template = read_jsonl_by_id(args.template)
    rater_a = read_csv_by_id(args.rater_a)
    rater_b = read_csv_by_id(args.rater_b)

    missing = sorted(set(template) - set(rater_a) | set(template) - set(rater_b))
    if missing:
        raise ValueError(f"Missing ids in rater files: {missing[:10]}")

    ids = sorted(template, key=lambda value: int(value) if value.isdigit() else value)
    labels_a = [require_label(rater_a[row_id].get("label", ""), row_id, "rater_a_label") for row_id in ids]
    labels_b = [require_label(rater_b[row_id].get("label", ""), row_id, "rater_b_label") for row_id in ids]

    agreement = sum(a == b for a, b in zip(labels_a, labels_b)) / len(ids)
    kappa = cohen_kappa(labels_a, labels_b)

    agreement_rows = [
        {
            "n": len(ids),
            "n_raters": 2,
            "label_scheme": "supported / partially_supported / unsupported",
            "raw_agreement": f"{agreement:.6f}",
            "cohen_kappa": f"{kappa:.6f}",
        }
    ]
    write_csv(
        args.outdir / "human_eval_agreement.csv",
        agreement_rows,
        ["n", "n_raters", "label_scheme", "raw_agreement", "cohen_kappa"],
    )

    correlation_rows: list[dict] = []
    distribution_rows: list[dict] = []
    summary_extra = ""

    if args.adjudicated.exists():
        adjudicated = read_csv_by_id(args.adjudicated)
        missing_adj = sorted(set(template) - set(adjudicated))
        if missing_adj:
            raise ValueError(f"Missing ids in adjudicated file: {missing_adj[:10]}")
        adjudicated_labels = [
            require_label(adjudicated[row_id].get("adjudicated_label", ""), row_id, "adjudicated_label")
            for row_id in ids
        ]
        human_scores = [LABEL_VALUES[label] for label in adjudicated_labels]
        distribution = Counter(adjudicated_labels)
        distribution_rows = [
            {"label": label, "count": distribution[label], "proportion": f"{distribution[label] / len(ids):.6f}"}
            for label in LABEL_VALUES
        ]
        for metric in AUTO_METRICS:
            metric_scores = [float(template[row_id][metric]) for row_id in ids]
            correlation_rows.append(
                {
                    "metric": metric,
                    "metric_name": METRIC_NAMES[metric],
                    "spearman_rho": f"{spearman(human_scores, metric_scores):.6f}",
                    "pearson_r": f"{pearson(human_scores, metric_scores):.6f}",
                    "kendall_tau_b": f"{kendall_tau_b(human_scores, metric_scores):.6f}",
                    "n": len(ids),
                }
            )
        write_csv(
            args.outdir / "human_eval_label_distribution.csv",
            distribution_rows,
            ["label", "count", "proportion"],
        )
        write_csv(
            args.outdir / "human_eval_metric_correlations.csv",
            correlation_rows,
            ["metric", "metric_name", "spearman_rho", "pearson_r", "kendall_tau_b", "n"],
        )
        write_csv(
            args.outdir / "human_eval_correlations.csv",
            correlation_rows,
            ["metric", "metric_name", "spearman_rho", "pearson_r", "kendall_tau_b", "n"],
        )
        summary_rows = [
            {
                "n_examples": len(ids),
                "n_raters": 2,
                "label_scheme": "supported / partially_supported / unsupported",
                "raw_agreement": f"{agreement:.6f}",
                "cohen_kappa": f"{kappa:.6f}",
                "final_label_source": "adjudicated_label",
            }
        ]
        write_csv(
            args.outdir / "human_eval_summary.csv",
            summary_rows,
            [
                "n_examples",
                "n_raters",
                "label_scheme",
                "raw_agreement",
                "cohen_kappa",
                "final_label_source",
            ],
        )
        summary_extra = "\n\n## Adjudicated Labels\n\n"
        summary_extra += "\n".join(
            f"- {row['label']}: {row['count']} ({row['proportion']})" for row in distribution_rows
        )
        summary_extra += "\n\n## Metric Correlations\n\n"
        summary_extra += "\n".join(
            f"- {row['metric']}: Spearman rho={row['spearman_rho']}, "
            f"Pearson r={row['pearson_r']}, Kendall tau-b={row['kendall_tau_b']} (n={row['n']})"
            for row in correlation_rows
        )
    else:
        summary_extra = (
            "\n\nNo adjudicated file was found, so metric correlations were not computed. "
            f"Create {args.adjudicated} with an adjudicated_label column to finish the analysis."
        )

    summary = (
        "# Fix 3 Human Evaluation Summary\n\n"
        f"- Examples: {len(ids)}\n"
        "- Raters: 2\n"
        "- Label scheme: supported / partially_supported / unsupported\n"
        f"- Raw agreement: {agreement:.6f}\n"
        f"- Cohen's kappa: {kappa:.6f}"
        f"{summary_extra}\n"
    )
    args.outdir.mkdir(parents=True, exist_ok=True)
    (args.outdir / "human_eval_summary.md").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
