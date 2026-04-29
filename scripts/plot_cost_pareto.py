"""
Regenerate Figure 2 (cost-aware Pareto plot) for papers/neurips.

Reads the matched-harness baseline audit summary
(papers/neurips/source_tables/h2h_summary_full_selfrag.csv),
filters out the harness-mismatched Self-RAG rows, and plots p99 latency vs.
faithfulness for the three remaining methods on SQuAD and HotpotQA.

Design choices for this revision (see SUBMISSION_READY.md, Fix 1):
  * Two legends: a Method legend (shape, neutral gray) and a Dataset legend
    (color, filled circle) — combined legends were ambiguous.
  * Manual x-axis range [2.8, 4.5] s with breathing room around both clusters,
    instead of autoscaled tight bounds that visually merge the clusters.
  * Marker size ~120 with thin black edge for print legibility.
  * Per-point method labels offset above (SQuAD) or below (HotpotQA) the marker
    using adjustText to avoid label collisions; falls back to fixed offsets if
    adjustText is not installed.
  * Exports both PDF and PNG at 300 DPI.

Run:
    python3 scripts/plot_cost_pareto.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_CSV = REPO_ROOT / "papers/neurips" / "source_tables" / "h2h_summary_full_selfrag.csv"
OUT_DIR = REPO_ROOT / "papers/neurips" / "figures"

METHOD_ORDER = ["crag", "hcpc_v2", "raptor_2l"]
METHOD_LABEL = {"crag": "CRAG", "hcpc_v2": "HCPC-v2", "raptor_2l": "RAPTOR-2L"}
METHOD_MARKER = {"crag": "o", "hcpc_v2": "s", "raptor_2l": "^"}

DATASET_ORDER = ["squad", "hotpotqa"]
DATASET_LABEL = {"squad": "SQuAD", "hotpotqa": "HotpotQA"}
DATASET_COLOR = {"squad": "#1f77b4", "hotpotqa": "#ff7f0e"}


def load_matched_harness(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df[df["condition"].isin(METHOD_ORDER)].copy()
    df["p99_latency_s"] = df["p99_latency_ms"] / 1000.0
    return df


def _try_adjust_text(ax, texts):
    try:
        from adjustText import adjust_text

        adjust_text(
            texts,
            ax=ax,
            only_move={"points": "y", "texts": "xy"},
            arrowprops=dict(arrowstyle="-", color="0.4", lw=0.4),
            expand_points=(1.1, 1.4),
            expand_text=(1.05, 1.2),
        )
        return True
    except Exception:
        return False


def plot_pareto(df: pd.DataFrame, out_pdf: Path, out_png: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 4.2))

    annotations = []
    for _, row in df.iterrows():
        method = row["condition"]
        dataset = row["dataset"]
        marker = METHOD_MARKER[method]
        color = DATASET_COLOR[dataset]
        ax.scatter(
            row["p99_latency_s"],
            row["faithfulness"],
            s=120,
            marker=marker,
            facecolor=color,
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )
        # Default offset: above for SQuAD (upper cluster), below for HotpotQA.
        dy = 0.004 if dataset == "squad" else -0.006
        annotations.append(
            ax.text(
                row["p99_latency_s"],
                row["faithfulness"] + dy,
                METHOD_LABEL[method],
                fontsize=8,
                ha="center",
                va="bottom" if dy > 0 else "top",
                zorder=4,
            )
        )

    ax.set_xlim(2.8, 4.5)
    ymin = float(df["faithfulness"].min())
    ymax = float(df["faithfulness"].max())
    pad = max((ymax - ymin) * 0.5, 0.02)
    ax.set_ylim(ymin - pad, ymax + pad)

    ax.set_xlabel("p99 latency (seconds)")
    ax.set_ylabel("faithfulness (DeBERTa-v3 NLI)")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    method_handles = [
        Line2D(
            [0],
            [0],
            marker=METHOD_MARKER[m],
            linestyle="",
            color="0.45",
            markeredgecolor="black",
            markeredgewidth=0.5,
            markersize=9,
            label=METHOD_LABEL[m],
        )
        for m in METHOD_ORDER
    ]
    dataset_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            color=DATASET_COLOR[d],
            markeredgecolor="black",
            markeredgewidth=0.5,
            markersize=9,
            label=DATASET_LABEL[d],
        )
        for d in DATASET_ORDER
    ]

    method_legend = ax.legend(
        handles=method_handles,
        title="Method",
        loc="upper right",
        frameon=False,
        fontsize=8,
        title_fontsize=9,
    )
    ax.add_artist(method_legend)
    ax.legend(
        handles=dataset_handles,
        title="Dataset",
        loc="upper left",
        frameon=False,
        fontsize=8,
        title_fontsize=9,
    )

    used_adjust = _try_adjust_text(ax, annotations)
    if not used_adjust:
        # Fallback: nudge HotpotQA HCPC-v2 / RAPTOR-2L apart manually.
        for txt, (_, row) in zip(annotations, df.iterrows()):
            if row["dataset"] == "hotpotqa" and row["condition"] == "raptor_2l":
                txt.set_position((row["p99_latency_s"] + 0.05, row["faithfulness"] - 0.006))
                txt.set_ha("left")

    fig.tight_layout()
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_matched_harness(SOURCE_CSV)
    if df.empty:
        raise SystemExit(f"No matched-harness rows found in {SOURCE_CSV}")
    out_pdf = OUT_DIR / "figure2_pareto_p99_log.pdf"
    out_png = OUT_DIR / "figure2_pareto_p99_log.png"
    plot_pareto(df, out_pdf, out_png)
    print(f"Wrote {out_pdf}")
    print(f"Wrote {out_png}")


if __name__ == "__main__":
    main()
