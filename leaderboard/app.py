"""
leaderboard/app.py — P2 Item 10b (community leaderboard)
=========================================================

Gradio app that lets researchers (a) browse the current paradox / faith
leaderboard, and (b) submit their own retriever's numbers as a pull
request to this repo.

Design
------
The leaderboard is stored as a single YAML file committed to the repo:

    release/context_coherence_bench_v1/leaderboard.yaml

Each entry has the shape:

    - method: "HCPC-v2 (ours)"
      authors: ["Maganti et al."]
      dataset: "squad"
      n_queries: 150
      faith_mean: 0.8077
      halluc_rate: 0.0133
      ccs_mean: 0.70
      paradox_drop: 0.030
      submitted: "2026-04-25"
      notes: "sim=0.45, ce=-0.20, protected_top_k=2"

The submission flow is intentionally "form → JSON blob → clipboard copy" —
we do not allow the app to write to the YAML directly because that would
mean trusting arbitrary form input.  Instead the user copies their entry
and opens a GitHub PR (one-click link that prefills the PR body).

Run locally
-----------
    pip install gradio pyyaml pandas
    python3 leaderboard/app.py

Deploy as a second HF Space (optional)
--------------------------------------
Copy this directory + `leaderboard.yaml` + a pinned `requirements.txt`
into a new Space; same CPU-basic tier is enough.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path

import gradio as gr
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LEADERBOARD_YAML = (
    ROOT / "release" / "context_coherence_bench_v1" / "leaderboard.yaml"
)

REPO_SLUG = "Saket-Maganti/rag-hallucination-detection"


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────

def _seed_entries() -> list[dict]:
    """Canonical seed rows extracted from our own results, so visitors see
    something meaningful on first load even if leaderboard.yaml has not yet
    been committed."""
    return [
        {"method": "baseline (fixed 1024)", "authors": ["Maganti et al."],
         "dataset": "squad", "n_queries": 150, "faith_mean": 0.7987,
         "halluc_rate": 0.0267, "ccs_mean": 0.71, "paradox_drop": 0.0,
         "submitted": "2026-04-25",
         "notes": "k=3, no rerank, baseline reference"},
        {"method": "HCPC-v1", "authors": ["Maganti et al."],
         "dataset": "squad", "n_queries": 150, "faith_mean": 0.7664,
         "halluc_rate": 0.0400, "ccs_mean": 0.72, "paradox_drop": 0.032,
         "submitted": "2026-04-25", "notes": "sim=0.50, ce=0.00"},
        {"method": "HCPC-v2", "authors": ["Maganti et al."],
         "dataset": "squad", "n_queries": 150, "faith_mean": 0.8077,
         "halluc_rate": 0.0133, "ccs_mean": 0.70, "paradox_drop": 0.030,
         "submitted": "2026-04-25",
         "notes": "sim=0.45, ce=-0.20, protected_top_k=2"},
        {"method": "baseline", "authors": ["Maganti et al."],
         "dataset": "pubmedqa", "n_queries": 150, "faith_mean": 0.6013,
         "halluc_rate": 0.1733, "ccs_mean": 0.68, "paradox_drop": 0.0,
         "submitted": "2026-04-25", "notes": "reference"},
        {"method": "HCPC-v2", "authors": ["Maganti et al."],
         "dataset": "pubmedqa", "n_queries": 150, "faith_mean": 0.5902,
         "halluc_rate": 0.1667, "ccs_mean": 0.68, "paradox_drop": 0.032,
         "submitted": "2026-04-25", "notes": "v2 recovery positive"},
    ]


def load_entries() -> list[dict]:
    if LEADERBOARD_YAML.exists():
        try:
            import yaml  # soft dep — fall back to seed if missing
            with LEADERBOARD_YAML.open() as fh:
                data = yaml.safe_load(fh) or []
            if isinstance(data, list) and data:
                return data
        except Exception as exc:
            print(f"[leaderboard] YAML load failed ({exc}); using seeds")
    return _seed_entries()


def entries_to_df(entries: list[dict], dataset_filter: str) -> pd.DataFrame:
    df = pd.DataFrame(entries)
    if dataset_filter and dataset_filter != "all" and "dataset" in df.columns:
        df = df[df["dataset"] == dataset_filter]
    keep_cols = ["method", "dataset", "n_queries", "faith_mean",
                 "halluc_rate", "ccs_mean", "paradox_drop",
                 "submitted", "notes", "authors"]
    df = df[[c for c in keep_cols if c in df.columns]]
    if "faith_mean" in df.columns:
        df = df.sort_values(["dataset", "faith_mean"],
                            ascending=[True, False]).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Submission form helpers
# ─────────────────────────────────────────────────────────────────────────────

SUBMISSION_TEMPLATE = """\
Proposed entry:

```yaml
- method: {method}
  authors: [{authors}]
  dataset: {dataset}
  n_queries: {n_queries}
  faith_mean: {faith_mean}
  halluc_rate: {halluc_rate}
  ccs_mean: {ccs_mean}
  paradox_drop: {paradox_drop}
  submitted: "{submitted}"
  notes: "{notes}"
```

**Reproducibility checklist** (reviewers will check):
- [ ] `n_queries` ≥ 30 per dataset
- [ ] `faith_mean` computed with `src/hallucination_detector.py` NLI scorer
- [ ] `ccs_mean` computed with `HCPCv2Retriever._compute_ccs` (or equivalent)
- [ ] Code or commit hash public
- [ ] Re-ran on exactly the datasets named in `release/context_coherence_bench_v1/`
"""


def format_submission(method, authors, dataset, n_queries, faith, halluc,
                      ccs, paradox_drop, notes):
    body = SUBMISSION_TEMPLATE.format(
        method=method.strip(),
        authors=", ".join(f'"{a.strip()}"' for a in authors.split(",") if a.strip()),
        dataset=dataset, n_queries=int(n_queries),
        faith_mean=round(float(faith), 4),
        halluc_rate=round(float(halluc), 4),
        ccs_mean=round(float(ccs), 4),
        paradox_drop=round(float(paradox_drop), 4),
        submitted=date.today().isoformat(),
        notes=notes.strip().replace('"', "'"),
    )
    # One-click GitHub PR link that prefills title + body.
    import urllib.parse
    pr_title = f"Leaderboard: add {method} ({dataset})"
    pr_url = (
        f"https://github.com/{REPO_SLUG}/issues/new?"
        + urllib.parse.urlencode({
            "title": pr_title,
            "body":  body,
            "labels": "leaderboard-submission",
        })
    )
    return body, pr_url


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────

def build_app() -> gr.Blocks:
    entries = load_entries()
    datasets = sorted({e.get("dataset", "?") for e in entries}) + ["all"]

    with gr.Blocks(title="Coherence Paradox Leaderboard") as app:
        gr.Markdown("# 🏆 Coherence Paradox Leaderboard\n"
                    "Community-submitted numbers on "
                    "`release/context_coherence_bench_v1/`.")

        with gr.Tab("Browse"):
            ds_dd = gr.Dropdown(sorted(set(datasets)), value="all",
                                label="Filter by dataset")
            board = gr.Dataframe(value=entries_to_df(entries, "all"),
                                 interactive=False, wrap=True)
            ds_dd.change(lambda d: entries_to_df(entries, d),
                         inputs=ds_dd, outputs=board)

        with gr.Tab("Submit"):
            gr.Markdown(
                "Fill in your retriever's numbers, then click **Generate "
                "submission** → **Open GitHub PR**.  We review via PR so "
                "the YAML file stays trustworthy."
            )
            with gr.Row():
                method = gr.Textbox(label="Method name (e.g. 'MyRetriever')")
                authors = gr.Textbox(label="Authors (comma-separated)")
            with gr.Row():
                dataset = gr.Dropdown(
                    ["squad", "pubmedqa", "naturalqs", "triviaqa",
                     "hotpotqa", "financebench", "qasper", "msmarco"],
                    label="Dataset")
                n_queries = gr.Number(label="n_queries", value=30)
            with gr.Row():
                faith = gr.Number(label="faith_mean (0-1)", value=0.75)
                halluc = gr.Number(label="halluc_rate (0-1)", value=0.1)
                ccs = gr.Number(label="ccs_mean", value=0.7)
            with gr.Row():
                paradox_drop = gr.Number(label="paradox_drop (faith_baseline - faith_yours)",
                                         value=0.0)
                notes = gr.Textbox(label="Notes (config, hyperparameters)")
            gen_btn = gr.Button("Generate submission", variant="primary")
            blob = gr.Textbox(label="YAML blob (copy into a PR to "
                              "`release/context_coherence_bench_v1/leaderboard.yaml`)",
                              lines=14)
            link = gr.Markdown()
            gen_btn.click(
                lambda *a: (format_submission(*a)[0],
                            f"**[→ open pre-filled GitHub issue]"
                            f"({format_submission(*a)[1]})**"),
                inputs=[method, authors, dataset, n_queries, faith, halluc,
                        ccs, paradox_drop, notes],
                outputs=[blob, link],
            )

        with gr.Tab("Rules"):
            gr.Markdown("""\
### What counts as a valid submission

1. **Score on at least one of** squad, pubmedqa, naturalqs, triviaqa, hotpotqa,
   qasper, msmarco.  FinanceBench is HF-gated and not required.
2. **n_queries ≥ 30**.  If you run fewer, we'll flag the row as preliminary.
3. **Use the provided evaluators**: `src/hallucination_detector.py` for
   faithfulness, and `HCPCv2Retriever._compute_ccs` (or the equivalent
   formula from §3 of the paper) for CCS.
4. **Include a commit hash or a link to the code** that produced your numbers.
5. **`paradox_drop`** = `faith_baseline − faith_yours`, where `faith_baseline`
   is the fixed-1024-chunk k=3 baseline on the same dataset.  Sign matters:
   positive = your method sacrificed faith; negative = your method improved.

Submissions that violate (1)-(3) will be left in an "unverified" band but
still visible.  The maintainers will attempt reproduction; pass → row is
promoted to the main table.
""")

    return app


if __name__ == "__main__":
    build_app().launch(server_name="0.0.0.0", server_port=7861)
