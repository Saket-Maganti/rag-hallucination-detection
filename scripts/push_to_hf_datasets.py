"""
scripts/push_to_hf_datasets.py — Phase 5 #2 (HF Datasets push)
==============================================================

Turn ContextCoherenceBench into a `load_dataset(...)`-able artifact.
After this runs, anyone can do:

    from datasets import load_dataset
    ds = load_dataset("saketmgnt/context-coherence-bench")
    print(ds["adversarial_drift"][0])

That's the single biggest community-leverage move for the benchmark:
right now it requires `git clone` + reading a README; this turns it
into a one-line dependency that downstream papers can just import.

What we push (one repo, multiple configurations):

    config "adversarial_drift"      from data/adversarial/drift.jsonl
    config "adversarial_disjoint"   from data/adversarial/disjoint.jsonl
    config "adversarial_contradict" from data/adversarial/contradict.jsonl
    config "adversarial_control"    from data/adversarial/control.jsonl
    config "coherence_paradox"      from results/multidataset/per_query.csv

Each config gets a typed schema, a description, and a README front-matter
that HF auto-renders.

Prerequisites:
    pip install datasets huggingface_hub
    huggingface-cli login        (or set HF_TOKEN env var)

Usage:
    python3 scripts/push_to_hf_datasets.py                 # dry-run preview
    python3 scripts/push_to_hf_datasets.py --push          # actually push
    python3 scripts/push_to_hf_datasets.py --push --repo your-name/your-bench
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REPO = "saketmgnt/context-coherence-bench"

ADVERSARIAL_DIR = ROOT / "data" / "adversarial"
PER_QUERY_CSV = ROOT / "results" / "multidataset" / "per_query.csv"

README_FRONTMATTER = """\
---
license: mit
task_categories:
  - question-answering
  - text-classification
language:
  - en
tags:
  - retrieval-augmented-generation
  - hallucination-detection
  - context-coherence
  - benchmark
pretty_name: ContextCoherenceBench
size_categories:
  - 1K<n<10K
configs:
  - config_name: adversarial_drift
    data_files: adversarial/drift.jsonl
  - config_name: adversarial_disjoint
    data_files: adversarial/disjoint.jsonl
  - config_name: adversarial_contradict
    data_files: adversarial/contradict.jsonl
  - config_name: adversarial_control
    data_files: adversarial/control.jsonl
  - config_name: coherence_paradox
    data_files: paradox/per_query.parquet
---

# ContextCoherenceBench

Companion benchmark for *"When Better Retrieval Hurts: Context Coherence
Drives Faithfulness in Retrieval-Augmented Generation"* (NeurIPS 2026
submission).

**Permanent DOI**: [10.5281/zenodo.19757291](https://doi.org/10.5281/zenodo.19757291)
**Code**: <https://github.com/Saket-Maganti/rag-hallucination-detection>
**Demo**: <https://huggingface.co/spaces/saketmgnt/sakkk>

## Configurations

### `adversarial_*` (4 splits)
Hand-validated adversarial coherence pairs across 4 categories:
- `drift` — passages drift off-topic mid-set
- `disjoint` — disjoint sub-clusters within retrieval set
- `contradict` — internally contradictory passages
- `control` — coherent baseline

129 validated cases total (40 + 50 + 10 + 29 after the NLI validator).

### `coherence_paradox`
Per-query records across 5 datasets × 3 generators × 3 retrieval
conditions (baseline, HCPC-v1, HCPC-v2). Includes faithfulness scores,
hallucination labels, retrieval similarity, refinement decisions, and
Context Coherence Score (CCS).

## Quick start

```python
from datasets import load_dataset

# All paradox per-query records
paradox = load_dataset("saketmgnt/context-coherence-bench",
                       "coherence_paradox", split="train")

# Just the adversarial drift cases
drift = load_dataset("saketmgnt/context-coherence-bench",
                     "adversarial_drift", split="train")
```

## Citation

```bibtex
@dataset{maganti2026ccbench,
  title     = {ContextCoherenceBench: a benchmark for evaluating
               context coherence in retrieval-augmented generation},
  author    = {Maganti, Saket},
  year      = {2026},
  publisher = {Zenodo},
  version   = {v2.0.0},
  doi       = {10.5281/zenodo.19757291},
}
```
"""


def _stage(staging: Path) -> Dict[str, int]:
    """Copy adversarial JSONLs + paradox CSV into a clean staging tree
    that mirrors the HF dataset repo layout. Returns row counts."""
    counts: Dict[str, int] = {}

    adv_dst = staging / "adversarial"
    adv_dst.mkdir(parents=True, exist_ok=True)
    for fname in ["drift.jsonl", "disjoint.jsonl",
                   "contradict.jsonl", "control.jsonl"]:
        src = ADVERSARIAL_DIR / fname
        if not src.exists():
            print(f"[hf-push] WARN: missing {src}")
            continue
        dst = adv_dst / fname
        dst.write_bytes(src.read_bytes())
        # Count lines = cases
        n = sum(1 for _ in dst.read_text().splitlines() if _.strip())
        counts[f"adversarial_{fname[:-6]}"] = n
        print(f"[hf-push]   {fname}: {n} cases")

    par_dst_dir = staging / "paradox"
    par_dst_dir.mkdir(parents=True, exist_ok=True)
    if PER_QUERY_CSV.exists():
        df = pd.read_csv(PER_QUERY_CSV)
        # Cast the bool column so HF schema is clean
        if "is_hallucination" in df.columns:
            df["is_hallucination"] = df["is_hallucination"].astype(bool)
        if "refined" in df.columns:
            df["refined"] = df["refined"].astype(bool)
        out = par_dst_dir / "per_query.parquet"
        df.to_parquet(out, index=False)
        counts["coherence_paradox"] = len(df)
        print(f"[hf-push]   coherence_paradox: {len(df)} rows -> {out.name}")

    (staging / "README.md").write_text(README_FRONTMATTER)
    return counts


def _preview(staging: Path) -> None:
    print(f"\n[hf-push] staging tree at {staging.relative_to(ROOT.parent)}:")
    for p in sorted(staging.rglob("*")):
        if p.is_file():
            sz = p.stat().st_size
            print(f"   {p.relative_to(staging)}  ({sz} bytes)")


def _push(staging: Path, repo: str, hf_token: str) -> None:
    from huggingface_hub import HfApi, login
    if hf_token:
        login(token=hf_token, add_to_git_credential=False)
    api = HfApi()

    # Create the dataset repo if it doesn't exist (idempotent).
    try:
        api.create_repo(repo_id=repo, repo_type="dataset",
                         exist_ok=True, private=False)
        print(f"[hf-push] repo ready: {repo}")
    except Exception as exc:
        print(f"[hf-push] create_repo: {exc}")

    print(f"[hf-push] uploading {staging} -> {repo}")
    api.upload_folder(
        folder_path=str(staging),
        repo_id=repo,
        repo_type="dataset",
        commit_message="ContextCoherenceBench v2.0.0 push",
        ignore_patterns=[".git*"],
    )
    print(f"[hf-push] ✅ done. Browse: https://huggingface.co/datasets/{repo}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--push", action="store_true",
                    help="actually upload (default: dry-run preview only)")
    ap.add_argument("--repo", default=DEFAULT_REPO,
                    help=f"target HF dataset repo (default: {DEFAULT_REPO})")
    ap.add_argument("--staging", default=str(ROOT / "hf_dataset_staging"),
                    help="local staging dir (clean rebuild on each run)")
    args = ap.parse_args()

    staging = Path(args.staging)
    if staging.exists():
        import shutil
        shutil.rmtree(staging)
    staging.mkdir(parents=True)

    counts = _stage(staging)
    _preview(staging)

    print(f"\n[hf-push] row counts: {counts}")
    print(f"[hf-push] total: {sum(counts.values())} rows across "
          f"{len(counts)} configurations")

    if not args.push:
        print("\n[hf-push] dry-run only — pass --push to upload.")
        return

    hf_token = os.environ.get("HF_TOKEN", "").strip()
    if not hf_token:
        print("[hf-push] HF_TOKEN env var not set; will rely on "
              "huggingface-cli login state.")
    _push(staging, args.repo, hf_token)


if __name__ == "__main__":
    main()
