"""
scripts/kaggle_phase7.py — Phase 7 Kaggle parallelization helper
=================================================================

Generates a self-contained Kaggle notebook (`.ipynb`) that runs the
Groq-backed Phase 7 experiments on Kaggle's free CPU runtime.

Why Kaggle (Groq)?
------------------
- Groq is API-only, so the bottleneck is rate limit, not local hardware.
- Kaggle's free tier gives:
    * 9-hour wall-clock budget (plenty for any single Phase 7 script)
    * stable internet (won't drop mid-run if your laptop sleeps)
    * free GitHub push from inside the notebook
    * free CPU (Groq does the GPU work via API)

What's runnable on Kaggle/Groq:
    7.1 synthetic causal      (~10 min, uses ~50k tokens)
    7.2a MMR baseline           (~10 min, uses ~30k tokens)
    7.2b CRAG baseline          (~10 min, uses ~30k tokens)
    7.3 scaled headline (n=300) (~30 min, uses ~150k tokens — over daily limit, NOT recommended on Kaggle)

What's local-only:
    7.4 CCS alternatives  (no LLM, ~30s local)
    7.6 human-eval samples (no LLM, ~5s local)
    7.7 failure typology   (no LLM, ~10s local)

Usage
-----
    # 1. Generate the notebook locally:
    python3 scripts/kaggle_phase7.py
    # → writes notebooks/phase7_kaggle.ipynb

    # 2. (optional) pick a subset of scripts:
    python3 scripts/kaggle_phase7.py --scripts synthetic_causal mmr_baseline
    python3 scripts/kaggle_phase7.py --scripts synthetic_causal crag_baseline mmr_baseline

    # 3. Upload to Kaggle:
    #    kaggle.com → New Notebook → File → Import → select the .ipynb
    #    Settings:
    #       Accelerator: None (CPU is fine — Groq does the GPU work)
    #       Internet:    On
    #       Persistence: Files only
    #    Add Kaggle Secrets:
    #       GROQ_API_KEY    (required)
    #       GH_TOKEN        (optional, for auto-push of results)
    #
    # 4. Save & Run All.

The notebook intentionally stays small so the prep flow is auditable.
Re-generate any time the experiment scripts change — the notebook
does `git pull` rather than embedding the scripts.

Daily Groq token budget (free tier): 100k tokens total per organization.
If you've already burned the budget today, the notebook will hit 429s.
The runs are checkpoint-resumable — re-running the same notebook
tomorrow picks up where it stopped.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "notebooks" / "phase7_kaggle.ipynb"
DEFAULT_REPO = "Saket-Maganti/rag-hallucination-detection"
ALL_SCRIPTS = ["synthetic_causal", "mmr_baseline", "crag_baseline"]


# ---- Cell builders --------------------------------------------------

def _cell(kind: str, source: str) -> dict:
    base = {
        "cell_type": kind,
        "metadata": {},
        "source": [line + "\n" for line in source.splitlines()],
    }
    if kind == "code":
        base["execution_count"] = None
        base["outputs"] = []
    return base


def _cell_run_script(name: str, datasets: str, n: int, model: str) -> dict:
    """Generate a code cell that invokes one Phase 7 runner via Groq."""
    return _cell("code", (
        f"# Run Phase 7: {name}\n"
        "import subprocess, sys, time\n"
        f"print('=== {name} starting ===')\n"
        "t0 = time.time()\n"
        "subprocess.check_call([sys.executable, "
        f"'experiments/run_{name}.py', "
        f"'--backend', 'groq', '--model', '{model}', "
        f"'--datasets'] + {datasets!r}.split() + ["
        f"'--n', '{n}'])\n"
        "print(f'=== {} done in {{:.1f}} min ==='.format("
        f"'{name}', (time.time() - t0) / 60))\n"
    ))


def build_notebook(repo: str, scripts: list, model: str,
                    datasets: str, n_synthetic: int, n_baseline: int) -> dict:
    cells = [
        _cell("markdown", (
            "# Phase 7 — Kaggle (Groq) runner\n"
            "\n"
            "Runs the Groq-backed Phase 7 experiments (synthetic causal, "
            "MMR baseline, CRAG baseline) on Kaggle's free CPU runtime "
            "with Groq doing all the LLM work via API.\n"
            "\n"
            "**Wall-clock**: ~30-60 min total depending on selected "
            "scripts. **Internet must be ON**, `GROQ_API_KEY` must be "
            "in Kaggle Secrets.\n"
            "\n"
            "If you also set `GH_TOKEN` (Kaggle Secret), the last cell "
            f"pushes results to a `kaggle-phase7` branch on `{repo}`.\n"
            "\n"
            "**Daily Groq budget**: 100k tokens/day total per org. Each "
            "Phase 7 script uses 30-50k tokens. Plan 1-2 scripts/day if "
            "you have other Groq usage.\n"
            "\n"
            "## Selected scripts\n"
            "\n"
            f"This notebook will run: **{', '.join(scripts)}**.\n"
        )),
        _cell("code", (
            "# 1. Pull the repo (read-only clone is fine)\n"
            "import os, subprocess, pathlib\n"
            f"REPO_URL = 'https://github.com/{repo}.git'\n"
            "ROOT = pathlib.Path('/kaggle/working/rag')\n"
            "if not ROOT.exists():\n"
            "    subprocess.check_call(['git', 'clone', '--depth', '1', "
            "REPO_URL, str(ROOT)])\n"
            "os.chdir(ROOT)\n"
            "subprocess.check_call(['git', 'log', '--oneline', '-1'])\n"
        )),
        _cell("code", (
            "# 2. Install lean dependencies (Groq is API-only — no torch GPU)\n"
            "import subprocess, sys\n"
            "subprocess.check_call([sys.executable, '-m', 'pip', 'install', "
            "'-q', 'groq', 'sentence-transformers', 'chromadb', 'datasets', "
            "'langchain', 'langchain-community', 'langchain-ollama', "
            "'pandas', 'scipy'])\n"
        )),
        _cell("code", (
            "# 3. Pull GROQ_API_KEY from Kaggle Secrets\n"
            "from kaggle_secrets import UserSecretsClient\n"
            "secrets = UserSecretsClient()\n"
            "import os\n"
            "os.environ['GROQ_API_KEY'] = secrets.get_secret('GROQ_API_KEY')\n"
            "print('GROQ_API_KEY set:', bool(os.environ.get('GROQ_API_KEY')))\n"
        )),
        _cell("code", (
            "# 4. Smoke test before the long runs\n"
            "import subprocess, sys\n"
            "subprocess.check_call([sys.executable, "
            "'scripts/smoke_test_groq.py', "
            f"'--models', '{model}'])\n"
        )),
    ]

    # Per-script run cells
    for name in scripts:
        n = n_synthetic if name == "synthetic_causal" else n_baseline
        cells.append(_cell_run_script(name, datasets, n, model))

    # Final cell: bundle + optional GH push
    output_dirs = " ".join(f"results/{d}" for d in [
        "synthetic_causal", "mmr_baseline", "crag_baseline"
    ])
    cells.append(_cell("code", (
        "# 5. Snapshot results into a downloadable zip + optional GH push\n"
        "import shutil, subprocess, os, pathlib\n"
        "OUT = pathlib.Path('/kaggle/working')\n"
        f"# Bundle whatever Phase 7 result dirs exist\n"
        "phase7_dirs = ['synthetic_causal', 'mmr_baseline', 'crag_baseline',\n"
        "                'ccs_alternatives', 'failure_typology']\n"
        "import os\n"
        "existing = [f'results/{d}' for d in phase7_dirs if os.path.isdir(f'results/{d}')]\n"
        "if existing:\n"
        "    subprocess.check_call(['tar', 'czf', str(OUT / 'phase7_results.tar.gz')] + existing)\n"
        "    print('Wrote', OUT / 'phase7_results.tar.gz')\n"
        "else:\n"
        "    print('No Phase 7 result dirs found yet')\n"
        "\n"
        "# Optional: push to GitHub if GH_TOKEN secret is present.\n"
        "try:\n"
        "    from kaggle_secrets import UserSecretsClient\n"
        "    gh_token = UserSecretsClient().get_secret('GH_TOKEN')\n"
        "except Exception:\n"
        "    gh_token = None\n"
        "if gh_token and existing:\n"
        f"    REPO = '{repo}'\n"
        "    branch = 'kaggle-phase7'\n"
        "    subprocess.check_call(['git', 'config', 'user.email', "
        "'kaggle@auto'])\n"
        "    subprocess.check_call(['git', 'config', 'user.name', "
        "'kaggle-runner'])\n"
        "    subprocess.check_call(['git', 'checkout', '-B', branch])\n"
        "    for d in existing:\n"
        "        subprocess.check_call(['git', 'add', d])\n"
        "    subprocess.check_call(['git', 'commit', '-m', "
        "'Phase 7 results from Kaggle/Groq'])\n"
        "    push_url = f'https://{gh_token}@github.com/{REPO}.git'\n"
        "    subprocess.check_call(['git', 'push', '-f', push_url, "
        "branch])\n"
        "    print(f'Pushed to {REPO} {branch}')\n"
        "else:\n"
        "    if not gh_token:\n"
        "        print('No GH_TOKEN — download phase7_results.tar.gz manually.')\n"
    )))
    cells.append(_cell("markdown", (
        "## Where the results land\n"
        "\n"
        "After this notebook completes (with `GH_TOKEN`):\n"
        "\n"
        f"  - GitHub branch `kaggle-phase7` on `{repo}` has all CSVs/JSON.\n"
        f"  - Local: `git pull origin kaggle-phase7` then `git merge kaggle-phase7` to bring into main.\n"
        "\n"
        "Without `GH_TOKEN`:\n"
        "\n"
        "  - Download `phase7_results.tar.gz` from the Output panel.\n"
        "  - Extract to your local `results/` and commit normally.\n"
    )))

    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(DEFAULT_OUT),
                    help=f"Output notebook path (default: {DEFAULT_OUT})")
    ap.add_argument("--repo", default=DEFAULT_REPO,
                    help=f"GitHub repo to clone (default: {DEFAULT_REPO})")
    ap.add_argument("--scripts", nargs="+", default=ALL_SCRIPTS,
                    choices=ALL_SCRIPTS,
                    help="Which Phase 7 scripts to include")
    ap.add_argument("--model", default="llama-3.3-70b",
                    help="Groq model (default: llama-3.3-70b)")
    ap.add_argument("--datasets", default="squad pubmedqa",
                    help="space-separated dataset list")
    ap.add_argument("--n_synthetic", type=int, default=100,
                    help="n queries for synthetic causal (default 100)")
    ap.add_argument("--n_baseline", type=int, default=30,
                    help="n queries for MMR + CRAG baselines (default 30)")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nb = build_notebook(args.repo, args.scripts, args.model,
                          args.datasets, args.n_synthetic, args.n_baseline)
    out_path.write_text(json.dumps(nb, indent=1))
    print(f"[kaggle-phase7] wrote {out_path}")
    print(f"[kaggle-phase7] selected scripts: {args.scripts}")
    print(f"[kaggle-phase7] model: {args.model}")
    print(f"[kaggle-phase7] datasets: {args.datasets}")
    print()
    print(f"[kaggle-phase7] upload to https://www.kaggle.com/code → "
           "New Notebook → File → Import → select this .ipynb")
    print(f"[kaggle-phase7] required Kaggle Secret: GROQ_API_KEY  "
           "(optional: GH_TOKEN for auto-push)")
    print(f"[kaggle-phase7] estimated wall-clock: ~{10 * len(args.scripts)} "
           "min with Groq 30-RPM rate limit + retries")


if __name__ == "__main__":
    main()
