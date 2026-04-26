"""
scripts/kaggle_frontier_scale.py — Path 2 Kaggle parallelization helper
=======================================================================

Generates a self-contained Kaggle notebook (`.ipynb`) that runs
`experiments/run_frontier_scale.py` on Kaggle's free CPU runtime.

Why Kaggle?
-----------
Groq is an *API*, so the local hardware doesn't matter — the bottleneck
is rate-limit (~30 RPM per model on the free tier).  Kaggle's free tier
gives us:

    1. A stable internet connection that won't drop mid-run when the
       laptop sleeps.
    2. 9-hour wall-clock budget (we need ~45 min, comfortable margin).
    3. Free `git push` to GitHub from inside the notebook so results
       land in the repo without manual zip-download.
    4. Optional GPU isn't needed — generation runs on Groq's servers.

Usage
-----
    # 1. Generate the notebook locally:
    python3 scripts/kaggle_frontier_scale.py
    # → writes notebooks/frontier_scale_kaggle.ipynb

    # 2. On kaggle.com → New Notebook → File → Import → upload the .ipynb
    #    Settings:
    #       Accelerator: None (CPU is fine — Groq does the GPU work)
    #       Internet:    On
    #       Persistence: Files only
    #    Add Kaggle Secrets:
    #       GROQ_API_KEY     (required)
    #       GH_TOKEN         (optional, for auto-push)
    #
    # 3. Save & Run All.  ~45 min wall-clock.
    #
    # 4. Either download `frontier_scale.zip` from the Output panel or
    #    let the notebook push results to a `kaggle-frontier-scale`
    #    branch on GitHub if GH_TOKEN was provided.

The notebook is intentionally small (six cells) so that the whole
prep flow stays auditable.  Re-generate any time the experiment script
changes — the notebook does `git pull` rather than embedding the script.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT = ROOT / "notebooks" / "frontier_scale_kaggle.ipynb"
DEFAULT_REPO = "Saket-Maganti/rag-hallucination-detection"


def _cell(kind: str, source: str) -> dict:
    """Build a single Jupyter notebook cell dict."""
    base = {
        "cell_type": kind,
        "metadata": {},
        "source": [line + "\n" for line in source.splitlines()],
    }
    if kind == "code":
        base["execution_count"] = None
        base["outputs"] = []
    return base


def build_notebook(repo: str) -> dict:
    cells = [
        _cell("markdown", (
            "# Frontier-scale ablation — Kaggle runner\n"
            "\n"
            "Runs `experiments/run_frontier_scale.py` on Kaggle so results "
            "land in the repo without tying up the laptop. Total wall-clock "
            "≈ 45 min. **Internet must be ON** and `GROQ_API_KEY` must be "
            "set as a Kaggle Secret.\n"
            "\n"
            "If you also set `GH_TOKEN`, the last cell pushes results to a "
            f"`kaggle-frontier-scale` branch on `{repo}`."
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
            "'langchain', 'langchain-community', 'pandas'])\n"
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
            "# 4. Smoke test before the long run\n"
            "import subprocess, sys\n"
            "subprocess.check_call([sys.executable, "
            "'scripts/smoke_test_groq.py', "
            "'--models', 'llama-3.3-70b', 'mixtral-8x7b'])\n"
        )),
        _cell("code", (
            "# 5. Run the frontier-scale experiment (~45 min)\n"
            "import subprocess, sys\n"
            "subprocess.check_call([sys.executable, "
            "'experiments/run_frontier_scale.py', "
            "'--datasets', 'squad', 'pubmedqa', "
            "'--models', 'llama-3.3-70b', 'mixtral-8x7b', "
            "'--n_questions', '30'])\n"
        )),
        _cell("code", (
            "# 6. Snapshot results into a downloadable zip + optional GH push\n"
            "import shutil, subprocess, os, pathlib\n"
            "OUT = pathlib.Path('/kaggle/working')\n"
            "shutil.make_archive(str(OUT / 'frontier_scale'), 'zip', "
            "'results/frontier_scale')\n"
            "print('Wrote', OUT / 'frontier_scale.zip')\n"
            "\n"
            "# Optional: push to GitHub if GH_TOKEN secret is present.\n"
            "try:\n"
            "    from kaggle_secrets import UserSecretsClient\n"
            "    gh_token = UserSecretsClient().get_secret('GH_TOKEN')\n"
            "except Exception:\n"
            "    gh_token = None\n"
            "if gh_token:\n"
            f"    REPO = '{repo}'\n"
            "    branch = 'kaggle-frontier-scale'\n"
            "    subprocess.check_call(['git', 'config', 'user.email', "
            "'kaggle@auto'])\n"
            "    subprocess.check_call(['git', 'config', 'user.name', "
            "'kaggle-runner'])\n"
            "    subprocess.check_call(['git', 'checkout', '-B', branch])\n"
            "    subprocess.check_call(['git', 'add', "
            "'results/frontier_scale'])\n"
            "    subprocess.check_call(['git', 'commit', '-m', "
            "'Frontier-scale results from Kaggle'])\n"
            "    push_url = f'https://{gh_token}@github.com/{REPO}.git'\n"
            "    subprocess.check_call(['git', 'push', '-f', push_url, "
            "branch])\n"
            "    print(f'Pushed to {REPO} {branch}')\n"
            "else:\n"
            "    print('No GH_TOKEN — download frontier_scale.zip manually.')\n"
        )),
    ]
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
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nb = build_notebook(args.repo)
    out_path.write_text(json.dumps(nb, indent=1))
    print(f"[kaggle] wrote {out_path}")
    print(f"[kaggle] upload to https://www.kaggle.com/code → New Notebook → "
          f"File → Import → select this .ipynb")
    print("[kaggle] required Kaggle Secret: GROQ_API_KEY  "
          "(optional: GH_TOKEN for auto-push)")


if __name__ == "__main__":
    main()
