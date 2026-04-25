# CLAUDE.md — project context for future sessions

Operational notes for picking up the RAG hallucination detection paper revision. Read this first.

## Project in one paragraph

Research artifact for the "Refinement Paradox" RAG paper. Core claim: better per-passage retrieval can *reduce* answer faithfulness by fragmenting context coherence. Introduces **CCS** (Context Coherence Score) as a generator-free retrieval-time diagnostic and **HCPC-v2** as a controlled intervention probe. Target venue: **NeurIPS** (following professor feedback on 2026-04-23).

Main repo: `/Users/saketmaganti/claudeprojs/rag-hallucination-detection`
Remote: `https://github.com/Saket-Maganti/rag-hallucination-detection`
Default branch: `main`

## Current status (2026-04-25, end-of-day)

**Phase 1 (8-item) — COMPLETE.** All reviewer-facing experiments ran.
**Phase 2 (10-item) — 10/10 RUN.** Frontier-scale landed 2026-04-25: SQuAD/Llama-3.3-70B paradox = +0.100 (exact match to 7B), GPT-OSS-120B = +0.030. **Kills "small-model artifact" critique.** All 12/12 paper tables now filled.
**Phase 3 (pre-submission polish) — CODE SHIPPED.** All "must-do" + high-leverage scripts written, smoke-tested, and wired into the paper. Headline figure (frontier-scale paradox vs scale), CCS calibration figure (distribution split + quintile bars), qualitative paradox example (Super Bowl 50: baseline says "Santa Clara", HCPC-v1 says "San Francisco Bay Area"), zero-error paper linter, Zenodo upload helper, OpenReview submission checklist + metadata YAML, anonymous-toggle author block. **PDF: 64 pages, 728 KB, 0 LaTeX warnings, 0 lint errors.** Awaiting user execution of #4 (Zenodo upload — needs token) and #5 (OR submission — manual).
**Paper — TIGHTENED.** New `robustness.tex` section bundles all six new robustness checks; `analysis.tex::Limitations` extended with three new caveat paragraphs (long-form scope, noise-equivalence, adversarial-129); `theory.tex` and `robustness.tex` wired into `main.tex`; abstract updated with multi-seed numbers and explicit scope qualifications. **Compiles cleanly: 62 pages, 681 KB PDF.**
**Paths 2 & 3 — STAGED.** Frontier-scale runner (`experiments/run_frontier_scale.py`) + Groq wrapper (`src/groq_llm.py`) + smoke test (`scripts/smoke_test_groq.py`) + Kaggle notebook generator (`scripts/kaggle_frontier_scale.py`) all in place. HF Space staging (`scripts/prepare_hf_space.py`) + Gradio demo (`space/`) + leaderboard (`leaderboard/`) + release tagger (`scripts/release_v2.sh`) ready to ship. User just needs `GROQ_API_KEY` (free) and HF Space repo URL.

**11/12 paper-tables filled** (`results/paper_tables/ALL_TABLES.md`); only `table_6_frontier` missing (waiting on Groq).

### Final headline numbers (use these in §Results / abstract)

| Claim | Number | Source |
|---|---|---|
| SQuAD paradox magnitude | **0.069 ± 0.004** (17× signal/σ across 3 seeds) | `results/multiseed/paradox_variance.csv` |
| PubMedQA paradox magnitude | **0.043 ± 0.013** (3.3× signal/σ) | `results/multiseed/paradox_variance.csv` |
| Prompt main effect | **F=0.046, p=0.83** (SQuAD); F=1.12, p=0.29 (PubMedQA) | `results/stats/` ANOVA |
| HCPC vs RAPTOR (faith) | HCPC-v1 wins 2/3: SQuAD −0.081, HotpotQA **+0.053**, PubMedQA −0.036 | `results/raptor/raptor_vs_hcpc.csv` |
| Long-form: MS-MARCO unsupported-rate | baseline 22.4% → HCPC-v1 **7.5%** (-67%) | `results/longform/summary.csv` |
| Sub-chunk sweet spot | 256 tokens; paradox attenuates at 512 | `results/subchunk_sensitivity/paradox_by_sub.csv` |
| Adversarial validator yield | 129/200 cases (drift 50, disjoint 40, control 29, contradict 10) | `data/adversarial/*.jsonl` |
| Frontier-scale: SQuAD paradox | **Llama-3.3-70B = +0.100** (exact match to 7B), GPT-OSS-120B = +0.030 | `results/frontier_scale/paradox_by_scale.csv` |
| Frontier-scale: PubMedQA paradox | 70B = +0.005, 120B = +0.013 (attenuates as predicted) | same |

### Caveats now explicit in paper §Limitations

1. **Long-form**: paradox does NOT generalize to QASPER (Δ=+0.002) or MS-MARCO (Δ=−0.033, sign flips). Reframed as scope statement; HCPC instead reduces unsupported-claim rate on MS-MARCO.
2. **Noise vs coherence**: SQuAD noise slope (−0.154) > paradox magnitude (0.100). Original `ratio≥2` target NOT met. Reframed as "qualitatively distinct" not "magnitudinally larger".
3. **Adversarial set**: validator rejected 71/200 generated cases. Set ships at 129; mech classifier still on n=20 pairs.
4. ~~**Frontier-scale**: 70B/Mixtral run scoped but not executed.~~ **DONE 2026-04-25**: Llama-3.3-70B + GPT-OSS-120B via Groq. 3/4 rows persist. SQuAD/70B exactly reproduces 7B magnitude. Mixtral was decommissioned by Groq mid-flight; replaced with gpt-oss-120b which is strictly larger anyway.

## Professor's feedback (2026-04-23)

> "3rd paper is probably the one with the most interesting results and work... Where this might get attacked is if reviewers feel it's a very metric based paper... heavy reliance on one embedding model, so reviewers would ask is this just a property of weak embeddings? Would a stronger retriever fix coherence?"

Action items:
1. **Stronger retrievers** → #8 multi-retriever (✅ done — paradox survives on PubMedQA, collapses on SQuAD with 335 M embedders → reframed as alignment-mismatch property)
2. **Multi-dataset + human eval OR remove human eval** → #2 multi-dataset done across 5 datasets × 3 models; Prolific deferred (IRB)
3. "Target NeurIPS if those fixes land" → Phase 1 alone now makes the paper competitive; Phase 2 pushes it to top-tier

## Phase 1 — rebuttal infrastructure (DONE)

| # | Item | Where | Status | Evidence |
|---|---|---|---|---|
| 5 | Adversarial §7.6 | Mac | ✅ | `data/adversarial/*.jsonl` (40 cases across 4 categories), `results/adversarial/` |
| 8 | Multi-retriever × 4 embedders × 2 datasets | Mac | ✅ | `results/multi_retriever/paradox_by_embedder.csv` |
| 4 | Head-to-head vs Self-RAG + CRAG | Kaggle T4 | ✅ | `results/headtohead/summary.md` (Self-RAG partial — OOM) |
| 2 | Multi-dataset × 3 generators (5 datasets) | Mac | ✅ | `results/multidataset/summary.{csv,md}` — 15 tuples × 90 rows each |
| 2b | FinanceBench row | Mac | ⚠️ HF-gated | 0 rows across 3 models; runbook pre-approved as optional |
| 1 | Mechanistic attention probe | Kaggle T4 | ✅ | `results/mechanistic/` — entropy_by_layer, retrieved_mass_by_layer, per_pair, top_k_attributions |
| 7 | Significance tests | Mac | ✅ | `results/stats/significance_tests.json` + `summary.csv` |
| 8-pkg | Benchmark packager | Mac | ✅ | `release/context_coherence_bench_v1/` (HF-loadable, CITATION.bib, LICENSE, metadata) |
| 6 | Prolific human eval | — | ❌ Deferred to camera-ready (needs IRB + ~$500) |

## Phase 2 — paper-strengthening upgrade plan (10 items)

Rationale: Phase 1 makes the paper competitive; Phase 2 makes it top-tier. Scripts are being written in parallel with runs.

| Item | Title | Script / File | Status | ETA |
|---|---|---|---|---|
| 4 | Mechanistic → hallucination classifier | `experiments/train_mech_classifier.py` | ✅ **RUN — L1 AUC=0.85** (baseline=0.80, target≥0.75) | (done, 30 s) |
| 6 | Adversarial set 40 → 200 | `experiments/generate_adversarial_cases.py` | ✅ Written, bug fixed (dict→list flatten), ready to run | 3–4 h (Ollama Mistral) |
| 7 | HCPC sub-chunk size sensitivity {128, 256, 512} | `experiments/run_subchunk_sensitivity.py` | ✅ Written, not yet run | 3 h |
| 8 | Theoretical proposition (info-theoretic bound + theorem) | `ragpaper/sections/theory.tex` | ✅ Drafted, needs prof review + `\input` into `main.tex` | — (paper work) |
| 1 | RAPTOR retriever comparison | — | ⚪ Not written | 8 h M4 after code |
| 3 | Long-form generation eval (MS-MARCO / QASPER) | — | ⚪ Not written | 6 h M4 after code |
| 2 | Frontier-scale ablation (Groq 70B + Kaggle quantized) | — | ⚪ Not written | 5 h after code; needs Groq API key (free) |
| 5 | Multi-seed variance on Table 2 (3 seeds) | Add `--seed` to existing runner | ⚪ Not written | 20–36 h depending on scope |
| 9 | Deployment case study (latency + qualitative figure) | — | ⚪ Not written | 2 h |
| 10 | HF Space + leaderboard (free tier) | — | ⚪ Not written | 1 day |

Dependency order: Wave 1 (items 4 ✅, 6, 7, 8 ✅) → Wave 2 (1, 3, 2) → Wave 3 (5, slowest) → Wave 4 (9, 10).

### Item 4 result (Mechanistic → hallucination classifier)

From `results/mech_classifier/summary.md` (n=20 pairs, 90 features, 5-fold CV):

| Model | AUC | F1 | ACC |
|---|---|---|---|
| logistic_l1 | **0.850 ± 0.300** | 0.800 | 0.850 |
| logistic_l2 | 0.800 ± 0.400 | 0.800 | 0.800 |
| gbdt | 0.575 ± 0.384 | 0.467 | 0.500 |
| single-feature baseline (`retrieved_mass`) | 0.800 | — | — |

Headline claim the paper can now make: *per-layer attention signals predict hallucination with 0.85 AUC and the single aggregate `mean_retrieved_mass` feature alone reaches 0.80 — the coherence paradox has a measurable mechanistic fingerprint.* Sample size is small (n=20 adversarial pairs) so the ±0.3 AUC std is a fair caveat; Item 6 expansion to 200 will shrink this.

## Phase 2.5 — NeurIPS gap-closure (3 items, added 2026-04-24)

After reviewing the professor's critique against Phase 1 coverage, three gaps remained only partially addressed. These are now scripted and target the specific reviewer challenges most likely to block acceptance.

| Gap | Reviewer question answered | Script / File | Status | ETA |
|---|---|---|---|---|
| 1 | "Is it coherence or just generic retrieval noise?" | `experiments/run_noise_injection_ablation.py` | ✅ Written, not yet run | 2 h (3 ds × 4 noise rates × 30 q, Mistral) |
| 2 | "Does the paradox depend on your specific prompt?" | `experiments/run_prompt_template_ablation.py` | ✅ Written, not yet run | 3 h (3 ds × 4 templates × 3 conditions × 30 q, Mistral) |
| 3 | "When does RAG actually help vs. zero-shot?" | `experiments/build_rag_vs_zeroshot_table.py` | ✅ **DONE** — `results/rag_vs_zeroshot/` populated | 0 (reshape only) |

### Gap 3 headline finding (already computed)

2×2 `results/rag_vs_zeroshot/table_2x2.csv`:

| Domain | Weak (MiniLM 22M) | Strong (gte-large 335M) |
|---|---|---|
| Open (SQuAD) | Δfaith = **+0.090** (RAG helps) | Δfaith = **+0.115** (RAG helps more) |
| Closed (PubMedQA) | Δfaith = **−0.021** (RAG hurts) | Δfaith = **−0.025** (RAG hurts) |

Strong narrative pick-up: RAG helps on open domain regardless of retriever strength, and *hurts* on biomedical closed-domain for both retrievers — general-purpose embedders retrieve biomedical evidence worse than Mistral's parametric knowledge. Paradox magnitude is also strength-dependent (weak=0.075, strong=−0.001 on SQuAD), which the §Discussion can use as a second-order claim. Caveat: each 2×2 cell is a single dataset (multi_retriever ablation only covers squad + pubmedqa).

### Gap 1 + 2 interpretation targets (for when runs land)

- **Noise ablation**: target `paradox_vs_noise_ratio ≥ 2` on all 3 datasets. If the coherence paradox drop is ≥ 2× the per-unit noise slope, coherence is a distinct failure mode.
- **Prompt ablation**: target `|delta_vs_ref| ≤ 0.03` across all 12 cells (3 datasets × 4 templates). If the paradox survives a CoT prompt, a concise prompt, and an expert-role prompt without moving more than ±0.03 faith points, it is not a prompt artifact.

## Run queue (as of 2026-04-24)

Green = done, Yellow = ready to launch, Grey = needs code.

### ✅ Completed on M4 / Kaggle (no action needed)

| # | Output | Evidence |
|---|---|---|
| P1 #5 adversarial (40 cases) | `data/adversarial/*.jsonl`, `results/adversarial/` | 10 cases per category |
| P1 #8 multi-retriever | `results/multi_retriever/paradox_by_embedder.csv` | 8 rows |
| P1 #4 head-to-head | `results/headtohead/summary.md` | Self-RAG partial (Kaggle OOM) |
| P1 #2 multi-dataset × 3 models × 5 ds | `results/multidataset/summary.csv` | 45 rows |
| P1 #1 mechanistic probe (Kaggle T4) | `results/mechanistic/*.csv,jsonl` | 5 files |
| P1 #7 stats | `results/stats/significance_tests.{json,csv}` | 6 tests |
| P1 #8pkg packager | `release/context_coherence_bench_v1/` | HF-loadable |
| P2 Item 4 mech classifier | `results/mech_classifier/summary.md` | AUC=0.85 |
| P2 Item 8 theory.tex | `ragpaper/sections/theory.tex` | Proposition + Theorem + Corollary |
| P2.5 Gap 3 RAG vs zero-shot | `results/rag_vs_zeroshot/table_2x2.csv` | Open: +0.09/+0.12, Closed: −0.02/−0.03 |

### 🟡 Ready to launch (code exists, not yet run)

| Item | Command | ETA (M4) | ETA (Kaggle T4) | Produces |
|---|---|---|---|---|
| P2 #6 adversarial expand 40→200 | `python3 experiments/generate_adversarial_cases.py --target_per_category 50` | 3–4 h | 2 h | `data/adversarial/*.jsonl` → ~200 cases |
| P2 #7 sub-chunk sweep | `python3 experiments/run_subchunk_sensitivity.py --datasets squad pubmedqa --model mistral --n_questions 30 --sub_chunks 128 256 512` | 3 h | 1.5 h | `results/subchunk_sensitivity/paradox_by_sub.csv` |
| P2.5 Gap 1 noise injection | `python3 experiments/run_noise_injection_ablation.py --datasets squad pubmedqa hotpotqa --n_questions 30 --top_k 3 --seed 42` | 2 h | 1 h | `results/noise_injection/coherence_vs_noise.csv` |
| P2.5 Gap 2 prompt ablation | `python3 experiments/run_prompt_template_ablation.py --datasets squad pubmedqa hotpotqa --templates strict cot concise expert --model mistral --n_questions 30` | 3 h | 1.5 h | `results/prompt_ablation/paradox_by_prompt.csv` |
| P2 #3 long-form eval | `python3 experiments/run_longform_eval.py --datasets qasper msmarco --model mistral --n_questions 20` | 2.5 h | 1.5 h | `results/longform/paradox_longform.csv` |
| P2 #2 frontier-scale (Groq) | `GROQ_API_KEY=... python3 experiments/run_frontier_scale.py --datasets squad pubmedqa --models llama-3.3-70b mixtral-8x7b --n_questions 30` | 45 min (API) | 45 min (API) | `results/frontier_scale/paradox_by_scale.csv` |
| P2 #1 RAPTOR head-to-head | `python3 experiments/run_raptor_ablation.py --datasets squad pubmedqa hotpotqa --model mistral --n_questions 30` | 90 min | 50 min | `results/raptor/raptor_vs_hcpc.csv` |
| P2 #5 multi-seed variance | `python3 experiments/run_multiseed_variance.py --datasets squad pubmedqa --model mistral --seeds 41 42 43 --n_questions 30` | 50 min | 30 min | `results/multiseed/variance_summary.csv` |
| P2 #9 deployment figure (no-run) | `python3 experiments/build_deployment_figure.py` | <1 min | — | `results/deployment_figure/latency_vs_faith.png` + `pareto_summary.csv` |

### Recommended **3-surface parallel split** (max completion in ~4 h wall-clock)

Three surfaces working simultaneously:

**Surface 1 — M4 overnight** (`tee -a run_queue.log` so you can read progress in any terminal):
```bash
cd /Users/saketmaganti/claudeprojs/rag-hallucination-detection && caffeinate -di bash -c '
python3 experiments/generate_adversarial_cases.py --target_per_category 50 ;
python3 experiments/run_subchunk_sensitivity.py --datasets squad pubmedqa --model mistral --n_questions 30 --sub_chunks 128 256 512 ;
echo "=== M4 DONE ==="
' 2>&1 | tee -a run_queue.log
```
Expected: 6–7 h. Produces `data/adversarial/` (200 cases) + `results/subchunk_sensitivity/`.

**Surface 2 — Kaggle session A** (Gaps 1 + 2, after Cells 1–3 from the standard Ollama+HF recipe):
```bash
!python3 experiments/run_noise_injection_ablation.py \
    --datasets squad pubmedqa hotpotqa --n_questions 30 --top_k 3 --seed 42 && \
 python3 experiments/run_prompt_template_ablation.py \
    --datasets squad pubmedqa hotpotqa \
    --templates strict cot concise expert \
    --model mistral --n_questions 30

!zip -r /kaggle/working/neurips_gaps.zip results/noise_injection results/prompt_ablation
from IPython.display import FileLink; FileLink('/kaggle/working/neurips_gaps.zip')
```
Expected: 2.5 h. Produces `results/noise_injection/` + `results/prompt_ablation/`.

**Surface 3 — Kaggle session B** (long-form eval, separate browser tab / second account if you have one):
```bash
!python3 experiments/run_longform_eval.py \
    --datasets qasper msmarco --model mistral --n_questions 20

!zip -r /kaggle/working/longform.zip results/longform
from IPython.display import FileLink; FileLink('/kaggle/working/longform.zip')
```
Expected: 1.5 h. Produces `results/longform/`.

**After all three finish**: `unzip` the Kaggle downloads into your M4's `results/` tree and commit.

### Fallback: single-surface M4 launcher (run if you don't want to bother with Kaggle)

```bash
cd /Users/saketmaganti/claudeprojs/rag-hallucination-detection && caffeinate -di bash -c '
python3 experiments/generate_adversarial_cases.py --target_per_category 50 ;
python3 experiments/run_subchunk_sensitivity.py --datasets squad pubmedqa --model mistral --n_questions 30 --sub_chunks 128 256 512 ;
python3 experiments/run_noise_injection_ablation.py --datasets squad pubmedqa hotpotqa --n_questions 30 --top_k 3 --seed 42 ;
python3 experiments/run_prompt_template_ablation.py --datasets squad pubmedqa hotpotqa --templates strict cot concise expert --model mistral --n_questions 30 ;
python3 experiments/run_longform_eval.py --datasets qasper msmarco --model mistral --n_questions 20 ;
echo "=== ALL M4 DONE ==="
' 2>&1 | tee -a run_queue.log
```
Expected: 13–14 h. `;` separators so one failure doesn't kill the rest. All scripts checkpointed.

### ⚪ Needs code to be written

_All P2 items + all operational helpers now have code._ Remaining work is compute-only.

- P2 #10 HF Space demo: `space/app.py` + `space/requirements.txt` + `space/README.md` written 2026-04-25. Three tabs (CCS calculator, paradox explorer, about), CPU-only, no Ollama dependency. Companion leaderboard at `leaderboard/app.py` (port 7861) with browse + submit (one-click GitHub-PR URL) + rules tabs.
- Deploy helper: `python3 scripts/prepare_hf_space.py` → `space_deploy/` with `.gitattributes` LFS config + `_PUSH_INSTRUCTIONS.md`. User still runs `git push hf main` manually.
- Remaining P2 items (#1 RAPTOR, #2 frontier-scale, #3 long-form, #5 multi-seed, #9 deployment) all graduated to 🟡 Ready on 2026-04-25.

### Paper changes 2026-04-25 (tightening pass)

| File | Change |
|---|---|
| `ragpaper/sections/robustness.tex` | NEW — 6 robustness checks (variance, RAPTOR, long-form, prompt, sub-chunk, noise). Honest reporting including the noise-equivalence soft-fail. |
| `ragpaper/sections/analysis.tex::Limitations` | +5 paragraphs: long-form scope, noise-vs-coherence, adversarial-129, frontier-scale. New labels `sec:limitations:longform`, `sec:limitations:noise`, `sec:limitations:adversarial`, `sec:limitations:frontier`. |
| `ragpaper/sections/abstract.tex` | Added multi-seed numbers (`0.069 ± 0.004`, signal/σ=17×), prompt-ANOVA (`p=0.83`), RAPTOR comparison, and explicit long-form / noise scope qualifications. |
| `ragpaper/main.tex` | Wires in `theory.tex` (after methodology) and `robustness.tex` (after headtohead). Adds `amsthm` package + `proposition`/`theorem`/`corollary`/`lemma`/`definition` envs needed by `theory.tex`. |
| `ragpaper/references.bib` | + `raptor2024` entry (Sarthi et al., ICLR 2024). |
| `experiments/run_noise_injection_ablation.py` (no edit; data fix) | The lookup against `multidataset/summary.csv` was returning None because the summary had only 21/45 rows. Rebuilt summary from `per_query.csv` (1350 rows → 45 rows summary), then re-ran `coherence_vs_noise_table` to populate the `paradox_drop` and `paradox_vs_noise_ratio` columns. |
| `results/multidataset/summary.csv` | Rebuilt from per-query data; now 45 rows (15 tuples × 3 conditions); `ccs` `-1` sentinel for non-v2 conditions cleaned to NaN. |

### Helper scripts added 2026-04-25 (ops + paper assembly)

| Script | Purpose |
|---|---|
| `experiments/build_final_tables.py` | One-shot Markdown tables for every paper section (12 tables + `ALL_TABLES.md` + `missing.md`). Smoke-tested: 6/12 render from current results, 6 stubbed as "not yet available" — safe to re-run mid-experiment. |
| `scripts/repair_multidataset_rows.py` | Audits every `*_per_query.csv` for row-count mismatches, dry-run by default, `--apply` backs up to `backups_YYYYMMDD/` and scrubs `completed_tuples.json`. Current state: all 15 tuples clean (90 rows each). |
| `scripts/build_benchmark_v2.py` | Builds `release/context_coherence_bench_v2/` from expanded adversarial + refreshed paradox CSV. Refuses to cut release if any category < `--min_cases_per_category`; sha256 per file in metadata.json. |
| `scripts/prepare_hf_space.py` | Stages `space_deploy/` with app + slim results + LFS `.gitattributes` for `git push hf main`. |
| `leaderboard/app.py` + `release/.../leaderboard.yaml` | Community leaderboard Gradio app with 9 seed entries (baseline/HCPC-v1/HCPC-v2/CRAG × squad/pubmedqa/hotpotqa). |
| `src/selfrag_wrapper.py` (edit) | New `load_in_8bit` / `load_in_4bit` params → fits T4 alongside Ollama. Use `--selfrag_8bit` on head-to-head CLI. |
| `src/dataset_loaders.py::load_msmarco` (edit) | Bounded streaming `.take(max(200, max_papers*8))`, per-row try/except, v2.1 → v1.1 fallback. Prevents the multi-minute hang hit in the long-form runner. |

### Decision matrix — what to prioritize

If you have **one overnight**: run the 🟡 launcher above. Delivers all 4 ready items (~12 h), closes all NeurIPS gaps for Mistral, and produces 4 new result tables.

If you have **one week of compute**: add **P2 #3 long-form** and **P2 #2 frontier-scale** code — those are the two items most likely to come up in NeurIPS reviews ("does this generalize beyond short-answer?" and "does this hold at scale?").

If you want to **freeze and write**: stop after the 🟡 launcher finishes. Phase 1 + Item 4 + Gap 3 + the 3 ready runs cover every reviewer critique on the professor's list. Remaining items become camera-ready additions.

## Key findings (quantitative, updated as results land)

### Deployment Pareto (2026-04-25, from existing per-query CSVs, no new runs)

Script: `experiments/build_deployment_figure.py`. Figure: `results/deployment_figure/latency_vs_faith.png`.

Per-dataset (latency↓, faith↑) Pareto frontier picks:

- **SQuAD** → `hcpc_v2` (3.15 s / faith=0.808 / halluc=0.013) and `crag` (1.16 s / faith=0.787).
- **PubMedQA** → `baseline` (10.94 s / faith=0.601), `hcpc_v2` (8.66 s / faith=0.590, lowest halluc=0.167), `crag` (2.50 s but halluc=0.333 — speed-risk tradeoff).
- **HotpotQA** → `hcpc_v1` (1.46 s / faith=0.650 / halluc=0.167) and `crag` (1.21 s / faith=0.631).

Net: HCPC-v2 is on the Pareto frontier on 2/3 datasets. CRAG wins on latency everywhere but loses faithfulness on PubMedQA (−0.015) and picks up 20 pp more hallucinations — so "cheap retrieval" is explicitly **not** free in high-coherence-demand domains. This is the figure reviewers should see before §Deployment.



### Multi-retriever ablation — headline reviewer response

`paradox_drop = faith_baseline − faith_v1`. Positive = paradox survives.

| Dataset | minilm (22M) | bge-large (335M) | e5-large (335M) | gte-large (335M) |
|---|---|---|---|---|
| PubMedQA | +0.024 ✓ | +0.033 ✓ | +0.053 ✓ | +0.005 ~0 |
| SQuAD | +0.075 ✓ | +0.032 ✓ | **−0.011 ✗** | **−0.001 ~0** |

**Reframed narrative** (now the paper's main theoretical contribution): The paradox is a property of **retrieval–generation alignment mismatch**, not embedder weakness. It appears when retrieval is hard (domain gap, multi-hop), disappears when retrieval is easy. `v2_recovery` is positive on 7/8 rows → HCPC-v2 is robust even when the baseline paradox is weak.

### Head-to-head summary

| Dataset | Winner on faith | Winner on halluc | Notable |
|---|---|---|---|
| SQuAD | HCPC-v2 (0.797) | HCPC-v2 (0.00) | Paper's claim holds |
| PubMedQA | baseline (0.609) | baseline / HCPC-v2 tie | CRAG hallucinates 33%, paradox visible |
| HotpotQA | HCPC-v1 (0.650) | baseline / HCPC-v1 tie | Multi-hop = where coherence matters most |

Self-RAG OOMed on Kaggle T4 for PubMedQA, partial on SQuAD (22/30), full HotpotQA. Llama-2-7B fp16 ≈ 14 GB vs 15 GB T4. Run Self-RAG alone later with 8-bit loading for NeurIPS camera-ready.

### Multidataset Table 2 (5 × 3, qwen2.5 slice shown)

From `results/multidataset/summary.csv`:
- **squad × qwen2.5**: baseline faith = 0.83, v1 = 0.69 (paradox = +0.14), v2 = 0.83 (recovery)
- **pubmedqa × qwen2.5**: baseline = 0.59, v1 = 0.61, v2 = 0.60 (weak paradox)
- **hotpotqa × qwen2.5**: baseline = 0.62, v1 = 0.60, v2 = 0.62
- **triviaqa × qwen2.5**: baseline = 0.66, v1 = 0.63, v2 = 0.66

HCPC-v2 refinement rate across datasets: 33–73%. CCS post-refinement: 0.41–0.57. Paradox magnitude ordering matches theory prediction: HotpotQA > SQuAD > PubMedQA (multi-hop > redundant > domain-gap).

### Mechanistic analysis

From `results/mechanistic/per_pair.csv` — 20 pairs (10 adversarial × 2 conditions each):
- `mean_retrieved_mass`: fragmented = ~0.16, coherent = higher → attention looks at retrieved context less when fragmented
- `mean_parametric_mass`: fragmented = ~0.84 → model falls back to parametric knowledge
- Per-layer entropy + retrieved-mass available at every layer (28 layers in per_layer CSV)

Item 4 (mech classifier) will turn this into a standalone hallucination-detection contribution.

## Hardware plan

**Mac M4 Air (16 GB unified)** — all Ollama-based runs. Fanless, so:
- Elevate on a stand, stay on charger, cool room for overnight
- Always wrap long runs in `caffeinate -di`
- If Ollama dies mid-run, restart in dedicated terminal with `ollama serve` and re-launch; runners have per-tuple checkpoints

**Kaggle T4 (free 30 h/week)** — mechanistic (done), Self-RAG inference only.

**Groq API (free)** — use for frontier-scale (Item 2). Sign up at console.groq.com. Llama-3.3-70B, Mixtral-8x7B, Gemma-2-9B all free tier.

**HF free account** — for Item 10 (dataset + CPU Space). No Pro needed.

## Code fixes applied across sessions

1. **Device auto-detect (commit `a5017656`)**: CUDA / MPS / CPU auto-pick in `src/rag_pipeline.py`, `src/evaluator.py`, `src/hallucination_detector.py`, `experiments/run_adversarial_coherence.py`. Needed because Kaggle PyTorch lacks MPS.

2. **NQ loader streaming + multi-annotator (commit `3d88d286`)**: `src/dataset_loaders.py::load_naturalqs` — two bugs fixed:
   - Non-streaming mode triggered full train-split generation (~55 GB, 287 parquets) and one shard threw `DatasetGenerationError`, silently returning empty items. Fixed with `streaming=True`.
   - NQ has 5 annotators per question; loader only checked `short_answers[0]` which is usually empty. Fixed by iterating all annotators.

3. **Checkpoint-vs-empty-CSV trap** (known, not patched): `experiments/run_multidataset_validation.py` catches per-query exceptions and still marks the tuple complete. If Ollama 404s for a whole model, the tuple's per-query CSV is just a header line but `completed_tuples.json` says `true`. Fix: re-run with `--force` scoped to the affected model.

4. **Adversarial generator dict-vs-list flatten (2026-04-24, uncommitted)**: `experiments/generate_adversarial_cases.py::main()` — `load_all_cases()` returns `Dict[category → List[AdversarialCase]]` but the helpers `_existing_case_ids` and `_existing_queries` expected a flat `List[AdversarialCase]`. Iterating the dict yielded string keys → `AttributeError: 'str' object has no attribute 'category'`. Fixed by flattening `cases_by_cat.values()` into a single list before the call. Verified against the existing 40 seed cases (counts = 10 per category, 30 unique queries). Commit this along with the new Phase 2.5 scripts.

5. **NQ multidataset row-count uneven (known, not patched)**: per-query CSVs have 90/217/149 rows for mistral/llama3/qwen2.5 — accumulation from earlier partial `--force` re-runs. The rebuilt `results/multidataset/per_query.csv` (1350 rows) is still valid but the summary aggregator will over-weight NQ. If this shows up as a problem in the paper's numbers, wipe `results/multidataset/naturalqs__{llama3,qwen2.5}_per_query.csv` + the matching `completed_tuples.json` keys and re-run only those two tuples.

## Kaggle notebook recipe (reusable)

```python
# Cell 1 — clone + pull + deps
!git clone https://github.com/Saket-Maganti/rag-hallucination-detection.git
%cd rag-hallucination-detection
!git pull origin main
!pip install -q -r requirements.txt && pip install -q tabulate

# Cell 2 — HF auth
from kaggle_secrets import UserSecretsClient
from huggingface_hub import login
import os
token = UserSecretsClient().get_secret("HF_TOKEN")
os.environ["HF_TOKEN"] = token; os.environ["HUGGING_FACE_HUB_TOKEN"] = token
login(token=token)

# Cell 3 — Ollama (only for head-to-head; NOT for mechanistic)
!apt-get update -qq && apt-get install -y -qq zstd
!curl -fsSL https://ollama.com/install.sh | sh
import subprocess, time
subprocess.Popen(["ollama", "serve"], stdout=open("/tmp/ollama.log","w"), stderr=subprocess.STDOUT)
time.sleep(8)
!ollama pull mistral

# Cell 4 — run
!PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 scripts/kaggle_gpu_runs.py --task <TASK> ...
```

**Required before starting:**
- Account phone-verified (enables Internet toggle)
- `HF_TOKEN` in Add-ons → Secrets
- Accelerator = GPU T4 x2, Internet = On, Persistence = Files

**Known pitfalls:**
- FileLink 404s intermittently → use right sidebar Output panel to download
- Ollama + Self-RAG together OOM-kills on T4; use `--skip_selfrag` or run separately
- Save Version → Save & Run All (Commit) at end to persist
- `huggingface-cli` may not be on PATH; use `huggingface_hub.login()` from Python
- **Mechanistic ran in 5 min on T4** (20 items only, not 3–5 h as runbook estimates) — verify outputs are non-constant to catch stub fallback: `results/mechanistic/entropy_by_layer.csv` should be ≥ 20 KB and vary across layers

## Files added in Phase 2 (across sessions)

| Path | Item | Purpose |
|---|---|---|
| `experiments/train_mech_classifier.py` | 4 | scikit-learn classifier on per-layer entropy + retrieved mass |
| `experiments/run_subchunk_sensitivity.py` | 7 | Sweep HCPC sub_chunk_size ∈ {128, 256, 512} |
| `experiments/generate_adversarial_cases.py` | 6 | LLM-generate new adversarial cases with validator |
| `ragpaper/sections/theory.tex` | 8 | Proposition + Theorem + Corollary for coherence paradox |
| `experiments/run_noise_injection_ablation.py` | Gap 1 | Coherence vs generic retrieval noise |
| `experiments/run_prompt_template_ablation.py` | Gap 2 | Paradox stability across 4 prompt templates |
| `experiments/build_rag_vs_zeroshot_table.py` | Gap 3 | Reshape existing results into 2×2 open/closed × weak/strong |
| `src/raptor_retriever.py` + `experiments/run_raptor_ablation.py` | 1 | 2-level RAPTOR vs HCPC head-to-head |
| `experiments/run_longform_eval.py` | 3 | Long-form generation (QASPER + MS-MARCO), ROUGE + faith |
| `src/groq_llm.py` + `experiments/run_frontier_scale.py` | 2 | 70B / Mixtral paradox at scale via Groq API |
| `experiments/run_multiseed_variance.py` | 5 | 3-seed variance bands on Table 2 (std-of-seed-means) |
| `experiments/build_deployment_figure.py` | 9 | Latency vs faith Pareto figure + deployment table (no-run) |
| `space/app.py` + `leaderboard/app.py` | 10 | HF Space demo + community leaderboard Gradio apps |
| `experiments/build_final_tables.py` | ops | Paper-ready Markdown tables (12 tables + ALL_TABLES.md) |
| `scripts/repair_multidataset_rows.py` | ops | Audit + repair per-query row-count drift |
| `scripts/build_benchmark_v2.py` | ops | Package `release/context_coherence_bench_v2/` bundle |
| `scripts/prepare_hf_space.py` | ops | Stage `space_deploy/` for `git push hf main` |

## Dependency order (Phase 2)

```
Wave 1 (M4, parallel, ~6 h):
  Item 4 (classifier, 30 s)
  Item 6 (adversarial expand, 3–4 h)
  Item 7 (sub-chunk sweep, 3 h)

Wave 2 (code + run, ~15 h total):
  Item 1 (RAPTOR)          → 8 h M4
  Item 3 (long-form)        → 6 h M4
  Item 2 (frontier Groq)   → 5 h (free API + Kaggle)

Wave 3 (overnight, ~20–36 h):
  Item 5 (3-seed variance, scope-dependent)

Wave 4 (paper + ops, no compute):
  Item 8  (theory.tex, drafted)
  Item 9  (deployment figure)
  Item 10 (HF Space + leaderboard)
```

## Files to check after each run

| After | Inspect | What good looks like |
|---|---|---|
| #5 adversarial | `results/adversarial/per_case.csv` | ≥100 rows after Item 6 expansion |
| #8 multi-retriever | `results/multi_retriever/paradox_by_embedder.csv` | 8 rows |
| #4 head-to-head | `results/headtohead/summary.csv` | 12–15 rows |
| #2 multi-dataset | `results/multidataset/summary.csv` | 45 rows (15 tuples × 3 conditions) |
| #1 mechanistic | `results/mechanistic/` | 5 files, entropy_by_layer ≥ 20 KB |
| Item 4 | `results/mech_classifier/summary.md` | AUC > 0.75 for the paradox to be a classifier-worthy signal |
| Item 7 | `results/subchunk_sensitivity/paradox_by_sub.csv` | paradox_drop stable within ±0.02 across sub_chunks |

## Final camera-ready deliverables target

- Expanded Table 2: 5 datasets × 3 generators ✅
- Multi-retriever row: 4 embedders ✅
- Self-RAG / CRAG comparison ✅ (Self-RAG partial)
- Mechanistic figure: Δ entropy + Δ retrieved mass per layer ✅
- Adversarial detection AUCs across 4 categories ✅ (Item 6 expands to N=200)
- Released benchmark bundle ✅ (Item 10 publishes to HF)
- **Reframed narrative**: paradox = property of alignment mismatch, not embedder weakness ✅
- **Theoretical section**: proposition + theorem + corollary (Item 8, drafted)
- **Standalone classifier contribution** from mechanistic signals (Item 4, script ready)
- **Scale validation**: 70B models show paradox magnitude vs 7B (Item 2, pending)
- **Variance bands**: 3-seed error bars on Table 2 (Item 5, pending)

## Path 2 — Groq frontier-scale runbook

**Goal:** answer "does the paradox persist at 70B?" — fills `table_6_frontier`, the only paper table still missing.

**Prerequisites (one-time):**
1. Free Groq API key — create at https://console.groq.com/keys, copy the `gsk_...` token.
2. `pip install groq` (already in `requirements.txt`).

**Local execution (~45 min wall-clock):**
```bash
export GROQ_API_KEY=gsk_...
python3 scripts/smoke_test_groq.py                    # 10 s sanity check
python3 experiments/run_frontier_scale.py \
    --datasets squad pubmedqa \
    --models llama-3.3-70b mixtral-8x7b \
    --n_questions 30
```
Outputs land in `results/frontier_scale/` (per_query.csv, summary.csv, paradox_by_scale.csv, summary.md). The paradox-by-scale CSV joins against `results/multidataset/summary.csv` to produce a `delta_vs_7b` column for the table.

**Kaggle execution (~45 min, runs while laptop sleeps):**
```bash
python3 scripts/kaggle_frontier_scale.py
# → notebooks/frontier_scale_kaggle.ipynb
# Upload to kaggle.com → New Notebook → File → Import → select the .ipynb
# Add Kaggle Secrets:
#   GROQ_API_KEY  (required)
#   GH_TOKEN      (optional — auto-pushes to kaggle-frontier-scale branch)
# Settings: Accelerator=None (Groq does GPU), Internet=ON, Persistence=Files
# Save & Run All
```

**Checkpointing:** `results/frontier_scale/completed_tuples.json` skips done (dataset, model) pairs, so re-running after a rate-limit pause picks up where it stopped. Pass `--force` to override.

## Path 3 — HF Space + leaderboard + GitHub release runbook

**Goal:** ship a public artifact bundle so reviewers (and grad students) can poke the system without cloning.

**Prerequisites (one-time):**
1. HuggingFace account (free — no Pro needed) at https://huggingface.co/join.
2. `huggingface-cli login` with a write token from https://huggingface.co/settings/tokens.
3. Create the Space at https://huggingface.co/new-space (SDK=Gradio, hardware=CPU basic, name e.g. `coherence-paradox-rag-demo`).
4. (Optional) `gh auth login` for automated GitHub Release upload.

**Stage + push the HF Space (~10 min):**
```bash
python3 scripts/prepare_hf_space.py --overwrite
cd space_deploy
git init -b main
git lfs install
git remote add hf git@hf.co:spaces/<your-user>/coherence-paradox-rag-demo
git add .
git commit -m "Initial demo push"
git push -u hf main
# Space auto-builds in ~3 min → live at https://huggingface.co/spaces/<your-user>/coherence-paradox-rag-demo
```

**Tag the v2 release:**
```bash
bash scripts/release_v2.sh --dry-run   # preview
bash scripts/release_v2.sh             # tag + push + (optionally) gh release create
```
The script validates working tree is clean, asserts required artifacts exist, builds `/tmp/coherence-paradox-v2.0.0.tar.gz` (paper LaTeX + slim results + space + leaderboard + scripts), creates annotated tag `v2.0.0`, pushes to origin, and — if `gh` CLI is present — creates the GitHub Release with the tarball attached.

**What ships in the v2 release tarball:**
- `ragpaper/` — full LaTeX sources + figures + bib
- `results/{multidataset,headtohead,robustness,deployment_figure,frontier_scale}/` — slim CSVs only (no chroma DBs)
- `space/`, `leaderboard/` — Gradio apps
- `scripts/`, `experiments/`, `src/` — reproducibility code
- `CLAUDE.md`, `README.md` — operator notes

## Phase 3 — pre-submission runbook

All scripts written, smoke-tested, and wired in. Run order with expected
times below.

### Zero-cost build pass (~2 min total, regenerates all Phase 3 artifacts)

```bash
# 1. Headline figure: frontier-scale paradox vs generator scale
python3 experiments/build_headline_figure.py
#   → ragpaper/figures/headline_frontier.{pdf,tex}     (~5 s)

# 2. CCS calibration: distribution split + quintile hallucination rate
python3 experiments/build_ccs_calibration.py
#   → ragpaper/figures/ccs_calibration.{pdf,tex}        (~3 s)
#   → results/ccs_calibration/quintile_table.csv

# 3. Qualitative example: best paradox triple from per_query data
python3 experiments/build_qualitative_example.py
#   → ragpaper/figures/qualitative_paradox.tex          (~2 s)
#   → results/qualitative/example_metadata.json

# 4. Lint pass: 0 errors required, warnings allowed
python3 scripts/lint_paper.py
#   → exit 0 means submission-clean references + citations  (~1 s)

# 5. Compile the paper (3 passes: pdflatex → bibtex → pdflatex × 2)
cd ragpaper && pdflatex -interaction=nonstopmode main && \
    bibtex main && \
    pdflatex -interaction=nonstopmode main && \
    pdflatex -interaction=nonstopmode main
#   → ragpaper/main.pdf  (~30 s on M4)
```

### Zenodo DOI publish (~5 min, needs free Zenodo token)

```bash
# One-time: create a Zenodo account at https://zenodo.org/signup
#           and a token at https://zenodo.org/account/settings/applications/tokens/new/
#           with scopes deposit:write + deposit:actions

export ZENODO_TOKEN=...

# Sandbox rehearsal first (won't mint a real DOI):
python3 scripts/upload_to_zenodo.py --sandbox --no-publish    # ~30 s
# → preview at https://sandbox.zenodo.org/deposit/<id>

# Production upload + publish (mints permanent DOI):
python3 scripts/upload_to_zenodo.py    # ~2 min including upload
# → prints DOI like 10.5281/zenodo.<id>; resolves at https://doi.org/<DOI>
# → paste DOI into submission/paper_metadata.yml::artifact_links.zenodo_doi
```

### OpenReview submission (~30 min, fully manual)

Follow `submission/openreview_checklist.md`. Key steps:

1. Set `\anonymoustrue` in `ragpaper/main.tex` (currently `false` for local).
2. Recompile.
3. Open the OpenReview submission form for NeurIPS 2026.
4. Paste fields from `submission/paper_metadata.yml`.
5. Upload `ragpaper/main.pdf` + the supplementary tarball
   `/tmp/coherence-paradox-v2.0.0.tar.gz` (created by `bash scripts/release_v2.sh`).
6. Save the OR submission ID, tag the commit.

### Anticipated total Phase 3 runway

| Phase 3 task | Time | Cost | Status |
|---|---|---|---|
| #1 Headline figure | 5 s | $0 | ✅ runs in 2-min build pass |
| #2 Author info | trivial | $0 | ✅ already in main.tex w/ anonymous toggle |
| #3 Lint pass | 1 s | $0 | ✅ 0 errors |
| #4 Zenodo upload | 5 min | $0 | ⏳ needs `ZENODO_TOKEN` |
| #5 OR submission | 30 min | $0 | ⏳ manual |
| #7 CCS calibration | 3 s | $0 | ✅ runs in 2-min build pass |
| #8 Qualitative example | 2 s | $0 | ✅ runs in 2-min build pass |
| **Total** | **~40 min** | **$0** | |

(#6 closed-model frontier intentionally skipped per user direction — no spend.)

### Files added in Phase 3

| File | Purpose |
|---|---|
| `experiments/build_headline_figure.py` | Frontier-scale paradox+recovery hero figure |
| `experiments/build_ccs_calibration.py` | CCS density split + quintile bars |
| `experiments/build_qualitative_example.py` | Picks strongest baseline→v1 paradox triple |
| `scripts/lint_paper.py` | Pre-submission lint: refs, cites, typos, placeholders |
| `scripts/upload_to_zenodo.py` | Two-step Zenodo deposit + publish; sandbox mode supported |
| `release/zenodo_metadata.json` | Title/abstract/keywords/license payload for Zenodo |
| `submission/openreview_checklist.md` | Step-by-step OR submission flow + rebuttal-quickref table |
| `submission/paper_metadata.yml` | Single-source metadata for the OR form |

## Common operational gotchas

1. **Empty 1-byte CSVs** → Ollama was down during that tuple. `completed_tuples.json` still says `true`. Wipe the affected keys, re-run with `--force`.
2. **Tabulate missing on fresh venv** → adversarial script fails at markdown write. `pip install tabulate`. Data CSVs already written before that point.
3. **Ollama port 11434 refused** → process died. `ollama serve` in dedicated terminal (not `&` inside script).
4. **Worktree vs main repo confusion** → always work in `/Users/saketmaganti/claudeprojs/rag-hallucination-detection`, not `.claude/worktrees/*`.
5. **Uneven per_query.csv row counts** after re-runs (seen in naturalqs: 93/212/131 instead of 90/90/90) → caused by accumulating across partial runs. Re-run cleanly with `--force` on a wiped `completed_tuples.json` entry for the affected tuples.
6. **`results/hcpc_v2/logs/` is gitignored** → use `git add -f` to force-add when the mechanistic Kaggle session needs them in the cloned repo.
