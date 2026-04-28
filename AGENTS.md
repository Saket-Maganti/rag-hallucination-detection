# AGENTS.md — project context for future sessions

Operational notes for picking up the RAG hallucination detection paper revision. Read this first.

## Project in one paragraph

Research artifact for the "Refinement Paradox" RAG paper. Core claim: better per-passage retrieval can *reduce* answer faithfulness by fragmenting context coherence. Introduces **CCS** (Context Coherence Score) as a generator-free retrieval-time diagnostic and **HCPC-v2** as a controlled intervention probe. Target venue: **NeurIPS** (following professor feedback on 2026-04-23).

Main repo: `/Users/saketmaganti/claudeprojs/rag-hallucination-detection`
Remote: `https://github.com/Saket-Maganti/rag-hallucination-detection`
Default branch: `main`

## Current status (2026-04-28) — senior-reviewer revision complete, paper revamp pending

### Read this first

The single source of truth for the senior-reviewer revision is now
[`docs/revision/README.md`](docs/revision/README.md). That book
consolidates the 10-weakness reviewer prompt, operating constraints,
per-fix deep dives with results, paper implications, what's remaining,
hardware recipes, and the file map. **Read it before doing anything
revision-adjacent.**

Sibling operational docs (moved from repo root into `docs/revision/`):

- [`docs/revision/status.md`](docs/revision/status.md) — per-fix scoreboard (was `REVISION_SUMMARY.md`)
- [`docs/revision/codex.md`](docs/revision/codex.md) — operational handoff (was `CODEX.md`)
- [`docs/revision/runbook.md`](docs/revision/runbook.md) — exact execution commands (was `REVISION_RUNBOOK.md`)
- [`docs/revision/snapshot.md`](docs/revision/snapshot.md) — reviewer-fix snapshot (was `fixes`)

### Recent commits on origin/main

- `7f1aca5e` — Consolidate revision docs into docs/revision/ and add comprehensive book
- `0fb01489` — Import Fix 2/3/4/5/9/11 outputs and stage Fix 6 Kaggle scaffolding

### Senior-reviewer revision (Fix 1 through Fix 11)

A pre-registered revision cycle scoping 11 fixes (W1–W10 plus an extra
W11 RAPTOR-row spinoff) into runnable scripts, paper sections, and
zero-dollar execution paths.

| Fix | Weakness | Status | Headline result |
| --- | --- | --- | --- |
| 1   | W1 causal vs correlational | **Done — null** | H1 unsupported. n=200 matched pairs, paired Wilcoxon p=0.628, Cohen's d_z=−0.017, 95% CI [−0.022, +0.017]. HIGH-CCS hallucinates more (16.5%) than LOW-CCS (9.0%). Causal/mechanistic language must be downgraded throughout. |
| 2   | W2 n=30 too small | **Done — paradox collapsed at scale** | n=500 × 5 seeds. Per-seed paradox (baseline−v1) magnitudes 0.006–0.020; only seed 44 (1/5) reaches p<0.05. v2−v1 recovery 3/5 seeds significant. Pooled bootstrap CI on paradox includes zero. |
| 3   | W3 single metric | **Done — DeBERTa is outlier** | n=7500 rescored across DeBERTa + roberta-large-mnli + RAGAS-judge (Mistral). Paradox magnitudes: 0.011 / 0.032 / 0.140. Pairwise Pearson r=0.18 (DeBERTa↔RAGAS), 0.26 (DeBERTa↔mnli), 0.67 (mnli↔RAGAS). 99-item human-eval template staged. |
| 4   | W4 τ leakage | **Done — flag SQuAD/PubMedQA/NaturalQS** | 5×5 τ matrix, n=7500. Diagonal-vs-offdiagonal recovery gap > 0.03 for SQuAD, PubMedQA, NaturalQS. TriviaQA/HotpotQA do not flag. |
| 5   | W5 noise slope | **Done — coherence carries signal beyond similarity** | n=1591. Random-noise faith slope −0.069; coherence-preserving slope −0.043 at matched rate. Coherence is not equivalent to similarity loss. |
| 6   | W6 baselines | **Done (no-Self-RAG path)** | n=1200 (CRAG / HCPC-v2 / RAPTOR-2L × SQuAD + HotpotQA × 200). HCPC-v2 does not dominate. SQuAD all three within 1.2 pp on faithfulness (0.698 / 0.708 / 0.710). HotpotQA: CRAG wins on faith (0.6427), halluc (10.5%), and latency (1940 ms mean). RAPTOR offline build 17–22× dense. Self-RAG smoke/full optional follow-up via `notebooks/revision_fix6_kaggle_t4x2_fresh.ipynb` stages `smoke_selfrag` / `selfrag`. |
| 7   | W7 70B reproduction | **Budget-blocked** | No genuinely free 70B-capable endpoint under zero-dollar mode. Disclose, do not fake. |
| 8   | W8 theory overclaim | **Paper-pending** | Mandatory rewrite because Fix 1 was null. Retitle §5 "Information-Theoretic Consistency Check"; rewrite Proposition 1 + Theorem 1 as sufficient/structural. |
| 9   | W9 confidence confounding | **Done — limited, suggestive only** | n=60, no-control Pearson r=0.36 p=0.005. Mean retrieval similarity + passage redundancy controls absent in input CSV; partial-correlation question unanswered. |
| 10  | W10 deployment scope | **Paper-pending** | Mandatory rewrite because Fix 1 + Fix 2 collapsed the headline. Abstract verb downgrade ("drives" → "predicts"), explicit scope to short-answer extractive QA. |
| 11  | W6 spinoff RAPTOR | **Done** | n=300 (3 datasets × 100). SQuAD 0.789 faith / 5% halluc / 1.19 s p50; PubMedQA 0.560 / 29% / 3.90 s; HotpotQA 0.617 / 21% / 1.97 s. RAPTOR tree-build 100–161 s. |

### What this means for the paper (post-revision)

The original v2.0 paper's three-pillar pitch — phenomenon, mechanism,
intervention — has serious problems after the revision:

1. **Phenomenon (Fix 2):** SQuAD/Mistral paradox magnitude collapsed
   from 0.069 (n=30) to per-seed magnitudes 0.006–0.020 (n=500 × 5),
   with only 1/5 seeds reaching p<0.05.
2. **Mechanism (Fix 1):** at fixed mean per-passage similarity, CCS
   does **not** predict faithfulness. The matched-similarity
   intervention designed to test the causal claim returned a null with
   the *wrong* sign on hallucination rate.
3. **Intervention (Fix 3):** HCPC-v2 recovery is metric-dependent.
   Under DeBERTa it is small (0.011); under RAGAS it is large (0.140);
   under mnli it is intermediate (0.032). The metrics correlate weakly
   (DeBERTa↔RAGAS Pearson r=0.18).

**Surviving positive results:**

- Fix 5: coherence-preserving uninformative noise produces a smaller
  faith drop (−0.043) than random off-topic noise (−0.069) at matched
  rate. Coherence carries signal independent of similarity.
- Frontier-scale (pre-revision): paradox magnitude reproduced at
  Llama-3.3-70B via Groq; not independently re-run because Fix 7 is
  budget-blocked.
- Multi-retriever ablation (pre-revision): paradox survives on
  PubMedQA with stronger embedders.
- Methodology contributions: pre-registered protocols, n=7500
  multi-metric triangulation, full RAPTOR cost analysis, released
  benchmark with DOI.

### NeurIPS submission strategy

**Recommended: NeurIPS Datasets & Benchmarks track (≈ 45–60% odds).**
The released benchmark (HF Dataset + Zenodo DOI), pre-registered
evaluations at scale, multi-metric triangulation, and the full Pareto
baseline comparison are all core D&B contributions. The benchmark
already exists at `saketmgnt/context-coherence-bench`.

NeurIPS main track (≈ 25–35% odds) is reachable but uphill. Reviewers
will see the Fix 1 null and Fix 2 collapse and ask "what's the
contribution?" The revamped pitch must lead with **Fix 3 metric
divergence** as the primary methodology contribution, not the
refinement-paradox phenomenon claim.

Safety nets: ACL / EMNLP / NAACL Findings track, TMLR (no novelty
bar; rigor bar — this work clears it comfortably), NeurIPS workshop
(80%+ odds at the right workshop).

**Mandatory remaining work before any NeurIPS submission:**

1. **Fix 8 paper integration**: theory.tex retitle + theorem rewrite.
2. **Fix 10 paper integration**: abstract scope rewrite + §8 long-form
   subsection promotion.
3. **Wire all completed-fix tables and figures into
   `ragpaper/main.tex`** via `\input{sections/revision/...}`. None are
   currently inputted. Fix 6 figure is at
   `ragpaper/figures/fix_06_pareto_faith_latency.pdf`.
4. **Cascading edits in `abstract.tex` / `paradox.tex` /
   `discussion.tex`** to match Fix 1 null + Fix 2 collapse + Fix 6
   no-clear-dominance.

**Optional follow-up (each adds 5–10% NeurIPS acceptance probability):**
- Fix 6 Self-RAG smoke + full paths via the existing Kaggle notebook
  to add a Self-RAG row to the head-to-head table.

**Strongly recommended (each adds 5–10% acceptance probability):**

6. Collect two-rater human-eval labels for the 99-item template at
   `data/revision/fix_03/human_eval_template.jsonl`. Yields Cohen's κ
   between raters and Spearman ρ vs each automated metric.
7. Regenerate `experiments/run_confidence_calibration.py` output with
   `mean_retrieval_similarity` + `passage_redundancy` columns, then
   re-run Fix 9 to actually answer W9 with full controls.
8. Optional second matched-similarity intervention on PubMedQA or with
   a second generator (Llama-3 / Qwen instead of Mistral) — even a
   second null gives the causal-claim disclosure two-domain coverage.

### Hard operational constraints (do not violate without explicit user approval)

- **Zero-dollar mode.** No paid APIs, no Groq quota burn, no
  Together.ai paid calls, no OpenAI / Anthropic judge calls, no paid
  Colab. Free Kaggle / Colab GPU, M4 Air local Ollama, free local
  Hugging Face models, and manual human annotation only.
- **Pre-registered statistics.** Sample sizes, seed counts, paired
  Wilcoxon tests, 10000-resample bootstrap CIs, Wilson CIs for binary
  rates, and effect sizes are recorded in each fix's pre-registration
  log *before* execution.
- **Honest reporting of null results.** Fix 1 null and Fix 2 collapse
  must propagate through paper language, not be papered over. If a
  later experiment changes a current finding, document the change
  explicitly with both numbers.
- **τ frozen** at the original SQuAD-50-held-out values everywhere
  *except* inside Fix 4 (the cross-dataset τ-generalization
  experiment).
- **Released artifacts intact.** `pip install context-coherence`,
  the LangChain integration, the HF dataset, the Zenodo DOI
  (`10.5281/zenodo.19757291`), and the HF Space all remain runnable
  and unchanged. The pip package version was bumped to `0.2.0` for
  the revision artifact line; the released API is unchanged.

### File map (revision-specific)

- **Per-fix scripts:** `experiments/fix_NN_*.py`
- **Per-fix pre-registration + post-run log:** `experiments/fix_NN_log.md`
- **Per-fix raw data:** `data/revision/fix_NN/`
- **Per-fix aggregated results:** `results/revision/fix_NN/`
- **Per-fix paper sections:** `ragpaper/sections/revision/fix_NN_*.tex`
- **Shared revision helpers:** `experiments/revision_utils.py`
- **Provider wrappers (zero-dollar-aware):** `src/{ragas_scorer,vectara_hem_scorer,together_llm,openai_llm,anthropic_llm,groq_llm}.py`
- **Fix 6 Kaggle scaffolding:** `notebooks/revision_fix6_kaggle_t4x2_fresh.ipynb`, `scripts/kaggle_fix6_t4x2.sh`, `scripts/kaggle_stream_fix6_t4x2.py`
- **Other Kaggle launchers:** `scripts/kaggle_fix{1,2,3_4,5_11}_t4x2.sh` and matching `scripts/kaggle_stream_fix*_t4x2.py`
- **Logs:** `logs/revision/*.log` (gitignored, produced by Kaggle/local runs)

---

## Historical status (pre-revision, 2026-04-25 end-of-day)

**Phase 1 (8-item) — COMPLETE.** All reviewer-facing experiments ran.
**Phase 2 (10-item) — 10/10 RUN.** Frontier-scale landed 2026-04-25: SQuAD/Llama-3.3-70B paradox = +0.100 (exact match to 7B), GPT-OSS-120B = +0.030. **Kills "small-model artifact" critique.** All 12/12 paper tables now filled.
**Phase 3 (pre-submission polish) — DONE except OpenReview click.** All scripts shipped, all figures built, lint clean, Zenodo DOI **`10.5281/zenodo.19757291`** minted and wired into paper + CITATION.bib + submission YAML. Only remaining Phase 3 item: the user manually clicking through the OpenReview submission form (script-friendly checklist at `submission/openreview_checklist.md`).
**Phase 7 (NeurIPS rigor upgrades) — CODE SHIPPED, RUNS PENDING.** All 9 scripts addressing the 6 reviewer-style critiques are written, syntax-clean, and on origin/main. User runs them on preferred hardware (local Mac / Groq / Kaggle). See "Phase 7" runbook below.
**Phase 5 (project hardening) — FULLY COMPLETE.** All TIER 1 + TIER 2 + TIER 2.5 done. Released artifacts: HF Dataset live at `saketmgnt/context-coherence-bench`, Zenodo DOI `10.5281/zenodo.19757291`, pip wheel built, Colab tutorial, LangChain integration, Docker + CI + Makefile + 30 tests passing.

**Phase 5 ablation findings (committed to results/):**
- Cross-encoder: paradox reranker-agnostic across MiniLM-L-6 / L-12 / BAAI bge (SQuAD: 0.071-0.113; PubMedQA: 0.046-0.052)
- Quantization: paradox quantization-agnostic on SQuAD (Q4/Q5/Q8: +0.095 / +0.088 / +0.093)
- **Temperature: paradox AMPLIFIES with T on SQuAD** (T=0: 0.065 → T=1.0: 0.115; PubMedQA stays ~0.04)
- **Confidence calibration: model self-confidence statistically correlated with CCS** (Pearson r=0.36 p=0.005, Spearman ρ=0.48 p=0.0001 pooled)

All 59 paradox-magnitude cells aggregated in `results/all_results_summary.{csv,md}`.
**Phase 4 (ChatGPT-review hardening) — DONE.** All 6 code files + 4 new figures + 5 paper sections shipped. **Top-k ablation finished 2026-04-25 16:00** (1 h 9 min wall-clock for 8 tuples × 4 conditions × 30 q = 960 generator calls). Headline: paradox magnitude **scales with k on SQuAD** (k=2: 0.063 → k=5: **0.124** → k=10: 0.117) — directional prediction of the coherence theory empirically confirmed. PubMedQA paradox vanishes at k=10 (−0.001), consistent with the domain-mismatched-encoder story (already-incoherent retrieval can't be made worse). CCS-only gate recovers most of HCPC-v2 at k≤3 but **HCPC-v2 strictly wins at k=5/10** (0.139 vs 0.042 recovery on SQuAD k=5), justifying the protected-neighbor rule as independent value beyond the gate decision. Awaiting user decision on whether to populate Table tab:topk in the paper via `python3 experiments/build_topk_table.py`. **PDF currently: 69 pages, 0 LaTeX warnings, 0 lint errors.**
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
- `AGENTS.md`, `README.md` — operator notes

## Phase 7 — NeurIPS rigor upgrades (post-review hardening)

External reviewer-style critique flagged 6 weaknesses for top-tier
acceptance. Phase 7 ships code for all 6 (no runs yet — user runs them
on their preferred hardware).

| # | Reviewer concern | Phase 7 response | Script |
|---|---|---|---|
| 7.1 | Causality not established (correlational only) | Synthetic causal experiment: matched-similarity sets that vary only in coherence | `experiments/run_synthetic_causal.py` |
| 7.2a | Missing baseline: MMR diversification | MMR head-to-head (5 conditions: baseline / v1 / v2 / mmr_05 / mmr_07) | `experiments/run_mmr_baseline.py` + `src/mmr_retriever.py` |
| 7.2b | Missing baseline: CRAG | CRAG head-to-head (4 conditions: baseline / v1 / v2 / crag) | `experiments/run_crag_baseline.py` (uses existing `src/crag_retriever.py`) |
| 7.3 | n=30 too small | Scale to n=300 on headline cells with 95% CIs + paired Wilcoxon | `experiments/run_scaled_headline.py` |
| 7.4 | CCS heuristic, why mean−std? | Validate CCS vs 5 alternative metrics (entropy, MMR diversity, graph connectivity, etc.) — pure analysis, no LLM calls | `experiments/run_ccs_alternatives.py` |
| 7.6 | NLI-only | Single-author validation: stratified 100-sample JSONL + agreement analyser (Cohen's kappa vs NLI) | `experiments/build_human_eval_samples.py` + `experiments/analyze_human_eval.py` |
| 7.7 | No failure analysis | Typology of HCPC-v2 misses (type-A/B/C, top-20 ranked by faith drop) | `experiments/build_failure_typology.py` |

### Phase 7 execution runbook (~3 days, $0)

```bash
# ─── Wave 1: instant analyses (~5 min, no LLM calls) ───────────────
python3 experiments/run_ccs_alternatives.py             # ~30 s, pure pandas
python3 experiments/build_failure_typology.py --top 20  # ~10 s, pure pandas
python3 experiments/build_human_eval_samples.py --n 100 # ~5 s, sampling only

# ─── Wave 2: medium experiments (~2-3 hr local Mac OR ~30 min Groq) ──
# Synthetic causal — the biggest reviewer fix
nohup python3 -u experiments/run_synthetic_causal.py \
    --datasets squad pubmedqa --n 100 --backend ollama --model mistral \
    > logs/synthetic_causal.log 2>&1 &
# OR via Groq (~10 min if budget available):
#   GROQ_API_KEY=... python3 experiments/run_synthetic_causal.py \
#       --backend groq --model llama-3.3-70b --n 100

# MMR head-to-head
nohup python3 -u experiments/run_mmr_baseline.py \
    --datasets squad pubmedqa --n 30 --backend ollama --model mistral \
    > logs/mmr_baseline.log 2>&1 &

# CRAG head-to-head
nohup python3 -u experiments/run_crag_baseline.py \
    --datasets squad pubmedqa --n 30 --backend ollama --model mistral \
    > logs/crag_baseline.log 2>&1 &

# Or run all three serially overnight on Mac:
#   for s in synthetic_causal mmr_baseline crag_baseline; do
#     python3 experiments/run_${s}.py --datasets squad pubmedqa
#   done

# ─── Wave 3: long compute (~6-8 hr local Mac, OVERNIGHT) ───────────
# Scale n=30 → n=300 — addresses the most damaging reviewer critique
nohup python3 -u experiments/run_scaled_headline.py \
    --datasets squad pubmedqa --n 300 --backend ollama --model mistral \
    > logs/scaled_headline.log 2>&1 &
# Survives terminal close (nohup); does NOT survive sleep — keep Mac awake
# (System Settings → Battery → Prevent automatic sleeping when display off)

# ─── Wave 4: human eval (single-rater, ~3 hr manual) ──────────────
# After Wave 1 has produced samples.jsonl:
# 1. Open results/human_eval/samples.csv in a spreadsheet
# 2. For each row, fill in human_label (faithful / hallucinated)
#    and human_faith (1 / 0)
# 3. Save as samples_rated.csv → convert back to JSONL or update samples.jsonl
# 4. Compute agreement:
python3 experiments/analyze_human_eval.py \
    --rated_jsonl results/human_eval/samples_rated.jsonl
# Output: results/human_eval/agreement.csv + agreement_report.md

# ─── Wave 5: aggregate + paper updates (~1 day, manual) ────────────
# After all Wave 1-4 complete:
python3 scripts/build_results_summary.py  # refresh results/all_results_summary.csv
# Update paper sections with new numbers/tables (~1 day writing)
```

### Kaggle fast-path (for 7.1 / 7.2a / 7.2b)

If Groq daily budget is fresh, run via Groq for 10× speedup:

```bash
export GROQ_API_KEY=<rotated-token>
# 7.1 synthetic causal — ~10 min on Groq
python3 experiments/run_synthetic_causal.py --backend groq \
    --model llama-3.3-70b --datasets squad pubmedqa --n 100

# 7.2a MMR baseline — ~10 min on Groq
python3 experiments/run_mmr_baseline.py --backend groq \
    --model llama-3.3-70b --datasets squad pubmedqa --n 30

# 7.2b CRAG baseline — ~10 min on Groq
python3 experiments/run_crag_baseline.py --backend groq \
    --model llama-3.3-70b --datasets squad pubmedqa --n 30

# 7.3 scaled headline — possible but burns daily budget; better local
```

For Kaggle: same `kaggle_frontier_scale.py` template adapted to call
each script. The Groq daily token budget is the bottleneck (100k/day);
each Phase-7 run uses ~30k-50k tokens, so 1-2 runs per day on free tier.

### Files added in Phase 7

| File | Purpose |
|---|---|
| `experiments/run_synthetic_causal.py` | Causal experiment (7.1) |
| `experiments/run_ccs_alternatives.py` | CCS metric validation (7.4) |
| `experiments/run_mmr_baseline.py` | MMR head-to-head (7.2a) |
| `experiments/run_crag_baseline.py` | CRAG head-to-head (7.2b) |
| `experiments/run_scaled_headline.py` | n=300 scaling (7.3) |
| `experiments/build_human_eval_samples.py` | Human-eval sampling (7.6) |
| `experiments/analyze_human_eval.py` | Human-eval agreement analyser (7.6) |
| `experiments/build_failure_typology.py` | HCPC-v2 failure typology (7.7) |
| `src/mmr_retriever.py` | MMRRetriever (LangChain-compatible) |

### Expected outcome after running all 7

| Reviewer concern | Before Phase 7 | After Phase 7 |
|---|---|---|
| Causality | correlational only | paired Wilcoxon p-value on matched-sim/varying-coherence pairs |
| Baselines | RAPTOR only | RAPTOR + MMR (×2 λ) + CRAG (+ optional Self-RAG) |
| n=30 too small | 30 per cell | 300 per headline cell with 95% CI |
| CCS heuristic | unjustified | ranked among 6 alternatives by Spearman ρ |
| Limited generalisation | already in limitations | strengthen with synthetic-result framing |
| NLI-only | NLI-only | NLI + single-rater agreement (Cohen's kappa) |

Reviewer score projection: **7.8/10 → 8.5-9.0/10** if Wave 1+2+3 land
with the predicted directions.

## Phase 5 — final-mile project hardening (post-paper)

The paper is done. Phase 5 turns the codebase into infrastructure
others can build on. Code-only — no paper edits unless you opt in.

**Status as of 2026-04-25 EOD: ALL TIER 1 + TIER 2 + TIER 2.5 CODE SHIPPED.**
30/30 tests pass, 0 lint errors. Awaiting user execution of runs.

### TIER 1 — engineering polish (DONE — code shipped)

| # | What | File | Status |
|---|---|---|---|
| 5.1 | Unit tests (30 tests) | `tests/test_ccs.py` (9), `test_lint_paper.py` (9), `test_builders.py` (5), `pip-package/tests/` (12) | ✅ all green |
| 5.2 | HuggingFace Datasets push | `scripts/push_to_hf_datasets.py` | ✅ ready (`--push` to upload) |
| 5.3 | Standalone pip package | `pip-package/` (pyproject.toml, src/context_coherence/, tests/) | ✅ 12/12 tests pass |
| 5.4 | Makefile + Dockerfile + .dockerignore | `Makefile`, `Dockerfile`, `.dockerignore` | ✅ `make help` works |
| 5.5 | GitHub Actions CI | `.github/workflows/ci.yml` (3 jobs: test, smoke, pip-package matrix) | ✅ pushes activate it |

### TIER 2 — new scientific findings (CODE READY, runs pending)

| # | Experiment | File | Run time |
|---|---|---|---|
| 5.6 | Quantization sensitivity (Q4 / Q5 / Q8) | `experiments/run_quantization_sensitivity.py` | ~1.5-2 hr Ollama |
| 5.7 | Temperature sensitivity (T = 0/0.3/0.7/1.0) | `experiments/run_temperature_sensitivity.py` (multi-backend: Ollama OR Groq) | ~1.8 hr Ollama, ~12 min Groq |
| 5.8 | Cross-encoder choice (3 rerankers) | `experiments/run_crossencoder_sensitivity.py` | ~1.4 hr |
| 5.9 | Confidence calibration (model self-confidence vs CCS) | `experiments/run_confidence_calibration.py` (multi-backend) | ~10 min Ollama, ~3 min Groq |

### TIER 2.5 — community/outreach (DONE)

| # | What | File | Status |
|---|---|---|---|
| 5.11 | Colab tutorial notebook | `scripts/build_colab_tutorial.py` → `notebooks/colab_tutorial.ipynb` | ✅ generated |
| 5.12 | LangChain integration (drop-in `CoherenceGatedRetriever`) + upstream PR notes | `integrations/langchain/coherence_gated_retriever.py` + `INTEGRATION_NOTES.md` | ✅ usable today, PR-draft ready |

### Phase 5 unified runbook

```bash
# ─── Wave A — instant validation (~30 s) ────────────────────────────
make tests                         # 30 tests, ~20 s
python3 scripts/lint_paper.py      # 0 errors required

# ─── Wave B — community shipping (~10 min, needs HF token) ──────────
export HF_TOKEN=...
python3 scripts/push_to_hf_datasets.py            # dry-run preview
python3 scripts/push_to_hf_datasets.py --push     # ~2 min upload
# → load_dataset("saketmgnt/context-coherence-bench") works globally

cd pip-package && python3 -m build               # builds wheel
# Upload to PyPI: python3 -m twine upload dist/*  (needs PYPI_TOKEN)

python3 scripts/build_colab_tutorial.py          # already done
# Push to main → Colab badge in README will resolve

# ─── Wave C — Docker reproducibility (~5 min build) ─────────────────
docker build -t coherence-paradox:v2.0.0 .
docker run --rm coherence-paradox:v2.0.0 make tests

# ─── Wave D — TIER 2 experiments (Ollama or Groq) ───────────────────
# Each is independent; all save to results/<name>/

# Local Ollama:
ollama serve &
ollama pull mistral:7b-instruct-q4_0
ollama pull mistral:7b-instruct-q5_K_M
ollama pull mistral:7b-instruct-q8_0
nohup python3 -u experiments/run_quantization_sensitivity.py \
    > logs/quantization.log 2>&1 &      # ~2 hr

nohup python3 -u experiments/run_temperature_sensitivity.py \
    --backend ollama --model mistral \
    > logs/temperature_ollama.log 2>&1 &  # ~1.8 hr

nohup python3 -u experiments/run_crossencoder_sensitivity.py \
    > logs/crossencoder.log 2>&1 &      # ~1.4 hr

nohup python3 -u experiments/run_confidence_calibration.py \
    --backend ollama --model mistral \
    > logs/confidence_ollama.log 2>&1 &  # ~10 min

# Or via Groq (free, ~10× faster, runs on Kaggle CPU):
export GROQ_API_KEY=...
python3 experiments/run_temperature_sensitivity.py \
    --backend groq --model llama-3.3-70b           # ~12 min
python3 experiments/run_confidence_calibration.py \
    --backend groq --model llama-3.3-70b           # ~3 min

# ─── Wave E — Kaggle parallelization (optional) ─────────────────────
# Adapt scripts/kaggle_frontier_scale.py for any TIER 2 experiment;
# the Groq-backed runners (temperature, confidence) work as-is on Kaggle
# free CPU since Groq does the GPU work via API.
```

### What's runnable on Kaggle (vs Mac-only)

| Runner | Mac (Ollama) | Kaggle (Groq) | Notes |
|---|---|---|---|
| 5.6 quantization | ✅ ~2 hr | ❌ | needs different Ollama tags; Groq doesn't expose quantization |
| 5.7 temperature | ✅ ~1.8 hr | ✅ ~12 min | use `--backend groq --model llama-3.3-70b` |
| 5.8 cross-encoder | ✅ ~1.4 hr | ⚠️ partial | reranker is local; only generation moves to Groq (not a big speedup) |
| 5.9 confidence | ✅ ~10 min | ✅ ~3 min | uses Groq for both answer + confidence calls |

### Files added in Phase 5

```
tests/
  conftest.py
  test_ccs.py
  test_lint_paper.py
  test_builders.py
pip-package/
  pyproject.toml
  README.md
  src/context_coherence/{__init__,core,gate}.py
  tests/test_core.py
scripts/
  push_to_hf_datasets.py
  build_colab_tutorial.py
experiments/
  run_quantization_sensitivity.py
  run_temperature_sensitivity.py
  run_crossencoder_sensitivity.py
  run_confidence_calibration.py
integrations/langchain/
  coherence_gated_retriever.py
  INTEGRATION_NOTES.md
notebooks/
  colab_tutorial.ipynb
.github/workflows/
  ci.yml
Makefile
Dockerfile
.dockerignore
```

## Phase 4 — top-tier polish (post-ChatGPT-review hardening)

External reviewer (ChatGPT) flagged 5 + 3 suggestions on 2026-04-25.
Triage and execution plan below. Key insight: 4 of the 8 are reframes
of work we already have; 1 is genuinely missing (top-k ablation); 3 are
cheap viz/code wins. Total budget: ~5 hr code + ~3 hr Ollama, $0 spend.

**Status as of 2026-04-25 EOD: ALL CODE SHIPPED AND PAPER UPDATED.**
Figures (4.2/4.3/4.7/4.8) live; paper edits (4.4/4.5/4.6) merged;
top-k ablation (4.1) running in background. PDF: 69 pages (was 64),
0 LaTeX warnings, 0 lint errors.

### TIER 1 — must do for top-tier acceptance

| # | Task | Status | File | Run time |
|---|---|---|---|---|
| 4.1 | Top-k ablation (k ∈ {2, 3, 5, 10}, SQuAD + PubMedQA) | 🔄 running PID 21170 | `experiments/run_topk_sensitivity.py` | ~2-3 hr Ollama |
| 4.2 | Disentanglement figure (fix similarity, vary CCS) | ✅ DONE | `experiments/build_disentanglement_figure.py` → `figures/disentanglement.{pdf,tex}` | ~5 s |
| 4.3 | Coherence heatmap (pairwise sim matrix, 2 examples) | ✅ DONE | `experiments/build_coherence_heatmap.py` → `figures/coherence_heatmap.{pdf,tex}` | ~30 s |
| 4.4 | CCS-as-policy reframe (abstract + new §, +bare CCS gate) | ✅ DONE | `analysis.tex::sec:ccs_policy` + `src/ccs_gate_retriever.py` | ~30 min |
| 4.5 | Stronger positioning sentence (abstract opener) | ✅ DONE | `abstract.tex` line 1: "RAG fails not because retrieval is inaccurate, but because retrieved evidence does not form a usable narrative for the generator." | 5 min |
| 4.6 | "When CCS fails" subsection (promote 3 negative results) | ✅ DONE | `analysis.tex::sec:ccs_fails` (3 regimes: redundant corpora, long-context single-doc, domain mismatch) | 30 min |

### TIER 2 — nice-to-have

| # | Task | Status | File | Run time |
|---|---|---|---|---|
| 4.7 | Expand qualitative builder to 5 case studies (1 per dataset) | ✅ DONE (4 cases — only 4 paradox triples in data) | extended `build_qualitative_example.py --top 5` → `figures/qualitative_cases.tex` | ~10 s |
| 4.8 | Embedding clustering plot (UMAP/t-SNE of retrieved chunks) | ✅ DONE | `experiments/build_embedding_clusters.py` → `figures/embedding_clusters.{pdf,tex}` | ~30 s |

### What ChatGPT got right vs already-addressed

| ChatGPT suggestion | Our status |
|---|---|
| #1 Disentanglement (fix sim, vary CCS) | Partially done via §10.6 noise-injection + multi-retriever; clean focused plot still missing → **4.2** |
| #2 Real-world case study | PubMedQA covers medical; expanding to 5 named cases via **4.7** |
| #3 CCS as decision policy | HCPC-v2 already IS this; just framed weakly → **4.4 reframe** |
| #4 Negative result | Have limitations but defensive; promote to dedicated subsection → **4.6** |
| #5 Top-k ablation | **Genuinely missing** — primary new experiment → **4.1** |
| A Visualization | Genuinely thin (no heatmaps); fixing via **4.3 + 4.8** |
| B Positioning sentence | Easy LaTeX win → **4.5** |
| C Theoretical framing | Already have `theory.tex` (proposition + theorem) — skip |

### What we deliberately skip (per ChatGPT's own warning)

- ❌ More datasets (no legal/financial — scope creep)
- ❌ More models (no extra scale runs — Phase 2 already settled this)
- ❌ Overcomplicating CCS (the simple def is the contribution)
- ❌ Rewriting from scratch (we're past that stage)

### Phase 4 + Phase 3 unified execution order (recommended)

This single recipe runs everything from Phase 3 (figures + Zenodo + OR
prep) **and** Phase 4 (top-k + disentanglement + heatmap + clusters +
case studies + paper reframes). Total clock time: ~3.5 hr (mostly
top-k waiting), $0.

```bash
# ─── Wave 0 ── prerequisites (one-time) ─────────────────────────────
ollama serve &                # in a dedicated terminal, leave running
ollama pull mistral           # only if not already present (~4.4 GB)

# ─── Wave 1 ── all Phase 3 + Phase 4 instant builders (~3 min) ──────
# These rebuild every figure from the per_query CSVs you already have.
python3 experiments/build_headline_figure.py             # P3 #1, ~5 s
python3 experiments/build_ccs_calibration.py             # P3 #7, ~3 s
python3 experiments/build_qualitative_example.py --top 5 # P3 #8 + P4 #4.7, ~5 s
python3 experiments/build_disentanglement_figure.py      # P4 #4.2, ~5 s
python3 experiments/build_coherence_heatmap.py           # P4 #4.3, ~30 s
python3 experiments/build_embedding_clusters.py          # P4 #4.8, ~30 s

# ─── Wave 2 ── lint + compile (~30 s) ────────────────────────────────
python3 scripts/lint_paper.py                 # MUST show "0 errors"
cd ragpaper && pdflatex -interaction=nonstopmode main && \
    bibtex main && \
    pdflatex -interaction=nonstopmode main && \
    pdflatex -interaction=nonstopmode main
cd ..
# → ragpaper/main.pdf, ~69 pages, 760 KB

# ─── Wave 3 ── top-k ablation (~2-3 hr Ollama, can run overnight) ────
# P4 #4.1 — the only Phase 4 task that requires real compute.
# Use nohup so it survives terminal close. Checkpoint-resumable.
nohup python3 -u experiments/run_topk_sensitivity.py \
    --k 2 3 5 10 \
    --datasets squad pubmedqa \
    --model mistral \
    --n_questions 30 \
    > logs/topk_sensitivity.log 2>&1 &
echo $!  # save the PID
tail -f logs/topk_sensitivity.log     # watch progress
# After it finishes, the paper auto-pulls the numbers from
# results/topk_sensitivity/paradox_by_k.csv into Table tab:topk.

# ─── Wave 4 ── re-compile after top-k lands (~30 s) ──────────────────
# Repeat Wave 2 to refresh main.pdf with the new top-k numbers.

# ─── Wave 5 ── Zenodo DOI (~5 min) ───────────────────────────────────
# Free token at https://zenodo.org/account/settings/applications/tokens/new/
export ZENODO_TOKEN=...
python3 scripts/upload_to_zenodo.py --sandbox --no-publish    # rehearsal
python3 scripts/upload_to_zenodo.py                           # real DOI
# Paste the printed DOI into submission/paper_metadata.yml::artifact_links.zenodo_doi

# ─── Wave 6 ── flip to anonymous + final compile + tag ──────────────
sed -i '' 's/\\anonymousfalse/\\anonymoustrue/' ragpaper/main.tex
cd ragpaper && pdflatex main && bibtex main && pdflatex main && pdflatex main
cd ..
bash scripts/release_v2.sh    # tags v2.0.0, builds tarball, optional gh release

# ─── Wave 7 ── OpenReview submission (~30 min, manual) ──────────────
# Follow submission/openreview_checklist.md
# Upload ragpaper/main.pdf + /tmp/coherence-paradox-v2.0.0.tar.gz
```

### What's running RIGHT NOW (background)

- Top-k ablation: `pgrep -f run_topk_sensitivity` → PID ~21170
- Logs: `logs/topk_sensitivity.log`
- Per-tuple CSVs land in `results/topk_sensitivity/squad__k2_per_query.csv` etc.
- After it finishes (~2-3 hr), re-run Wave 2 to refresh PDF.

### Files added in Phase 4

| File | Purpose |
|---|---|
| `src/ccs_gate_retriever.py` | Bare CCS-gate baseline (HCPC-v$1.5$): if CCS<τ refine all, else baseline |
| `experiments/run_topk_sensitivity.py` | Sweep k ∈ {2, 3, 5, 10} with baseline/v1/CCS-gate/v2 conditions |
| `experiments/build_disentanglement_figure.py` | Bucket queries by similarity quartile, plot CCS→faith within each |
| `experiments/build_coherence_heatmap.py` | Pairwise sim matrix for 1 coherent + 1 incoherent example query |
| `experiments/build_embedding_clusters.py` | t-SNE/UMAP of retrieved chunks per condition, colored by query |
| `ragpaper/figures/{disentanglement,coherence_heatmap,embedding_clusters,qualitative_cases}.{pdf,tex}` | Generated figures |
| `ragpaper/sections/analysis.tex` | New §`sec:ccs_policy` (CCS as policy) and §`sec:ccs_fails` (when CCS fails) |
| `ragpaper/sections/abstract.tex` | New opening sentence (ChatGPT's framing) |
| `ragpaper/sections/robustness.tex` | New §`sec:rob:topk` stub (table populates after top-k run) |
| `ragpaper/sections/appendix.tex` | New §`app:case_studies` referencing qualitative_cases figure |

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
4. **Worktree vs main repo confusion** → always work in `/Users/saketmaganti/claudeprojs/rag-hallucination-detection`, not `.Codex/worktrees/*`.
5. **Uneven per_query.csv row counts** after re-runs (seen in naturalqs: 93/212/131 instead of 90/90/90) → caused by accumulating across partial runs. Re-run cleanly with `--force` on a wiped `completed_tuples.json` entry for the affected tuples.
6. **`results/hcpc_v2/logs/` is gitignored** → use `git add -f` to force-add when the mechanistic Kaggle session needs them in the cloned repo.
