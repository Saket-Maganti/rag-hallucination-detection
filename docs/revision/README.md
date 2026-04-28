# NeurIPS Revision Book

This is the single source of truth for the senior-reviewer revision of
*"When Better Retrieval Hurts: Context Coherence Drives Faithfulness in
Retrieval-Augmented Generation."* It consolidates everything needed to
read, run, interpret, and ship the revision: the original reviewer
prompt, operational constraints, per-fix deep dives with results, paper
implications, what's remaining, and the file map.

If you only read one file in `docs/revision/`, read this one.

## Table of contents

1. [Why this revision exists](#1-why-this-revision-exists)
2. [Operating constraints](#2-operating-constraints)
3. [Scoreboard at a glance](#3-scoreboard-at-a-glance)
4. [Per-fix deep dives](#4-per-fix-deep-dives)
   - [Fix 1 — Causal coherence intervention](#fix-1--causal-coherence-intervention)
   - [Fix 2 — Scaled headline cell (n=500 × 5 seeds)](#fix-2--scaled-headline-cell-n500--5-seeds)
   - [Fix 3 — Multi-metric faithfulness](#fix-3--multi-metric-faithfulness)
   - [Fix 4 — Cross-dataset τ generalization](#fix-4--cross-dataset-τ-generalization)
   - [Fix 5 — Coherence-preserving noise](#fix-5--coherence-preserving-noise)
   - [Fix 6 — Baseline head-to-head + Pareto](#fix-6--baseline-head-to-head--pareto)
   - [Fix 7 — Together.ai 70B reproduction](#fix-7--togetherai-70b-reproduction)
   - [Fix 8 — Information-theory section reframe](#fix-8--information-theory-section-reframe)
   - [Fix 9 — Self-confidence partial correlations](#fix-9--self-confidence-partial-correlations)
   - [Fix 10 — Deployment scope](#fix-10--deployment-scope)
   - [Fix 11 — RAPTOR full table](#fix-11--raptor-full-table)
5. [What this means for the paper](#5-what-this-means-for-the-paper)
6. [What's remaining](#6-whats-remaining)
7. [Hardware + execution recipes](#7-hardware--execution-recipes)
8. [File map](#8-file-map)

---

## 1. Why this revision exists

The senior reviewer flagged **10 weaknesses (W1–W10)** on the original
NeurIPS submission. The revision plan scoped 11 fixes (W1–W10 plus an
extra W11 RAPTOR row pulled out of the original W6) into runnable
scripts, paper sections, and zero-dollar execution paths.

| #   | Reviewer concern (verbatim, condensed) | Maps to |
| --- | -------------------------------------- | ------- |
| W1  | Causal vs correlational: existing evidence does not establish that CCS *causes* unfaithfulness at fixed similarity | Fix 1 |
| W2  | n=30 headline cells too small to support the abstract's claims | Fix 2 |
| W3  | Faithfulness depends on a single NLI scorer (DeBERTa) | Fix 3 |
| W4  | τ tuned on SQuAD-50 held-out is leakage; cross-dataset generalization unknown | Fix 4 |
| W5  | §8 admits noise slope > paradox magnitude on SQuAD — the paradox might be "noisy retrieval" | Fix 5 |
| W6  | RAPTOR is a one-line mention; Self-RAG / CRAG missing or partial | Fix 6 + Fix 11 |
| W7  | Frontier 70B reproduction relies on a single Groq run | Fix 7 |
| W8  | §5 information-theory section overclaims (Proposition 1, Theorem 1) | Fix 8 |
| W9  | Self-confidence × CCS correlation could be mediated by similarity / redundancy | Fix 9 |
| W10 | Deployment claim ("optimizing the wrong quantity") is overscoped | Fix 10 |

The 11th fix (Fix 11 — RAPTOR full per-(dataset, metric) table) is a
P2 polish item that splits cleanly out of W6.

## 2. Operating constraints

These are hard constraints, not preferences. Any deviation must be
called out explicitly in the per-fix log.

- **Zero dollars.** No paid APIs, no Groq quota burn, no Together.ai
  paid calls, no OpenAI/Anthropic judge calls, no paid Colab.
- **Free compute only.** M4 Air local Ollama, free Kaggle/Colab GPU
  sessions, free local Hugging Face models, manual human annotation.
- **Fix 1 is the gating experiment.** No causal-language paper edits
  land before Fix 1 reads. (Fix 1 has read; H1 is null.)
- **Pre-registered statistics.** Sample sizes, seed counts, paired
  Wilcoxon tests, 10000-resample bootstrap CIs, Wilson CIs for binary
  rates, and effect sizes are recorded in each fix's pre-registration
  log *before* execution.
- **Honest reporting of null results.** If a hypothesis fails, the
  paper's language is downgraded throughout, not papered over.
- **No re-tuning τ.** Fix 4 is the only place τ varies; everywhere
  else uses the original SQuAD-50-held-out values
  (HCPC-v1: τ_sim=0.50, τ_ce=0.00; HCPC-v2: τ_sim=0.45, τ_ce=−0.20).
- **Fix 7 is budget-blocked** under zero-dollar mode unless free 70B
  compute appears. Document, do not fake.
- **Released artifacts intact.** `pip install context-coherence`,
  the LangChain integration, the HF dataset, and the Zenodo DOI
  (`10.5281/zenodo.19757291`) all remain runnable. Pip package version
  bumped to 0.2.0 for the revision artifact line.

## 3. Scoreboard at a glance

Status legend: **Done** = ran and committed; **Code** = scaffold ready,
execution pending; **Paper** = paper-only edit pending; **Blocked** =
infeasible under current constraints.

| Fix | Weakness | Status | One-line assessment |
| --- | --- | --- | --- |
| 1   | W1 (causal) | **Done, null** | H1 unsupported. Causal/mechanistic language must be downgraded throughout. |
| 2   | W2 (n=30 too small) | **Done** | Paradox collapsed at scale: 1/5 seeds significant on baseline−v1; 3/5 on v2−v1 recovery. |
| 3   | W3 (single metric) | **Done** | DeBERTa is the outlier; RAGAS shows much larger paradox; pairwise r=0.18–0.67. |
| 4   | W4 (τ leakage) | **Done** | SQuAD/PubMedQA/NaturalQS flag for §8; TriviaQA/HotpotQA do not. |
| 5   | W5 (noise slope) | **Done** | Coherence-preserving slope (−0.043) < random (−0.069) at matched rate. |
| 6   | W6 (baselines) | **Code, scaffolded** | Kaggle T4×2 notebook + runner ready; no-Self-RAG run pending. |
| 7   | W7 (70B repro) | **Blocked** | No genuinely free 70B endpoint; will disclose, not fake. |
| 8   | W8 (theory) | **Paper-pending** | Theory.tex retitle + theorem rewrite mandatory because Fix 1 was null. |
| 9   | W9 (confidence) | **Done, limited** | No-control r=0.36 p=0.005; controls absent in input; report as suggestive. |
| 10  | W10 (deployment scope) | **Paper-pending** | Abstract scope rewrite mandatory because Fix 1 + Fix 2 collapsed the paradox. |
| 11  | W6 spinoff (RAPTOR) | **Done** | Per-dataset RAPTOR table (faith, halluc, p50/p99, idx cost, size). |

## 4. Per-fix deep dives

Each fix below has the same four-part structure: pre-registration,
result, interpretation, paper implication.

---

### Fix 1 — Causal coherence intervention

**Weakness W1.** Tests whether CCS *causes* unfaithfulness at fixed
mean per-passage similarity.

**Pre-registration:** [`experiments/fix_01_log.md`](../../experiments/fix_01_log.md)
**Code:** [`experiments/fix_01_causal_matched_pairs.py`](../../experiments/fix_01_causal_matched_pairs.py)
**Data:** `data/revision/fix_01/`
**Results:** `results/revision/fix_01/`
**Paper section:** `ragpaper/sections/revision/fix_01_causal_intervention.tex`

**Hypothesis (frozen):** $\mathbb{E}[\text{faith} | \text{HIGH-CCS}] - \mathbb{E}[\text{faith} | \text{LOW-CCS}] > 0$ at fixed mean query similarity (within ±0.02). H1 supported iff paired Wilcoxon p<0.05 AND Cohen's $d_z$>0.2 AND bootstrap 95% CI excludes 0.

**Sample:** 200 matched query pairs from SQuAD; seed 42; Mistral-7B; DeBERTa-v3 NLI.

**Construction diagnostics (`results/revision/fix_01/match_diagnostics.csv`):**

| pairs | mean abs sim gap | max abs sim gap | mean CCS gap | min CCS gap | max overlap |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 200 | 0.006351 | 0.018512 | 0.532634 | 0.264139 | 1 |

**Result (`results/revision/fix_01/paired_wilcoxon.csv`):**

| n_pairs | high faith | low faith | high−low | Wilcoxon p (greater) | Cohen's $d_z$ | bootstrap 95% CI | H1 |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 200 | 0.6362 | 0.6386 | **−0.0024** | **0.628** | **−0.017** | **[−0.022, +0.017]** | **False** |

Hallucination rate: HIGH-CCS = 16.5%, LOW-CCS = 9.0% (HIGH actually
hallucinates *more* — exactly the wrong direction).

**Interpretation:** the matched-similarity construction succeeded
(mean similarity gap 0.006, CCS gap 0.53), but the intervention does
not show that higher CCS causally improves faithfulness at fixed
similarity. The mechanism the paper attributed to coherence is not
visible in this controlled experiment.

**Paper implication:** the central claim of the v2.0 abstract is not
causally validated. Per the pre-registration:

- "Drives" → "predicts" in the abstract.
- Theorem 1 in `theory.tex` reframed as a sufficient condition.
- Proposition 1 retained but flagged as not empirically validated as
  causal by Fix 1.
- §5 retitled "Information-Theoretic Consistency Check" (also Fix 8).
- §`sec:causal_intervention` added with the null result, the paired-
  difference plot (`ragpaper/figures/fix_01_paired_diff.pdf`), and a
  short concession paragraph.

---

### Fix 2 — Scaled headline cell (n=500 × 5 seeds)

**Weakness W2.** Tests whether the SQuAD/Mistral paradox survives at
10× the original sample size and 5 seeds.

**Pre-registration:** [`experiments/fix_02_log.md`](../../experiments/fix_02_log.md)
**Code:** [`experiments/fix_02_scaled_headline_n500.py`](../../experiments/fix_02_scaled_headline_n500.py)
**Data:** `data/revision/fix_02/per_query.csv` (7500 evaluations)
**Results:** `results/revision/fix_02/`
**Paper section:** `ragpaper/sections/revision/fix_02_scaled_headline.tex`

**Sample:** 5 seeds (41–45) × 3 conditions (baseline / HCPC-v1 / HCPC-v2)
× 500 SQuAD queries. Mistral-7B via Ollama. τ frozen at paper values.

**Pooled headline (`headline_table.csv`, n=2500/condition):**

| condition | faith | faith CI95 | halluc | halluc Wilson95 | retr sim | refine rate |
| --- | ---: | --- | ---: | --- | ---: | ---: |
| baseline | 0.6609 | [0.6549, 0.6668] | 0.146 | [0.133, 0.160] | 0.5322 | 0.000 |
| hcpc_v1  | 0.6503 | [0.6447, 0.6560] | 0.147 | [0.134, 0.162] | 0.5689 | 0.000 |
| hcpc_v2  | 0.6612 | [0.6553, 0.6671] | 0.143 | [0.130, 0.157] | 0.5322 | 0.705 |

**Per-seed paired contrasts (`paired_contrasts.csv`):**

| seed | baseline−v1 | p | $d_z$ | v2−v1 | p | $d_z$ |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 41 | 0.0107 | 0.083 | 0.072 | 0.0132 | 0.025 | 0.088 |
| 42 | 0.0063 | 0.234 | 0.043 | 0.0030 | 0.504 | 0.021 |
| 43 | 0.0078 | 0.174 | 0.055 | 0.0059 | 0.215 | 0.042 |
| 44 | 0.0204 | **0.008** | 0.134 | 0.0191 | **0.008** | 0.126 |
| 45 | 0.0073 | 0.203 | 0.048 | 0.0125 | **0.036** | 0.081 |

**Interpretation:** the SQuAD/Mistral paradox **collapsed at scale**.
Per-seed magnitudes are 0.006–0.020; the pre-revision n=30 estimate of
0.069 was inflated by sample size. Only seed 44 (1/5) yields p<0.05 on
the paradox contrast. v2−v1 recovery is more robust (3/5 seeds
significant). The pooled bootstrap CI on the paradox contains zero.

**Paper implication:** combined with Fix 1, the paper's central
empirical claim is much weaker than v2.0 implied. The n=30 estimate
must be retired and replaced with this n=500×5 result. The abstract's
"drives" verb is no longer defensible (also forced by Fix 1).

---

### Fix 3 — Multi-metric faithfulness

**Weakness W3.** Re-scores the Fix 2 cell with two additional
faithfulness metrics to test whether the DeBERTa-only headline depends
on the choice of scorer.

**Pre-registration:** [`experiments/fix_03_log.md`](../../experiments/fix_03_log.md)
**Code:** [`experiments/fix_03_multimetric_faithfulness.py`](../../experiments/fix_03_multimetric_faithfulness.py)
**Helpers:** [`src/ragas_scorer.py`](../../src/ragas_scorer.py), [`src/vectara_hem_scorer.py`](../../src/vectara_hem_scorer.py)
**Data:** `data/revision/fix_03/per_query.csv` (7500 rows scored)
**Results:** `results/revision/fix_03/`
**Paper section:** `ragpaper/sections/revision/fix_03_multimetric.tex`

Vectara HEM required custom remote code; the second NLI fell back to
`roberta-large-mnli` per the scorer's documented fallback path. The
RAGAS judge runs locally on Mistral-7B (zero-dollar substitute for
GPT-4o-mini).

**Per-condition means (`table1_multimetric.csv`):**

| condition | n | DeBERTa | second NLI (mnli) | RAGAS judge |
| --- | ---: | ---: | ---: | ---: |
| baseline | 2500 | 0.6609 | 0.3501 | 0.7296 |
| hcpc_v1  | 2500 | 0.6503 | 0.3184 | 0.5904 |
| hcpc_v2  | 2500 | 0.6612 | 0.3509 | 0.7279 |

Paradox magnitudes (baseline − v1):
- DeBERTa: **0.011**
- second NLI: **0.032**
- RAGAS: **0.140**

**Pairwise correlations (`metric_correlations.csv`, n=7500):**

| metric A | metric B | Pearson r | Spearman ρ |
| --- | --- | ---: | ---: |
| DeBERTa | second NLI | 0.259 | 0.265 |
| DeBERTa | RAGAS | **0.182** | 0.212 |
| second NLI | RAGAS | 0.674 | 0.651 |

**Interpretation:** DeBERTa is the **outlier metric**. RAGAS and second
NLI agree at r=0.67; both correlate weakly with DeBERTa (r=0.18–0.26).
The paradox is metric-dependent: under DeBERTa it is small (0.011);
under RAGAS it is large (0.140); under mnli it is intermediate
(0.032). The Fix 2 collapse claim therefore deserves a caveat — it
holds under DeBERTa, weakens but does not vanish under mnli, and
*reverses sign-of-magnitude* under RAGAS.

A 99-item two-rater human-eval template is staged at
`data/revision/fix_03/human_eval_template.jsonl`. Filling it with two
annotators yields Cohen's κ between raters and Spearman ρ vs each
automated metric.

**Paper implication:** the paper must report all three metrics in
parallel rather than collapse to DeBERTa. The honest framing is "the
paradox is visible under metric A, weakens under metric B, and shifts
under metric C" rather than "the paradox is visible."

---

### Fix 4 — Cross-dataset τ generalization

**Weakness W4.** τ for HCPC was tuned on SQuAD's 50-question held-out
set. Fix 4 builds the 5×5 tune-on-X / eval-on-Y recovery matrix to
test whether the SQuAD-tuned τ transfers.

**Pre-registration:** [`experiments/fix_04_log.md`](../../experiments/fix_04_log.md)
**Code:** [`experiments/fix_04_tau_generalization.py`](../../experiments/fix_04_tau_generalization.py)
**Data:** `data/revision/fix_04/per_query.csv` (7500 evaluations)
**Results:** `results/revision/fix_04/`
**Paper section:** `ragpaper/sections/revision/fix_04_tau_generalization.tex`

**Sample:** 5 datasets (SQuAD, PubMedQA, HotpotQA, NaturalQS, TriviaQA)
× 5 τ values (0.30, 0.40, 0.50, 0.60, 0.70) × 300 queries.

Recovery is defined as
`(faith_ccs_gate − faith_hcpc_v1) / (faith_baseline − faith_hcpc_v1)`.
A "must-flag" dataset has `diag_recovery − offdiag_mean_recovery > 0.03`
(in-distribution τ tuning materially beats out-of-distribution).

**Honest-flag table (`generalization_flags.csv`):**

| tune dataset | diag recovery | offdiag mean recovery | diag − offdiag | flag in §8 |
| --- | ---: | ---: | ---: | :---: |
| pubmedqa  |  1.452 |  0.451 |  1.002 | **True** |
| naturalqs |  1.401 |  0.211 |  1.191 | **True** |
| squad     |  0.823 | −0.141 |  0.963 | **True** |
| triviaqa  |  0.464 |  0.698 | −0.234 | False |
| hotpotqa  | −0.112 |  0.093 | −0.205 | False |

**Interpretation:** τ tuned on SQuAD, PubMedQA, or NaturalQS does
**not** transfer cleanly to the other four datasets. The original
SQuAD-50-held-out τ that anchored the paper's HCPC numbers is one of
the leaky cases. TriviaQA and HotpotQA τ choices generalize.

**Paper implication:** §8 must disclose the diagonal-vs-off-diagonal
gap. The paper should report **off-diagonal recovery** as the primary
number rather than the SQuAD-50-held-out diagonal that originally
tuned τ.

---

### Fix 5 — Coherence-preserving noise

**Weakness W5.** §8 of the v2.0 paper admits that the SQuAD random-
noise slope exceeds the paradox magnitude; a reviewer could conclude
"the paradox is just retrieval noise." Fix 5 separates similarity
loss from coherence loss.

**Pre-registration:** [`experiments/fix_05_log.md`](../../experiments/fix_05_log.md)
**Code:** [`experiments/fix_05_coherence_preserving_noise.py`](../../experiments/fix_05_coherence_preserving_noise.py)
**Data:** `data/revision/fix_05/per_query.csv` (1591 evaluations)
**Results:** `results/revision/fix_05/`
**Paper section:** `ragpaper/sections/revision/fix_05_noise_slope.tex`

**Sample:** 200 SQuAD queries; baseline + coherence-preserving noise
(same-topic answer-absent passages from the query's top-20 pool, 3
noise levels) + random off-topic noise (3 levels) + HCPC-v1 refinement.

**Slope response (`slope_response.csv`):**

| condition | faith slope per noise rate | sim slope per noise rate | faith drop at full noise |
| --- | ---: | ---: | ---: |
| random_noise                 | **−0.069** | **−0.481** | 0.052 |
| coherent_uninformative_noise | **−0.043** | **−0.113** | 0.042 |
| hcpc_v1_refinement           | n/a | n/a | 0.001 |

**Interpretation:** at matched noise rate the coherence-preserving
condition degrades faithfulness more slowly (−0.043) than random
off-topic noise (−0.069). Crucially, random noise also drops
similarity dramatically (−0.481) while coherence-preserving noise
keeps similarity stable (−0.113). The §8 admission therefore deserves
a tightening: random off-topic injection removes both similarity and
coherence simultaneously, so the original noise-slope number does not
disentangle them. The coherence-preserving condition isolates
coherence and shows a smaller, but non-zero, faith drop.

**Paper implication:** §7.6 robustness section can be strengthened.
The paradox is not equivalent to "random noise hurt faithfulness" —
coherence carries some of the signal even when similarity is held
near-fixed. This is a partial recovery of the coherence story even
under Fix 1 + Fix 2 nulls.

---

### Fix 6 — Baseline head-to-head + Pareto

**Weakness W6.** RAPTOR was a one-line mention; Self-RAG/CRAG were
missing or partial. Fix 6 builds a full head-to-head with latency,
hallucination, and a Pareto plot.

**Pre-registration:** [`experiments/fix_06_log.md`](../../experiments/fix_06_log.md)
**Code:** [`experiments/fix_06_baseline_h2h_pareto.py`](../../experiments/fix_06_baseline_h2h_pareto.py) (now writes periodic partial CSVs via `--save_every`)
**Kaggle scaffolding:** [`notebooks/revision_fix6_kaggle_t4x2_fresh.ipynb`](../../notebooks/revision_fix6_kaggle_t4x2_fresh.ipynb), [`scripts/kaggle_fix6_t4x2.sh`](../../scripts/kaggle_fix6_t4x2.sh), [`scripts/kaggle_stream_fix6_t4x2.py`](../../scripts/kaggle_stream_fix6_t4x2.py)
**Paper section:** `ragpaper/sections/revision/fix_06_baselines.tex`

**Status:** **CODE ONLY** — execution pending.

**Plan:** run the no-Self-RAG path first on Kaggle T4×2
(~2–4 h), download the package, then optionally attempt the Self-RAG
smoke test, then optionally the full Self-RAG run if smoke passes.

**Conditions to compare:** baseline, HCPC-v1, HCPC-v2, CRAG, RAPTOR,
(optional) Self-RAG.
**Datasets:** SQuAD, HotpotQA.
**Sample:** n=200 per (dataset, condition).
**Reports:** faithfulness, hallucination rate, p50/p99 latency,
indexing cost, Pareto frontier (faith vs latency).

---

### Fix 7 — Together.ai 70B reproduction

**Weakness W7.** The frontier 70B paradox magnitude in v2.0 came from
a single Groq run, leaving the paper exposed to "you cherry-picked a
backend" in rebuttal.

**Pre-registration:** [`experiments/fix_07_log.md`](../../experiments/fix_07_log.md)
**Code:** [`experiments/fix_07_together_70b_reproduction.py`](../../experiments/fix_07_together_70b_reproduction.py)
**Wrapper:** [`src/together_llm.py`](../../src/together_llm.py)
**Paper section:** `ragpaper/sections/revision/fix_07_together.tex`

**Status:** **BUDGET-BLOCKED.** A 70B model does not fit on free T4 /
P100 / L4 / M4 Air, and Together.ai is paid. Until free 70B-capable
compute appears, this fix cannot be executed under the zero-dollar
constraint.

**Disclosure for the paper:** the v2.0 70B Groq number stands as a
single-backend observation. The revision discloses the missing
independent reproduction in §8 limitations and notes the budget
constraint explicitly.

---

### Fix 8 — Information-theory section reframe

**Weakness W8.** §5 of v2.0 made claims about Proposition 1 and
Theorem 1 that implied predictive power Fix 1 has now disconfirmed.

**Pre-registration:** [`experiments/fix_08_log.md`](../../experiments/fix_08_log.md)
**Code:** none (paper-only)
**Paper section:** `ragpaper/sections/revision/fix_08_theory_reframe.tex`

**Status:** **PAPER-PENDING.** Mandatory because Fix 1 was null.

**Plan:**
- Rename §5 from "An Information-Theoretic Account" to "Information-
  Theoretic Consistency Check."
- Rewrite Proposition 1 to explicitly state what it does **not**
  prove (prevalence, magnitude). Frame Theorem 1 the same way.
- Move any predictive language to clearly-flagged conjectures.

---

### Fix 9 — Self-confidence partial correlations

**Weakness W9.** v2.0 claimed CCS predicts model self-confidence; this
might be mediated by per-passage similarity or pairwise redundancy.

**Pre-registration:** [`experiments/fix_09_log.md`](../../experiments/fix_09_log.md)
**Code:** [`experiments/fix_09_partial_correlations.py`](../../experiments/fix_09_partial_correlations.py)
**Data:** `data/revision/fix_09/input_copy.csv`
**Results:** `results/revision/fix_09/partial_correlations.csv`
**Paper section:** `ragpaper/sections/revision/fix_09_partial_confidence.tex`

**Status:** **DONE, LIMITED.**

The available input
`results/confidence_calibration/per_query.csv` does **not** contain
`mean_retrieval_similarity` or `passage_redundancy`. The script
gracefully falls back to the no-control association:

| n | controls | partial Pearson r | partial Pearson p | partial Spearman ρ | partial Spearman p | survives |
| ---: | --- | ---: | ---: | ---: | ---: | --- |
| 60 | none | 0.360 | 0.005 | 0.481 | 0.0001 | True |

**Interpretation:** the no-control association exists. The partial-
correlation question (controlling for similarity and redundancy) is
**not** answered by this run because the controls are missing from the
input. Treat the confidence-calibration finding as **suggestive only**
unless a richer confidence-calibration CSV is regenerated.

**Paper implication:** §confidence-calibration must be downgraded to
"suggestive correlation" and the missing-control limitation noted
inline.

**Optional follow-up:** add `mean_retrieval_similarity` and
`passage_redundancy` columns to `experiments/run_confidence_calibration.py`
output, re-run on M4 (the original cell is small, ~10 min), then re-run
Fix 9 with full controls.

---

### Fix 10 — Deployment scope

**Weakness W10.** v2.0 abstract over-claimed by saying "RAG evaluation
has been optimizing the wrong quantity" — but the long-form non-result
is buried, and Fix 1 + Fix 2 have weakened the claim further.

**Pre-registration:** [`experiments/fix_10_log.md`](../../experiments/fix_10_log.md)
**Code:** none (paper-only)
**Paper section:** `ragpaper/sections/revision/fix_10_scope_deployment.tex`

**Status:** **PAPER-PENDING.** Mandatory because Fix 1 + Fix 2 collapsed
the headline.

**Plan:**
- Rewrite the abstract: replace the over-broad claim with a scoped
  version that names **short-answer extractive QA**.
- Restructure §8: promote the long-form non-result from a buried
  paragraph to a clearly labeled subsection "Scope of the Paradox."
- Update the broader-impact paragraph accordingly.

---

### Fix 11 — RAPTOR full table

**Weakness W6 spinoff.** The original RAPTOR mention was one line;
this fix populates the full per-(dataset, metric) table.

**Pre-registration:** [`experiments/fix_11_log.md`](../../experiments/fix_11_log.md)
**Code:** [`experiments/fix_11_raptor_full_table.py`](../../experiments/fix_11_raptor_full_table.py)
**Data:** `data/revision/fix_11/per_query.csv` (300 evaluations)
**Results:** `results/revision/fix_11/raptor_full_table.csv`, `raptor_indexing_costs.csv`
**Paper section:** `ragpaper/sections/revision/fix_11_raptor_full_table.tex`

**Sample:** 3 datasets × 100 queries.

**Full table (`raptor_full_table.csv`):**

| dataset  | n   | faith | halluc | p50 (ms) | p99 (ms) | dense idx (s) | RAPTOR idx (s) | size (MB) | clusters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| hotpotqa | 100 | 0.617 | 0.21 | 1972.55 | 4274.46 |  7.14 |  99.74 | 16.59 | 6 |
| pubmedqa | 100 | 0.560 | 0.29 | 3900.97 | 6811.90 |  1.18 | 108.26 |  3.10 | 6 |
| squad    | 100 | 0.789 | 0.05 | 1190.46 | 4942.80 |  8.33 | 161.20 |  1.83 | 6 |

**Interpretation:** RAPTOR's offline tree-build cost (100–161 s) is
**two orders of magnitude** above plain dense indexing (1–8 s); query
latency is competitive across the three datasets. SQuAD faithfulness
under RAPTOR (0.789) is comparable to baseline; PubMedQA hallucination
rate (29%) is well above the SQuAD figure, consistent with the broader
multi-dataset pattern.

**Paper implication:** §10.2 RAPTOR row in the v2.0 paper can be
replaced with this full table including indexing cost, index size, and
p99 latency.

## 5. What this means for the paper

Bringing the per-fix results together, the paper changes are:

1. **Abstract.** Verb downgrade ("drives" → "predicts"); explicit
   scope to short-answer extractive QA; remove the
   "optimizing-the-wrong-quantity" claim. Forced by Fix 1 + Fix 10.
2. **§Theory (theory.tex).** Retitle "Information-Theoretic
   Consistency Check"; rewrite Proposition 1 + Theorem 1 with
   explicit non-implications; move predictive language to flagged
   conjectures. Forced by Fix 1 + Fix 8.
3. **§Paradox (paradox.tex).** Replace the n=30 headline with the
   n=500 × 5-seed numbers; add the "1/5 seeds significant on
   baseline−v1" disclosure; cross-reference Fix 3 metric variation.
   Forced by Fix 2 + Fix 3.
4. **§Method (method.tex / multi-metric).** Add the three-metric
   triangulation table; report DeBERTa as the outlier and RAGAS as the
   alternative scoring that shows a larger paradox. Forced by Fix 3.
5. **§Robustness (robustness.tex).** Add the noise-slope comparison
   from Fix 5: coherence-preserving (−0.043) vs random (−0.069).
   Strengthens §7.6 disclosure. Sourced by Fix 5.
6. **§Discussion / §8 Limitations.** Add the τ-generalization gap
   table from Fix 4 (SQuAD/PubMedQA/NaturalQS flagged); add the
   Fix 7 budget-blocked disclosure; add the long-form non-result
   subsection. Forced by Fix 4 + Fix 7 + Fix 10.
7. **§Confidence calibration.** Downgrade to "suggestive" and note
   the missing-control limitation. Forced by Fix 9.
8. **§Head-to-head (headtohead.tex / appendix).** Add full RAPTOR
   table with indexing cost and p99 latency. Sourced by Fix 11. Add
   Fix 6 head-to-head once it runs.
9. **§Causal intervention (new subsection).** Add the matched-
   similarity HIGH/LOW CCS experiment with the null result, the
   paired-difference figure, and the concession paragraph. Sourced
   by Fix 1.

## 6. What's remaining

Ordered by priority and dependency.

**Must do (paper-only, no compute):**
- Wire `ragpaper/sections/revision/fix_08_theory_reframe.tex` into
  `theory.tex` (rename + theorem rewrite).
- Wire `ragpaper/sections/revision/fix_10_scope_deployment.tex` into
  `abstract.tex` and `discussion.tex` (scope narrowing).
- Wire `ragpaper/sections/revision/fix_01_causal_intervention.tex`
  into the paradox/method narrative.
- Apply the cascading edits in §5 above (abstract, paradox, method,
  robustness, discussion).
- Wire all completed-fix tables into `ragpaper/main.tex` via
  `\input{sections/revision/fix_NN_*}`.

**Must do (single Kaggle compute slot):**
- Run Fix 6 no-Self-RAG path on a fresh Kaggle T4×2 session via
  `notebooks/revision_fix6_kaggle_t4x2_fresh.ipynb`.
- Download `/kaggle/working/fix6_t4x2_outputs.zip`, import locally,
  update `experiments/fix_06_log.md` with the result block, wire the
  table into `ragpaper/sections/revision/fix_06_baselines.tex`.

**Optional:**
- Collect two-rater labels for `data/revision/fix_03/human_eval_template.jsonl`
  (99 items). Compute Cohen's κ and Spearman vs each automated metric.
- Regenerate `experiments/run_confidence_calibration.py` output with
  `mean_retrieval_similarity` and `passage_redundancy` columns and
  re-run Fix 9 with full controls.
- Attempt Fix 6 Self-RAG smoke test, then optionally the full
  Self-RAG run.

**Blocked:**
- Fix 7 (Together.ai 70B reproduction) stays blocked unless free 70B
  compute appears. Disclose, do not fake.

## 7. Hardware + execution recipes

### Zero-dollar paths

**Kaggle T4×2 (preferred for compute-heavy fixes):**

```bash
# In a fresh Kaggle cell with Internet ON, GPU = T4 x 2
%%bash
set -euo pipefail
cd /kaggle/working
if [ ! -d rag-hallucination-detection/.git ]; then
  git clone --branch main https://github.com/Saket-Maganti/rag-hallucination-detection.git
else
  git -C rag-hallucination-detection pull --ff-only origin main
fi
cd rag-hallucination-detection
bash scripts/kaggle_fix6_t4x2.sh   # for Fix 6
```

**M4 Air (for analysis-only and small re-runs):**

| Task | Runtime |
| --- | ---: |
| Validation/tests | 1–3 min |
| Fix 1 analysis (already done) | seconds–2 min |
| Fix 9 partial-correlation rerun | seconds–5 min |
| Fix 8 / Fix 10 manual paper integration | 30–90 min |

**Ollama recovery on Kaggle:**

```bash
%%bash
set -euo pipefail
cd /kaggle/working/rag-hallucination-detection
git pull --ff-only origin main
bash scripts/kaggle_ollama_guard.sh mistral
tail -n 80 /kaggle/working/ollama.log || true
ollama list
```

For the long-form runbook see [`runbook.md`](runbook.md). For the
operational handoff with Kaggle session-by-session detail see
[`codex.md`](codex.md). For the reviewer-fix snapshot summary see
[`snapshot.md`](snapshot.md).

## 8. File map

```
docs/revision/
├── README.md          ← this book (single source of truth)
├── codex.md           ← operational handoff (was CODEX.md)
├── runbook.md         ← exact execution commands (was REVISION_RUNBOOK.md)
├── status.md          ← scoreboard with assessments (was REVISION_SUMMARY.md)
└── snapshot.md        ← reviewer-fix snapshot (was `fixes`)

experiments/
├── fix_01_causal_matched_pairs.py
├── fix_02_scaled_headline_n500.py
├── fix_03_multimetric_faithfulness.py
├── fix_04_tau_generalization.py
├── fix_05_coherence_preserving_noise.py
├── fix_06_baseline_h2h_pareto.py
├── fix_07_together_70b_reproduction.py
├── fix_09_partial_correlations.py
├── fix_11_raptor_full_table.py
├── revision_utils.py
└── fix_NN_log.md       ← per-fix pre-registration + result append

src/
├── ragas_scorer.py
├── vectara_hem_scorer.py
├── together_llm.py
├── openai_llm.py
└── anthropic_llm.py

data/revision/
├── fix_01/  matched_pairs, per_query, COLUMNS, skipped_queries
├── fix_02/  per_query (5 seeds × 3 conds × 500), per_query_gpu0/1, partial seeds
├── fix_03/  per_query (rescored), human_eval_template.jsonl (99 items)
├── fix_04/  per_query (5×5×300), 25 tau partials
├── fix_05/  per_query (n=1591), per_query_partial
├── fix_09/  input_copy.csv
└── fix_11/  per_query (3×100), per-dataset partials

results/revision/
├── fix_01/  paired_wilcoxon, match_diagnostics, bootstrap_ci, summary
├── fix_02/  headline_table, paired_contrasts, summary (+ gpu shards)
├── fix_03/  table1_multimetric, metric_correlations, summary
├── fix_04/  tau_summary, tau_transfer_matrix, generalization_flags, summary
├── fix_05/  noise_summary, slope_response, summary
├── fix_09/  partial_correlations, summary
└── fix_11/  raptor_full_table, raptor_indexing_costs, summary

ragpaper/sections/revision/
├── fix_01_causal_intervention.tex
├── fix_02_scaled_headline.tex
├── fix_03_multimetric.tex
├── fix_04_tau_generalization.tex
├── fix_05_noise_slope.tex
├── fix_06_baselines.tex
├── fix_07_together.tex
├── fix_08_theory_reframe.tex
├── fix_09_partial_confidence.tex
├── fix_10_scope_deployment.tex
└── fix_11_raptor_full_table.tex

notebooks/
├── revision_session1_kaggle_fresh.ipynb     ← Fix 1 + Fix 5 + Fix 11 (already used)
├── revision_fix2_kaggle_t4x2_fresh.ipynb    ← Fix 2 (already used)
├── revision_fix3_4_kaggle_t4x2_fresh.ipynb  ← Fix 3 + Fix 4 (already used)
├── revision_fix5_11_kaggle_t4x2_fresh.ipynb ← Fix 5 + Fix 11 (already used)
└── revision_fix6_kaggle_t4x2_fresh.ipynb    ← Fix 6 (PENDING execution)

scripts/
├── kaggle_session1_fresh.sh
├── kaggle_fix1_only.sh
├── kaggle_fix1_parallel_t4x2.sh
├── kaggle_fix2_t4x2.sh
├── kaggle_fix3_4_t4x2.sh
├── kaggle_fix5_11_t4x2.sh
├── kaggle_fix6_t4x2.sh           ← Fix 6 (NEW)
├── kaggle_stream_fix1_t4x2.py
├── kaggle_stream_fix2_t4x2.py
├── kaggle_stream_fix3_4_t4x2.py
├── kaggle_stream_fix5_11_t4x2.py
├── kaggle_stream_fix6_t4x2.py    ← Fix 6 (NEW)
├── kaggle_ollama_guard.sh
├── kaggle_session1_background.sh
├── kaggle_tail_session1.sh
└── local_session1_zero_dollar.sh

logs/revision/   ← gitignored; produced by Kaggle runs
```

---

*Last update reflects commit `0fb01489` and the doc reorganization that
followed. Permanent identifiers and released artifacts are unchanged
from v2.0.0:*

- *DOI: [10.5281/zenodo.19757291](https://doi.org/10.5281/zenodo.19757291)*
- *HF Dataset: [`saketmgnt/context-coherence-bench`](https://huggingface.co/datasets/saketmgnt/context-coherence-bench)*
- *HF Space: [`saketmgnt/sakkk`](https://huggingface.co/spaces/saketmgnt/sakkk)*
- *Pip: `pip install context-coherence` (revision line bumped to 0.2.0)*
