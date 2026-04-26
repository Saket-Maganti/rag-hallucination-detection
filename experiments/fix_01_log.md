# Fix 1 Log - Causal Coherence Intervention

**Pre-registration date:** 2026-04-26  
**Status:** designed; matched-pair construction complete; generation/NLI pending  
**Reviewer weakness addressed:** W1, causal-vs-correlational concern  
**Primary script:** `experiments/fix_01_causal_matched_pairs.py`

## Handoff Audit Of Claude Markdown

Claude's `CLAUDE.md` usefully records prior project context, but it is not
source-of-truth for this new 10-weakness revision. The current file describes
an earlier Phase 7 plan with six reviewer-style critiques, while the user's
new task defines a stricter P0/P1/P2 priority order and makes Fix 1 the gating
result.

The Claude worktree
`.claude/worktrees/laughing-haslett-ade7d7/` contains a plausible first draft
for Fixes 1-3, but I am not copying it verbatim for three reasons:

1. Its `REVISION_SUMMARY.md` says all Fix-NN scripts are built, but only
   `fix_01`, `fix_02`, and `fix_03` scripts actually exist in that worktree.
2. The Fix 1 log/script use HIGH minus LOW and a one-sided Wilcoxon
   `alternative="greater"`, while the summary table says `alt=less`. This is
   a reproducibility hazard, so the revised harness uses HIGH minus LOW and
   `greater` consistently.
3. The draft loads only `max_papers=30` for SQuAD while targeting `n=200`
   valid pairs. That can silently underpower the experiment. The revised
   preregistered command loads 400 SQuAD contexts and considers up to 400
   shuffled candidate queries.

The useful parts retained are the central combinatorial-search design, the
two-row-per-query CSV contract, the honest downgrade commitment, and the
paired Wilcoxon/Cohen's dz/bootstrap analysis.

## Hypothesis

At fixed mean per-passage query similarity, lower context coherence causes
lower faithfulness.

Operationally, for each matched query pair:

`diff_i = faithfulness(HIGH_CCS_i) - faithfulness(LOW_CCS_i)`

The preregistered directional hypothesis is:

`H1: median(diff_i) > 0`

The mechanistic claim is considered supported only if all of the following
hold on the primary SQuAD cell:

- Paired Wilcoxon signed-rank test on `diff_i`, `alternative="greater"`,
  gives `p < 0.05`.
- Cohen's `d_z > 0.2`.
- The 10000-resample bootstrap 95% CI for mean `diff_i` excludes 0.
- The constructed pairs satisfy `max(abs(mean_sim_high - mean_sim_low)) <= 0.02`.

If these conditions fail, the paper must replace causal language such as
"drives" with correlational language such as "predicts" and reframe the
theory section as a consistency check rather than a mechanism proof.

## Protocol

Primary cell:

- Dataset: SQuAD validation split.
- Retrieval corpus: first 400 unique SQuAD validation contexts, loaded by the
  existing SQuAD loader.
- Candidate query order: QA pairs from those contexts, shuffled with seed 42.
- Candidate query cap: first 400 shuffled QA pairs.
- Target sample size: at least 200 valid matched query pairs.
- Retriever: existing ChromaDB + `sentence-transformers/all-MiniLM-L6-v2`.
- Retrieval depth: top-20 passages per query.
- Triple size: 3 passages.
- Matching tolerance: absolute mean query-similarity gap <= 0.02.
- Minimum CCS gap: `CCS_high - CCS_low >= 0.05`.
- Passage overlap: at most one shared passage between HIGH and LOW triples.
- Generator: Mistral-7B via Ollama, temperature 0.0.
- Faithfulness metric: existing DeBERTa-v3 NLI scorer.

Construction routine:

1. Retrieve the top-20 passages for a query.
2. Embed the query and all 20 passages with MiniLM.
3. Enumerate all `C(20, 3) = 1140` triples.
4. For each triple, compute mean query similarity and CCS.
5. Bucket triples by mean query similarity in 0.02-wide buckets.
6. Within every bucket, search all valid LOW/HIGH triple pairs and keep the
   pair with the largest CCS gap, breaking ties by smaller similarity gap and
   smaller passage overlap.
7. Skip queries with no valid matched pair and record the reason.

## Statistical Test

Primary test:

- Paired Wilcoxon signed-rank on `faith_high - faith_low`,
  `alternative="greater"`.

Effect sizes and intervals:

- Cohen's `d_z = mean(diff) / sd(diff)` for continuous faithfulness.
- Matched-pair hallucination odds ratio
  `(low_only + 0.5) / (high_only + 0.5)` for binary hallucination.
- 10000-resample percentile bootstrap CI for mean paired faithfulness
  difference.

Similarity matching diagnostic:

- The hard construction criterion is `max_abs_similarity_delta <= 0.02`.
- A two-sided Wilcoxon test on similarity deltas is reported as a diagnostic
  only. It is not treated as proof of equivalence because non-significance is
  not an equivalence test.

## Execution Log

Preregistered full command:

```bash
python3 experiments/fix_01_causal_matched_pairs.py \
  --stage full \
  --dataset squad \
  --n_target 200 \
  --seed 42 \
  --max_contexts 400 \
  --candidate_limit 400 \
  --backend ollama \
  --model mistral
```

Partial construction smoke command:

```bash
python3 experiments/fix_01_causal_matched_pairs.py \
  --stage construct \
  --dataset squad \
  --n_target 10 \
  --seed 42 \
  --max_contexts 80 \
  --candidate_limit 40 \
  --run_tag smoke
```

Key numbers are written to:

- `results/revision/fix_01/construction_summary.csv`
- `results/revision/fix_01/match_diagnostics.csv`
- `results/revision/fix_01/paired_wilcoxon.csv` after generation/NLI

Construction execution completed on 2026-04-26:

```bash
python3 experiments/fix_01_causal_matched_pairs.py \
  --stage construct \
  --dataset squad \
  --n_target 200 \
  --seed 42 \
  --max_contexts 400 \
  --candidate_limit 400 \
  --run_tag primary_n200
```

Construction key numbers:

- Valid matched pairs: `200/200`.
- Skipped queries: `0`.
- Top-20 retrieval pools came from `400` SQuAD validation contexts.
- Mean absolute similarity gap: `0.006351`.
- Maximum absolute similarity gap: `0.018512`, below the preregistered
  `0.02` tolerance.
- Mean CCS gap: `0.532634`.
- Minimum CCS gap: `0.264139`, above the preregistered `0.05` minimum.
- Mean passage overlap: `0.395`; max overlap: `1`.

These numbers establish that the controlled dataset construction is feasible
at the required sample size. They do **not** establish the causal hypothesis;
that requires the full Mistral generation and DeBERTa NLI pass.

## Interpretation Template

If H1 is supported, the paper can claim that CCS has causal evidence in the
controlled SQuAD intervention, scoped to short-answer extractive QA and the
tested generator/scorer.

If H1 is not supported, the honest interpretation is that CCS remains a useful
predictor/deployment diagnostic, but the paper has not shown that coherence is
the causal mediator at fixed query similarity. In that case, the abstract,
theory section, and discussion must be downgraded before any resubmission.
