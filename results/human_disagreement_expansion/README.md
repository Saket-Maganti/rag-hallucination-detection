# Human Disagreement Expansion

This directory contains a targeted annotation batch for scorer-disagreement
calibration. It is not a broad human-evaluation study.

## Source

- Input rows: `data/revision/fix_03/per_query.csv`
- Selected file: `annotation_batch_disagreement_100.csv`
- Previously used IDs excluded when available: 187

## Selection Rule

Rows were ranked by the absolute disagreement between the stored DeBERTa score
and the local RAGAS-style score. The sampler then mixed the strongest
disagreements across:

- DeBERTa high / RAGAS-style low
- RAGAS-style high / DeBERTa low
- near-threshold disagreements around the 0.5 hallucination boundary
- baseline, HCPC-v1, and HCPC-v2 conditions

The goal is to stress-test the central metric-fragility claim where automated
scorers disagree most, not to estimate overall population faithfulness.

## Batch Counts by Condition

- baseline: 34
- hcpc_v1: 33
- hcpc_v2: 33

## Batch Counts by Disagreement Type

- deberta_high_ragas_low: 34
- near_threshold_disagreement: 33
- ragas_high_deberta_low: 33

## Annotation Instructions

Annotators should read the question, retrieved context, generated answer, and
gold answer. Label the generated answer against the retrieved context, not
against parametric knowledge.

Use `annotator_label` values:

- `faithful`: every factual claim in the generated answer is supported by the retrieved context.
- `hallucinated`: at least one factual claim is unsupported or contradicted by the retrieved context.
- `unclear`: the context or answer is too ambiguous for a reliable binary decision.

Use `annotator_confidence` as a number from 0 to 1. Use `annotator_notes` for
short rationale or ambiguity notes.

For two-rater annotation, duplicate the three annotator columns with suffixes
such as `_rater1` and `_rater2`, or add an `adjudicated_label` column after
discussion.

## After Annotation

Run:

```bash
python3 scripts/analyze_human_disagreement_labels.py \
  --input results/human_disagreement_expansion/annotation_batch_disagreement_100.csv
```

The analysis script reports inter-rater agreement when two rater columns exist,
and scorer alignment metrics against an adjudicated or majority label when
labels are available.
