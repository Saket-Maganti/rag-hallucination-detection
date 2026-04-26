# Fix 9 Log - Self-Confidence Partial Correlations

**Status:** code written, execution pending.  
**Weakness addressed:** W9, CCS-confidence correlation might be mediated by
similarity or redundancy.

## Protocol

- Input: confidence-calibration per-query CSV.
- Outcome: model self-confidence.
- Predictor: CCS.
- Controls:
  - mean query-passage similarity;
  - passage redundancy, defined as mean pairwise passage similarity without
    the standard-deviation term.

## Command

```bash
python3 experiments/fix_09_partial_correlations.py \
  --input results/confidence_calibration/per_query.csv
```

If the full-control partial correlation does not survive, the paper demotes
the confidence result to "suggestive."
