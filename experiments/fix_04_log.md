# Fix 4 Log - Cross-Dataset Hyperparameter Generalization

**Status:** code written, execution pending.  
**Weakness addressed:** W4, possible tau-tuning leakage.

## Protocol

- Datasets: SQuAD, PubMedQA, HotpotQA, Natural Questions, TriviaQA.
- Tune threshold tau on each dataset and evaluate that frozen tau on every
  other dataset.
- Default grid: `0.30 0.40 0.50 0.60 0.70`.
- Sample: `n=100` per dataset/tau.
- Generator: Mistral-7B unless backend override is supplied.

## Recovery Definition

`recovery = (faith_ccs_gate - faith_hcpc_v1) / (faith_baseline - faith_hcpc_v1)`

## Flag Rule

If diagonal recovery exceeds mean off-diagonal recovery by more than `0.03`,
the gap must be flagged honestly in the discussion/limitations section.

## Command

```bash
python3 experiments/fix_04_tau_generalization.py \
  --datasets squad pubmedqa hotpotqa naturalqs triviaqa \
  --taus 0.30 0.40 0.50 0.60 0.70 \
  --n 100 \
  --backend ollama \
  --model mistral
```

## Output

- `data/revision/fix_04/per_query.csv`
- `results/revision/fix_04/tau_summary.csv`
- `results/revision/fix_04/tau_transfer_matrix.csv`
- `results/revision/fix_04/generalization_flags.csv`
