# Fix 6 Log - Proper Baseline Head-To-Head

**Status:** code written, execution pending.  
**Weakness addressed:** W6, RAPTOR/Self-RAG/CRAG comparison too thin.

## Protocol

- Datasets: SQuAD and HotpotQA.
- Sample: `n>=200` per dataset.
- Conditions implemented in the same harness:
  - HCPC-v2;
  - CRAG;
  - RAPTOR-2L.
  - Self-RAG when `--include_selfrag` is passed. This should be run on a CUDA
    host because it loads the fine-tuned Self-RAG checkpoint.

## Command

```bash
python3 experiments/fix_06_baseline_h2h_pareto.py \
  --datasets squad hotpotqa \
  --n 200 \
  --backend ollama \
  --model mistral
```

CUDA Self-RAG variant:

```bash
python3 experiments/fix_06_baseline_h2h_pareto.py \
  --datasets squad hotpotqa \
  --n 200 \
  --backend ollama \
  --model mistral \
  --include_selfrag \
  --selfrag_8bit
```

## Output

- `data/revision/fix_06/per_query.csv`
- `results/revision/fix_06/h2h_summary.csv`
- `results/revision/fix_06/pareto_faithfulness_latency.pdf`
