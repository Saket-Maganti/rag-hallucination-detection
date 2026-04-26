# Fix 11 Log - RAPTOR Full Table

**Status:** code written, execution pending.  
**Weakness addressed:** W11, RAPTOR was only mentioned in one line.

## Protocol

- Datasets: SQuAD, PubMedQA, HotpotQA by default.
- RAPTOR: two-level tree with one summary layer.
- Metrics: faithfulness, hallucination rate, p50/p99 latency, dense index
  build time, RAPTOR tree build time, index size.

## Command

```bash
python3 experiments/fix_11_raptor_full_table.py \
  --datasets squad pubmedqa hotpotqa \
  --n 100 \
  --backend ollama \
  --model mistral
```

## Output

- `data/revision/fix_11/per_query.csv`
- `results/revision/fix_11/raptor_full_table.csv`
- `results/revision/fix_11/raptor_indexing_costs.csv`
