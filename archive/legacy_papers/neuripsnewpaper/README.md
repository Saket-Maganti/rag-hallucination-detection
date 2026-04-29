# NeurIPS Submission Draft

Research-paper draft built from the senior-reviewer revision results.

## Research Thesis

> Local retrieval scores do not identify RAG faithfulness.

The paper is no longer centered on "CCS drives faithfulness." The controlled
results do not support that. The stronger research result is that several
common evidential shortcuts in RAG papers fail under controls:

- higher mean query-passage similarity does not identify answer support;
- higher context coherence does not causally guarantee faithfulness;
- a single automatic faithfulness metric does not identify effect size;
- a single tuned threshold does not cleanly transfer across datasets;
- one baseline row does not identify deployment superiority once latency and
  indexing cost are included.

CCS survives as a useful diagnostic, especially through the coherence-preserving
noise result, but not as a validated causal mechanism.

## Latest Results Snapshot

| Fix | Result | Research implication |
| --- | --- | --- |
| Fix 1 causal coherence | HIGH-CCS faithfulness 0.636 vs LOW-CCS 0.639; HIGH-LOW -0.002; Wilcoxon p=0.628; CI [-0.022, 0.017]; HIGH hallucination 16.5% vs LOW 9.0%. | Reject causal "CCS drives faithfulness" language. |
| Fix 2 scale-up | SQuAD/Mistral n=500 x 5 seeds: baseline 0.6609, HCPC-v1 0.6503, HCPC-v2 0.6612; only 1/5 baseline-v1 seeds significant. | The old n=30 headline was a pilot; DeBERTa effect is small at scale. |
| Fix 3 multi-metric | Paradox magnitude: DeBERTa 0.011, second NLI 0.032, RAGAS-style judge 0.140; DeBERTa-RAGAS Pearson r=0.182. | Faithfulness conclusions are metric-dependent. |
| Fix 4 threshold transfer | SQuAD, PubMedQA, and NaturalQuestions have large diagonal-vs-off-diagonal recovery gaps. | A single tuned threshold is not broadly validated. |
| Fix 5 noise | Coherent uninformative noise slope -0.043 vs random noise -0.069. | Coherence remains a diagnostic signal, not a causal proof. |
| Fix 6 full baselines | Full Self-RAG package verified: 1600 rows, 8 summary rows, 0 errors. SQuAD: RAPTOR-2L 0.710, HCPC-v2 0.708, CRAG 0.698, Self-RAG 0.574. HotpotQA: CRAG 0.643, HCPC-v2 0.632, RAPTOR-2L 0.631, Self-RAG 0.574. | No method dominance; Self-RAG underperforms in this matched short-answer harness. |
| Fix 7 70B repro | Budget-blocked under zero-dollar mode. | Disclose; do not fake independent 70B reproduction. |
| Fix 9 confidence | No-control Pearson r=0.360, p=0.0047; controls absent. | Treat self-confidence as suggestive only. |
| Fix 11 RAPTOR | SQuAD RAPTOR faith 0.789 / 5% halluc; PubMedQA 0.560 / 29%; HotpotQA 0.617 / 21%; tree build 99.7-161.2 s. | RAPTOR needs full cost accounting. |

## Build

```bash
cd /Users/saketmaganti/claudeprojs/rag-hallucination-detection/neuripsnewpaper
pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## Main Files

- `main.tex` - paper entry point
- `supplement.tex` - supplementary artifact and source-table material
- `sections/` - research-paper sections
- `main.pdf` - compiled draft
- `source_tables/` - copied result CSVs used in the draft
- `figures/` - revision figures
- `CLAIMS_AUDIT.md` - guardrail against overclaiming

## Submission Positioning

This is closer to a NeurIPS research paper than the earlier artifact-style
draft. The core contribution is a falsification/identifiability result for RAG
faithfulness evaluation, with the benchmark as evidence and reusable protocol.

## Submission Notes

1. Main paper and supplement are separate: build `main.tex` for the research
   narrative and `supplement.tex` for artifact/source-table details.
2. Add completed two-rater human-evaluation results only if labels are
   actually collected; current rater fields are blank.
3. Replace the fallback style with official `neurips_2026.sty` before final
   submission.
4. The full Self-RAG summary and per-query artifact have been imported into
   repository-local files; see `source_tables/CHECKSUMS.md`.
