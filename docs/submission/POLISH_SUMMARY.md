# Polish Summary

Submission directory: `papers/neurips`

Title: "When Retrieval Quality Decouples from Faithfulness: A Pre-Registered Audit of RAG Evaluation"

## Six Fixes

| Fix | Reviewer concern addressed | What changed | Status |
| --- | --- | --- | --- |
| Fix 1: Metric-fragility framing | The paper could read like a self-audit whose pilot collapsed, rather than a positive methodological contribution. | Reframed the abstract, introduction, contributions, and conclusion around the fixed-generation multi-scorer result and human calibration. The matched null, scale collapse, threshold transfer, and cost audit now support the under-identification thesis. | Fully resolved as framing; evidence remains conservatively qualified because Table 6 intervals overlap. |
| Fix 2: Bootstrap CIs | Table 5 and Table 6 point estimates could overstate scorer ordering, especially at \(n=99\). | Added 10,000-resample paired-index percentile bootstrap CIs to cross-scorer and human-calibration correlations; added a methods note and prose warning that Table 6 ordering is suggestive. | Fully resolved statistically; wide human-calibration intervals remain an honest limitation. |
| Fix 3: CCS definition mismatch | The paper's simplified CCS definition did not match the original implementation. | Restored the original \(mean - sd\) off-diagonal pairwise cosine definition in Section 2 and verified Fix 1 used that definition. | Fully resolved; no package version bump needed. |
| Fix 4: Self-RAG fairness | Main-table/figure placement made a harness-mismatched Self-RAG row look like a like-for-like underperformance claim. | Removed Self-RAG from main Table 9 and Figure 2; added a supplement harness-mismatch table and caveat explaining backbone, quantization, fine-tuning, and scoring differences. | Fully resolved for fairness; Self-RAG remains traceable in the supplement. |
| Fix 5: HIGH-CCS hallucination mechanism | The wrong-sign hallucination result was treated only as a negative. | Computed answer-span presence in the matched 200 pairs and added inline Section 4 numbers: HIGH 35/200, LOW 54/200, exact McNemar \(p=0.011\). The paper frames lower answer coverage in HIGH-CCS sets as a data-suggested explanation. | Partially mitigated; it is an explanatory diagnostic, not a tested generator mechanism. |
| Fix 6: Minimum reporting standard | The most reusable recommendation was buried late in the paper. | Promoted the seven-item minimum reporting standard to Section 1 and referenced it in the abstract. Section 10 now points back to the Section 1 block. | Fully resolved as positioning; the standard is now visible early without expanding page count. |

## P2 Polish

- Table 7 caption now includes a worked recovery-score example.
- Limitations disclose that pilot 70B/120B observations were not re-verified under the zero-dollar revision constraint.
- Figure 1 caption now explains the diagonal and above/below interpretation.
- Key numeric claims were cross-checked against source CSVs and source-trace files.

## Page-Budget Note

The fallback local build is 10 pages including references. To recover the page after adding CIs and the reporting standard, the duplicated Section 10 minimum-reporting prose was shortened and the compact related-work paragraph was pruned to submission-essential citations. No margin changes or global font shrinking were introduced.
