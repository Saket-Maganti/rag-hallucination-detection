# OpenReview submission checklist — NeurIPS 2026

Pre-submission checklist for *"When Better Retrieval Hurts: Context Coherence
Drives Faithfulness in Retrieval-Augmented Generation"* (commit: `v2.0.0`).

## 0. Pre-flight (do these once, save outputs)

- [ ] `bash scripts/release_v2.sh --dry-run` — sanity check the release script
- [ ] `python3 scripts/lint_paper.py` — zero `[ERR]` lines
- [ ] `python3 experiments/build_headline_figure.py` — produces `papers/arxiv_longform/figures/headline_frontier.{pdf,tex}`
- [ ] `python3 experiments/build_ccs_calibration.py` — produces `papers/arxiv_longform/figures/ccs_calibration.{pdf,tex}`
- [ ] `python3 experiments/build_qualitative_example.py` — produces `papers/arxiv_longform/figures/qualitative_paradox.tex`
- [ ] Wire the three new figures into the paper (one `\input{...}` line each in `results.tex` or `analysis.tex`)
- [ ] `cd papers/arxiv_longform && pdflatex main && bibtex main && pdflatex main && pdflatex main` — clean build, all `??` and `Citation undefined` warnings resolved
- [ ] Final author block in `main.tex` (no `\thanks{TBD}`)
- [ ] `python3 scripts/upload_to_zenodo.py --sandbox` rehearsal succeeds; then production for real DOI
- [ ] `bash scripts/release_v2.sh` — tag pushed, tarball at `/tmp/coherence-paradox-v2.0.0.tar.gz`

## 1. OpenReview metadata (paste into the form)

See `submission_packages/neurips/openreview/paper_metadata.yml` for canonical values. The big ones:

| Field | Value |
|---|---|
| Title | When Better Retrieval Hurts: Context Coherence Drives Faithfulness in Retrieval-Augmented Generation |
| Track | Main conference (Datasets & Benchmarks track also viable — it's a benchmark + analysis paper) |
| Primary keywords | retrieval-augmented generation; hallucination; context coherence; benchmarks; evaluation |
| Secondary keywords | natural language inference; faithfulness; mechanistic interpretability |
| TL;DR | Better per-passage retrieval can *reduce* answer faithfulness by fragmenting context coherence; we name this the *refinement paradox*, introduce a generator-free retrieval-time diagnostic (CCS), and show a coherence-gated retriever (HCPC-v2) recovers faithfulness across SQuAD, PubMedQA, and frontier-scale generators (Llama-3.3-70B, GPT-OSS-120B). |
| Abstract | Use `papers/arxiv_longform/sections/abstract.tex` verbatim; OpenReview strips LaTeX so paste the plain-text version (script: `python3 -c "import re; t=open('papers/arxiv_longform/sections/abstract.tex').read(); print(re.sub(r'\\\\[a-zA-Z]+\\{?|\\$|\\}', '', t))"`). |

## 2. Required form fields

- [ ] **Conflicts of interest** — list co-authors / advisor / institution emails for the past 3 years
- [ ] **Reproducibility statement** — paste from §Appendix; mention: code at GitHub, benchmark with DOI on Zenodo, demo on HF Spaces, all seeds documented in `results/multiseed/`
- [ ] **Computational resources statement** — paste from `experimental_setup.tex`: M4 Pro local + Kaggle T4 + Groq free-tier API; total compute ≪ 100 GPU-hours
- [ ] **Limitations** — paste from §`sec:limitations` (already includes long-form scope, noise-equivalence, adversarial-129, frontier-scale)
- [ ] **Broader impact** — write a short paragraph (template below)
- [ ] **Ethics statement** — no human subjects in the released set (Prolific deferred), all datasets are public, benchmark is MIT-licensed
- [ ] **Funding disclosure** — independent researcher; no funding to disclose
- [ ] **Code & data availability** — link GitHub repo + Zenodo DOI + HF Space

## 3. Broader impact paragraph (template — adapt to taste)

> RAG systems are increasingly deployed in high-stakes settings (medical
> question answering, legal research, customer support) where unsupported
> claims have direct downstream cost. Our work identifies a counter-
> intuitive failure mode in the dominant deployment pattern (rerank +
> per-passage refinement) and provides a lightweight, generator-free
> diagnostic (CCS) that can be computed at retrieval time. We expect the
> primary positive impact to be on deployers who can now flag low-coherence
> retrieval sets before they reach the generator. A potential negative
> impact is mis-use of the diagnostic to over-confidently certify "high-
> coherence" outputs as faithful; we explicitly mark CCS as a *predictive*
> signal, not a guarantee, in §\ref{sec:ccs:limits}.

## 4. Supplementary upload

Bundle for upload (≤ 100 MB per OpenReview limit):

```bash
bash scripts/release_v2.sh    # produces /tmp/coherence-paradox-v2.0.0.tar.gz (~1.6 MB)
```

The tarball already excludes artifacts/generated/chroma_db, caches, and the archived paper zip
build snapshot. If the form wants a single PDF supplement, also include
`papers/arxiv_longform/main.pdf` separately.

## 5. Post-submission

- [ ] Save the OpenReview submission ID somewhere durable (CLAUDE.md)
- [ ] Tag commit with the OR submission id: `git tag -a or-<id> -m "OR submission <id>"; git push origin or-<id>`
- [ ] Watch for reviewer assignments (~2 weeks)
- [ ] Begin rebuttal prep:
  - [ ] Closed-model frontier row (GPT-4o or Claude-3.5-Sonnet) — Phase 3 #6, ~$10
  - [ ] Prolific human eval — Phase 3 #9 (deferred), ~1 week + $500 + IRB

## 6. Rebuttal-phase quick-wins (already pre-bought via Phase 1/2)

We pre-emptively ran experiments that anticipate the most likely reviewer
asks. If a reviewer raises any of these, the answer is "see §X":

| Likely reviewer ask | Pre-built answer |
|---|---|
| "Multi-seed variance?" | §10.1, Table `tab:variance` |
| "Comparison to RAPTOR?" | §10.2, Table `tab:raptor` |
| "Long-form generation?" | §10.3, Table `tab:longform` (qualifying scope) |
| "Prompt template artifact?" | §10.4 (ANOVA p=0.83) |
| "Sub-chunk size sensitivity?" | §10.5 |
| "Generic retrieval noise?" | §10.6 (qualifying claim) |
| "Frontier scale?" | §10.7, Table `tab:frontier` (paradox persists at 70B & 120B) |
| "Multi-retriever?" | §multi-retriever in main results |
| "Self-RAG / CRAG comparison?" | §headtohead |
| "Mechanistic explanation?" | §mechanistic |

Have these section-pointer responses pre-drafted in `submission/rebuttal_quickrefs.md` (TODO: write during rebuttal phase).
