# Fix 5 Log: Answer-Span Presence in Matched HIGH/LOW CCS Contexts

Date: 2026-04-29

Purpose: interpret the wrong-sign hallucination result in the matched CCS test
without claiming a new generator-side mechanism.

Source file:

- `data/revision/fix_01/matched_pairs.csv`

Protocol:

- For each of the 200 matched pairs, concatenate the three HIGH-CCS passages
  and the three LOW-CCS passages.
- Normalize the gold answer and context by lowercasing, removing punctuation,
  dropping English articles, and collapsing whitespace.
- Mark span presence when the normalized gold answer appears in the normalized
  context.
- Test discordant pairs with an exact two-sided McNemar/binomial test.
- Output details: `results/revision/fix_01/answer_span_presence.csv`.

Four-way table:

|  | Answer in HIGH | Answer not in HIGH |
| --- | ---: | ---: |
| Answer in LOW | 19 | 35 |
| Answer not in LOW | 16 | 130 |

Summary:

- P(answer in HIGH-CCS context) = 35 / 200 = 0.175.
- P(answer in LOW-CCS context) = 54 / 200 = 0.270.
- Exact McNemar p-value = 0.0109735629.

Interpretation: answer-span coverage is not comparable across HIGH/LOW CCS
sets; LOW-CCS contexts contain the gold answer more often. This offers a
simple explanation for the higher HIGH-CCS hallucination rate (16.5% vs. 9.0%)
and should be framed as a data-suggested hypothesis, not a tested mechanism.
