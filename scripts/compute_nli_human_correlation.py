"""
Compute correlation between NLI faithfulness scores and human judgments.
Run this after filling in results/human_validation/validation_sheet.csv
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv("results/human_validation/validation_sheet.csv")

# Filter to graded rows only
graded = df[df["human_faithful"].notna() & (df["human_faithful"] != "")].copy()
graded["human_faithful"] = graded["human_faithful"].astype(int)

if len(graded) < 10:
    print("Need at least 10 graded rows. Please fill in the human_faithful column.")
    exit()

nli_scores   = graded["nli_faithfulness_score"].values
human_labels = graded["human_faithful"].values

# Pearson correlation (continuous NLI score vs binary human label)
r, p_value = stats.pearsonr(nli_scores, human_labels)

# Spearman correlation (rank-based)
rho, p_spearman = stats.spearmanr(nli_scores, human_labels)

# Agreement at threshold 0.5
nli_binary = (nli_scores >= 0.5).astype(int)
agreement  = (nli_binary == human_labels).mean()

# Precision / Recall of NLI detector vs human
from sklearn.metrics import classification_report, confusion_matrix
print("\n===== NLI vs Human Validation =====")
print(f"Graded samples: {len(graded)}")
print(f"\nPearson r:    {r:.4f}  (p={p_value:.4f})")
print(f"Spearman rho: {rho:.4f} (p={p_spearman:.4f})")
print(f"Agreement @ threshold 0.5: {agreement*100:.1f}%")
print("\nClassification Report (NLI vs Human):")
print(classification_report(human_labels, nli_binary,
      target_names=["Hallucinated", "Faithful"]))
print("Confusion Matrix:")
print(confusion_matrix(human_labels, nli_binary))

# Save results
results = {
    "n_graded": len(graded),
    "pearson_r": round(r, 4),
    "pearson_p": round(p_value, 4),
    "spearman_rho": round(rho, 4),
    "spearman_p": round(p_spearman, 4),
    "agreement_pct": round(agreement * 100, 2),
}
import json
with open("results/human_validation/correlation_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to results/human_validation/correlation_results.json")