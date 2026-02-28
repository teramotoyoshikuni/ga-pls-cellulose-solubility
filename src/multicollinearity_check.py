# -*- coding: utf-8 -*-
"""
Multicollinearity check for selected descriptors.

This script:
  1) loads a dataset CSV (1st column = y, remaining columns = X)
  2) computes Pearson correlation matrix of X
  3) saves the correlation matrix as CSV
  4) saves a correlation heatmap figure
  5) extracts strongly correlated variable pairs (|r| >= threshold) and saves them as CSV

Outputs:
  - correlation_matrix.csv
  - correlation_heatmap.png
  - strongly_correlated_pairs.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ===== User settings =====
DATASET_CSV = "dataset_selected_for_final_model.csv"
THRESHOLD_ABS_R = 0.85

# Output filenames
OUT_CORR_MATRIX = "correlation_matrix.csv"
OUT_HEATMAP_PNG = "correlation_heatmap.png"
OUT_STRONG_PAIRS = "strongly_correlated_pairs.csv"


def main() -> None:
    # =========================
    # 1) Load dataset
    # =========================
    dataset = pd.read_csv(DATASET_CSV, index_col=0)

    # 1st column = y, remaining columns = X
    X = dataset.iloc[:, 1:].astype(float)

    # =========================
    # 2) Correlation matrix
    # =========================
    corr_matrix = X.corr(method="pearson")
    corr_matrix.to_csv(OUT_CORR_MATRIX)

    # =========================
    # 3) Heatmap
    # =========================
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix,
        cmap="coolwarm",
        center=0,
        square=True
    )
    plt.title("Correlation matrix of selected variables (Pearson r)")
    plt.tight_layout()
    plt.savefig(OUT_HEATMAP_PNG, dpi=300)
    plt.close()

    # =========================
    # 4) Extract strongly correlated pairs
    # =========================
    strong_pairs = []
    cols = corr_matrix.columns

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            r = float(corr_matrix.iloc[i, j])
            if abs(r) >= THRESHOLD_ABS_R:
                strong_pairs.append((cols[i], cols[j], r))

    strong_df = pd.DataFrame(strong_pairs, columns=["Var1", "Var2", "Correlation"])
    strong_df = strong_df.sort_values(by="Correlation", key=np.abs, ascending=False)

    print(f"\nStrongly correlated pairs (|r| >= {THRESHOLD_ABS_R:.2f}):")
    if len(strong_df) == 0:
        print("  (none)")
    else:
        print(strong_df)

    strong_df.to_csv(OUT_STRONG_PAIRS, index=False)

    print("\nSaved files:")
    print(f" - {OUT_CORR_MATRIX}")
    print(f" - {OUT_HEATMAP_PNG}")
    print(f" - {OUT_STRONG_PAIRS}")


if __name__ == "__main__":
    main()