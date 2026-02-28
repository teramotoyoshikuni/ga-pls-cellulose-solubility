# -*- coding: utf-8 -*-
"""
y_randomization_loocv.py

Y-randomization test for PLS regression with LOOCV component selection.

Pipeline:
- Autoscaling (X and y)
- Component selection by LOOCV (maximize R2cv on original y-scale)
- Y-permutation repeated N_PERM times
- Empirical p-value calculation
- Histogram visualization
- CSV export of detailed results + summary + metadata

Tested with:
- Python 3.9+
- scikit-learn 1.0+
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict, LeaveOneOut


# =========================
# User settings
# =========================
DATASET_CSV = "final_model_dataset.csv"
MAX_COMPONENTS_CAP = 10
N_PERM = 200
SEED0 = 1

OUTDIR = Path("outputs_y_randomization")
OUT_RESULTS = OUTDIR / "y_randomization_results.csv"
OUT_SUMMARY = OUTDIR / "y_randomization_summary_stats.csv"
OUT_META = OUTDIR / "y_randomization_meta.csv"
OUT_HIST = OUTDIR / "y_randomization_hist_R2cv.png"


# =========================
# Utility functions
# =========================
def ensure_outdir(outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)


def autoscale_X_y(X: pd.DataFrame, y: pd.Series):
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0, ddof=1).replace(0, np.nan)
    X_scaled = (X - X_mean) / X_std
    X_scaled = X_scaled.dropna(axis=1)

    y_mean = float(y.mean())
    y_std = float(y.std(ddof=1))
    if y_std == 0:
        raise ValueError("y has zero variance.")

    y_scaled = (y - y_mean) / y_std
    return X_scaled, y_scaled, y_mean, y_std


def compute_metrics(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))
    return float(r2), float(rmse), float(mae)


def select_best_components_loocv(X_scaled, y_scaled, y_original, y_mean, y_std, max_comp):
    cv = LeaveOneOut()
    best_r2 = -np.inf
    best_comp = 1
    best_rmse = None
    best_mae = None

    for n_comp in range(1, max_comp + 1):
        model = PLSRegression(n_components=n_comp)
        y_cv_scaled = cross_val_predict(model, X_scaled, y_scaled, cv=cv).ravel()
        y_cv = y_cv_scaled * y_std + y_mean
        r2, rmse, mae = compute_metrics(y_original, y_cv)

        if r2 > best_r2:
            best_r2 = r2
            best_comp = n_comp
            best_rmse = rmse
            best_mae = mae

    return best_comp, best_r2, best_rmse, best_mae


# =========================
# Main
# =========================
def main():
    ensure_outdir(OUTDIR)

    # ---- Load data
    dataset = pd.read_csv(DATASET_CSV, index_col=0)
    y = dataset.iloc[:, 0].astype(float)
    X = dataset.iloc[:, 1:].astype(float)

    X_scaled, y_scaled, y_mean, y_std = autoscale_X_y(X, y)

    max_comp = min(np.linalg.matrix_rank(X_scaled), MAX_COMPONENTS_CAP)
    if max_comp < 1:
        raise ValueError("max_comp < 1.")

    # ---- Observed model performance
    best_comp, best_r2cv, best_rmsecv, best_maecv = \
        select_best_components_loocv(
            X_scaled, y_scaled, y.values, y_mean, y_std, max_comp
        )

    print("Observed model performance:")
    print("Best components:", best_comp)
    print("R2cv:", round(best_r2cv, 3))
    print("RMSEcv:", round(best_rmsecv, 3))
    print("MAEcv:", round(best_maecv, 3))

    # ---- Y-randomization
    rng = np.random.default_rng(SEED0)
    perm_results = []

    for k in range(N_PERM):
        y_perm = rng.permutation(y_scaled.values)

        best_comp_p, best_r2cv_p, best_rmsecv_p, best_maecv_p = \
            select_best_components_loocv(
                X_scaled, y_perm, y.values, y_mean, y_std, max_comp
            )

        perm_results.append(
            [k + 1, best_comp_p, best_r2cv_p, best_rmsecv_p, best_maecv_p]
        )

    perm_df = pd.DataFrame(
        perm_results,
        columns=["perm_id", "best_comp", "R2cv", "RMSEcv", "MAEcv"]
    )
    perm_df.to_csv(OUT_RESULTS, index=False)

    # ---- Summary statistics
    summary_stats = perm_df[["R2cv", "RMSEcv", "MAEcv"]].describe()
    summary_stats.to_csv(OUT_SUMMARY)

    # ---- Empirical p-value
    count_ge = int(np.sum(perm_df["R2cv"].values >= best_r2cv))
    p_emp_raw = count_ge / N_PERM
    p_emp_add1 = (count_ge + 1) / (N_PERM + 1)
    p_emp_lower = max(1 / N_PERM, p_emp_raw)

    meta_df = pd.DataFrame([{
        "N_PERM": N_PERM,
        "SEED0": SEED0,
        "best_comp_obs": best_comp,
        "R2cv_obs": best_r2cv,
        "RMSEcv_obs": best_rmsecv,
        "MAEcv_obs": best_maecv,
        "count_perm_ge_obs": count_ge,
        "p_emp_raw": p_emp_raw,
        "p_emp_add1": p_emp_add1,
        "p_emp_lower_bounded_max_1_over_N": p_emp_lower,
    }])
    meta_df.to_csv(OUT_META, index=False)

    print("\nY-randomization summary:")
    print(summary_stats)
    print("\nEmpirical p-values:")
    print("Raw:", p_emp_raw)
    print("Add-one corrected:", p_emp_add1)
    print("Lower-bounded:", p_emp_lower)

    # ---- Histogram
    plt.figure()
    plt.hist(perm_df["R2cv"].values, bins=30)
    plt.axvline(best_r2cv, linestyle="--")
    plt.xlabel("R2cv (LOOCV) after y-permutation")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUT_HIST, dpi=300)
    plt.show()


if __name__ == "__main__":
    main()