# -*- coding: utf-8 -*-
"""
final_pls_loocv_vip.py

PLS regression on a pre-selected (stable) descriptor set with:
- Autoscaling (X and y)
- Component selection by LOOCV (maximize R2cv on original y-scale)
- LOOCV predictions at best component number
- Final model fit on all data (for VIP and standardized regression coefficients)
- Exports: predictions, scatter plot, VIP table, model summary

Tested with:
- Python 3.9.12
- scikit-learn 1.0.2
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
DATASET_CSV = "final_model_dataset.csv"   # 1st col: y, remaining cols: X (descriptors)

MAX_COMPONENTS_CAP = 10                  # max PLS components to test (capped by matrix rank)

# Output directory and files
OUTDIR = Path("outputs_pls")
OUT_CV_PRED = OUTDIR / "cv_predictions_bestcomp_loocv.csv"
OUT_CV_SCATTER_PNG = OUTDIR / "actual_vs_cv_predicted_loocv.png"
OUT_VIP = OUTDIR / "final_model_vip_and_coefficients.csv"
OUT_SUMMARY = OUTDIR / "model_summary.csv"
OUT_FINAL_FIT_PRED = OUTDIR / "final_fit_predictions_insample.csv"


# =========================
# Utility functions
# =========================
def autoscale_X_y(X: pd.DataFrame, y: pd.Series):
    """Autoscale X and y using mean/std (ddof=1). Drops zero-variance X columns."""
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0, ddof=1).replace(0, np.nan)
    X_scaled = (X - X_mean) / X_std
    X_scaled = X_scaled.dropna(axis=1)  # drop zero-variance columns

    used_features = X_scaled.columns.tolist()

    y_mean = float(y.mean())
    y_std = float(y.std(ddof=1))
    if y_std == 0:
        raise ValueError("y has zero variance; cannot scale.")

    y_scaled = (y - y_mean) / y_std

    return X_scaled, y_scaled, used_features, y_mean, y_std


def r2_rmse_mae(y_true: np.ndarray, y_pred: np.ndarray):
    """Metrics on original scale."""
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    return float(r2), rmse, mae


def calculate_vip(pls_model: PLSRegression) -> np.ndarray:
    """
    Compute VIP scores for PLSRegression.
    Reference formula widely used in chemometrics.

    Returns:
      vip: shape (n_features,)
    """
    T = pls_model.x_scores_          # (n_samples, n_comp)
    W = pls_model.x_weights_         # (n_features, n_comp)
    Q = pls_model.y_loadings_        # (n_targets, n_comp) in sklearn (usually (1, n_comp))

    p, h = W.shape

    # Explained sum of squares of Y for each component:
    # S_h = (t_h^T t_h) * (q_h^T q_h)
    # Here Q is (1, h), so q_h^T q_h is scalar per component.
    S = np.diag(T.T @ T @ Q.T @ Q)   # (h,)

    total_S = np.sum(S)
    if total_S == 0:
        return np.zeros(p, dtype=float)

    vip = np.zeros(p, dtype=float)
    for i in range(p):
        vip[i] = np.sqrt(p * np.sum((W[i, :] ** 2) * S) / total_S)

    return vip


def ensure_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)


# =========================
# Main
# =========================
def main():
    ensure_outdir(OUTDIR)

    # ---- Load data
    dataset = pd.read_csv(DATASET_CSV, index_col=0)
    y = dataset.iloc[:, 0].astype(float)
    X = dataset.iloc[:, 1:].astype(float)

    # ---- Autoscaling
    X_scaled, y_scaled, used_features, y_mean, y_std = autoscale_X_y(X, y)

    # ---- LOOCV component selection
    cv = LeaveOneOut()
    max_comp = min(np.linalg.matrix_rank(X_scaled), MAX_COMPONENTS_CAP)
    if max_comp < 1:
        raise ValueError("max_comp < 1 (check X after autoscaling / dropping zero-variance columns).")

    best_r2cv = -np.inf
    best_comp = 1
    best_rmsecv = None
    best_maecv = None

    for n_comp in range(1, max_comp + 1):
        model = PLSRegression(n_components=n_comp)

        # cross_val_predict gives predictions for each left-out sample (on scaled y)
        y_cv_scaled = cross_val_predict(model, X_scaled, y_scaled, cv=cv).ravel()

        # back-transform to original y scale
        y_cv = y_cv_scaled * y_std + y_mean

        r2cv, rmsecv, maecv = r2_rmse_mae(y.values, y_cv)

        if r2cv > best_r2cv:
            best_r2cv = float(r2cv)
            best_comp = int(n_comp)
            best_rmsecv = float(rmsecv)
            best_maecv = float(maecv)

    print("Best components (LOOCV):", best_comp)
    print("R2cv:", round(best_r2cv, 3))
    print("RMSEcv:", round(best_rmsecv, 3))
    print("MAEcv:", round(best_maecv, 3))

    # ---- LOOCV predictions at best_comp (export + plot)
    best_model_for_cv = PLSRegression(n_components=best_comp)
    y_cv_scaled_best = cross_val_predict(best_model_for_cv, X_scaled, y_scaled, cv=cv).ravel()
    y_cv_best = y_cv_scaled_best * y_std + y_mean

    cv_pred_df = pd.DataFrame({
        "Sample": dataset.index.astype(str),
        "y_true": y.values,
        "y_cv_pred": y_cv_best
    })
    cv_pred_df.to_csv(OUT_CV_PRED, index=False)

    plt.figure()
    plt.scatter(y.values, y_cv_best)
    mn = min(float(y.values.min()), float(y_cv_best.min()))
    mx = max(float(y.values.max()), float(y_cv_best.max()))
    plt.plot([mn, mx], [mn, mx], "k-")
    plt.xlabel("Actual y")
    plt.ylabel(f"CV-predicted y (LOOCV, best_comp={best_comp})")
    plt.tight_layout()
    plt.savefig(OUT_CV_SCATTER_PNG, dpi=300)
    plt.show()

    # ---- Fit final model on all data for interpretation (VIP + coefficients)
    final_model = PLSRegression(n_components=best_comp)
    final_model.fit(X_scaled, y_scaled)

    vip_scores = calculate_vip(final_model)

    vip_df = pd.DataFrame({
        "Feature": used_features,
        "VIP": vip_scores,
        # NOTE: coef_ are in scaled space -> standardized regression coefficients
        "Coefficient": final_model.coef_.ravel()
    }).sort_values(by="VIP", ascending=False)

    print("\nVIP table (top 10):")
    print(vip_df.head(10))

    vip_df.to_csv(OUT_VIP, index=False)

    # ---- Optional: in-sample fit predictions (NOT CV)
    y_fit_scaled = final_model.predict(X_scaled).ravel()
    y_fit = y_fit_scaled * y_std + y_mean

    fit_df = pd.DataFrame({
        "Sample": dataset.index.astype(str),
        "y_true": y.values,
        "y_fit_pred": y_fit
    })
    fit_df.to_csv(OUT_FINAL_FIT_PRED, index=False)

    # ---- Save summary
    summary_df = pd.DataFrame({
        "n_samples": [int(len(y))],
        "n_features_used": [int(len(used_features))],
        "cv_type": ["LOOCV"],
        "best_components": [int(best_comp)],
        "R2cv": [float(best_r2cv)],
        "RMSEcv": [float(best_rmsecv)],
        "MAEcv": [float(best_maecv)]
    })
    summary_df.to_csv(OUT_SUMMARY, index=False)

    print("\nSaved files:")
    print(" -", OUT_CV_PRED)
    print(" -", OUT_CV_SCATTER_PNG)
    print(" -", OUT_VIP)
    print(" -", OUT_SUMMARY)
    print(" -", OUT_FINAL_FIT_PRED)


if __name__ == "__main__":
    main()