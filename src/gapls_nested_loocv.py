# -*- coding: utf-8 -*-
"""
Nested LOOCV + GA-PLS variable selection (binary GA version)

- Outer CV: LOOCV
- Inner CV: K-fold (inner_fold_number) for GA fitness evaluation and component selection
- Autoscaling is done using training data only (to avoid leakage)
- GA individual: binary (0/1) bitstring (selected if bit==1)

Outputs:
  - nested_loocv_predictions.csv
  - variable_selection_frequency.csv
  - nested_loocv_scatter.png
"""

import random
import numpy as np
import pandas as pd
from deap import base, creator, tools
from sklearn import model_selection
from sklearn.cross_decomposition import PLSRegression
import matplotlib.pyplot as plt
import collections


# ===== Settings =====
number_of_population = 50
number_of_generation = 30
inner_fold_number = 5
max_number_of_components = 10
probability_of_crossover = 0.5
probability_of_mutation = 0.2

DATASET_CSV = "dataset_logS0.csv"

SEED_BASE = 100

# ===== Output filenames =====
OUT_PRED = "nested_loocv_predictions.csv"
OUT_FREQ = "variable_selection_frequency.csv"
OUT_SCATTER = "nested_loocv_scatter.png"


# =========================
# 1) GA-PLS function
# =========================
def run_gapls_and_select_variables(X_train_raw, y_train_raw, random_seed=None):

    X_mean = X_train_raw.mean(axis=0)
    X_std = X_train_raw.std(axis=0, ddof=1).replace(0, np.nan)
    X_train = (X_train_raw - X_mean) / X_std

    y_mean = y_train_raw.mean()
    y_std = y_train_raw.std(ddof=1)
    if y_std == 0:
        return []

    y_train = (y_train_raw - y_mean) / y_std

    X_train = X_train.dropna(axis=1)
    kept_columns = X_train.columns

    n_bits = X_train.shape[1]
    if n_bits == 0:
        return []

    if "FitnessMax" not in creator.__dict__:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    if "Individual" not in creator.__dict__:
        creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    def create_ind_bitstring():
        ind = [random.randint(0, 1) for _ in range(n_bits)]
        if sum(ind) == 0:
            ind[random.randrange(n_bits)] = 1
        return ind

    toolbox.register("individual", tools.initIterate, creator.Individual, create_ind_bitstring)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    def eval_individual(individual):

        individual_array = np.array(individual, dtype=int)
        selected_idx = np.where(individual_array == 1)[0]

        if len(selected_idx) == 0:
            return (-999.0,)

        X_sel = X_train.iloc[:, selected_idx]

        max_comp = min(np.linalg.matrix_rank(X_sel), max_number_of_components)
        if max_comp < 1:
            return (-999.0,)

        r2_cv_all = []
        for n_comp in range(1, max_comp + 1):
            model = PLSRegression(n_components=n_comp)

            y_cv_scaled = np.ndarray.flatten(
                model_selection.cross_val_predict(
                    model, X_sel, y_train, cv=inner_fold_number
                )
            )

            y_cv = y_cv_scaled * y_std + y_mean

            denom = np.sum((y_train_raw - y_mean) ** 2)
            r2 = 1 - np.sum((y_train_raw - y_cv) ** 2) / denom if denom != 0 else -999.0
            r2_cv_all.append(r2)

        return (float(np.max(r2_cv_all)),)

    toolbox.register("evaluate", eval_individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=1.0 / n_bits)
    toolbox.register("select", tools.selTournament, tournsize=3)

    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    pop = toolbox.population(n=number_of_population)

    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    for _ in range(number_of_generation):

        offspring = toolbox.select(pop, len(pop))
        offspring = list(map(toolbox.clone, offspring))

        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < probability_of_crossover:
                toolbox.mate(c1, c2)
                del c1.fitness.values
                del c2.fitness.values

        for mutant in offspring:
            if random.random() < probability_of_mutation:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid)
        for ind, fit in zip(invalid, fitnesses):
            ind.fitness.values = fit

        pop[:] = offspring

    best = tools.selBest(pop, 1)[0]
    best_array = np.array(best, dtype=int)
    selected_idx = np.where(best_array == 1)[0]

    if len(selected_idx) == 0:
        return []

    selected_colnames = kept_columns[selected_idx]
    selected_original_indices = [X_train_raw.columns.get_loc(c) for c in selected_colnames]

    return selected_original_indices


# =========================
# 2) Main execution block
# =========================
if __name__ == "__main__":

    dataset = pd.read_csv(DATASET_CSV, index_col=0)
    y_all = dataset.iloc[:, 0].astype(float)
    X_all = dataset.iloc[:, 1:].astype(float)

    n = len(y_all)
    y_true = np.zeros(n)
    y_pred = np.zeros(n)
    selected_features_each_fold = []

    for test_idx in range(n):

        train_idx = [i for i in range(n) if i != test_idx]

        X_train_raw = X_all.iloc[train_idx, :]
        y_train_raw = y_all.iloc[train_idx]
        X_test_raw = X_all.iloc[[test_idx], :]
        y_test = float(y_all.iloc[test_idx])

        selected_vars = run_gapls_and_select_variables(
            X_train_raw, y_train_raw, random_seed=SEED_BASE + test_idx
        )
        selected_features_each_fold.append(selected_vars)

        X_mean = X_train_raw.mean(axis=0)
        X_std = X_train_raw.std(axis=0, ddof=1).replace(0, np.nan)
        y_mean = y_train_raw.mean()
        y_std = y_train_raw.std(ddof=1)

        if y_std == 0 or len(selected_vars) == 0:
            y_true[test_idx] = y_test
            y_pred[test_idx] = y_mean
            continue

        X_train = (X_train_raw - X_mean) / X_std
        X_test = (X_test_raw - X_mean) / X_std

        X_train_sel = X_train.iloc[:, selected_vars].dropna(axis=1)
        if X_train_sel.shape[1] == 0:
            y_true[test_idx] = y_test
            y_pred[test_idx] = y_mean
            continue

        X_test_sel = X_test[X_train_sel.columns]
        y_train = (y_train_raw - y_mean) / y_std

        max_comp = min(np.linalg.matrix_rank(X_train_sel), max_number_of_components)
        if max_comp < 1:
            y_true[test_idx] = y_test
            y_pred[test_idx] = y_mean
            continue

        best_r2 = -np.inf
        best_comp = 1
        for n_comp in range(1, max_comp + 1):
            model = PLSRegression(n_components=n_comp)
            y_cv_scaled = np.ndarray.flatten(
                model_selection.cross_val_predict(model, X_train_sel, y_train, cv=inner_fold_number)
            )
            y_cv = y_cv_scaled * y_std + y_mean
            denom = np.sum((y_train_raw - y_mean) ** 2)
            r2 = 1 - np.sum((y_train_raw - y_cv) ** 2) / denom if denom != 0 else -np.inf
            if r2 > best_r2:
                best_r2 = r2
                best_comp = n_comp

        final_model = PLSRegression(n_components=best_comp)
        final_model.fit(X_train_sel, y_train)
        y_hat_scaled = float(final_model.predict(X_test_sel).ravel()[0])
        y_hat = y_hat_scaled * y_std + y_mean

        y_true[test_idx] = y_test
        y_pred[test_idx] = y_hat

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    q2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    mae = np.mean(np.abs(y_true - y_pred))

    print("Nested LOOCV results (binary GA)")
    print(f"Q2 (R2cv) = {q2:.3f}")
    print(f"RMSEcv    = {rmse:.3f}")
    print(f"MAEcv     = {mae:.3f}")

    pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(OUT_PRED, index=False)

    plt.figure()
    plt.scatter(y_true, y_pred)
    mn = min(y_true.min(), y_pred.min())
    mx = max(y_true.max(), y_pred.max())
    plt.plot([mn, mx], [mn, mx], "k-")
    plt.xlabel("Actual y")
    plt.ylabel("Predicted y (nested LOOCV)")
    plt.tight_layout()
    plt.savefig(OUT_SCATTER, dpi=300)
    plt.close()

    all_selected = [idx for fold in selected_features_each_fold for idx in fold]
    freq_counter = collections.Counter(all_selected)

    feature_names = X_all.columns
    freq_table = [(feature_names[idx], count) for idx, count in freq_counter.items()]
    freq_df = pd.DataFrame(freq_table, columns=["Feature", "Selection_Frequency"])
    freq_df = freq_df.sort_values(by="Selection_Frequency", ascending=False)

    freq_df.to_csv(OUT_FREQ, index=False)

    print("\nSaved files:")
    print(f" - {OUT_PRED}")
    print(f" - {OUT_FREQ}")
    print(f" - {OUT_SCATTER}")