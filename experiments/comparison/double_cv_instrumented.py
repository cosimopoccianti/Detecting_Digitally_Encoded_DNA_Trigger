# Instrumented double cross-validation: same outer GroupKFold(5) / inner
# GroupKFold(4)+GridSearchCV structure as ../double_cross_validation.py, plus
# per-fold train/predict timing and peak-RSS (via utils.PeakRSSSampler), and
# summed/maxed totals on the aggregate row. Kept separate instead of editing
# double_cross_validation.py directly.
#
# algo may be a config.REGISTRY key (TF-IDF path: estimator/param_grid looked
# up as usual) or a free label with an explicit estimator/param_grid override
# (spectrum kernel / alignment-free baselines).

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from config import REGISTRY
from utils import PeakRSSSampler, _safe_div
import json
import pandas as pd
import time
import numpy as np


def double_cross_validation_instrumented(full_dataset, groups, X, y, algo, k, n, k_mer_list,
                                           estimator=None, param_grid=None,
                                           outer_folds_number=5, inner_folds_number=4):

    start_time = time.time()
    fold_time = start_time

    metrics_columns = ["model", "k", "n", "Outer_KFold", "Model_params", "TN", "FP", "FN", "TP",
                        "Accuracy", "Precision", "Recall", "F1", "Runtime", "Selected_k_mers",
                        "Train_time_s", "Train_peak_rss_kb", "Predict_time_s", "Predict_peak_rss_kb"]

    metrics_df = pd.DataFrame(columns=metrics_columns)

    if n <= X.shape[1]:

        entry = REGISTRY.get(algo, {})
        estimator_cls = estimator if estimator is not None else entry['estimator']
        grid = param_grid if param_grid is not None else entry['param_grid']
        preprocess = entry.get("preprocess", None)

        gkf = GroupKFold(n_splits=outer_folds_number)

        for split, (train_idx, test_idx) in enumerate(gkf.split(full_dataset, groups=groups)):

            print(f'Fitting split number: {split}')

            X_train, X_test = X[train_idx, :], X[test_idx, :]
            y_train, y_test = y[train_idx], y[test_idx]

            if preprocess is not None:
                X_train = preprocess(X_train)
                X_test = preprocess(X_test)

            train_groups = groups[train_idx]

            model = estimator_cls()

            inner_gkf = GroupKFold(n_splits=inner_folds_number)

            grid_search = GridSearchCV(estimator=model, param_grid=grid, cv=inner_gkf, scoring='f1', verbose=1, n_jobs=12)

            grid_search.fit(X_train, y_train, groups=train_groups)

            best_model = grid_search.best_estimator_

            print("\n... Refitting on the total training set ...")

            train_start = time.perf_counter()
            with PeakRSSSampler() as rss_sampler:
                best_model.fit(X_train, y_train)
            train_time = time.perf_counter() - train_start
            train_peak_rss_kb = rss_sampler.peak_kb

            predict_start = time.perf_counter()
            with PeakRSSSampler() as rss_sampler:
                y_pred = best_model.predict(X_test)
            predict_time = time.perf_counter() - predict_start
            predict_peak_rss_kb = rss_sampler.peak_kb

            f1 = f1_score(y_test, y_pred, average='binary', zero_division=np.nan)
            print(f'F1 for {split} split')
            print('>f1=%.4f' % (f1))
            print(f'Train time: {train_time:.4f}s | Predict time: {predict_time:.4f}s')

            TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()

            best_params = grid_search.best_params_
            params_dict = json.dumps(best_params)

            print(f"Best parameter: {params_dict}")

            fold_elapsed_minutes = round((time.time() - fold_time) / 60, 2)
            fold_time = time.time()

            metrics = [algo, k, n, split, params_dict, TN, FP, FN, TP,
                       accuracy_score(y_test, y_pred),
                       precision_score(y_test, y_pred, average='binary', zero_division=np.nan),
                       recall_score(y_test, y_pred, average='binary', zero_division=np.nan),
                       f1_score(y_test, y_pred, average='binary', zero_division=np.nan),
                       fold_elapsed_minutes, k_mer_list,
                       train_time, train_peak_rss_kb, predict_time, predict_peak_rss_kb]

            metrics_row = pd.DataFrame([metrics], columns=metrics_columns)
            metrics_df = pd.concat([metrics_df, metrics_row])
            print("\n")

        all_fold = metrics_df[metrics_df['k'] == k]
        TN = all_fold['TN'].sum()
        FP = all_fold['FP'].sum()
        FN = all_fold['FN'].sum()
        TP = all_fold['TP'].sum()
        accuracy = _safe_div(TN + TP, TN + TP + FN + FP, zero_value=np.nan)
        precision = _safe_div(TP, TP + FP, zero_value=np.nan)
        recall = _safe_div(TP, TP + FN, zero_value=np.nan)
        f1 = _safe_div(2 * (precision * recall), precision + recall, zero_value=np.nan)

        print('Average F1: %.3f' % f1)

        end_time = time.time()
        elapsed_minutes = round((end_time - start_time) / 60, 2)

        # Time totals sum across folds (folds run sequentially); peak-RSS
        # totals take the max across folds (peaks don't add).
        total_train_time_s = all_fold['Train_time_s'].sum()
        total_predict_time_s = all_fold['Predict_time_s'].sum()
        total_train_peak_rss_kb = all_fold['Train_peak_rss_kb'].max()
        total_predict_peak_rss_kb = all_fold['Predict_peak_rss_kb'].max()

        metrics = [algo, k, n, 'total', None, TN, FP, FN, TP,
                   accuracy, precision, recall, f1, elapsed_minutes, k_mer_list,
                   total_train_time_s, total_train_peak_rss_kb,
                   total_predict_time_s, total_predict_peak_rss_kb]

        metrics_row = pd.DataFrame([metrics], columns=metrics_columns)
        metrics_df = pd.concat([metrics_df, metrics_row])

        print(f"\nTime for k = {k}, n = {n}: {elapsed_minutes:.2f} minutes")
        print("\n")

    return metrics_df
