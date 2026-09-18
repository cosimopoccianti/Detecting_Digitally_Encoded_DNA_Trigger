# Islam et al. 1D-CNN, outer-only 5-fold LOGO double-CV comparison: runs
# cnn_islam_worker.py once per outer fold (architecture/hyperparameters fixed,
# no inner tuning - matches Islam et al.), in the dedicated Python 3.8 /
# TensorFlow 2.4.0 venv (CPU-only), then aggregates across folds the same way
# double_cv_instrumented.py aggregates its outer loop.

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import time
import numpy as np
import pandas as pd
import subprocess
from utils import _safe_div

VENV_PYTHON = '/home/cosimo/Desktop/PhD/Cyberbiosecurity/DNA_attacks/experiments/Islam/venv_cnn/bin/python'
WORKER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cnn_islam_worker.py')
WORK_DIR = '/home/cosimo/Desktop/PhD/Cyberbiosecurity/DNA_attacks/experiments/Islam'
BASE_DATASET_DIR = '/home/cosimo/Desktop/PhD/Cyberbiosecurity/DNA_attacks/experiments/datasets'


def cnn_islam_double_cv(groups, dataset, f_l, r_p, e_k, outer_folds_number=5):

    dataset_dir = os.path.join(BASE_DATASET_DIR, dataset, f"fragment_len_{f_l}", f"retention_pos_{r_p}", f"encryption_key_{e_k}", "dataset_tot")
    clean_path = os.path.join(dataset_dir, "non_trojan_dataset.txt")
    trojan_path = os.path.join(dataset_dir, "nw_best_greedy_trojan_insertion_dataset.txt")

    groups_path = os.path.join(WORK_DIR, "tmp_groups.npy")
    np.save(groups_path, groups)

    start_time = time.time()

    metrics_df = pd.DataFrame()

    for fold in range(outer_folds_number):

        print(f"====== algo: cnn_islam - fold {fold} ====== (running in dedicated venv, this can take a long time on CPU)")

        log_dir = os.path.join(WORK_DIR, "logs", f"fl{f_l}_rp{r_p}_ek{e_k}_fold{fold}")
        out_csv = os.path.join(WORK_DIR, "results", f"cnn_fl{f_l}_rp{r_p}_ek{e_k}_fold{fold}.csv")
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)

        cmd = [VENV_PYTHON, WORKER_SCRIPT,
               "--clean", clean_path, "--trojan", trojan_path,
               "--groups", groups_path, "--test_fold", str(fold),
               "--log_dir", log_dir, "--out_csv", out_csv]

        subprocess.run(cmd, check=True)

        fold_df = pd.read_csv(out_csv)
        metrics_df = pd.concat([metrics_df, fold_df])

    TN = metrics_df['TN'].sum()
    FP = metrics_df['FP'].sum()
    FN = metrics_df['FN'].sum()
    TP = metrics_df['TP'].sum()
    accuracy = _safe_div(TN + TP, TN + TP + FN + FP, zero_value=np.nan)
    precision = _safe_div(TP, TP + FP, zero_value=np.nan)
    recall = _safe_div(TP, TP + FN, zero_value=np.nan)
    f1 = _safe_div(2 * (precision * recall), precision + recall, zero_value=np.nan)

    elapsed_minutes = round((time.time() - start_time) / 60, 2)

    # Time totals sum across folds (folds run sequentially); peak-RSS totals
    # take the max across folds (peaks don't add).
    total_row = metrics_df.iloc[[0]].copy()
    total_row['Outer_KFold'] = 'total'
    total_row['Model_params'] = None
    total_row['TN'], total_row['FP'], total_row['FN'], total_row['TP'] = TN, FP, FN, TP
    total_row['Accuracy'], total_row['Precision'], total_row['Recall'], total_row['F1'] = accuracy, precision, recall, f1
    total_row['Runtime'] = elapsed_minutes
    total_row['Train_time_s'] = metrics_df['Train_time_s'].sum()
    total_row['Predict_time_s'] = metrics_df['Predict_time_s'].sum()
    total_row['Train_peak_rss_kb'] = metrics_df['Train_peak_rss_kb'].max()
    total_row['Predict_peak_rss_kb'] = metrics_df['Predict_peak_rss_kb'].max()
    total_row['Vectorize_time_s'] = metrics_df['Vectorize_time_s'].sum()
    total_row['Vectorize_peak_rss_kb'] = metrics_df['Vectorize_peak_rss_kb'].max()
    total_row['Transform_time_s_total'] = metrics_df['Transform_time_s_total'].sum()
    total_row['Peak_rss_kb_total'] = metrics_df['Peak_rss_kb_total'].max()

    metrics_df = pd.concat([metrics_df, total_row])

    return metrics_df
