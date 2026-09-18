# fp_error_analysis.py - standalone false-positive error analysis for the
# best TF-IDF SVM model at a given (dataset, fragment_len, retention_pos,
# encryption_key, k, n). double_cross_validation.py only saves aggregate
# TN/FP/FN/TP counts, not per-read predictions, so this replays the outer
# GroupKFold(5) split and refits each fold with the hyperparameters already
# found and saved there (no need to redo the inner GridSearchCV) to recover
# which reads are false positives, then reports how the top-N SHAP k-mers
# (from shap_analysis.py's saved output) show up across TN/FP/FN/TP.

import argparse
import json
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.svm import SVC

from utils import merge_datasets, get_groups, k_mers_sparse_matrix, tf_idf_k_mers_scores, top_kmers

#------------------------------- CLI ----------------------------


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ecoli")
    parser.add_argument("--algo", type=str, default="svm")
    parser.add_argument("--fragment_len", type=int, default=5)
    parser.add_argument("--retention_pos", type=int, default=5)
    parser.add_argument("--encryption_key", type=int, default=0)
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--top_n_shap", type=int, default=20)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--out_dir", type=str, default="/home/cosimo/Desktop/PhD/Cyberbiosecurity/DNA_attacks/experiments/results")
    return parser.parse_args()


#------------------------------------------------------------------------------


def main():

    args = parse_args()

    metrics_folder = os.path.join(
        args.out_dir,
        f"metrics/{args.algo}/{args.dataset}/fragment_len_{args.fragment_len}/retention_pos_{args.retention_pos}/encryption_key_{args.encryption_key}/"
    )
    shap_folder = os.path.join(metrics_folder, "shap", "All_split")
    out_folder = os.path.join(metrics_folder, "fp_error_analysis")
    os.makedirs(out_folder, exist_ok=True)

    tag = f"{args.algo}_{args.dataset}_fl{args.fragment_len}_rp{args.retention_pos}_ek{args.encryption_key}_k{args.k}_n{args.n}"

    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(str(msg))

    # --- Rebuild the exact same feature space used by the original run ---
    full_dataset, clean_dataset, infected_dataset = merge_datasets(
        dataset=args.dataset, fragment_len=args.fragment_len,
        retention_pos=args.retention_pos, encryption_key=args.encryption_key
    )
    groups = get_groups(clean_dataset)

    X, y, k_mers_list = k_mers_sparse_matrix(
        k=args.k, dataset_full=full_dataset, dataset_clean=clean_dataset, dataset_infected=infected_dataset
    )
    features_scores = tf_idf_k_mers_scores(X, k_mers_list)
    top_n_kmers = top_kmers(score_df=features_scores, n=args.n)
    X_reduced = X[:, top_n_kmers]
    top_kmers_list = [k_mers_list[i] for i in top_n_kmers]
    log(f"Feature matrix shape: {X_reduced.shape}")

    # --- Load the per-fold hyperparameters already found by double_cross_validation.py ---
    results_path = os.path.join(
        metrics_folder,
        f'results_{args.dataset}_dataset_length_20000_algo_{args.algo}_fl_{args.fragment_len}_rp_{args.retention_pos}_ek{args.encryption_key}.csv'
    )
    results_df = pd.read_csv(results_path)
    results_df['k'] = pd.to_numeric(results_df['k'], errors='coerce')
    results_df['n'] = pd.to_numeric(results_df['n'], errors='coerce')

    same_config = (results_df['k'] == args.k) & (results_df['n'] == args.n)
    fold_rows = results_df[same_config & (results_df['Outer_KFold'].astype(str) != 'total')]
    fold_params = {int(row['Outer_KFold']): json.loads(row['Model_params']) for _, row in fold_rows.iterrows()}

    total_row = results_df[same_config & (results_df['Outer_KFold'].astype(str) == 'total')].iloc[0]
    known_fp = int(total_row['FP'])

    # --- Replay the outer CV split, refit with the already-known hyperparameters, predict ---
    gkf = GroupKFold(n_splits=5)

    y_pred_full = np.full(len(y), -1, dtype=int)
    proba_full = np.full(len(y), np.nan)
    decision_full = np.full(len(y), np.nan)
    fold_of_read = np.full(len(y), -1, dtype=int)

    for split, (train_idx, test_idx) in enumerate(gkf.split(full_dataset, groups=groups)):
        hyperp = dict(fold_params[split])
        hyperp['probability'] = True
        hyperp['random_state'] = args.random_state

        model = SVC(**hyperp)
        model.fit(X_reduced[train_idx], y[train_idx])

        y_pred_full[test_idx] = model.predict(X_reduced[test_idx])
        proba_full[test_idx] = model.predict_proba(X_reduced[test_idx])[:, 1]
        decision_full[test_idx] = model.decision_function(X_reduced[test_idx])
        fold_of_read[test_idx] = split

        log(f"Fold {split}: refit with {fold_params[split]}")

    # --- Confusion groups ---
    tn_idx = np.where((y == 0) & (y_pred_full == 0))[0]
    fp_idx = np.where((y == 0) & (y_pred_full == 1))[0]
    fn_idx = np.where((y == 1) & (y_pred_full == 0))[0]
    tp_idx = np.where((y == 1) & (y_pred_full == 1))[0]

    log(f"Reconstructed confusion counts: TN={len(tn_idx)} FP={len(fp_idx)} FN={len(fn_idx)} TP={len(tp_idx)}")
    log(f"Known aggregate FP from double_cross_validation.py results: {known_fp}")
    if len(fp_idx) != known_fp:
        log(f"WARNING: reconstructed FP count ({len(fp_idx)}) does not match the known aggregate ({known_fp}) - predictions may not be exactly reproduced.")
    else:
        log("Reconstructed FP count matches the known aggregate exactly.")

    # --- Load top-N SHAP k-mers: magnitude (mean_abs_shap) + direction (mean signed shap) ---
    summary_path = os.path.join(shap_folder, f"shap_kmer_importance_SUMMARY_{tag}.csv")
    per_read_path = os.path.join(shap_folder, f"shap_per_read_ALL_{tag}.csv")

    shap_summary = pd.read_csv(summary_path)
    topN = shap_summary.sort_values('mean', ascending=False).head(args.top_n_shap).copy()
    topN = topN.rename(columns={'mean': 'mean_abs_shap'})

    per_read_shap = pd.read_csv(per_read_path)
    signed = per_read_shap.groupby('k_mer')['shap_value'].mean().rename('mean_signed_shap')
    topN = topN.merge(signed, on='k_mer', how='left')

    # --- Presence/frequency of each top-N k-mer across TN/FP/FN/TP ---
    kmer_col_idx = {kmer: i for i, kmer in enumerate(top_kmers_list)}
    X_csc = X_reduced.tocsc()
    confusion_groups = {'TN': tn_idx, 'FP': fp_idx, 'FN': fn_idx, 'TP': tp_idx}

    stats_rows = []
    for _, row in topN.iterrows():
        kmer = row['k_mer']
        col = kmer_col_idx.get(kmer)
        stat_row = {
            'k_mer': kmer,
            'mean_abs_shap': row['mean_abs_shap'],
            'mean_signed_shap': row.get('mean_signed_shap', np.nan),
        }
        if col is None:
            log(f"WARNING: top-SHAP k-mer '{kmer}' not found among the top-{args.n} TF-IDF features - skipping presence stats.")
        else:
            col_values = X_csc[:, col].toarray().ravel()
            for group_name, idx in confusion_groups.items():
                group_counts = col_values[idx]
                n_group = len(idx)
                stat_row[f'{group_name}_n_reads'] = n_group
                stat_row[f'{group_name}_presence_count'] = int((group_counts > 0).sum())
                stat_row[f'{group_name}_presence_rate'] = float((group_counts > 0).mean()) if n_group else np.nan
                stat_row[f'{group_name}_total_count'] = int(group_counts.sum())
                stat_row[f'{group_name}_mean_count'] = float(group_counts.mean()) if n_group else np.nan
        stats_rows.append(stat_row)

    kmer_stats_df = pd.DataFrame(stats_rows)
    kmer_stats_path = os.path.join(out_folder, f"fp_top{args.top_n_shap}_kmer_stats_{tag}.csv")
    kmer_stats_df.to_csv(kmer_stats_path, index=False)
    log(f"Saved top-{args.top_n_shap} SHAP k-mer stats across TN/FP/FN/TP to: {kmer_stats_path}")

    # --- Per-FP-read detail: original sequence, prediction, confidence scores, full feature vector ---
    fp_vectors = X_reduced[fp_idx].toarray()
    fp_detail_df = pd.DataFrame(fp_vectors, columns=[f'kmer__{kmer}' for kmer in top_kmers_list])
    fp_detail_df.insert(0, 'read_id', fp_idx)
    fp_detail_df.insert(1, 'sequence', [full_dataset[i] for i in fp_idx])
    fp_detail_df.insert(2, 'true_label', y[fp_idx])
    fp_detail_df.insert(3, 'predicted_label', y_pred_full[fp_idx])
    fp_detail_df.insert(4, 'predicted_probability_positive', proba_full[fp_idx])
    fp_detail_df.insert(5, 'decision_function_score', decision_full[fp_idx])
    fp_detail_df.insert(6, 'outer_fold', fold_of_read[fp_idx])

    fp_detail_path = os.path.join(out_folder, f"fp_reads_detail_{tag}.csv")
    fp_detail_df.to_csv(fp_detail_path, index=False)
    log(f"Saved {len(fp_idx)} FP reads (sequence, prediction, confidence scores, top-{args.n} k-mer vector) to: {fp_detail_path}")

    log_path = os.path.join(out_folder, f"fp_error_analysis_{tag}.log")
    with open(log_path, 'w') as f:
        f.write("\n".join(log_lines) + "\n")
    print(f"Saved run log to: {log_path}")


if __name__ == "__main__":
    main()
