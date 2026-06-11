# shap_analysis.py
import os
import json
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score
from joblib import Parallel, delayed
import shap

from config import REGISTRY
from utils import merge_datasets, get_groups, k_mers_sparse_matrix, tf_idf_k_mers_scores, top_kmers


def _process_split(split, train_idx, test_idx, full_dataset, X_reduced, y, groups,
                   algo, dataset, f_l, r_p, e_k, k, n, top_kmers_list, hyperp,
                   shap_folder, random_state, background_size, n_explain, nsamples,
                   positive_class):
    """Esegue fit + SHAP + salvataggi per UN singolo split.
    Restituisce (importance_df, per_read_df, shap_values, test_subset) per gli aggregati."""

    print(f'Fitting split number: {split}')
    split_start = time.time()

    X_train, X_test = X_reduced[train_idx, :], X_reduced[test_idx, :]
    y_train, y_test = y[train_idx], y[test_idx]

    entry = REGISTRY[algo]
    preprocess = entry.get("preprocess", None)
    if preprocess is not None:
        X_train = preprocess(X_train)
        X_test = preprocess(X_test)

    estimator_cls = entry["estimator"]
    model = estimator_cls(**hyperp)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    print(f"Split {split} F1: {f1:.4f}")

    # --- SHAP
    # summarize background with kmeans to cut runtime
    background = shap.kmeans(X_train, background_size)

    # random, traceable explained subset
    n_explain_split = min(n_explain, X_test.shape[0])
    rng_test = np.random.RandomState(random_state)
    explain_idx = rng_test.choice(X_test.shape[0], n_explain_split, replace=False)
    test_subset = X_test[explain_idx].toarray()
    explained_original_idx = test_idx[explain_idx]  # map back to original dataset rows

    explainer = shap.KernelExplainer(model.predict_proba, background)
    shap_values_all = explainer.shap_values(test_subset, nsamples=nsamples)

    # binary classification -> take positive class
    if isinstance(shap_values_all, list):
        shap_values = shap_values_all[positive_class]
    else:
        shap_values = shap_values_all[:, :, positive_class]

    ev = explainer.expected_value
    expected_value = ev[positive_class] if np.ndim(ev) > 0 else ev

    print(f"Split {split} SHAP runtime: {time.time() - split_start:.1f}s")

    # --- Save raw SHAP values + base value + traceability ---
    np.savez_compressed(
        os.path.join(shap_folder,
            f"shap_values_{algo}_{dataset}_fl{f_l}_rp{r_p}_ek{e_k}_k{k}_n{n}_split{split}.npz"),
        shap_values=shap_values,
        test_subset=test_subset,
        feature_names=np.array(top_kmers_list),
        expected_value=expected_value,
        explained_original_idx=explained_original_idx,
        y_test_subset=y_test[explain_idx],
    )

    # --- Save summary plots (with split info) ---
    shap.summary_plot(shap_values, test_subset,
                      feature_names=top_kmers_list, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(shap_folder,
                f"shap_summary_beeswarm_{algo}_{dataset}_fl{f_l}_rp{r_p}_ek{e_k}_split{split}.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    shap.summary_plot(shap_values, test_subset, feature_names=top_kmers_list,
                      plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(shap_folder,
                f"shap_summary_bar_{algo}_{dataset}_fl{f_l}_rp{r_p}_ek{e_k}_split{split}.png"),
                dpi=300, bbox_inches="tight")
    plt.close()

    # --- Save kmer feature importances (mean |SHAP|, with split info) ---
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "split": split,
        "k_mer": top_kmers_list,
        "mean_abs_shap": mean_abs_shap
    }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)

    importance_path = os.path.join(
        shap_folder,
        f"shap_kmer_importance_{algo}_{dataset}_fl{f_l}_rp{r_p}_ek{e_k}_k{k}_n{n}_split{split}.csv"
    )
    importance_df.to_csv(importance_path, index=False)
    print(f"Saved SHAP importances to: {importance_path}")

    # --- Per-read SHAP values (long format) ---
    n_reads, n_feat = shap_values.shape
    per_read_df = pd.DataFrame({
        "split": split,
        "read_id": np.repeat(explained_original_idx, n_feat),
        "label": np.repeat(y_test[explain_idx], n_feat),
        "k_mer": np.tile(top_kmers_list, n_reads),
        "feature_value": test_subset.reshape(-1),
        "shap_value": shap_values.reshape(-1),
    })

    per_read_path = os.path.join(
        shap_folder,
        f"shap_per_read_{algo}_{dataset}_fl{f_l}_rp{r_p}_ek{e_k}_k{k}_n{n}_split{split}.csv"
    )
    per_read_df.to_csv(per_read_path, index=False)
    print(f"Saved per-read SHAP values to: {per_read_path}")

    return importance_df, per_read_df, shap_values, test_subset


def shap_analysis(algo, dataset, f_l, r_p, e_k, out_dir, random_state=42,
                  outer_folds_number=5, n_shap_folds=5, n_explain=400,
                  background_size=50, nsamples="auto", positive_class=1,
                  n_jobs=-1):

    metrics_folder = os.path.join(
        out_dir,
        f"metrics/{algo}/{dataset}/fragment_len_{f_l}/retention_pos_{r_p}/encryption_key_{e_k}/"
    )

    # SHAP output folder (images + kmer importances)
    shap_folder = os.path.join(metrics_folder, "shap/8giugno")
    os.makedirs(shap_folder, exist_ok=True)

    # --- Load cross-validation results to pick the best model ---
    result_df = pd.read_csv(
        os.path.join(
            metrics_folder,
            f'results_{dataset}_dataset_length_20000_algo_{algo}_fl_{f_l}_rp_{r_p}_ek{e_k}.csv'
        ),
        index_col=0
    )

    cross_model = result_df[result_df['Outer_KFold'] == 'total']
    best_cross_model = cross_model.sort_values(by='F1', ascending=False).iloc[0]
    k, n = best_cross_model['k'], best_cross_model['n']

    folds = result_df[
        (result_df['Outer_KFold'] != 'total') &
        (result_df['k'] == k) &
        (result_df['n'] == n)
    ]
    params_df = pd.DataFrame([json.loads(p) for p in folds['Model_params']])
    hyperp = {col: params_df[col].mode().iloc[0] for col in params_df.columns}

    if algo == 'svm':
        hyperp['probability'] = True

    # --- Rebuild dataset and feature matrix ---
    full_dataset, clean_dataset, infected_dataset = merge_datasets(
        dataset=dataset, fragment_len=f_l, retention_pos=r_p, encryption_key=e_k
    )
    groups = get_groups(clean_dataset)

    X, y, k_mers_list = k_mers_sparse_matrix(
        k=k, dataset_full=full_dataset,
        dataset_clean=clean_dataset, dataset_infected=infected_dataset
    )
    features_scores = tf_idf_k_mers_scores(X, k_mers_list)

    print(f"====== k: {k} - n: {n} - algo: {algo} ======")

    top_n_kmers = top_kmers(score_df=features_scores, n=n)
    X_reduced = X[:, top_n_kmers]
    top_kmers_list = [k_mers_list[i] for i in top_n_kmers]
    print(f"Selected top {n} k-mers by TF-IDF score")
    print(f"Feature matrix shape: {X_reduced.shape}")

    if n > X_reduced.shape[1]:
        print(f"Skipping: n ({n}) > available features ({X_reduced.shape[1]})")
        return None

    if n_shap_folds > outer_folds_number:
        print(f"Capping n_shap_folds ({n_shap_folds}) to outer_folds_number ({outer_folds_number})")
        n_shap_folds = outer_folds_number

    # --- Outer CV:
    gkf = GroupKFold(n_splits=outer_folds_number)
    splits = [(split, train_idx, test_idx)
              for split, (train_idx, test_idx) in enumerate(gkf.split(full_dataset, groups=groups))
              if split < n_shap_folds]

    # Parallel execution
    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_split)(
            split, train_idx, test_idx, full_dataset, X_reduced, y, groups,
            algo, dataset, f_l, r_p, e_k, k, n, top_kmers_list, hyperp,
            shap_folder, random_state, background_size, n_explain, nsamples,
            positive_class
        )
        for (split, train_idx, test_idx) in splits
    )
    
    results = sorted(results, key=lambda r: r[0]["split"].iloc[0])

    importance_dfs = [r[0] for r in results]
    per_read_dfs = [r[1] for r in results]
    shap_all_list = [r[2] for r in results]
    feat_all_list = [r[3] for r in results]

    if importance_dfs:
        combined = pd.concat(importance_dfs, ignore_index=True)
        summary = (combined.groupby("k_mer")["mean_abs_shap"]
                   .agg(["mean", "std", "count"])
                   .sort_values("mean", ascending=False)
                   .reset_index())
        summary.to_csv(os.path.join(shap_folder,
            f"shap_kmer_importance_SUMMARY_{algo}_{dataset}_fl{f_l}_rp{r_p}_ek{e_k}_k{k}_n{n}.csv"),
            index=False)
        print("Saved cross-fold SHAP importance summary")


        per_read_all = pd.concat(per_read_dfs, ignore_index=True)
        per_read_all_path = os.path.join(shap_folder,
            f"shap_per_read_ALL_{algo}_{dataset}_fl{f_l}_rp{r_p}_ek{e_k}_k{k}_n{n}.csv")
        per_read_all.to_csv(per_read_all_path, index=False)
        print(f"Saved combined per-read SHAP values (all splits) to: {per_read_all_path}")

        # --- Beeswarm 
        shap_all = np.vstack(shap_all_list)   
        feat_all = np.vstack(feat_all_list)

    
        np.savez_compressed(
            os.path.join(shap_folder,
                f"shap_values_ALL_{algo}_{dataset}_fl{f_l}_rp{r_p}_ek{e_k}_k{k}_n{n}.npz"),
            shap_values=shap_all,
            test_subset=feat_all,
            feature_names=np.array(top_kmers_list),
        )

        shap.summary_plot(shap_all, feat_all, feature_names=top_kmers_list,
                          max_display=20, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(shap_folder,
                    f"shap_summary_beeswarm_ALL_{algo}_{dataset}_fl{f_l}_rp{r_p}_ek{e_k}_k{k}_n{n}.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()

        shap.summary_plot(shap_all, feat_all, feature_names=top_kmers_list,
                          plot_type="bar", max_display=20, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(shap_folder,
                    f"shap_summary_bar_ALL_{algo}_{dataset}_fl{f_l}_rp{r_p}_ek{e_k}_k{k}_n{n}.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Saved combined beeswarm/bar over {shap_all.shape[0]} reads from {len(shap_all_list)} splits")

        return combined

    return None