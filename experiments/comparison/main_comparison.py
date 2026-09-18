# main_comparison.py - double-CV comparison of TF-IDF+SVM (or other
# REGISTRY algo) against the spectrum kernel, alignment-free, and Islam et
# al. CNN baselines, mirroring ../main.py's CLI and loop structure.

import argparse
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from main import algo_type, fragment_len_type, retention_pos_type, encryption_key_type, n_feature_type
from utils import merge_datasets, k_mers_sparse_matrix, tf_idf_k_mers_scores, top_kmers, get_groups
from double_cv_instrumented import double_cross_validation_instrumented
from spectrum_kernel import spectrum_kernel_double_cv
from alignment_free import alignment_free_double_cv
from cnn_islam import cnn_islam_double_cv

#------------------------------- CLI ----------------------------


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", type=algo_type, default=["svm"], help="TF-IDF classifier; spectrum kernel / alignment-free / cnn are always SVM / fixed architecture")
    parser.add_argument("--dataset", type=str, default="ecoli", help="- 'ecoli' → E. coli reads dataset - 'lentivirus' → Lentivirus reads dataset")
    parser.add_argument("--k_min", type=int, default=3)
    parser.add_argument("--k_max", type=int, default=25)
    parser.add_argument("--fragment_len", type=fragment_len_type, default=[5])
    parser.add_argument("--retention_pos", type=retention_pos_type, default=[5])
    parser.add_argument("--encryption_key", type=encryption_key_type, default=[0])
    parser.add_argument("--n_features", type=n_feature_type, default=[*range(5, 50, 5), 50, 75, 100])
    parser.add_argument("--out_dir", type=str, default="/home/cosimo/Desktop/PhD/Cyberbiosecurity/DNA_attacks/experiments/results")
    args = parser.parse_args()

    return args


#------------------------------------------------------------------------------


def main():

    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    k_list = list(range(args.k_min, args.k_max + 1))

    for algo in args.algo:
        print(f"\n############ {algo} ############")
        for f_l in args.fragment_len:
            for r_p in args.retention_pos:
                if r_p <= f_l + 1:
                    for e_k in args.encryption_key:

                        comparison_folder = os.path.join(args.out_dir, f"comparison/{args.dataset}/fragment_len_{f_l}/retention_pos_{r_p}/encryption_key_{e_k}/")
                        os.makedirs(comparison_folder, exist_ok=True)

                        full_dataset, clean_dataset, infected_dataset = merge_datasets(dataset=args.dataset, fragment_len=f_l, retention_pos=r_p, encryption_key=e_k)

                        result_path = os.path.join(comparison_folder, f'comparison_{args.dataset}_algo_{algo}_fl_{f_l}_rp_{r_p}_ek{e_k}.csv')

                        # Create group assignments based on unique sequences (for Leave-One-Group-Out CV)
                        groups = get_groups(clean_dataset)

                        comparison_df = pd.DataFrame()

                        # --- Islam et al. 1D-CNN baseline: outer-only 5-fold LOGO, dedicated venv ---
                        cnn_df = cnn_islam_double_cv(groups=groups, dataset=args.dataset, f_l=f_l, r_p=r_p, e_k=e_k)
                        comparison_df = pd.concat([comparison_df, cnn_df])
                        comparison_df.to_csv(result_path, index=False)

                        for k in k_list:

                            # Create reads frequency matrix (shared by spectrum kernel, alignment-free, and TF-IDF)
                            X, y, k_mers_list = k_mers_sparse_matrix(k=k, dataset_full=full_dataset, dataset_clean=clean_dataset, dataset_infected=infected_dataset)

                            # --- Spectrum kernel SVM: full k-mer count matrix, linear kernel ---
                            print(f"====== k: {k} - algo: svm_spectrum ======")
                            spectrum_df = spectrum_kernel_double_cv(full_dataset=full_dataset, groups=groups, X=X, y=y, k=k, k_mer_list=k_mers_list)
                            comparison_df = pd.concat([comparison_df, spectrum_df])
                            comparison_df.to_csv(result_path, index=False)

                            # --- Alignment-free baseline: row-normalized k-mer frequency matrix, RBF kernel ---
                            print(f"====== k: {k} - algo: svm_alignment_free ======")
                            alignment_free_df = alignment_free_double_cv(full_dataset=full_dataset, groups=groups, X=X, y=y, k=k, k_mer_list=k_mers_list)
                            comparison_df = pd.concat([comparison_df, alignment_free_df])
                            comparison_df.to_csv(result_path, index=False)

                            # Compute k-mer TF-IDF scores from the frequency matrix
                            features_scores = tf_idf_k_mers_scores(X, k_mers_list)

                            for n in args.n_features:

                                print(f"====== k: {k} - n: {n} - algo: {algo} ======")

                                top_n_kmers = top_kmers(score_df=features_scores, n=n)
                                # Reduce X to top-n kmers based on TF-IDF
                                X_reduced = X[:, top_n_kmers]

                                # Reduce kmers_list consistently
                                top_kmers_list = [k_mers_list[i] for i in top_n_kmers]

                                n_df = double_cross_validation_instrumented(full_dataset=full_dataset, groups=groups, X=X_reduced, y=y, algo=algo, k=k, n=n, k_mer_list=top_kmers_list)
                                comparison_df = pd.concat([comparison_df, n_df])
                                comparison_df.to_csv(result_path, index=False)


if __name__ == "__main__":
    main()
