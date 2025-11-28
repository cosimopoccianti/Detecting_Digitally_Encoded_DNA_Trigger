from utils_dataset import join_datasets,k_mers_sparse_matrix
import numpy as np
import pandas as pd
from pathlib import Path
import os


def shannon_entropy_k(k_list, full, clean, trojan):

    print("\nStarting Shannon Entropy calculations...\n")

    E_c_l = []

    X_l = []

    for k in k_list:

        X,y,_= k_mers_sparse_matrix(k=k,dataset_full=full,dataset_clean=clean,dataset_trojan=trojan)

        C_j = X.sum(axis=0)

        C = C_j.sum()

        M = C_j.shape[1]

        S = 0
        for j in range(0,M):
            S += C_j[0,j]*np.log(C_j[0,j]/C)

        E_c = -(1/C)*S

        E_c_l.append(E_c)

        X_l.append(X)
    
    return E_c_l, X_l


def pi_k(k_list,X_l):

    print("\nComputing pi for all k...")
    pi = []
    for mat,k in zip(X_l,k_list):
        pi.append(mat.shape[1]/(5**k))

    return pi

def summary_df(S_d,pi_d,folder,pattern,f_l,r_p,e_k):

    folder = Path(folder)
    files = list(folder.glob(pattern))

    df = pd.read_csv(files[0])
    df_tot = df[df["Outer_KFold"] == "mean"].copy()
    df_tot.insert(0, "fragment_len", f_l)
    df_tot.insert(1, "retention_position", r_p)
    df_tot.insert(2, "encryption_key", e_k)

    df_tot["Shannon_entropy"] = df_tot["k"].map(S_d)
    df_tot["pi"] = df_tot["k"].map(pi_d)

    df_tot.drop(columns=["Outer_KFold","C","gamma","kernel"], inplace=True)

    return df_tot

def main(fragment_len, retention_position,encryption_key):

    k_list = [2,3,4]#[i for i in range(2,26)]
    outer_dfs = []

    for f_l in fragment_len:
        for r_p in retention_position:
            if r_p <= f_l + 1:
                for e_k in encryption_key:

                    full, clean, trojan = join_datasets(dataset='greedy', dataset_number=10, encryption_key=e_k,fragment_len=f_l,retention_pos=r_p)

                    E_c_l, X_l = shannon_entropy_k(k_list,full,clean,trojan)
                    pi = pi_k(k_list,X_l)
                    
                    S_d = dict(zip(k_list,E_c_l))
                    pi_d = dict(zip(k_list,pi))

                    folder = f"DNA_attacks/metrics/SVM/Balanced/fragment_len_{f_l}"
                    pattern = f"metrics_len_{f_l}_pos_{r_p}_key_{e_k}_double_cv(5,4)_dataset_len_*_greedy.csv"
                    
                    df_inner = summary_df(S_d=S_d, pi_d=pi_d,folder=folder, pattern=pattern, f_l=f_l, r_p=r_p,e_k=e_k)
                    outer_dfs.append(df_inner)

    df_outer = pd.concat(outer_dfs, ignore_index=True)
    df_outer.to_csv(f"DNA_attacks/metrics/SVM/Balanced/metrics_summary_double_cv_entropy_pi.csv", index=False)

                    
if __name__ == "__main__":

    fragment_len = [5]
    retention_position = [0,1,2]
    encryption_key = [0,10]

    main(fragment_len=fragment_len,retention_position=retention_position,encryption_key=encryption_key)

