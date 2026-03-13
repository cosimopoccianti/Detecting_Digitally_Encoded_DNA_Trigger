import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import VarianceThreshold
from scipy.sparse import vstack
from sklearn.utils import shuffle
import os
from pathlib import Path
from collections import defaultdict

# Functions 'padding' and 'dataset_loading' are taken from Islam et al.'s repository: https://github.com/sibleeislam/trojan-malware-in-bio-cyber-attacks

def padding(seq_size_max, line):
    less = seq_size_max-len(line)
    padding_value = line[:less]
    line = line+padding_value
    if(len(line) < seq_size_max):
        line = padding(seq_size_max, line)
    return line


def dataset_loading(file_path_clean = '', file_path_infected = ''):
    SEQUENCE_SIZE = 1000

    def load_sequence_dataset_from_file(file_path):
        seq_size_max = SEQUENCE_SIZE
        sequences = []
        with open(file_path) as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                if len(line) >= seq_size_max:
                    sequences.append(line[:seq_size_max])
                else:
                    line = padding(seq_size_max, line)
                    sequences.append(line)
        
        return sequences

    def check_dataset_validity(dataset, file_path):
        i = 1
        for sequence in dataset:
            if any(x not in ['A', 'T', 'C', 'G', 'N'] for x in set(list(sequence))) or len(sequence)!=SEQUENCE_SIZE:
                print('Following problem is found in dataset at ', file_path)
                print('At line number:', i)
                print('Charaters :', set(sequence))
                print('Length: ', len(sequence), 'where sequence len should be', SEQUENCE_SIZE)
                print('Sequence', sequence)
                exit()
            i += 1

    dataset_clean = load_sequence_dataset_from_file(file_path_clean)
    check_dataset_validity(dataset_clean, file_path_clean)

    
    dataset_infected = load_sequence_dataset_from_file(file_path_infected)
    check_dataset_validity(dataset_infected, file_path_infected)

    dataset_full = dataset_clean + dataset_infected

    return dataset_full, dataset_clean, dataset_infected

#'k_mers_sparse_matrix' takes as input a dataset of DNA reads and returns a matrix of k-mer frequencies across the reads, given a k-mer length k

def k_mers_sparse_matrix(k, dataset_full, dataset_clean, dataset_infected, unique_kmers='empty', variance_treshold=None, features_limit=None):
    print(f"=== k: {k} ===")

    if variance_treshold!=None and features_limit!=None:
        raise ValueError('Please choose only one between variance threshold and features limit')
    elif variance_treshold == None and features_limit == None:
        variance_treshold = 0
        
    # If no k-mer vocabulary is provided, the unique k-mers are extracted from the full dataset
    k_mers_computed = False
    if unique_kmers=='empty':
        k_mers_computed = True
        unique_kmers = set()
        for string in (dataset_full):
            for index in range(len(string)-k+1):
                unique_kmers.add(string[index: index+k])

    print('Unique kmers: ', len(unique_kmers))

    k_mers = list(unique_kmers)

    # Compute the k-mer frequency matrices for the clean and infected samples separately, using the same vocabulary
    vectorizer = CountVectorizer(vocabulary=k_mers, lowercase=False)
    selector = VarianceThreshold(threshold=variance_treshold) 

    genomes_split = [
        " ".join([genome[i:i+k] for i in range(len(genome) - k + 1)])
        for genome in dataset_clean
    ]

    X_clean = vectorizer.fit_transform(genomes_split)
    y_clean = np.zeros(len(dataset_clean), int)

    genomes_split_infected = [
        " ".join([genome[i:i+k] for i in range(len(genome) - k + 1)])
        for genome in dataset_infected
    ]

    X_infected = vectorizer.transform(genomes_split_infected)
    y_infected = np.ones(len(dataset_infected), int)
    X_full = vstack([X_clean, X_infected])

    # If specified, the number of features can be reduced either by keeping only k-mers with variance above a threshold, or by keeping the top-n k-mers by variance. Otherwise, no reduction is performed.
    if k_mers_computed and variance_treshold != None:
        X = selector.fit_transform(X_full)
        feature_mask = selector.get_support()
        retained_kmers = [kmer for kmer, kept in zip(k_mers, feature_mask) if kept]
    elif k_mers_computed and features_limit != None:
        mean = X_full.mean(axis=0).A1
        mean_sq = X_full.multiply(X_full).mean(axis=0).A1
        variances = mean_sq - mean**2
        top_idx = np.argsort(variances)[::-1][:features_limit]
        X = X_full[:, top_idx]
        retained_kmers = [k_mers[i] for i in top_idx]
    else:
        X = X_full
        retained_kmers = unique_kmers

    y = np.hstack([y_clean, y_infected])

    print(f"K-mers frequency matrix shape: {X.shape}")

    return X,y,retained_kmers

# Datasets are stored in 10 different folders for each combination of fragment length, retention position, and encryption key, following the structure used by Islam et al. ("Using Deep Learning to Detect Digitally Encoded DNA Triggers for Trojan Malware in Bio-Cyber Attacks").
# This function loads and merges all datasets into a single dataset.
def merge_datasets(dataset:['','ecoli','lentivirus'],fragment_len=5, retention_pos=5, encryption_key=0,  dataset_number=10):
    
    base_path = f"DNA_attacks/experiments/datasets/{dataset}/fragment_len_{fragment_len}/retention_pos_{retention_pos}/encryption_key_{encryption_key}"

    dataset_clean_tot = []
    dataset_infected_tot = []
    dataset_num = 0
    
    # Text files containing infected reads
    file_type = "nw_best_greedy_trojan_insertion_dataset.txt"
        

    while True and dataset_num<dataset_number:
        dataset_folder = os.path.join(base_path, f"dataset_{dataset_num}")
        
        if not os.path.exists(dataset_folder):
            print(f"No more datasets found. Stopped at dataset_{dataset_num}")
            break
        
        # Loop through files in the current dataset folder
        for filename in os.listdir(dataset_folder):
            file_path = os.path.join(dataset_folder, filename)

            
            if filename == 'non_trojan_dataset.txt': # Text files containing clean reads
                file_clean = filename
            if filename == file_type:
                file_infected = filename

        path_clean = os.path.join(os.path.join(base_path, f"dataset_{dataset_num}"),file_clean)
        path_infected = os.path.join(os.path.join(base_path, f"dataset_{dataset_num}"),file_infected)

        _, dataset_clean, dataset_infected = dataset_loading(file_path_clean=path_clean,file_path_infected=path_infected)

        dataset_clean_tot = dataset_clean_tot + dataset_clean
        dataset_infected_tot = dataset_infected_tot + dataset_infected
                    
        dataset_num += 1


    base = Path(os.path.join(base_path, "dataset_tot"))

    base.mkdir(parents=True, exist_ok=True)

    with open(os.path.join(base,"non_trojan_dataset.txt"), 'w') as f:
        for item in dataset_clean_tot:
            f.write(str(item) + '\n')

    with open(os.path.join(base,file_type), 'w') as f:
        for item in dataset_infected_tot:
            f.write(str(item) + '\n')
    

    # Merge infected and clean reads
    dataset_tot = dataset_clean_tot + dataset_infected_tot

    print(f"Succesfully loaded {len(dataset_tot)} samples, {len(dataset_clean_tot)} clean and {len(dataset_infected_tot)} infected -- dataset: {dataset}, fragment length: {fragment_len}, retention position: {retention_pos}, encryption_key: {encryption_key}")
    
    return dataset_tot, dataset_clean_tot, dataset_infected_tot

# tf_idf_k_mers_scores computes the TF-IDF score for each k-mer based on its frequency across the dataset.
# The result is a DataFrame where each k-mer is associated with its TF-IDF score.
def tf_idf_k_mers_scores(X,kmers_list) -> pd.DataFrame:

    n_samples, _ = X.shape

    #Document frequency. In how many rows the k-mer is present
    df_i = np.asarray(X.getnnz(axis=0)).ravel().astype(np.float64)

    #TF-IDF number of documents over the number of documents that contains the term i, log
    idf = np.log(float(n_samples) / (df_i))

    #getting the log matrix
    X_log = X.tocsc(copy=True)
    X_log.data = np.log1p(X_log.data)
    col_sum_log = np.asarray(X_log.sum(axis=0)).ravel().astype(np.float64)

    mean_log_tf = col_sum_log / float(n_samples)

    tfidf_unsup = idf * mean_log_tf

    df_out = pd.DataFrame(
            {
                "TF-IDF": tfidf_unsup, # unsupervised (mean log-scaled TF)
                "label": kmers_list
            }
        )
    df_out.index.name = "feature_idx"

    return df_out

# Select the top n k-mers based on TF-IDF scores
def top_kmers(score_df: pd.DataFrame, n:int,score_col='TF-IDF'):
    return(
        score_df.sort_values(by=score_col, ascending=False)
                .head(n)
                .index
                .to_numpy(dtype=int)
    )

# Assign each read to a group such that all duplicate reads belong to the same group,
# and no duplicates appear across different groups.
#Returns the group assignment for each dataset entry.
def get_groups(dataset_clean, folds_number=5):

    N = len(dataset_clean)

    groups_dict = defaultdict(list)

    for idx, seq in enumerate(dataset_clean):
        groups_dict[seq].append(idx)

    unique_groups = list(groups_dict.values())

    num_folds = folds_number
    fold_sizes = [0] * num_folds   
    fold_assignment = {}

    sorted_groups = sorted(unique_groups, key=len, reverse=True)

    for g_idx, group in enumerate(sorted_groups):
        
        fold_id = np.argmin(fold_sizes)
        fold_assignment[g_idx] = fold_id
        fold_sizes[fold_id] += len(group)

    clean_groups = [None] * N

    for g_idx, group in enumerate(sorted_groups):
        fold_id = fold_assignment[g_idx]
        for idx in group:
            clean_groups[idx] = fold_id

    clean_groups = np.array(clean_groups)

    full_groups = np.concatenate([clean_groups, clean_groups])

    return full_groups
