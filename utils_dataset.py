import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import VarianceThreshold
from scipy.sparse import vstack
from sklearn.utils import shuffle
import os
from pathlib import Path


def padding(seq_size_max, line):
    less = seq_size_max-len(line)
    padding_value = line[:less]
    line = line+padding_value
    if(len(line) < seq_size_max):
        line = padding(seq_size_max, line)
    return line


def dataset_loading(file_path_clean = 'dataset_4356/non_trojan_dataset.txt', file_path_trojan = 'dataset_4356/nw_best_greedy_trojan_insertion_dataset.txt'):
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
    '''unique_ds_clean = set(dataset_clean)
    if len(dataset_clean) == len(unique_ds_clean):
        print('There are no duplicates in the clean dataset!')
    else:
        print(f'There are {len(dataset_clean) - len(unique_ds_clean)} duplicates in the clean dataset, removing ...')
        dataset_clean = list(unique_ds_clean)'''
    
    dataset_trojan = load_sequence_dataset_from_file(file_path_trojan)
    check_dataset_validity(dataset_trojan, file_path_trojan)
    '''unique_ds_trojan = set(dataset_trojan)
    if len(dataset_trojan) == len(unique_ds_trojan):
        print('There are no duplicates in the trojan dataset!')
    else:
        print(f'There are {len(dataset_trojan) - len(unique_ds_trojan)} duplicates in the trojan dataset, removing ...')
        dataset_trojan = list(unique_ds_trojan)'''

    print('dataset clean len:', len(dataset_clean), '\ndataset trojan len:', len(dataset_trojan))
    label_a = list(np.zeros(len(dataset_trojan)))
    label_b = list(np.ones(len(dataset_trojan)))

    dataset_full = dataset_clean + dataset_trojan
    '''
    unique_ds_full = set(dataset_full)
    if len(dataset_full) == len(unique_ds_full):
        print('There are no duplicates in the full dataset!')
    else:
        print(f'==== ATTENTION: There are {len(dataset_full) - len(unique_ds_full)} duplicates in the full dataset ====')
    '''
    return dataset_full, dataset_clean, dataset_trojan


def k_mers_sparse_matrix(k, dataset_full, dataset_clean, dataset_trojan, unique_kmers='empty', variance_treshold=None, features_limit=None):
    print(f"k = {k}")

    if variance_treshold!=None and features_limit!=None:
        raise ValueError('Please choose only one between variance threshold and features limit')
    elif variance_treshold == None and features_limit == None:
        print('No variance threshold or features limit selected, features will not be reduced')
        variance_treshold = 0
        

    k_mers_computed = False
    if unique_kmers=='empty':
        k_mers_computed = True
        unique_kmers = set()
        for string in (dataset_full):
            for index in range(len(string)-k+1):
                unique_kmers.add(string[index: index+k])

    print('Unique kmers: ', len(unique_kmers))

    k_mers = list(unique_kmers)

    vectorizer = CountVectorizer(vocabulary=k_mers, lowercase=False)
    selector = VarianceThreshold(threshold=variance_treshold) 

    genomes_split = [
        " ".join([genome[i:i+k] for i in range(len(genome) - k + 1)])
        for genome in dataset_clean
    ]

    X_clean = vectorizer.fit_transform(genomes_split)
    y_clean = np.zeros(len(dataset_clean), int)

    genomes_split_trojan = [
        " ".join([genome[i:i+k] for i in range(len(genome) - k + 1)])
        for genome in dataset_trojan
    ]

    X_trojan = vectorizer.transform(genomes_split_trojan)
    y_trojan = np.ones(len(dataset_trojan), int)
    X_full = vstack([X_clean, X_trojan])

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

    y = np.hstack([y_clean, y_trojan])

    #print("Number of features: ", len(X.shape[0]))

    print(f"Final X shape: {X.shape}")

    X, y = shuffle(X, y, random_state=42)

    return X,y,retained_kmers


def join_datasets(dataset:['','greedy','straight'],fragment_len, retention_pos, encryption_key,  dataset_number=2):
    if encryption_key in [0,10,20,30,40,50]:
        base_path = f"/home/cosimo/Desktop/PhD/Cyberbiosecurity/trojan-malware-in-bio-cyber-attacks/trojan_attack/experiment_data/datasets_{dataset}/fragment_len_{fragment_len}/retention_pos_{retention_pos}/encryption_key_{encryption_key}"

        dataset_clean_tot = []
        dataset_trojan_tot = []
        dataset_num = 0
        
        if dataset == 'greedy':
            file_type = "nw_best_greedy_trojan_insertion_dataset.txt"
        elif dataset == 'straight':
            file_type = "nw_best_straight_forward_trojan_insertion_dataset.txt"

        while True and dataset_num<dataset_number:
            dataset_folder = os.path.join(base_path, f"dataset_{dataset_num}")
            
            if not os.path.exists(dataset_folder):
                print(f"No more datasets found. Stopped at dataset_{dataset_num}")
                break
            
            print(f"\nProcessing {dataset_folder}")
            
            # Loop through files in the current dataset folder
            for filename in os.listdir(dataset_folder):
                file_path = os.path.join(dataset_folder, filename)
                
                # Check if it's a file (not a subdirectory)
                if os.path.isfile(file_path):
                    print(f"  Found file: {filename}")
                    
                if filename == 'non_trojan_dataset.txt':
                    file_clean = filename
                if filename == file_type:
                    file_trojan = filename

            path_clean = os.path.join(os.path.join(base_path, f"dataset_{dataset_num}"),file_clean)
            path_trojan = os.path.join(os.path.join(base_path, f"dataset_{dataset_num}"),file_trojan)

            dataset_full, dataset_clean, dataset_trojan = dataset_loading(file_path_clean=path_clean,file_path_trojan=path_trojan)

            dataset_clean_tot = dataset_clean_tot + dataset_clean
            dataset_trojan_tot = dataset_trojan_tot + dataset_trojan
            print("Len clean dataset: ", len(dataset_clean_tot))
            print("Len trojan dataset: ", len(dataset_trojan_tot))
                        
            dataset_num += 1


        base = Path(os.path.join(base_path, "dataset_tot"))

        base.mkdir(parents=True, exist_ok=True)

        with open(os.path.join(base,"non_trojan_dataset.txt"), 'w') as f:
            for item in dataset_clean_tot:
                f.write(str(item) + '\n')

        with open(os.path.join(base,file_type), 'w') as f:
            for item in dataset_trojan_tot:
                f.write(str(item) + '\n')
        
    else:  # encryption_key == 'mix'
        encryption_keys_list = [10, 20, 30, 40, 50]
        dataset_clean_tot = []
        dataset_trojan_tot = []
        
        # Iterate through each encryption key
        for enc_key in encryption_keys_list:
            base_path = f"trojan-malware-in-bio-cyber-attacks/trojan_attack/experiment_data/datasets_{dataset}/fragment_len_{fragment_len}/retention_pos_{retention_pos}/encryption_key_{enc_key}"
            
            # Process dataset_0 to dataset_(dataset_number-1) for each encryption key
            for dataset_num in range(dataset_number):
                dataset_folder = os.path.join(base_path, f"dataset_{dataset_num}")
                
                if not os.path.exists(dataset_folder):
                    print(f"Warning: {dataset_folder} not found, skipping")
                    continue
                
                print(f"\nProcessing {dataset_folder}")
                
                file_clean = None
                file_trojan = None
                
                # Loop through files in the current dataset folder
                for filename in os.listdir(dataset_folder):
                    file_path = os.path.join(dataset_folder, filename)
                    
                    # Check if it's a file (not a subdirectory)
                    if os.path.isfile(file_path):
                        print(f"  Found file: {filename}")
                        
                    if filename == 'non_trojan_dataset.txt':
                        file_clean = filename
                    if filename == file_type:
                        file_trojan = filename
                
                if file_clean is None or file_trojan is None:
                    print(f"Warning: Required files not found in {dataset_folder}, skipping")
                    continue
                
                path_clean = os.path.join(dataset_folder, file_clean)
                path_trojan = os.path.join(dataset_folder, file_trojan)

                dataset_full, dataset_clean, dataset_trojan = dataset_loading(file_path_clean=path_clean, file_path_trojan=path_trojan)

                dataset_clean_tot = dataset_clean_tot + dataset_clean
                dataset_trojan_tot = dataset_trojan_tot + dataset_trojan
                print("Len clean dataset: ", len(dataset_clean_tot))
                print("Len trojan dataset: ", len(dataset_trojan_tot))
        
        # Save the mixed dataset
        base_path_mix = f"trojan-malware-in-bio-cyber-attacks/trojan_attack/experiment_data/datasets_{dataset}/fragment_len_{fragment_len}/retention_pos_{retention_pos}/encryption_key_mix"
        base = Path(os.path.join(base_path_mix, "dataset_mix"))
        base.mkdir(parents=True, exist_ok=True)

        with open(os.path.join(base,"non_trojan_dataset.txt"), 'w') as f:
            for item in dataset_clean_tot:
                f.write(str(item) + '\n')

        with open(os.path.join(base,trojan_file), 'w') as f:
            for item in dataset_trojan_tot:
                f.write(str(item) + '\n')
        
    dataset_tot = dataset_clean_tot + dataset_trojan_tot
    
    return dataset_tot, dataset_clean_tot, dataset_trojan_tot
    
