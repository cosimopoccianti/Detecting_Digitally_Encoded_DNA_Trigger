import os
from utils import merge_datasets, k_mers_sparse_matrix, tf_idf_k_mers_scores, top_kmers, deduplicate_train_test, _safe_div, _pick_mode
import pandas as pd
import json
import time
from config import REGISTRY
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import json
import time
import numpy as np


def encryption_key_eval(algo, args, f_l, r_p, train_e_k, avb_key):
                            
    metrics_columns = ["model","k","n","train_encryption_key","test_encryption_key","Model_params","TN","FP","FN","TP","Accuracy", "Precision", "Recall", "F1", "Runtime","Selected_k_mers"]

    metrics_df = pd.DataFrame(columns=metrics_columns)

    metrics_folder_model =  os.path.join(args.out_dir,f"metrics/{algo}/{args.dataset}/fragment_len_{f_l}/retention_pos_{r_p}/encryption_key_0/")
    result_df = pd.read_csv(os.path.join(metrics_folder_model, f'results_{args.dataset}_dataset_length_20000_algo_{algo}_fl_{f_l}_rp_{r_p}_ek0.csv'), index_col=0)
    cross_model = result_df[result_df['Outer_KFold'] == 'total']
    best_cross_model = cross_model.sort_values(by='F1', ascending=False).iloc[0]
    k, n = best_cross_model['k'], best_cross_model['n']
    folds = result_df[(result_df['Outer_KFold'] != 'total') & (result_df['k'] == k) & (result_df['n'] == n)]
    params_df = pd.DataFrame([json.loads(p) for p in folds['Model_params']])
    hyperp = {col: _pick_mode(params_df[col]) for col in params_df.columns}
    
    start_time = time.time()
    fold_time = start_time

    for test_e_k in [k for k in avb_key if k != train_e_k]:

        print(f"\n====== k: {k} - n: {n} - algo: {algo} - train encryption key: {train_e_k} - test enrcyption key: {test_e_k}======")

        metrics_folder =  os.path.join(args.out_dir,f"metrics/mixed_key/{algo}/{args.dataset}/fragment_len_{f_l}/retention_pos_{r_p}/train_encryption_key_{train_e_k}")
        os.makedirs(metrics_folder, exist_ok=True)

        full_dataset_train, clean_dataset_train, infected_dataset_train = merge_datasets(dataset=args.dataset, fragment_len=f_l, retention_pos=r_p, encryption_key=train_e_k, dataset_number=10)

        full_dataset_test, clean_dataset_test, infected_dataset_test = merge_datasets(dataset=args.dataset, fragment_len=f_l,retention_pos=r_p, encryption_key=test_e_k, dataset_number=10)
        

        (full_dataset_train, clean_dataset_train, infected_dataset_train,
        full_dataset_test,  clean_dataset_test,  infected_dataset_test) = deduplicate_train_test(clean_dataset_train, infected_dataset_train,
            clean_dataset_test,  infected_dataset_test)

    
        # Create reads frequency matrix
        print("\n--- Train matrix before TF-IDF ---")
        X_train,y_train,k_mers_list = k_mers_sparse_matrix(k=k, dataset_full=full_dataset_train, dataset_clean=clean_dataset_train, dataset_infected=infected_dataset_train)

        #Compute k-mer TF-IDF scores from the frequency matrix
        features_scores = tf_idf_k_mers_scores(X_train,k_mers_list)



        top_n_kmers = top_kmers(score_df=features_scores,n=n)
        # Reduce X to top-n kmers based on TF-IDF
        X_train = X_train[:, top_n_kmers]

        # Reduce kmers_list consistently
        top_kmers_list = [k_mers_list[i] for i in top_n_kmers]
        print(f"\nSelected top {n} k-mers by TF-IDF score")
        print(f"Train feature matrix shape: {X_train.shape}")

        print("\n--- Test matrix ---")
        X_test, y_test, k_mers_test_check = k_mers_sparse_matrix(k=k, dataset_full=full_dataset_test, dataset_clean=clean_dataset_test, dataset_infected=infected_dataset_test, unique_kmers=top_kmers_list)

        print("Same kmers used in train and test: ",top_kmers_list==k_mers_test_check)

        
        
        acc_outer = list()
        f1_outer = list()

        # Define output DataFrame columns with run metadata

        if n <= X_train.shape[1]:
        

            # Retrieve the parameter search space for the selected algorithm and
            # determine whether the feature matrix X must be converted from sparse to dense
            entry = REGISTRY[algo]
            preprocess = entry.get("preprocess", None) 

            if preprocess is not None:
                X_train = preprocess(X_train)
                X_test  = preprocess(X_test)

            estimator_cls = REGISTRY[algo]["estimator"]
            model = estimator_cls(**hyperp)

            # Refit the model after grid search
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)


            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='binary', zero_division=np.nan)
            # store the result
            acc_outer.append(acc)
            f1_outer.append(f1)
            # report progress
            print(f'F1 for test encryption key = {test_e_k}')
            print('>f1=%.4f' % (f1))
            

            TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()
            
            
            fold_elapsed_minutes = round((time.time() - fold_time) / 60,2)
            fold_time = time.time()

            # Save experiment metrics for the current outer-loop split
            metrics = [algo,
                    k,
                    n,
                    train_e_k,
                        test_e_k,
                            hyperp,
                            TN,
                            FP,
                            FN,
                            TP,
                            accuracy_score(y_test, y_pred),
                            precision_score(y_test, y_pred, average='binary', zero_division=np.nan),
                            recall_score(y_test, y_pred, average='binary', zero_division=np.nan),
                            f1_score(y_test, y_pred, average='binary', zero_division=np.nan),
                            fold_elapsed_minutes,
                            top_kmers_list]
            
            metrics_row = pd.DataFrame([metrics], columns=metrics_columns)

            metrics_df = pd.concat([metrics_df, metrics_row])
            print("\n")
        # summarize the estimated performance of the model
            
    all_fold = metrics_df[metrics_df['k']==k]
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
    elapsed_minutes = round((end_time - start_time) / 60,2)
    
    # Save the average experiment metrics 
    metrics = [algo,
            k,
            n,
            train_e_k,
                'all',
                None,
                TN,
                FP,
                FN,
                TP,
                accuracy,
                precision,
                recall,
                f1,
                elapsed_minutes,
                top_kmers_list]
    
    
        
    metrics_row = pd.DataFrame([metrics], columns=metrics_columns)
    metrics_df = pd.concat([metrics_df, metrics_row])


    print(f"\nTime for k = {k}, n = {n}: {elapsed_minutes:.2f} minutes")

    print("\n")
    
    return metrics_df