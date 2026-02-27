# SVM
import pandas as pd
from utils_dataset import dataset_loading,join_datasets, k_mers_sparse_matrix_no_shuffle
from sklearn.model_selection import GridSearchCV, StratifiedKFold, GroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.svm import SVC 
import time
import warnings
import os
warnings.filterwarnings("ignore")
import numpy as np
from collections import defaultdict
from utils_feat_sel import k_mers_scores,top_kmers

def run_exp(k_list, full, clean, trojan, metrics_csv, metrics_columns,score_feat_eng,feat_limit):

    
    metrics_df = pd.DataFrame(columns=metrics_columns)


    N = len(clean)

    groups_dict = defaultdict(list)

    for idx, seq in enumerate(clean):
        groups_dict[seq].append(idx)

    unique_groups = list(groups_dict.values())

    num_folds = 5
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

    full = clean + trojan
    full_groups = np.concatenate([clean_groups, clean_groups])


    for k in k_list:

        start_time = time.time()

        X,y,kmers_list = k_mers_sparse_matrix_no_shuffle(k, full, clean, trojan)
        # --- UNSUPERVISED FEATURE SELECTION: keep only top-10 k-mers ---
        scores = k_mers_scores(X,kmers_list)

        feat = top_kmers(
            scores,
            score_col=score_feat_eng,
            n=feat_limit,
        )

        # Reduce X to top-10 columns (keep sparse)
        X = X[:, feat]

        # Reduce kmers_list consistently
        kmers_list = [kmers_list[i] for i in feat]
        print(f"Reducing features according to: {score_feat_eng} \nKeeping only the first {feat_limit} features")
        print("Final X shape after feature reduction:", X.shape)
        

        acc_outer = list()
        f1_outer = list()
        
        gkf = GroupKFold(n_splits=5)

        for split, (train_idx, test_idx) in enumerate(gkf.split(full, groups=full_groups)):

            print(f'Fitting split number: {split}')

            X_train, X_test = X[train_idx, :], X[test_idx, :]
            y_train, y_test = y[train_idx], y[test_idx]

            model = SVC(C=10,gamma=0.1,kernel='rbf')
            '''
            param_grid = {'C': [1,10,100], 
                    'gamma': [0.1,0.01,0.001,0.0001], 
                    'kernel': ['rbf']}#,'linear']}
                                #'degree':[0, 1, 6]} 

            skf_inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=skf_inner, scoring='f1', verbose=1, n_jobs=-1)

            grid_search.fit(X_train, y_train)
            
            best_model = grid_search.best_estimator_

            print("\n... Refitting on the total training set ...")
            
            model_t = SVC(C=best_model.C, gamma=best_model.gamma,kernel=best_model.kernel)
            '''
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)


            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='binary')
            # store the result
            acc_outer.append(acc)
            f1_outer.append(f1)
            # report progress
            print(f'F1 for {split} split')
            print('>f1=%.4f' % (f1))

            TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()


            metrics = [k,
                        split,
                            model.C,
                            model.gamma,
                            model.kernel,
                            TN,
                            FP,
                            FN,
                            TP,
                            accuracy_score(y_test, y_pred),
                            precision_score(y_test, y_pred, average='binary'),
                            recall_score(y_test, y_pred, average='binary'),
                            f1_score(y_test, y_pred, average='binary')]
            
            metrics_row = pd.DataFrame([metrics], columns=metrics_columns)

            metrics_df = pd.concat([metrics_df, metrics_row])
            metrics_df.to_csv(metrics_csv, index=False)
            print("\n")
        # summarize the estimated performance of the model
        
        all_fold = metrics_df[metrics_df['k']==k]
        TN = all_fold['TN'].sum()
        FP = all_fold['FP'].sum()
        FN = all_fold['FN'].sum()
        TP = all_fold['TP'].sum()
        accuracy = (TN+TP)/(TN+TP+FN+FP)
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        f1 = 2*((precision*recall)/(precision+recall))

        print('Average F1: %.3f' % f1)

        metrics = [k,
                    'total',
                    '',
                    '',
                    '',
                    TN,
                    FP,
                    FN,
                    TP,
                    accuracy,
                    precision,
                    recall,
                    f1]
            
        metrics_row = pd.DataFrame([metrics], columns=metrics_columns)
        metrics_df = pd.concat([metrics_df, metrics_row])
        metrics_df.to_csv(metrics_csv, index=False)

        end_time = time.time()
        elapsed_minutes = (end_time - start_time) / 60

        print(f"\nTime for k = {k}: {elapsed_minutes:.2f} minutes")

        print("\n")
        
    return metrics_df

def main(fragment_len, retention_pos, encryption_key, dataset_type,dataset_len, algo, k_list, metrics_columns,score_feat_eng,feat_limit):

    full,clean,trojan = join_datasets(dataset=dataset_type, 
                                        dataset_number=int(dataset_len/2000), 
                                        fragment_len=fragment_len, 
                                        retention_pos=retention_pos, 
                                        encryption_key=encryption_key)

    metrics_folder = f"/home/cosimo/Desktop/PhD/Cyberbiosecurity/DNA_attacks/metrics/{algo}/Balanced/{dataset_type}/GroupKFold_feat_eng/fragment_len_{fragment_len}/"
    os.makedirs(metrics_folder, exist_ok=True)

    metrics_csv = os.path.join(metrics_folder, 
                            f"metrics_len_{fragment_len}_pos_{retention_pos}_key_{encryption_key}_LOGO(5)_dataset_len_{len(full)}_{dataset_type}_GKFold_{score_feat_eng}_{feat_limit}.csv")
    
    print(f"\n========== Running {algo} for fragment len = {fragment_len}, retention position = {retention_pos}, encryption key = {encryption_key}, feature reduction = {score_feat_eng}, feature number = {feat_limit} ==========")

    run_exp(k_list, full, clean, trojan, metrics_csv, metrics_columns,score_feat_eng,feat_limit)
    
if __name__ == "__main__":

    #dataset_num_clean = 0
    #dataset_num_trojan = 0
    dataset_type = 'greedy'
    algo = 'SVM'
    dataset_len = 20000
    score_feat_eng =  "TF-IDF unsupervised (mean log-scaled TF)"
    feat_limit=100
    #file_type = f"{dataset_type}_forward" if dataset_type == 'straight' else dataset_type

    #full,clean,trojan = dataset_loading(file_path_clean=f'/home/cosimo/Desktop/PhD/Cyberbiosecurity/trojan-malware-in-bio-cyber-attacks/trojan_attack/experiment_data/datasets_{dataset_type}/fragment_len_{fragment_len}/retention_pos_{retention_pos}/encryption_key_{encryption_key}/dataset_{dataset_num_clean}/non_trojan_dataset.txt',
    #                                                       file_path_trojan=f'/home/cosimo/Desktop/PhD/Cyberbiosecurity/trojan-malware-in-bio-cyber-attacks/trojan_attack/experiment_data/datasets_{dataset_type}/fragment_len_{fragment_len}/retention_pos_{retention_pos}/encryption_key_{encryption_key}/dataset_{dataset_num_trojan}/nw_best_{file_type}_trojan_insertion_dataset.txt')

    
    metrics_columns = ["k","Outer_KFold","C", "gamma", "kernel","TN","FP","FN","TP","Accuracy", "Precision", "Recall", "F1"]

    k_list = [i for i in range(2,26)] #+ [i for i in range(13, 26, 3)]

        
    main(fragment_len=5,
         retention_pos=5, 
         encryption_key=10, 
        dataset_type='greedy', 
        dataset_len=dataset_len, 
        algo=algo,  
        k_list=k_list, 
        metrics_columns=metrics_columns,
        score_feat_eng=score_feat_eng,
        feat_limit=feat_limit)