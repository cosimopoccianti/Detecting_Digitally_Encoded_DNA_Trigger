# SVM
import pandas as pd
from utils_dataset import dataset_loading,join_datasets, k_mers_sparse_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.svm import SVC 
import time
import warnings
import os
warnings.filterwarnings("ignore")



def run_exp(k_list, full, clean, trojan, metrics_csv, metrics_columns):

    
    metrics_df = pd.DataFrame(columns=metrics_columns)

    for k in k_list:

        start_time = time.time()

        X,y,kmers_list = k_mers_sparse_matrix(k, full, clean, trojan, features_limit=100)#, features_limit=100)

        skf_outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=18)

        acc_outer = list()
        f1_outer = list()
       
        for split, (train_ix, test_ix) in enumerate(skf_outer.split(X, y), start=1):
            
            X_train, X_test = X[train_ix, :], X[test_ix, :]
            y_train, y_test = y[train_ix], y[test_ix]

            model = SVC()
            
            param_grid = {'C': [1,10,100], 
                    'gamma': [0.1,0.01,0.001,0.0001], 
                    'kernel': ['rbf']}#,'linear']}
                                #'degree':[0, 1, 6]} 

            skf_inner = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=skf_inner, scoring='f1', verbose=1, n_jobs=18)

            grid_search.fit(X_train, y_train)

            best_model = grid_search.best_estimator_

            print("\n... Refitting on the total training set ...")

            model_t = SVC(C=best_model.C, gamma=best_model.gamma,kernel=best_model.kernel)

            model_t.fit(X_train, y_train)

            y_pred = model_t.predict(X_test)


            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='binary')
            # store the result
            acc_outer.append(acc)
            f1_outer.append(f1)
            # report progress
            print('>acc=%.4f, est=%.3f, cfg=%s' % (acc, grid_search.best_score_, grid_search.best_params_))

            TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()


            metrics = [k,
                        split,
                            best_model.C,
                            best_model.gamma,
                            best_model.kernel,
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

        print('Average accuracy: %.3f' % accuracy)

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


def main(fragment_len, retention_pos, encryption_key, dataset_type, algo, full, clean, trojan, k_list, metrics_columns):

    metrics_folder = f"/home/cosimo/Desktop/PhD/Cyberbiosecurity/DNA_attacks/metrics/{algo}/Balanced/{dataset_type}/fragment_len_{fragment_len}/"
    os.makedirs(metrics_folder, exist_ok=True)

    metrics_csv = os.path.join(metrics_folder, 
                            f"metrics_len_{fragment_len}_pos_{retention_pos}_key_{encryption_key}_double_cv(5,4)_dataset_len_{len(full)}_{dataset_type}_no_duplicates.csv")
    
    print(f"\n========== Running {algo} for fragment len = {fragment_len}, retention position = {retention_pos}, encryption key = {encryption_key} ==========")

    run_exp(k_list, full, clean, trojan, metrics_csv, metrics_columns)
    
if __name__ == "__main__":

    fragment_len = [2]
    retention_pos = [1]
    encryption_key = [0]
    #dataset_num_clean = 0
    #dataset_num_trojan = 0
    dataset_type = 'unique'
    algo = 'SVM'
    dataset_len = 20000
    #file_type = f"{dataset_type}_forward" if dataset_type == 'straight' else dataset_type

    #full,clean,trojan = dataset_loading(file_path_clean=f'/home/cosimo/Desktop/PhD/Cyberbiosecurity/trojan-malware-in-bio-cyber-attacks/trojan_attack/experiment_data/datasets_{dataset_type}/fragment_len_{fragment_len}/retention_pos_{retention_pos}/encryption_key_{encryption_key}/dataset_{dataset_num_clean}/non_trojan_dataset.txt',
    #                                                       file_path_trojan=f'/home/cosimo/Desktop/PhD/Cyberbiosecurity/trojan-malware-in-bio-cyber-attacks/trojan_attack/experiment_data/datasets_{dataset_type}/fragment_len_{fragment_len}/retention_pos_{retention_pos}/encryption_key_{encryption_key}/dataset_{dataset_num_trojan}/nw_best_{file_type}_trojan_insertion_dataset.txt')

    
    metrics_columns = ["k","Outer_KFold","C", "gamma", "kernel","TN","FP","FN","TP","Accuracy", "Precision", "Recall", "F1"]

    k_list = [i for i in range(2, 26)] #+ [i for i in range(13, 26, 3)]

    for frag_l in fragment_len:
        for ret_p in retention_pos:
            if ret_p <= frag_l + 1:
                for enc_k in encryption_key:

                    full,clean,trojan = join_datasets(dataset=dataset_type, 
                                        dataset_number=int(dataset_len/2000), 
                                        fragment_len=frag_l, 
                                        retention_pos=ret_p, 
                                        encryption_key=enc_k)
        
                    main(frag_l, ret_p, enc_k, 
                        dataset_type=dataset_type, algo=algo, 
                        full=full, clean=clean, trojan=trojan, 
                        k_list=k_list, metrics_columns=metrics_columns)