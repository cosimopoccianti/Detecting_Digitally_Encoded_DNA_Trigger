from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from config import REGISTRY
import json
import joblib
import pandas as pd
import time


def double_cross_validation(full_dataset, groups, X, y, algo, k,n, k_mer_list, outer_folds_number=5, inner_folds_number=4):
        
        start_time = time.time()
        fold_time = start_time
        
        acc_outer = list()
        f1_outer = list()

        # Define output DataFrame columns with run metadata

        metrics_columns = ["model","k","n","Outer_KFold","Model_params","TN","FP","FN","TP","Accuracy", "Precision", "Recall", "F1", "Runtime","Selected_k_mers"]

        metrics_df = pd.DataFrame(columns=metrics_columns)
        
        # Define the dataset partitioning for the outer cross-validation loop
        gkf = GroupKFold(n_splits=outer_folds_number)

        for split, (train_idx, test_idx) in enumerate(gkf.split(full_dataset, groups=groups)):

            print(f'Fitting split number: {split}')

            X_train, X_test = X[train_idx, :], X[test_idx, :]
            y_train, y_test = y[train_idx], y[test_idx]

            # Retrieve the parameter search space for the selected algorithm and
            # determine whether the feature matrix X must be converted from sparse to dense
            entry = REGISTRY[algo]
            preprocess = entry.get("preprocess", None) 

            if preprocess is not None:
                X_train = preprocess(X_train)
                X_test  = preprocess(X_test)

            train_groups = groups[train_idx]

            model = entry['estimator']()
            param_grid = entry['param_grid']

            # Define the dataset partitioning for the inner cross-validation loop
            inner_gkf = GroupKFold(n_splits=inner_folds_number)

            grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=inner_gkf, scoring='f1', verbose=1, n_jobs=-1) 

            grid_search.fit(X_train, y_train, groups=train_groups)
            
            best_model = grid_search.best_estimator_

            print("\n... Refitting on the total training set ...")

            # Refit the model after grid search
            best_model.fit(X_train, y_train)

            y_pred = best_model.predict(X_test)


            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='binary')
            # store the result
            acc_outer.append(acc)
            f1_outer.append(f1)
            # report progress
            print(f'F1 for {split} split')
            print('>f1=%.4f' % (f1))
            

            TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()

            best_params = grid_search.best_params_
            params_dict = json.dumps(best_params)

            print(f"Best parameter: {params_dict}")
            
            
            fold_elapsed_minutes = round((time.time() - fold_time) / 60,2)
            fold_time = time.time()

            # Save experiment metrics for the current outer-loop split
            metrics = [algo,
                       k,
                       n,
                        split,
                            params_dict,
                            TN,
                            FP,
                            FN,
                            TP,
                            accuracy_score(y_test, y_pred),
                            precision_score(y_test, y_pred, average='binary'),
                            recall_score(y_test, y_pred, average='binary'),
                            f1_score(y_test, y_pred, average='binary'),
                            fold_elapsed_minutes,
                            k_mer_list]
            
            metrics_row = pd.DataFrame([metrics], columns=metrics_columns)

            metrics_df = pd.concat([metrics_df, metrics_row])
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

        end_time = time.time()
        elapsed_minutes = round((end_time - start_time) / 60,2)
        
        # Save the average experiment metrics across all outer-loop splits
        metrics = [algo,
                   k,
                   n,
                    'total',
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
                    k_mer_list]
        
        
            
        metrics_row = pd.DataFrame([metrics], columns=metrics_columns)
        metrics_df = pd.concat([metrics_df, metrics_row])


        print(f"\nTime for k = {k}, n = {n}: {elapsed_minutes:.2f} minutes")

        print("\n")

        return metrics_df