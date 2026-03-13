# main.py
import argparse
import os
from utils import merge_datasets, k_mers_sparse_matrix, tf_idf_k_mers_scores, top_kmers, get_groups
from double_cross_validation import double_cross_validation
import pandas as pd

#------------------------------- CLI ----------------------------

def algo_type(value):
    if value == "all":
        return ["naive_bayes",'logistic','knn','svm', "decision_tree","random_forest","xgboost"]
    
    try:
        if "," in value:
            return [v for v in value.split(",")]
        
        return [value]
    
    except ValueError:
        raise argparse.ArgumentTypeError("algo must be 'all', or one or more of the available algorithms, divided by a comma")

def fragment_len_type(value):
    if value == "all":
        return [1, 2, 3, 4, 5]
    
    try:
        if "," in value:
            return [int(v) for v in value.split(",")]
        
        return [int(value)]
    
    except ValueError:
        raise argparse.ArgumentTypeError("fragment_len must be 'all', an integer, or a comma-separated list (e.g. 1,2,3,4,5), min 1 - max 5")
    
def retention_pos_type(value):
    if value == "all":
        return [0,1,2,3,4,5]
    try:
        if "," in value:
            return [int(v) for v in value.split(",")]
        
        return [int(value)]
    
    except ValueError:
        raise argparse.ArgumentTypeError("retention_pos must be 'all', an integer, or a comma-separated list (e.g. 0,1,2,3,4,5), min 0 - max 5")
    
def encryption_key_type(value):
    if value == "all":
        return [0,10,20,30,40,50]
    try:
        if "," in value:
            return [int(v) for v in value.split(",")]
        
        return [int(value)]
    
    except ValueError:
        raise argparse.ArgumentTypeError("encryption_key must be 'all', an integer, or a comma-separated list (e.g. 0,10,20,30,40,50)")
    
def n_feature_type(value):
    if value == "all":
        return [*range(5,50,5), 50,75,100]
    
    try:
        if "," in value:
            return [int(v) for v in value.split(",")]
        
        return [int(value)]
    
    except ValueError:
        raise argparse.ArgumentTypeError("n_feature must be 'all', an integer or a comma-separeted list (e.g. 5,10,15,20,25,...). Must be >=0, default value = [5,10,15,...,50,75,100]")
    

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--algo",    type=algo_type, default=["svm"], help="Available algortihms: 'svm', 'knn',")   # "svm", "knn", or "all"
    parser.add_argument("--dataset", type=str, default="ecoli", help="- 'ecoli' → E. coli reads dataset - 'lentivirus' → Lentivirus reads dataset")
    parser.add_argument("--k_min",   type=int, default=3)
    parser.add_argument("--k_max",   type=int, default=25)
    parser.add_argument(
        "--fragment_len",
        type=fragment_len_type,
        default=[5],
        help="fragment_len must be 'all', an integer, or a comma-separated list (e.g. 1,2,3), min 1 - max 5"
    )
    parser.add_argument(
        "--retention_pos",
        type=retention_pos_type,
        default=[5],
        help="retention_pos must be 'all', an integer, or a comma-separated list (e.g. 0,1,2,3,4,5), min 0 - max 5"
    )
    parser.add_argument("--encryption_key", type=encryption_key_type, 
                        default=[0],
                        help="encryption_key")
    parser.add_argument("--n_features", type=n_feature_type, default= [*range(5,50,5), 50,75,100], help="encryption_key must be 'all', an integer, or a comma-separated list (e.g. 0,10,20,30,40,50)")
    parser.add_argument("--out_dir", type=str, default="DNA_attacks/experiments/results")
    parser.add_argument("--random_state", type=int, default=42)
    args = parser.parse_args()

    return args


#------------------------------------------------------------------------------


def main():

    args=parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)

    k_list = list(range(args.k_min,args.k_max+1))

    for algo in args.algo:
        print(f"######## {algo} ########")
        for f_l in args.fragment_len:
            for r_p in args.retention_pos:
                if r_p <= f_l + 1: 
                    for e_k in args.encryption_key:


                        metrics_folder =  os.path.join(args.out_dir,f"metrics/{algo}/{args.dataset}/fragment_len_{f_l}/retention_pos_{r_p}/encryption_key_{e_k}/")
                        os.makedirs(metrics_folder, exist_ok=True)

                        full_dataset, clean_dataset, infected_dataset = merge_datasets(dataset=args.dataset, fragment_len=f_l, retention_pos=r_p, encryption_key=e_k)

                        result_df_path = os.path.join(metrics_folder, f'results_{args.dataset}_dataset_length_{len(full_dataset)}_algo_{algo}_fl_{f_l}_rp_{r_p}_ek{e_k}.csv')
                        
                        # Create group assignments based on unique sequences (for Leave-One-Group-Out CV)
                        groups = get_groups(clean_dataset)

                        k_df = pd.DataFrame()

                        for k in k_list:

                            # Create reads frequency matrix
                            X,y,k_mers_list = k_mers_sparse_matrix(k=k, dataset_full=full_dataset, dataset_clean=clean_dataset, dataset_infected=infected_dataset)

                            #Compute k-mer TF-IDF scores from the frequency matrix
                            features_scores = tf_idf_k_mers_scores(X,k_mers_list)

                            for n in args.n_features:

                                print(f"=== k: {k} - n: {n} ===")

                                top_n_kmers = top_kmers(score_df=features_scores,n=n)
                                # Reduce X to top-n kmers based on TF-IDF
                                X_reduced = X[:, top_n_kmers]

                                # Reduce kmers_list consistently
                                top_kmers_list = [k_mers_list[i] for i in top_n_kmers]
                                print(f"Selected top {n} k-mers by TF-IDF score")
                                print(f"Feature matrix shape: {X_reduced.shape}")

                                n_df = double_cross_validation(full_dataset=full_dataset, groups=groups, X=X_reduced, y=y, algo=algo, k=k, n=n, k_mer_list=top_kmers_list)
                                k_df = pd.concat([k_df,n_df])

                                k_df.to_csv(result_df_path)
                        
                        

if __name__ == "__main__":
    main()