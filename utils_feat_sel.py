import pandas as pd
import numpy as np

def k_mers_scores(X,kmers_list) -> pd.DataFrame:

    n_samples, n_features = X.shape

    #Document frequency. In how many rows the k-mer is present
    df_i = np.asarray(X.getnnz(axis=0)).ravel().astype(np.float64)

    #Prevalence
    p = df_i / float(n_samples)

    # Balanced DF
    balanced_df = 4.0 * p * (1.0 - p)

    # Column sums: how many times that specific kmers appears in the dataset
    col_sum = np.asarray(X.sum(axis=0)).ravel().astype(np.float64)

    #Mean intensity
    tf_mean_pos = np.zeros(n_features, dtype=np.float64)
    nonzero_df_mask = df_i > 0
    tf_mean_pos[nonzero_df_mask] = col_sum[nonzero_df_mask] / df_i[nonzero_df_mask]

    #Median intensity !!!! QUi la calcola sulle righe quindi è sbagliata?
    Xc = X.tocsc()  # CSC: indptr is column pointer
    tf_median_pos = np.zeros(n_features, dtype=np.float64)

    for j in range(n_features):
        start, end = Xc.indptr[j], Xc.indptr[j + 1]
        if end > start:
            tf_median_pos[j] = float(np.median(Xc.data[start:end]))
        else:
            tf_median_pos[j] = 0.0

    bdf_x_intensity_mean = balanced_df * np.log1p(tf_mean_pos)

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
                "Balanced DF": balanced_df,
                "Balanced DF × Intensità": bdf_x_intensity_mean,
                "TF-IDF unsupervised":idf,
                "TF-IDF unsupervised (mean log-scaled TF)": tfidf_unsup,
                #"Balanced DF × Intensità (mediana)": bdf_x_intensity_median,
                "label": kmers_list
            }
        )
    df_out.index.name = "feature_idx"

    return df_out


def top_kmers(score_df: pd.DataFrame, score_col: str, n:int):
    return(
        score_df.sort_values(by=score_col, ascending=False)
                .head(n)
                .index
                .to_numpy(dtype=int)
    )