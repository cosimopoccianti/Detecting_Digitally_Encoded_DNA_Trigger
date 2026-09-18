# Alignment-free baseline: plain k-mer relative-frequency vector (every row
# sums to 1, all columns kept, no feature selection), SVM with RBF kernel -
# the "genomic signature" / feature frequency profile (FFP) representation
# (Sims et al. 2009). Reuses the shared instrumented double-CV loop with the
# same RBF param grid already used for the TF-IDF SVM (config.REGISTRY).

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.sparse import diags
from sklearn.svm import SVC
from config import REGISTRY
from double_cv_instrumented import double_cross_validation_instrumented


def alignment_free_double_cv(full_dataset, groups, X, y, k, k_mer_list):
    row_sums = np.asarray(X.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1
    X_freq = diags(1.0 / row_sums) @ X

    return double_cross_validation_instrumented(
        full_dataset=full_dataset, groups=groups, X=X_freq, y=y,
        algo='svm_alignment_free', k=k, n=X_freq.shape[1], k_mer_list=k_mer_list,
        estimator=SVC, param_grid=REGISTRY['svm']['param_grid'],
    )
