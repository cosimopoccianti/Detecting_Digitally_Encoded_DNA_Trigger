# Spectrum kernel SVM: linear kernel over the full, un-reduced k-mer count
# matrix (no TF-IDF reweighting, no top-n selection) - the spectrum kernel is
# exactly the linear kernel over exact k-mer count vectors. Reuses the shared
# instrumented double-CV loop, tuning only C - kernel is fixed to 'linear',
# since that's what makes it a spectrum kernel in the first place.

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.svm import SVC
from double_cv_instrumented import double_cross_validation_instrumented

SPECTRUM_PARAM_GRID = {'C': [1, 10, 100], 'kernel': ['linear']}


def spectrum_kernel_double_cv(full_dataset, groups, X, y, k, k_mer_list):
    return double_cross_validation_instrumented(
        full_dataset=full_dataset, groups=groups, X=X, y=y,
        algo='svm_spectrum', k=k, n=X.shape[1], k_mer_list=k_mer_list,
        estimator=SVC, param_grid=SPECTRUM_PARAM_GRID,
    )
