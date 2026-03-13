from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from scipy.sparse import issparse
from xgboost import XGBClassifier


# Define algorithms and their hyperparameter search spaces
REGISTRY = {

    "naive_bayes": {
    "estimator": GaussianNB,
    "param_grid": {"var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3]},
    "preprocess": lambda X: X.toarray() if issparse(X) else X,
    },

    "logistic": {
        "estimator": LogisticRegression,
        "param_grid": [
        {   # L2 penalty
            "penalty": ["l2"],
            "C": [0.1, 1, 10],
            "solver": ["lbfgs", "newton-cg", "sag", "saga"],
        },
        {   # L1 penalty
            "penalty": ["l1"],
            "C": [0.1, 1, 10],
            "solver": ["liblinear", "saga"],
        },
        {   # ElasticNet
            "penalty": ["elasticnet"],
            "C": [0.1, 1, 10],
            "solver": ["saga"],
            "l1_ratio": [0.25, 0.5, 0.75],
        },
        {   # No penalty
            "penalty": [None],
            "solver": ["lbfgs", "newton-cg", "sag", "saga"],
        },
    ],
    },

    "knn": {
        "estimator": KNeighborsClassifier,
        "param_grid": {"n_neighbors": list(range(2,11,2))},
    },

    "svm": {
        "estimator": SVC,
        "param_grid": {'C': [1,10,100], 
                    'gamma': [0.1,0.01,0.001,0.0001], 
                    'kernel': ['rbf']},
    },
    

    "decision_tree": {
        "estimator": DecisionTreeClassifier,
        "param_grid": {
            "max_depth":        [None, 5, 10, 20],
            "min_samples_leaf": [1, 5, 10],
            "criterion":        ["gini", "entropy"],
        },
    },
    "random_forest": {
        "estimator": RandomForestClassifier,
        "param_grid": {
            "n_estimators": [50, 100, 200],
            "max_depth":    [None, 5, 10],
            "criterion":    ["gini", "entropy"],
        }
    },

    "xgboost": {
        "estimator": XGBClassifier,
        "param_grid": {
            "n_estimators":  [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth":     [3, 5, 7],
            "subsample":     [0.8, 1.0],
        },
    },
}