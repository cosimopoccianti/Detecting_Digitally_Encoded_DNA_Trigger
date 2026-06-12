# Detecting Digitally Encoded DNA Trigger for Trojan Malware in Bio-Cyber Attacks

Code and data for the paper *"Detecting Digitally Encoded DNA Triggers for Trojan Malware in Bio-Cyber Attacks."*

## Method overview

1. **Encoding** — clean and infected reads are converted into k-mer count matrices (`k` from `k_min` to `k_max`).
2. **Feature selection** — k-mers are scored by TF-IDF; the top-`n` are retained.
3. **Classification** — LOGO cross-validation with `GroupKFold` (grouping by source sequence to avoid leakage); inner loop performs `GridSearchCV` hyperparameter tuning.
4. **Payload encryption** — `mixed_key` mode trains on one encryption key and tests across the others.
5. **Interpretability** — `shap` mode computes per-fold SHAP values and k-mer importances.

Models: Naive Bayes, Logistic Regression, k-NN, SVM (RBF), Decision Tree, Random Forest, XGBoost (see `experiments/config.py`).

## Repository structure

```
experiments/
├── main.py                     # entry point / CLI
├── config.py                   # classifier registry + hyperparameter grids
├── utils.py                    # data loading, k-mer matrices, TF-IDF
├── double_cross_validation.py  # LOGO CV
├── encryption_key_eval.py      # cross-key (mixed_key) evaluation
├── shap_analysis.py            # SHAP interpretability
├── datasets/                   # E. coli and Lentivirus reads
│   └── <dataset>/fragment_len_<f>/retention_pos_<r>/encryption_key_<e>/
└── results/metrics/            # per-model output CSVs
```

Datasets are organized by **fragment length** (1–5), **retention position** (0–5), and **encryption key** (0–50). Each leaf contains a clean (`non_trojan_dataset.txt`) and an infected (`nw_best_greedy_trojan_insertion_dataset.txt`) file.

## Requirements

Python 3.12+ with:

```
numpy pandas scipy scikit-learn xgboost shap matplotlib joblib
```

## Usage

Run from the `experiments/` directory.

**LOGO cross-validation (default):**
```bash
python3 main.py --algo svm --dataset ecoli --fragment_len 5 --retention_pos 5 --encryption_key 0
```


**Cross-key robustness:**
```bash
python3 main.py --mode mixed_key --algo svm --dataset ecoli
```

**SHAP interpretability:**
```bash
python3 main.py --fragment_len 5 --retention_pos 5 --encryption_key 0 --mode shap --algo svm --n_shap_folds 5
```

### Main arguments

| Argument | Default | Description |
|---|---|---|
| `--algo` | `svm` | `all`, or one/more of: `naive_bayes`, `logistic`, `knn`, `svm`, `decision_tree`, `random_forest`, `xgboost` |
| `--dataset` | `ecoli` | `ecoli` or `lentivirus` |
| `--k_min`, `--k_max` | `3`, `25` | k-mer length range |
| `--n_features` | `5,10,20,…50,75,100` | number of top TF-IDF k-mers |
| `--fragment_len` | `5` | `all`, int, or comma list (1–5) |
| `--retention_pos` | `5` | `all`, int, or comma list (0–5) |
| `--encryption_key` | `0` | `all`, int, or comma list (0–50) - [0,10,...,50]|
| `--mode` | `double_cross_val` | `double_cross_val`, `mixed_key`, `shap` |
| `--out_dir` | — | output directory |
| `--random_state` | `42` | random seed |

Results are written as CSVs under `results/metrics/<algo>/<dataset>/...`.

## Acknowledgements

- Poccianti et al., "Detecting Digitally Encoded DNA Triggers for Trojan Malware in Bio-Cyber Attacks".

- The dataset injection methods and the padding/dataset_loading functions are from Islam et al., "Using Deep Learning to Detect Digitally Encoded DNA Triggers for Trojan Malware in Bio-Cyber Attacks" (https://github.com/sibleeislam/trojan-malware-in-bio-cyber-attacks).


## License

This work is licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).

[![CC BY 4.0](https://licensebuttons.net/l/by/4.0/88x31.png)](https://creativecommons.org/licenses/by/4.0/)
