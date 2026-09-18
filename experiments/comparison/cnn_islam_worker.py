# Copy of ../Islam/modified_x_comparision/trojan_cnn_instrumented.py, kept
# separate (original untouched) and parameterized by --test_fold instead of
# the hardcoded groups.max(), so cnn_islam.py can run it once per outer LOGO
# fold for double cross-validation. Everything else - architecture,
# hyperparameters, training procedure, venv requirement - is unchanged.
#
# Runs standalone in the dedicated Python 3.8 / TensorFlow 2.4.0 venv
# (experiments/Islam/venv_cnn), CPU-only.

import argparse
import json
import os
import sys
import threading
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Conv1D, Dense, MaxPooling1D, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras as keras

SEQUENCE_SIZE = 1000


class PeakRSSSampler:
    """
    Duplicated verbatim from experiments/utils.py's PeakRSSSampler (not
    imported - this script runs standalone in the dedicated Python 3.8 /
    TensorFlow 2.4.0 venv, isolated from the rest of the project).
    """

    def __init__(self, interval_s=0.005):
        self.interval_s = interval_s
        self._peak_kb = 0
        self._stop = threading.Event()
        self._thread = None
        self._prev_switch_interval = None

    @staticmethod
    def _current_rss_kb():
        try:
            with open('/proc/self/status') as f:
                for line in f:
                    if line.startswith('VmRSS:'):
                        return int(line.split()[1])  # kB, per `man proc`
        except (OSError, ValueError, IndexError):
            pass
        return 0

    def _poll(self):
        while not self._stop.is_set():
            rss = self._current_rss_kb()
            if rss > self._peak_kb:
                self._peak_kb = rss
            self._stop.wait(self.interval_s)

    def __enter__(self):
        self._peak_kb = self._current_rss_kb()
        self._stop.clear()
        self._prev_switch_interval = sys.getswitchinterval()
        sys.setswitchinterval(min(self._prev_switch_interval, 0.0005))
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop.set()
        self._thread.join()
        sys.setswitchinterval(self._prev_switch_interval)
        rss = self._current_rss_kb()
        if rss > self._peak_kb:
            self._peak_kb = rss
        return False

    @property
    def peak_kb(self):
        return self._peak_kb


# --- verbatim from trojan_cnn.py ---

def padding(seq_size_max, line):
    less = seq_size_max - len(line)
    padding_value = line[:less]
    line = line + padding_value
    if (len(line) < seq_size_max):
        line = padding(seq_size_max, line)
    return line


def load_sequence_dataset_from_file(file_path):
    seq_size_max = SEQUENCE_SIZE
    sequences = []
    with open(file_path) as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if len(line) >= seq_size_max:
                sequences.append(line[:seq_size_max])
            else:
                line = padding(seq_size_max, line)
                sequences.append(line)
    return sequences


def check_dataset_validity(dataset, file_path):
    i = 1
    for sequence in dataset:
        if any(x not in ['A', 'T', 'C', 'G', 'N'] for x in set(list(sequence))) or len(sequence) != SEQUENCE_SIZE:
            print('Following problem is found in dataset at ', file_path)
            print('At line number:', i)
            print('Charaters :', set(sequence))
            print('Length: ', len(sequence), 'where sequence len should be', SEQUENCE_SIZE)
            print('Sequence', sequence)
            exit()
        i += 1


def encode_reads(dataset_a, dataset_b):
    """Verbatim one-hot encoding logic from trojan_cnn.py main()."""
    label_a = list(np.zeros(len(dataset_a)))
    label_b = list(np.ones(len(dataset_b)))

    sequences = dataset_a + dataset_b
    labels = label_a + label_b

    one_hot_encoder = OneHotEncoder()
    input_features = []

    one_hot_encoder.fit(np.array(list('ACTGN')).reshape(-1, 1))
    for sequence in sequences:
        one_hot_encoded = one_hot_encoder.transform(np.array(list(sequence)).reshape(-1, 1)).toarray()
        input_features.append(one_hot_encoded)
    input_features = np.stack(input_features)

    label_one_hot_encoder = OneHotEncoder()
    labels = np.array(labels).reshape(-1, 1)
    input_labels = label_one_hot_encoder.fit_transform(labels).toarray()

    return input_features, input_labels


def build_model(input_len, epochs=3000, lrate=0.001):
    """Verbatim CNN architecture, hyperparameters and optimizer from trojan_cnn.py main()."""
    model = Sequential()

    model.add(Conv1D(filters=32, kernel_size=5, padding='same',
                      kernel_initializer='he_uniform',
                      input_shape=(input_len, 5), activation='relu',  # 5 because A C T G N
                      kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.2))
    model.add(Conv1D(filters=32, kernel_size=5, padding='same',
                      kernel_initializer='he_uniform',
                      activation='relu',
                      kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.2))
    model.add(MaxPooling1D(pool_size=2))

    model.add(Flatten())
    model.add(Dense(16, kernel_initializer='he_uniform', activation='relu',
                     kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.1))
    model.add(Dense(2, activation='softmax'))

    decay = lrate / epochs
    sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)

    model.compile(loss='binary_crossentropy', optimizer=sgd, metrics=['binary_accuracy'])
    model.summary()
    return model


def train_model(model, train_features, train_labels, epochs, log_dir):
    """Verbatim training call from trojan_cnn.py main()."""
    early_stopping = EarlyStopping(
        monitor='val_binary_accuracy',
        patience=100,
        mode='max',  # We want to maximize accuracy
        restore_best_weights=True,
        verbose=1
    )
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir)
    history = model.fit(train_features, train_labels, epochs=epochs, batch_size=100,
                         callbacks=[tensorboard_callback, early_stopping], validation_split=0.10)
    return history


# --- instrumentation / orchestration (not part of the original code) ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", required=True, help="path to non_trojan_dataset.txt")
    parser.add_argument("--trojan", required=True, help="path to the trojan-insertion dataset txt")
    parser.add_argument("--groups", required=True, help="path to a .npy file with the group id per read (clean+infected order)")
    parser.add_argument("--test_fold", type=int, required=True, help="which outer LOGO fold id (0..outer_folds_number-1) is held out as test")
    parser.add_argument("--log_dir", required=True)
    parser.add_argument("--out_csv", required=True)
    args = parser.parse_args()

    dataset_a = load_sequence_dataset_from_file(args.clean)
    check_dataset_validity(dataset_a, args.clean)
    dataset_b = load_sequence_dataset_from_file(args.trojan)
    check_dataset_validity(dataset_b, args.trojan)

    groups = np.load(args.groups)

    # --- Time & measure true peak RSS: one-hot encoding (strings -> numeric vectors) ---
    encode_start = time.perf_counter()
    with PeakRSSSampler() as rss_sampler:
        input_features, input_labels = encode_reads(dataset_a, dataset_b)
    encode_time = time.perf_counter() - encode_start
    encode_peak_rss_kb = rss_sampler.peak_kb

    # Same duplicate-aware group split used for SVM / spectrum kernel /
    # alignment-free baseline, one outer LOGO fold held out as test
    test_fold = args.test_fold
    train_idx = np.where(groups != test_fold)[0]
    test_idx = np.where(groups == test_fold)[0]

    train_features, test_features = input_features[train_idx], input_features[test_idx]
    train_labels, test_labels = input_labels[train_idx], input_labels[test_idx]

    # Shuffle only within the training split (train/test assignment stays
    # fixed by the LOGO groups, so no leakage) - otherwise Keras's
    # validation_split takes the last 10% of the array *before* shuffling,
    # which here would be a single-class tail (clean reads then infected
    # reads, unshuffled), making val_binary_accuracy meaningless.
    shuffle_idx = np.random.RandomState(42).permutation(train_features.shape[0])
    train_features = train_features[shuffle_idx]
    train_labels = train_labels[shuffle_idx]

    epochs = 3000
    model = build_model(input_len=train_features.shape[1], epochs=epochs)

    # --- Time & measure true peak RSS for CNN training ---
    train_start = time.perf_counter()
    with PeakRSSSampler() as rss_sampler:
        train_model(model, train_features, train_labels, epochs=epochs, log_dir=args.log_dir)
    train_time = time.perf_counter() - train_start
    train_peak_rss_kb = rss_sampler.peak_kb

    # --- Time & measure true peak RSS for prediction on new data ---
    predict_start = time.perf_counter()
    with PeakRSSSampler() as rss_sampler:
        predicted_labels = model.predict(test_features)
    predict_time = time.perf_counter() - predict_start
    predict_peak_rss_kb = rss_sampler.peak_kb

    y_true = np.argmax(test_labels, axis=1)
    y_pred = np.argmax(predicted_labels, axis=1)
    TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()

    accuracy = (TN + TP) / (TN + FP + FN + TP)
    precision = TP / (TP + FP) if (TP + FP) > 0 else float('nan')
    recall = TP / (TP + FN) if (TP + FN) > 0 else float('nan')
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else float('nan')

    print('>f1=%.4f' % f1)
    print(f'Fold {test_fold} | Encode time: {encode_time:.4f}s | Train time: {train_time:.4f}s | Predict time: {predict_time:.4f}s')

    metrics_columns = ["model", "k", "n", "Outer_KFold", "Model_params", "TN", "FP", "FN", "TP", "Accuracy", "Precision", "Recall", "F1", "Selected_k_mers",
                        "Train_time_s", "Train_peak_rss_kb",
                        "Predict_time_s", "Predict_peak_rss_kb",
                        "Vectorize_time_s", "Vectorize_peak_rss_kb",
                        "TFIDF_score_time_s", "TFIDF_score_peak_rss_kb",
                        "TopN_select_time_s", "TopN_select_peak_rss_kb",
                        "Transform_time_s_total", "Peak_rss_kb_total"]

    model_params = {
        "architecture": "Conv1D(32,5)-Dropout(.2)-Conv1D(32,5)-Dropout(.2)-MaxPool(2)-Flatten-Dense(16)-Dropout(.1)-Dense(2,softmax)",
        "optimizer": "SGD", "lr": 0.001, "momentum": 0.9, "decay": 0.001 / epochs, "nesterov": False,
        "epochs": epochs, "batch_size": 100, "early_stopping_patience": 100,
    }

    # True peak RSS for the whole cnn_islam pipeline: the worst moment across
    # encode/train/predict (peaks combine via max, not sum - unlike time,
    # they aren't additive), matching the SVM-side Peak_rss_kb_total.
    peak_rss_kb_total = max(encode_peak_rss_kb, train_peak_rss_kb, predict_peak_rss_kb)

    row = ["cnn_islam", None, input_features.shape[1], test_fold, json.dumps(model_params),
           TN, FP, FN, TP, accuracy, precision, recall, f1, None,
           train_time, train_peak_rss_kb,
           predict_time, predict_peak_rss_kb,
           encode_time, encode_peak_rss_kb,
           0.0, 0,
           0.0, 0,
           encode_time, peak_rss_kb_total]

    out_df = pd.DataFrame([row], columns=metrics_columns)
    out_df.to_csv(args.out_csv, index=False)
    print(f"Saved results to {args.out_csv}")


if __name__ == '__main__':
    main()
