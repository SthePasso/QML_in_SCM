#!/usr/bin/env python3
"""
run_quantum_experiments.py
==========================
Run SVC+QuantumKernel and/or QSVC experiments on the malware dataset.
Designed to be executed on a supercomputer from the terminal.

REQUIRED PACKAGES — install once before running
------------------------------------------------
    pip install "qiskit>=1.0,<3.0"
    pip install "qiskit-machine-learning==0.8.2"
    pip install "qiskit-algorithms==0.3.1"
    pip install "qiskit-aer>=0.15"
    pip install scikit-learn pandas numpy scipy

Or all at once:
    pip install "qiskit>=1.0,<3.0" "qiskit-machine-learning==0.8.2" \
                "qiskit-algorithms==0.3.1" "qiskit-aer>=0.15" \
                scikit-learn pandas numpy scipy

Why these versions?
  qiskit >= 1.0          removed the old qiskit.primitives.Sampler (V1)
  qiskit-ml == 0.8.2     last release fully compatible with Qiskit 1.x
  qiskit-aer >= 0.15     provides AerSampler for fast statevector simulation
  qiskit-algorithms 0.3.1 provides ComputeUncompute fidelity

Usage examples
--------------
# Run both quantum models, all strategies, dimensions 2-10
python run_quantum_experiments.py --dataset_path /path/to/dataset

# Run only QSVC on IBM hardware, high_corr strategy, dimensions 2-5
python run_quantum_experiments.py \\
    --dataset_path /path/to/dataset \\
    --model qsvc \\
    --strategy high_corr \\
    --dim_min 2 --dim_max 5 \\
    --ibm_token YOUR_TOKEN \\
    --backend ibm_brisbane

# Resume interrupted run (already-saved folds are automatically skipped)
python run_quantum_experiments.py --dataset_path /path/to/dataset --model svc_qkernel

# Use full dataset (no sampling)
python run_quantum_experiments.py --dataset_path /path/to/dataset --samples 0

Arguments
---------
--dataset_path  Path to directory containing the malware .csv file  [required]
--model         Which model(s) to run: svc_qkernel | qsvc | both    [default: both]
--strategy      Feature strategy: high_corr | high_low | hierarchical | all [default: all]
--dim_min       Minimum number of features / dimensions              [default: 2]
--dim_max       Maximum number of features / dimensions              [default: 10]
--samples       Balanced samples to use (even number, 0 = all data) [default: 1000]
--n_splits      Number of CV folds                                   [default: 10]
--results_dir   Directory to write result CSVs                       [default: results]
--ibm_token     IBM Quantum API token (optional, uses simulator if omitted)
--ibm_channel   IBM channel: ibm_quantum or ibm_cloud               [default: ibm_quantum]
--backend       IBM backend name                                     [default: ibm_brisbane]
--seed          Random seed                                          [default: 42]
--target_col    Target column name in CSV                            [default: class]
--exclude_cols  Comma-separated columns to drop (e.g. e_magic,e_crlc)
"""

import argparse
import logging
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                              precision_score, recall_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Canonical CSV schema
# ─────────────────────────────────────────────────────────────────────────────

RESULT_COLUMNS = [
    "Model", "Fold",
    "TP", "TN", "FP", "FN",
    "Accuracy", "Precision", "Sensitivity", "Specificity", "F1 Score",
    "Elapsed Time (s)", "Usage (s)", "Estimated Usage (s)",
    "Num Qubits", "Median T1", "Median T2", "Median Read Out Error",
]


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(dataset_path, target_col="class",
                 exclude_cols=None, num_samples=1000, seed=42):
    """
    Load the malware CSV, drop exclude_cols, create a balanced sample.

    Parameters
    ----------
    dataset_path : str   directory that contains the .csv file
    target_col   : str   name of the label column
    exclude_cols : list  columns to drop before any processing
    num_samples  : int   total balanced rows to keep (0 = keep all)
    seed         : int   random seed for sampling

    Returns
    -------
    df : pd.DataFrame   balanced dataset (features + target)
    """
    csv_files = [f for f in os.listdir(dataset_path) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in: {dataset_path}")

    csv_path = os.path.join(dataset_path, csv_files[0])
    log.info(f"Loading dataset: {csv_path}")
    df = pd.read_csv(csv_path)
    log.info(f"Raw shape: {df.shape}")

    # Drop explicitly excluded columns
    if exclude_cols:
        df = df.drop(columns=[c for c in exclude_cols if c in df.columns],
                     errors="ignore")

    # Drop columns that are entirely NaN
    df = df.dropna(axis=1, how="all")

    # Drop ALL non-numeric feature columns (object, StringDtype, category)
    non_numeric = [c for c in df.columns
                   if c != target_col
                   and not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        log.info(f"Dropping {len(non_numeric)} non-numeric column(s) "
                 f"at load time: {non_numeric}")
        df = df.drop(columns=non_numeric)

    if num_samples > 0:
        if num_samples % 2 != 0:
            raise ValueError("--samples must be an even number.")
        class_0 = df[df[target_col] == 0]
        class_1 = df[df[target_col] == 1]
        half = num_samples // 2
        min_size = min(len(class_0), len(class_1))
        if min_size < half:
            raise ValueError(
                f"Not enough samples: need {half} per class, "
                f"have {min_size}."
            )
        df = pd.concat([
            class_0.sample(n=half, random_state=seed),
            class_1.sample(n=half, random_state=seed),
        ]).sample(frac=1, random_state=seed).reset_index(drop=True)

    log.info(f"Dataset after balancing: {df.shape}")
    log.info(f"Class distribution:\n{df[target_col].value_counts()}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Feature-selection strategies
# ─────────────────────────────────────────────────────────────────────────────

def get_highest_corr_features(df, target_col, n):
    """Top-n features by |correlation with target|."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    corr = X.corrwith(y).abs().sort_values(ascending=False)
    return corr.index[:n].tolist()


def get_highest_lowest_corr_features(df, target_col, n):
    """Top n//2 + bottom n//2 features by |correlation with target|."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    corr = X.corrwith(y).abs().sort_values(ascending=False)
    top    = n // 2
    bottom = n - top
    return corr.index[:top].tolist() + corr.index[-bottom:].tolist()


def get_hierarchical_corr_features(df, target_col, n):
    """
    Ward hierarchical clustering on feature-feature correlation matrix.
    One representative per cluster: the feature with highest |corr with target|.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    X = X.loc[:, X.std() > 1e-6]          # remove constant columns

    corr_ff = X.corr().abs().fillna(0).clip(0, 1)
    dissim  = 1 - corr_ff
    dissim  = (dissim + dissim.T) / 2
    np.fill_diagonal(dissim.values, 0)
    dissim  = dissim.clip(0, None)

    try:
        Z = linkage(squareform(dissim), "ward")
    except ValueError:
        Z = linkage(squareform(dissim), "average")

    labels      = fcluster(Z, n, criterion="maxclust")
    corr_target = X.corrwith(y).abs()

    clusters = {}
    for feat, lab in zip(X.columns, labels):
        clusters.setdefault(lab, []).append(feat)

    representatives = [
        max(clusters[lab], key=lambda f: corr_target.get(f, 0))
        for lab in sorted(clusters)
    ]
    return representatives[:n]


FEATURE_STRATEGIES = {
    "high_corr"   : get_highest_corr_features,
    "high_low"    : get_highest_lowest_corr_features,
    "hierarchical": get_hierarchical_corr_features,
}


def clean_df(df, target_col):
    """
    Prepare dataframe for ML:
      1. Drop columns that are entirely NaN
      2. Drop non-numeric columns (e.g. string packer labels) — keep target
      3. Median-impute remaining NaN values
    Returns a clean copy — does NOT modify the original df.
    """
    df = df.copy()

    # 1. Drop all-NaN columns
    df = df.dropna(axis=1, how="all")

    # 2. Drop ALL non-numeric columns (object, string, StringDtype, category)
    #    Use is_numeric_dtype so pandas StringDtype is also caught.
    non_numeric = [c for c in df.columns
                   if c != target_col
                   and not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        log.info(f"  Dropping {len(non_numeric)} non-numeric column(s): "
                 f"{non_numeric}")
        df = df.drop(columns=non_numeric)

    # 3. Median-impute remaining NaNs — recompute num_cols AFTER the drop
    num_cols = [c for c in df.columns
               if c != target_col
               and pd.api.types.is_numeric_dtype(df[c])]
    if num_cols:
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    return df


# ─────────────────────────────────────────────────────────────────────────────
# Metrics helpers
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = 0
        tp = len(y_true)
    accuracy    = accuracy_score(y_true, y_pred)
    precision   = precision_score(y_true, y_pred, average="weighted",
                                   zero_division=0)
    sensitivity = recall_score(y_true, y_pred, average="weighted",
                                zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1          = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    return dict(
        TP=int(tp), TN=int(tn), FP=int(fp), FN=int(fn),
        Accuracy=accuracy, Precision=precision,
        Sensitivity=sensitivity, Specificity=specificity,
        **{"F1 Score": f1},
    )


def empty_quantum_metrics():
    return {
        "Usage (s)": 0, "Estimated Usage (s)": 0, "Num Qubits": 0,
        "Median T1": 0, "Median T2": 0, "Median Read Out Error": 0,
    }


def make_row(model_name, fold_label, metrics_dict, elapsed, quantum_dict=None):
    qm  = quantum_dict or empty_quantum_metrics()
    row = {
        "Model": model_name, "Fold": fold_label,
        **metrics_dict,
        "Elapsed Time (s)": elapsed,
        **qm,
    }
    return pd.DataFrame([{col: row.get(col, 0) for col in RESULT_COLUMNS}])


# ─────────────────────────────────────────────────────────────────────────────
# CSV helpers
# ─────────────────────────────────────────────────────────────────────────────

def init_csv(path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    if not os.path.exists(path):
        pd.DataFrame(columns=RESULT_COLUMNS).to_csv(path, index=False)
        log.info(f"Created {path}")


def append_row(path, row_df):
    row_df.to_csv(path, mode="a", header=False, index=False)


def row_exists(path, model_name, fold_label):
    if not os.path.exists(path):
        return False
    df = pd.read_csv(path, usecols=["Model", "Fold"])
    return (
        (df["Model"] == model_name) &
        (df["Fold"].astype(str) == str(fold_label))
    ).any()


# ─────────────────────────────────────────────────────────────────────────────
# Model trainers
# ─────────────────────────────────────────────────────────────────────────────

def _build_kernel_matrix(X_train, X_test, n_features):
    """
    Compute quantum kernel matrices using Qiskit Aer statevector simulation.

    Avoids qiskit.primitives.Sampler entirely — works with Qiskit 1.x + 2.x.
    Uses ZZFeatureMap with reps=2.

    Returns K_train (n_train x n_train) and K_test (n_test x n_train).
    """
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit_machine_learning.kernels import FidelityQuantumKernel

    # ── pick the best available statevector sampler ───────────────────────
    # Priority: Aer (fast, C++ backend) > StatevectorSampler (pure Python)
    sampler = None
    try:
        from qiskit_aer.primitives import Sampler as AerSampler
        sampler = AerSampler()
        sampler.set_options(shots=None)   # statevector mode (exact)
    except Exception:
        pass

    if sampler is None:
        try:
            from qiskit.primitives import StatevectorSampler
            sampler = StatevectorSampler()
        except ImportError:
            pass

    feature_map = ZZFeatureMap(feature_dimension=n_features, reps=2)

    # ── build FidelityQuantumKernel ───────────────────────────────────────
    try:
        from qiskit_algorithms.state_fidelities import ComputeUncompute
        fidelity = ComputeUncompute(sampler=sampler)
        qkernel  = FidelityQuantumKernel(
            feature_map=feature_map, fidelity=fidelity)
    except Exception:
        qkernel = FidelityQuantumKernel(feature_map=feature_map)

    K_train = qkernel.evaluate(x_vec=X_train)
    K_test  = qkernel.evaluate(x_vec=X_test, y_vec=X_train)
    return K_train, K_test


def _build_kernel_matrix_ibm(X_train, X_test, n_features, service, backend_name):
    """Compute quantum kernel matrices on IBM hardware."""
    from qiskit import transpile
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit_ibm_runtime import SamplerV2 as IBMSampler
    from qiskit_machine_learning.kernels import FidelityQuantumKernel

    backend    = service.backend(backend_name)
    feature_map = ZZFeatureMap(feature_dimension=n_features, reps=2)
    fm_compiled = transpile(feature_map, backend=backend)

    try:
        from qiskit_algorithms.state_fidelities import ComputeUncompute
        fidelity = ComputeUncompute(sampler=IBMSampler(backend))
        qkernel  = FidelityQuantumKernel(
            feature_map=fm_compiled, fidelity=fidelity)
    except Exception:
        qkernel = FidelityQuantumKernel(feature_map=fm_compiled)

    K_train = qkernel.evaluate(x_vec=X_train)
    K_test  = qkernel.evaluate(x_vec=X_test, y_vec=X_train)
    return K_train, K_test


def train_svc_qkernel(X_tr, y_tr, X_te, n_features,
                      service=None, backend_name=None):
    """
    SVC with ZZFeatureMap precomputed quantum kernel.
    Simulator: uses qiskit-aer AerSampler (statevector, exact).
    Hardware:  uses IBM SamplerV2 when service + backend_name are given.
    """
    imputer = SimpleImputer(strategy="median")
    scaler  = StandardScaler()
    X_tr_s  = scaler.fit_transform(imputer.fit_transform(X_tr))
    X_te_s  = scaler.transform(imputer.transform(X_te))

    if service is not None and backend_name is not None:
        K_train, K_test = _build_kernel_matrix_ibm(
            X_tr_s, X_te_s, n_features, service, backend_name)
    else:
        K_train, K_test = _build_kernel_matrix(X_tr_s, X_te_s, n_features)

    model = SVC(kernel="precomputed", random_state=42)
    model.fit(K_train, y_tr)
    return model.predict(K_test), empty_quantum_metrics()


def train_qsvc(X_tr, y_tr, X_te, n_features,
               service=None, backend_name=None):
    """
    QSVC with ZZFeatureMap quantum kernel.
    Simulator: uses qiskit-aer AerSampler (statevector, exact).
    Hardware:  uses IBM SamplerV2 when service + backend_name are given.
    """
    from qiskit_machine_learning.algorithms import QSVC

    imputer = SimpleImputer(strategy="median")
    scaler  = StandardScaler()
    X_tr_s  = scaler.fit_transform(imputer.fit_transform(X_tr))
    X_te_s  = scaler.transform(imputer.transform(X_te))

    # QSVC needs the kernel object itself, not the matrices
    from qiskit.circuit.library import ZZFeatureMap
    from qiskit_machine_learning.kernels import FidelityQuantumKernel

    feature_map = ZZFeatureMap(feature_dimension=n_features, reps=2)
    sampler = None
    try:
        from qiskit_aer.primitives import Sampler as AerSampler
        sampler = AerSampler()
        sampler.set_options(shots=None)
    except Exception:
        pass
    if sampler is None:
        try:
            from qiskit.primitives import StatevectorSampler
            sampler = StatevectorSampler()
        except ImportError:
            pass

    try:
        from qiskit_algorithms.state_fidelities import ComputeUncompute
        fidelity = ComputeUncompute(sampler=sampler)
        qkernel  = FidelityQuantumKernel(
            feature_map=feature_map, fidelity=fidelity)
    except Exception:
        qkernel = FidelityQuantumKernel(feature_map=feature_map)

    model = QSVC(quantum_kernel=qkernel)
    model.fit(X_tr_s, y_tr)
    return model.predict(X_te_s), empty_quantum_metrics()


# ─────────────────────────────────────────────────────────────────────────────
# Core CV loop
# ─────────────────────────────────────────────────────────────────────────────

def run_cv(df, target_col, feature_strategy, n_features,
           model_type, csv_path, n_splits=10,
           service=None, backend_name=None):
    """
    10-fold Stratified CV for one (model × strategy × dimension) combination.

    Writes rows to csv_path immediately after each fold.
    Skips folds already present in the CSV (safe to resume).

    Model name format : <PREFIX>_<strategy>_<N>D
    Retrained row     : <name>_retrained, Fold = "<best_fold>_retrained"
    """
    df = clean_df(df, target_col)

    feat_fn  = FEATURE_STRATEGIES[feature_strategy]
    features = feat_fn(df, target_col, n_features)
    X        = df[features].values
    y        = df[target_col]

    prefix     = {"svc_qkernel": "SVCQK", "qsvc": "QSVC"}[model_type]
    model_name = f"{prefix}_{feature_strategy}_{n_features}D"
    retrain_nm = f"{model_name}_retrained"

    log.info(f"{'─'*60}")
    log.info(f"  {model_name}  |  features: {features}")

    skf    = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    splits = list(skf.split(X, y))

    fold_records = []   # (fold_num, precision, train_idx, test_idx)

    for fold_idx, (train_idx, test_idx) in enumerate(splits):
        fold_num = fold_idx + 1

        if row_exists(csv_path, model_name, fold_num):
            saved = pd.read_csv(csv_path)
            mask  = (
                (saved["Model"] == model_name) &
                (saved["Fold"].astype(str) == str(fold_num))
            )
            prec = float(saved.loc[mask, "Precision"].iloc[0])
            fold_records.append((fold_num, prec, train_idx, test_idx))
            log.info(f"  fold {fold_num:2d}  SKIP (already saved)"
                     f"  Precision={prec:.4f}")
            continue

        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

        t0 = time.time()
        if model_type == "svc_qkernel":
            y_pred, q_m = train_svc_qkernel(
                X_tr, y_tr, X_te, n_features, service, backend_name)
        else:
            y_pred, q_m = train_qsvc(
                X_tr, y_tr, X_te, n_features, service, backend_name)
        elapsed = time.time() - t0

        metrics = compute_metrics(y_te, y_pred)
        append_row(csv_path, make_row(model_name, fold_num,
                                      metrics, elapsed, q_m))

        prec = metrics["Precision"]
        fold_records.append((fold_num, prec, train_idx, test_idx))
        log.info(
            f"  fold {fold_num:2d}  Precision={prec:.4f}"
            f"  Accuracy={metrics['Accuracy']:.4f}"
            f"  ({elapsed:.1f}s)"
        )

    # ── Best / Worst retrain ──────────────────────────────────────────────
    precisions = [r[1] for r in fold_records]
    best_idx   = int(np.argmax(precisions))
    worst_idx  = int(np.argmin(precisions))
    best_fold  = fold_records[best_idx]
    worst_fold = fold_records[worst_idx]
    rt_label   = f"{best_fold[0]}_retrained"

    log.info(f"  Best  fold: {best_fold[0]}  (Precision={best_fold[1]:.4f})")
    log.info(f"  Worst fold: {worst_fold[0]}  (Precision={worst_fold[1]:.4f})")

    if row_exists(csv_path, retrain_nm, rt_label):
        log.info("  Retrained row already saved — skipping")
    else:
        X_best_tr  = X[best_fold[2]];  y_best_tr  = y.iloc[best_fold[2]]
        X_worst_te = X[worst_fold[3]]; y_worst_te = y.iloc[worst_fold[3]]

        t0 = time.time()
        if model_type == "svc_qkernel":
            y_pred_r, q_m_r = train_svc_qkernel(
                X_best_tr, y_best_tr, X_worst_te,
                n_features, service, backend_name)
        else:
            y_pred_r, q_m_r = train_qsvc(
                X_best_tr, y_best_tr, X_worst_te,
                n_features, service, backend_name)
        elapsed_r = time.time() - t0

        metrics_r = compute_metrics(y_worst_te, y_pred_r)
        append_row(csv_path, make_row(retrain_nm, rt_label,
                                      metrics_r, elapsed_r, q_m_r))
        log.info(
            f"  Retrained  Precision={metrics_r['Precision']:.4f}"
            f"  Accuracy={metrics_r['Accuracy']:.4f}"
            f"  ({elapsed_r:.1f}s)"
        )


# ─────────────────────────────────────────────────────────────────────────────
# CLI argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Run SVC+QKernel / QSVC malware experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Dataset
    p.add_argument("--dataset_path", required=True,
                   help="Directory containing the malware .csv file")
    p.add_argument("--target_col", default="class",
                   help="Target column name (default: class)")
    p.add_argument("--exclude_cols", default="e_magic,e_crlc",
                   help="Comma-separated columns to drop "
                        "(default: e_magic,e_crlc)")
    p.add_argument("--samples", type=int, default=1000,
                   help="Balanced sample size, 0 = use all data "
                        "(default: 1000)")

    # Experiment scope
    p.add_argument("--model", default="both",
                   choices=["svc_qkernel", "qsvc", "both"],
                   help="Which model to run (default: both)")
    p.add_argument("--strategy", default="all",
                   choices=["high_corr", "high_low", "hierarchical", "all"],
                   help="Feature strategy (default: all)")
    p.add_argument("--dim_min", type=int, default=2,
                   help="Min number of features / dimensions (default: 2)")
    p.add_argument("--dim_max", type=int, default=10,
                   help="Max number of features / dimensions (default: 10)")
    p.add_argument("--n_splits", type=int, default=10,
                   help="CV folds (default: 10)")

    # Output
    p.add_argument("--results_dir", default="results",
                   help="Directory for result CSVs (default: results)")

    # IBM Quantum (optional)
    p.add_argument("--ibm_token", default=None,
                   help="IBM Quantum API token (omit to use simulator)")
    p.add_argument("--ibm_channel", default="ibm_quantum",
                   choices=["ibm_quantum", "ibm_cloud"],
                   help="IBM channel (default: ibm_quantum)")
    p.add_argument("--backend", default="ibm_brisbane",
                   help="IBM backend name (default: ibm_brisbane)")

    # Misc
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed (default: 42)")

    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Print experiment config ───────────────────────────────────────────
    log.info("=" * 65)
    log.info("  Quantum Malware Experiments")
    log.info("=" * 65)
    log.info(f"  dataset_path : {args.dataset_path}")
    log.info(f"  model        : {args.model}")
    log.info(f"  strategy     : {args.strategy}")
    log.info(f"  dimensions   : {args.dim_min} → {args.dim_max}")
    log.info(f"  samples      : {args.samples if args.samples > 0 else 'all'}")
    log.info(f"  cv folds     : {args.n_splits}")
    log.info(f"  results_dir  : {args.results_dir}")
    log.info(f"  IBM backend  : "
             f"{'simulator (no token)' if not args.ibm_token else args.backend}")
    log.info("=" * 65)

    # ── IBM Quantum service setup ─────────────────────────────────────────
    service = None
    backend_name = None
    if args.ibm_token:
        try:
            from qiskit_ibm_runtime import QiskitRuntimeService
            service = QiskitRuntimeService(
                channel=args.ibm_channel,
                token=args.ibm_token,
            )
            backend_name = args.backend
            log.info(f"IBM Quantum connected — backend: {backend_name}")
        except Exception as e:
            log.error(f"IBM Quantum connection failed: {e}")
            log.warning("Falling back to local simulator.")
            service = None

    # ── Load & balance dataset ────────────────────────────────────────────
    exclude = [c.strip() for c in args.exclude_cols.split(",") if c.strip()]
    df = load_dataset(
        dataset_path=args.dataset_path,
        target_col=args.target_col,
        exclude_cols=exclude,
        num_samples=args.samples,
        seed=args.seed,
    )

    # ── Resolve model and strategy lists ─────────────────────────────────
    models_to_run = (
        ["svc_qkernel", "qsvc"] if args.model == "both"
        else [args.model]
    )
    strategies_to_run = (
        list(FEATURE_STRATEGIES.keys()) if args.strategy == "all"
        else [args.strategy]
    )
    dim_range = range(args.dim_min, args.dim_max + 1)

    csv_map = {
        "svc_qkernel": os.path.join(args.results_dir, "df_svc_qkernel.csv"),
        "qsvc"       : os.path.join(args.results_dir, "df_qsvc.csv"),
    }

    # Initialise CSVs
    os.makedirs(args.results_dir, exist_ok=True)
    for model_type in models_to_run:
        init_csv(csv_map[model_type])

    # ── Experiment grid ───────────────────────────────────────────────────
    total = len(models_to_run) * len(strategies_to_run) * len(dim_range)
    done  = 0
    t_start = time.time()

    for model_type in models_to_run:
        csv_path = csv_map[model_type]
        log.info("")
        log.info("=" * 65)
        log.info(f"  MODEL: {model_type.upper()}  →  {csv_path}")
        log.info("=" * 65)

        for strategy in strategies_to_run:
            log.info(f"\n  Strategy: {strategy}")
            for n_feat in dim_range:
                run_cv(
                    df=df,
                    target_col=args.target_col,
                    feature_strategy=strategy,
                    n_features=n_feat,
                    model_type=model_type,
                    csv_path=csv_path,
                    n_splits=args.n_splits,
                    service=service,
                    backend_name=backend_name,
                )
                done += 1
                elapsed_total = time.time() - t_start
                avg_per_combo = elapsed_total / done
                remaining     = (total - done) * avg_per_combo
                log.info(
                    f"  Progress: {done}/{total} combinations  "
                    f"| elapsed {elapsed_total/60:.1f} min  "
                    f"| ETA ~{remaining/60:.1f} min"
                )

    log.info("")
    log.info("=" * 65)
    log.info("  DONE")
    log.info(f"  Total time: {(time.time() - t_start)/60:.1f} min")
    for model_type in models_to_run:
        log.info(f"  {csv_map[model_type]}")
    log.info("=" * 65)


if __name__ == "__main__":
    main()