from qiskit_aer import AerSimulator
from qiskit_aer.primitives import Sampler as AerSampler
from qiskit_ibm_runtime import Sampler, QiskitRuntimeService
from qiskit.circuit.library import ZZFeatureMap, RealAmplitudes
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from qiskit_machine_learning.algorithms import QSVC
from qiskit.compiler import transpile
from qiskit.circuit import QuantumCircuit
from qiskit_algorithms.optimizers import COBYLA

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from s01_0qml_in_scm import calculate_metrics, calculate_metrics_hardware, dataset_path, ClaMPDataset, ClaMPDatasetGPT, backends, X, time, train_test_split

class MinimalDataProcessor:
    def __init__(self, dataset_path, target_col='class', exclude_cols=None, num_samples=1000):
        self.dataset_path = dataset_path
        self.target_col = target_col
        self.exclude_cols = exclude_cols or ["e_magic", "e_crlc"]
        self.num_samples = num_samples
        self.df = None
        self.feature_2to10 = []
        self.all_features = []
        # ── new attributes ──
        self.X = None          # feature DataFrame (all columns)
        self.y = None          # target Series
        self.list_atributs = [] # ordered representative features (2-D … 10-D)

    def stratified_ordered_split(self, X, test_size=0.2, random_state=42):
        """
        Stratified train/test split.
        Works with the class-level self.y so callers only pass X.
        Returns X_train, X_test, y_train, y_test as numpy arrays.
        """
        X_train, X_test, y_train, y_test = train_test_split(
            X, self.y,
            test_size=test_size,
            random_state=random_state,
            stratify=self.y
        )
        return (
            X_train.to_numpy(),
            X_test.to_numpy(),
            y_train.to_numpy(),
            y_test.to_numpy(),
        )
    
    def load_and_balance_data(self):
        # Load CSV
        csv_files = [f for f in os.listdir(self.dataset_path) if f.endswith('.csv')]
        self.df = pd.read_csv(os.path.join(self.dataset_path, csv_files[0]))
        
        # Drop excluded columns and balance dataset
        self.df = self.df.drop(columns=self.exclude_cols, errors='ignore')
        class_0 = self.df[self.df[self.target_col] == 0].sample(n=self.num_samples//2)
        class_1 = self.df[self.df[self.target_col] == 1].sample(n=self.num_samples//2)
        self.df = pd.concat([class_0, class_1]).sample(frac=1).reset_index(drop=True)
        
        # Store all features (excluding target)
        self.all_features = [col for col in self.df.columns if col != self.target_col]
        
        return self.df
    
    def _get_representative_feature(self, cluster_features, corr_matrix):
        if len(cluster_features) == 1:
            return cluster_features[0]
        scores = {f: np.mean([abs(corr_matrix.loc[f, other]) 
                            for other in cluster_features if other != f]) 
                 for f in cluster_features}
        return max(scores, key=scores.get)
    
    def generate_feature_clusters(self, min_clusters=2, max_clusters=10):
        data = self.df.drop(columns=[self.target_col])
        
        # Remove constant features that cause correlation issues
        data = data.loc[:, data.std() > 1e-6]
        
        correlations = data.corr().abs().fillna(0)
        
        # Clip correlations to valid range [0, 1]
        correlations = correlations.clip(0, 1)
        
        # Create dissimilarity matrix
        dissimilarity = 1 - correlations
        
        # Ensure perfect symmetry and valid distance properties
        dissimilarity = (dissimilarity + dissimilarity.T) / 2
        np.fill_diagonal(dissimilarity.values, 0)  # Distance to self = 0
        
        # Ensure all values are non-negative
        dissimilarity = dissimilarity.clip(0, None)
        
        try:
            Z = linkage(squareform(dissimilarity), 'ward')
        except ValueError:
            # Fallback: use average linkage if ward fails
            Z = linkage(squareform(dissimilarity), 'average')
        
        for n_clusters in range(min_clusters, max_clusters + 1):
            labels = fcluster(Z, n_clusters, criterion='maxclust')
            
            # Group features by cluster
            clusters = {}
            for idx, feature in enumerate(data.columns):
                cluster_id = labels[idx]
                clusters.setdefault(cluster_id, []).append(feature)
            
            # Get representative features
            representatives = [self._get_representative_feature(features, correlations) 
                             for features in clusters.values()]
            self.feature_2to10.append(representatives)
        
        return self.feature_2to10
    
    def _cluster_and_reorder_features(self, features, n_clusters=None):
        """Cluster features and return them ordered by cluster groups"""
        if len(features) <= 1:
            return features
        
        data = self.df[features]
        correlations = data.corr().abs().fillna(0).clip(0, 1)
        dissimilarity = 1 - correlations
        dissimilarity = (dissimilarity + dissimilarity.T) / 2
        np.fill_diagonal(dissimilarity.values, 0)
        dissimilarity = dissimilarity.clip(0, None)
        
        try:
            Z = linkage(squareform(dissimilarity), 'ward')
        except ValueError:
            Z = linkage(squareform(dissimilarity), 'average')
        
        # Determine number of clusters
        if n_clusters is None:
            n_clusters = min(10, max(2, len(features) // 5))
        
        cluster_labels = fcluster(Z, n_clusters, criterion='maxclust')
        
        # Group features by cluster and order them
        cluster_groups = {}
        for feature, cluster_id in zip(features, cluster_labels):
            cluster_groups.setdefault(cluster_id, []).append(feature)
        
        # Create ordered feature list (all cluster 1, then all cluster 2, etc.)
        ordered_features = []
        for cluster_id in sorted(cluster_groups.keys()):
            ordered_features.extend(sorted(cluster_groups[cluster_id]))
        
        return ordered_features
    
    
    def get_features(self, n_clusters):
        """Get representative features for n clusters (2-10) or all features"""
        if n_clusters == "all":
            return self.all_features
        return self.feature_2to10[n_clusters - 2] if 2 <= n_clusters <= 10 else None
    
    def get_subset(self, n_clusters, include_target=True):
        """Get DataFrame with representative features"""
        features = self.get_features(n_clusters)
        if include_target and features:
            return self.df[features + [self.target_col]]
        return self.df[features] if features else None
    
    def _select_features_for_dimensions(self, n_dimensions):
        """Select features based on dimension count with intelligent selection"""
        if n_dimensions == "all":
            return self.all_features
            
        if n_dimensions <= 10:
            # Use feature_2to10 for dimensions 2-10
            return self.feature_2to10[n_dimensions - 2]
        else:
            # For >10 dimensions: use 10D features + add least correlated from each cluster
            base_features = self.feature_2to10[8]  # 10D features (index 8)
            selected_features = base_features.copy()
            
            if n_dimensions <= len(base_features):
                return selected_features[:n_dimensions]
            
            # Need more features - get original clustering data
            data = self.df.drop(columns=[self.target_col])
            data = data.loc[:, data.std() > 1e-6]
            correlations = data.corr().abs().fillna(0).clip(0, 1)
            dissimilarity = 1 - correlations
            dissimilarity = (dissimilarity + dissimilarity.T) / 2
            np.fill_diagonal(dissimilarity.values, 0)
            dissimilarity = dissimilarity.clip(0, None)
            
            try:
                Z = linkage(squareform(dissimilarity), 'ward')
            except ValueError:
                Z = linkage(squareform(dissimilarity), 'average')
            
            # Get 10 clusters to match our base features
            cluster_labels = fcluster(Z, 10, criterion='maxclust')
            
            # Group all features by their 10 clusters
            clusters = {}
            for idx, feature in enumerate(data.columns):
                cluster_id = cluster_labels[idx]
                clusters.setdefault(cluster_id, []).append(feature)
            
            # Map each base feature to its cluster
            feature_to_cluster = {}
            for cluster_id, cluster_features in clusters.items():
                for base_feature in base_features:
                    if base_feature in cluster_features:
                        feature_to_cluster[base_feature] = cluster_id
                        break
            
            # Add additional features from each cluster (least correlated to existing)
            additional_needed = n_dimensions - len(selected_features)
            added_count = 0
            
            # Cycle through clusters to add features evenly
            for cycle in range((additional_needed // len(feature_to_cluster)) + 1):
                if added_count >= additional_needed:
                    break
                    
                for base_feature in base_features:
                    if added_count >= additional_needed:
                        break
                        
                    cluster_id = feature_to_cluster.get(base_feature)
                    if cluster_id is None:
                        continue
                        
                    cluster_features = clusters[cluster_id]
                    available_features = [f for f in cluster_features if f not in selected_features]
                    
                    if not available_features:
                        continue
                    
                    # Find feature least correlated with the base feature from this cluster
                    correlations_with_base = {}
                    for candidate in available_features:
                        corr_val = abs(correlations.loc[base_feature, candidate])
                        correlations_with_base[candidate] = corr_val
                    
                    # Select feature with lowest correlation to base feature
                    least_correlated = min(correlations_with_base.keys(), 
                                         key=correlations_with_base.get)
                    selected_features.append(least_correlated)
                    added_count += 1
            
            return selected_features[:n_dimensions]  # Ensure exact count
    
    def _force_cluster_grouping(self, features, n_clusters=None):
        """Force features to be grouped by clusters and return ordered features with cluster assignments"""
        if len(features) <= 1:
            return features, [1] * len(features)
        
        data = self.df[features]
        correlations = data.corr().abs().fillna(0).clip(0, 1)
        dissimilarity = 1 - correlations
        dissimilarity = (dissimilarity + dissimilarity.T) / 2
        np.fill_diagonal(dissimilarity.values, 0)
        dissimilarity = dissimilarity.clip(0, None)
        
        try:
            Z = linkage(squareform(dissimilarity), 'ward')
        except ValueError:
            Z = linkage(squareform(dissimilarity), 'average')
        
        # Determine number of clusters
        if n_clusters is None:
            n_clusters = min(10, max(2, len(features) // 5))
        
        cluster_labels = fcluster(Z, n_clusters, criterion='maxclust')
        
        # Group features by cluster and force them to be together
        cluster_groups = {}
        for feature, cluster_id in zip(features, cluster_labels):
            cluster_groups.setdefault(cluster_id, []).append(feature)
        
        # Create ordered feature list (all cluster 1, then all cluster 2, etc.)
        ordered_features = []
        ordered_cluster_assignments = []
        
        for cluster_id in sorted(cluster_groups.keys()):
            cluster_features = sorted(cluster_groups[cluster_id])  # Sort within cluster
            ordered_features.extend(cluster_features)
            ordered_cluster_assignments.extend([cluster_id] * len(cluster_features))
        
        print(f"FORCED grouping: Features ordered by clusters")
        for cluster_id in sorted(cluster_groups.keys()):
            print(f"  Cluster {cluster_id}: {cluster_groups[cluster_id]}")
        
        return ordered_features, ordered_cluster_assignments
    
    def _get_cluster_boundaries(self, cluster_assignments):
        """Get cluster boundary positions from cluster assignments"""
        boundaries = []
        current_cluster = cluster_assignments[0]
        for i, cluster_id in enumerate(cluster_assignments):
            if cluster_id != current_cluster:
                boundaries.append(i)
                current_cluster = cluster_id
        return boundaries
    
    def _add_forced_cluster_labels(self, ax, cluster_assignments, cluster_boundaries):
        """Add cluster labels based on forced cluster grouping"""
        cluster_starts = [0] + cluster_boundaries + [len(cluster_assignments)]
        
        for i in range(len(cluster_starts) - 1):
            start = cluster_starts[i]
            end = cluster_starts[i + 1]
            mid_point = (start + end) / 2
            cluster_id = cluster_assignments[start]
            
            # Row labels (LEFT)
            ax.text(-1.5, mid_point, f'C{cluster_id}', 
                   fontweight='bold', fontsize=12, ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8))
            
            # Column labels (TOP)
            ax.text(mid_point, -1.5, f'C{cluster_id}', 
                   fontweight='bold', fontsize=12, ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8))
    
    def _print_forced_cluster_summary(self, ordered_features, cluster_assignments, n_dimensions):
        """Print summary of forced cluster grouping"""
        print(f"\nFORCED Cluster Summary for {n_dimensions}:")
        print(f"Features are GUARANTEED to be grouped by cluster (no dendrogram reordering)")
        print(f"Ordered features: {ordered_features}")
        
        # Group features by cluster for summary
        cluster_groups = {}
        for feature, cluster_id in zip(ordered_features, cluster_assignments):
            cluster_groups.setdefault(cluster_id, []).append(feature)
        
        print(f"\nCluster groups (in matrix order):")
        for cluster_id in sorted(cluster_groups.keys()):
            features = cluster_groups[cluster_id]
            print(f"Cluster {cluster_id} ({len(features)} features): {features}")
    
    
    def _find_boundaries(self, cluster_labels):
        """Find cluster boundary positions"""
        boundaries = []
        current_cluster = cluster_labels[0]
        for i, cluster_id in enumerate(cluster_labels):
            if cluster_id != current_cluster:
                boundaries.append(i)
                current_cluster = cluster_id
        return boundaries
    
    def _add_external_cluster_labels(self, ax, row_boundaries, col_boundaries, row_labels, col_labels):
        """Add cluster labels OUTSIDE the correlation matrix (to the left and top)"""
        # Row cluster labels on the LEFT (outside the matrix)
        row_starts = [0] + row_boundaries + [len(row_labels)]
        for i in range(len(row_starts) - 1):
            start = row_starts[i]
            end = row_starts[i + 1]
            mid_point = (start + end) / 2
            cluster_id = row_labels[start] if start < len(row_labels) else row_labels[-1]
            
            ax.text(-1.5, mid_point, f'C{cluster_id}', 
                   fontweight='bold', fontsize=12, ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8))
        
        # Column cluster labels at the TOP (outside the matrix)
        col_starts = [0] + col_boundaries + [len(col_labels)]
        for i in range(len(col_starts) - 1):
            start = col_starts[i]
            end = col_starts[i + 1]
            mid_point = (start + end) / 2
            cluster_id = col_labels[start] if start < len(col_labels) else col_labels[-1]
            
            ax.text(mid_point, -1.5, f'C{cluster_id}', 
                   fontweight='bold', fontsize=12, ha='center', va='center',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.8))
    
    def build_list_atributs(self):
        """
        Flatten feature_2to10 into a single ordered list of unique
        representative features (2-D representative first, then 3-D, …).
        This gives a stable column order so dataset(dimension) slices
        the first `dimension` columns correctly.

        Strategy:
          - dimension 2  → feature_2to10[0]  (2 features)
          - dimension 3  → feature_2to10[1]  (3 features)
          - …
          - dimension 10 → feature_2to10[8] (10 features)

        list_atributs is set to the 10-D representative list so that
        dataset(n) returns the first n features of that ordered list.
        This matches the way ClaMPDataset / ClaMPDatasetGPT_ behave.
        """
        if not self.feature_2to10:
            raise RuntimeError("Call generate_feature_clusters() before build_list_atributs().")

        # Use the 10-D feature set as the master ordered list
        # (it contains 10 representative features, one per cluster)
        self.list_atributs = self.feature_2to10[-1]   # index 8 → 10 features
        return self.list_atributs

    # ------------------------------------------------------------------ #
    # NEW: dataset(dimension) — called inside the QSVC training loop      #
    # ------------------------------------------------------------------ #
    def dataset(self, dimension):
        """
        Return a stratified train/test split for the first `dimension`
        features in list_atributs.

        Args:
            dimension: int — number of features to use (2 … len(list_atributs))

        Returns:
            X_train, X_test, y_train, y_test  (numpy arrays)
        """
        if self.X is None or self.y is None:
            raise RuntimeError("Call run_all() or load_and_balance_data() first.")
        if not self.list_atributs:
            raise RuntimeError("Call build_list_atributs() first.")

        selected = self.list_atributs[:dimension]
        # Guard: make sure all selected features exist in X
        missing = [f for f in selected if f not in self.X.columns]
        if missing:
            raise ValueError(f"Features not found in dataset: {missing}")

        X_dim = self.X[selected]
        return self.stratified_ordered_split(X_dim, test_size=0.2)

    # ------------------------------------------------------------------ #
    # UPDATED: run_all — now also builds X, y, list_atributs              #
    # ------------------------------------------------------------------ #
    def run_all(self, plotd=10):
        """Complete pipeline: load → cluster → (plot) → build split attrs."""
        self.load_and_balance_data()

        # Populate self.X and self.y right after loading
        self.X = self.df.drop(columns=[self.target_col])
        self.y = self.df[self.target_col]

        if plotd != "all":
            self.generate_feature_clusters()
            print(f"Generated feature_2to10 with {len(self.feature_2to10)} entries")
            for i, features in enumerate(self.feature_2to10):
                print(f"  {i+2}D: {len(features)} features - {features}")
        else:
            print(f"Processing ALL features: {len(self.all_features)} total")
            self.generate_feature_clusters()

        # Build the master ordered attribute list used by dataset()
        self.build_list_atributs()
        print(f"list_atributs ({len(self.list_atributs)} features): {self.list_atributs}")

        if plotd:
            if plotd == "all":
                print(f"\nGenerating correlation plot for ALL {len(self.all_features)} features...")
            else:
                print(f"\nGenerating {plotd}D correlation plots...")

            if plotd == 10:
                self.plot_10d_correlation_heatmap()
                self.plot_10d_clustermap()

            self.plot_features_clustermap(plotd)

        return self.feature_2to10 if plotd != "all" else self.all_features

def inicial_df():
  df_results = pd.DataFrame({
    'Hardware': [],
    'Dimension': [],
    'TP': [],
    'TN': [],
    'FP': [],
    'FN': [],
    'Accuracy': [],
    'Precision': [],
    'Sensitivity': [],
    'Specificity': [],
    'F1 Score': [],
    'Elapsed Time (s)':[],
    'Usage (s)':[],
    'Estimated Usage (s)': [],
    'Num Qubits': [],
    'Median T1':[],
    'Median T2':[],
    'Median Read Out Error':[]
  })
  return df_results

def metrics_of_evaluation(name, dimension, qsvc,end,start,test_features, test_labels, backend):
  usage_seconds, estimated_running_time_seconds,mean_readout_error, mean_t1, mean_t2 = calculate_metrics_hardware(backend)
  enlapse_time = end-start
  predictions = qsvc.predict(test_features)
  TP, TN, FP, FN, accuracy, precision, sensitivity, specificity, f1_score = calculate_metrics(test_labels, predictions)

  # Create a DataFrame to store the results
  df_results = pd.DataFrame({
      'Hardware':name,
      'Dimension':dimension,
      'TP': [TP],
      'TN': [TN],
      'FP': [FP],
      'FN': [FN],
      'Accuracy': [accuracy],
      'Precision': [precision],
      'Sensitivity': [sensitivity],
      'Specificity': [specificity],
      'F1 Score': [f1_score],
      'Elapsed Time (s)': [enlapse_time],
      'Usage (s)':[usage_seconds],
      'Estimated Usage (s)': [estimated_running_time_seconds],
      'Num Qubits': [backend.num_qubits],
      'Median T1':[mean_t1],
      'Median T2':[mean_t2],
      'Median Read Out Error':[mean_readout_error]

  })
  return df_results

def metrics_of_evaluation_classicaly(svc,dimension,end,start,test_features, test_labels):
  predictions = svc.predict(test_features)
  TP, TN, FP, FN, accuracy, precision, sensitivity, specificity, f1_score = calculate_metrics(test_labels, predictions)
  usage_time = end-start
  # Create a DataFrame to store the results
  df_results = pd.DataFrame({
      'Dimension':dimension,
      'TP': [TP],
      'TN': [TN],
      'FP': [FP],
      'FN': [FN],
      'Accuracy': [accuracy],
      'Precision': [precision],
      'Sensitivity': [sensitivity],
      'Specificity': [specificity],
      'F1 Score': [f1_score],
      'Elapsed Time (s)': [usage_time],
      'Usage (s)':[usage_time],


  })
  return df_results

def should_skip_execution(results_file, samples, start_at):
    if os.path.exists(results_file):
        df_existing = pd.read_csv(results_file)
        # If there are existing records with the same sample count and dimensions greater than or equal to start_at, return True
        if ((df_existing["Samples"] == samples) & (df_existing["Dimension"] >= start_at)).any():
            return True
    
    return False

SIMULATOR_BACKEND_NAME = "aer_simulator"

#     return qsvc, backend
def qsvc_with_ibm_hardware(name, X_train, y_train, X_test, y_test, dimension, service=None):
    use_simulator = name in (SIMULATOR_BACKEND_NAME, "simulator")

    # Always assign backend before any branching
    backend = None

    if use_simulator:
        backend = AerSimulator()
        quantum_kernel = FidelityQuantumKernel(
            feature_map=ZZFeatureMap(feature_dimension=dimension, reps=2)
        )
    else:
        if service is None:
            raise ValueError("A QiskitRuntimeService instance is required for hardware backends.")
        backend = service.backend(name)

        number_qubits    = 127 if name != "ibm_torino" else 133
        feature_map      = ZZFeatureMap(feature_dimension=dimension, reps=2)
        feature_map_compiled   = transpile(feature_map, backend=backend)
        quantum_kernel_circuit = QuantumCircuit(number_qubits)
        quantum_kernel_circuit.append(feature_map_compiled, range(number_qubits))

        fidelity_quantum_kernel = FidelityQuantumKernel()
        fidelity_quantum_kernel._quantum_circuit = quantum_kernel_circuit
        quantum_kernel = fidelity_quantum_kernel

    print("Ok training QSVC")
    qsvc = QSVC(quantum_kernel=quantum_kernel)
    qsvc.fit(X_train, y_train)
    print("finish training QSVC")

    return qsvc, backend  # backend is always defined now

def main_qsvc_update_correlation(
    computers="all", dataset=0, samples=500, start_at=0, end_at=20
):
    # ------------------------------------------------------------------ #
    # Dataset selection                                                    #
    # ------------------------------------------------------------------ #
    if dataset == 1:
        results_file = "qsvc_results_correlation_10_10.csv"
        if should_skip_execution(results_file, samples, end_at):
            return True
        malware = ClaMPDataset(target="class", cut=samples) #wrong
    elif dataset == 0:
        results_file = "qsvc_results_correlation_10_0.csv"
        if should_skip_execution(results_file, samples, end_at):
            return True
        malware = ClaMPDatasetGPT(target="class", cut=samples)
    else:
        results_file = "qsvc_results_hierarchical.csv"
        if should_skip_execution(results_file, samples, end_at):
            return True
        print("HERE0")
        processor = MinimalDataProcessor(
            dataset_path=dataset_path,
            target_col='class',
            num_samples=samples        # ← pass samples so the cut matches
        )
        processor.run_all(plotd=False) # plotd=False skips all plotting
        print("HERE1")
        malware = processor             # ← processor now has .dataset(dimension)
        # return True  # placeholder — remove if more logic follows
    print("HEREyeyy")
    # ------------------------------------------------------------------ #
    # Backend selection                                                    #
    # ------------------------------------------------------------------ #
    if computers == "simulator":
        # Local Aer simulation — no IBM credentials needed
        q_hardwares = [SIMULATOR_BACKEND_NAME]
        service     = None
    elif computers == "all":
        q_hardwares, service = backends()
    else:
        _, service  = backends()
        q_hardwares = [computers]

    print("QSVM with high correlation features:")
    file_exists = os.path.exists(results_file)

    # ------------------------------------------------------------------ #
    # Main training loop                                                   #
    # ------------------------------------------------------------------ #
    for dimension in range(start_at, end_at):
        if should_skip_execution(results_file, samples, dimension):
            continue

        X_train, X_test, y_train, y_test = malware.dataset(dimension)
        print("Shape:", X_train.shape, X_test.shape, y_train.shape, y_test.shape)

        for q_hardware in q_hardwares:
            start = time.time()
            print("HERE3")
            qsvc, backend = qsvc_with_ibm_hardware(
                q_hardware, X_train, y_train, X_test, y_test, dimension, service
            )

            df_model = metrics_of_evaluation(
                q_hardware, dimension, qsvc,
                time.time(), start, X_test, y_test, backend
            )
            df_model["Samples"] = samples

            df_model.to_csv(
                results_file, mode="a", index=False, header=not file_exists
            )
            file_exists = True  # write header only on first append

    return True

TOTAL_SAMPLES = X.shape[0]

# for i in range(0, 2): # Type of features selection: (High High) vs (High Low)
for j in range(1, 11): # Size of dataset: 10%, 20%, ..., 100%
    samples = int((TOTAL_SAMPLES * j) / 10)   
    df_result_qsvc0 = main_qsvc_update_correlation("simulator", dataset=2, samples = samples, start_at=2, end_at=11)
print("**********************************************")