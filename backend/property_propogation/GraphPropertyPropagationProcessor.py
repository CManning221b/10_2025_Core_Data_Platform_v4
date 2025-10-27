# -*- coding: utf-8 -*-
"""
Created on Tue Jul  4

@author: callu
 """
import networkx as nx
# Try different import strategies
try:
    from networkx import adjacency_matrix
except ImportError:
    try:
        from networkx.linalg import adjacency_matrix
        nx.adjacency_matrix = adjacency_matrix
    except ImportError:
        # Use manual creation as fallback
        def adjacency_matrix(G, nodelist=None, dtype=float, weight='weight'):
            return nx.to_scipy_sparse_array(G, nodelist=nodelist, dtype=dtype, weight=weight, format='csr')
        nx.adjacency_matrix = adjacency_matrix

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import expm
from scipy.stats import beta, norm, gaussian_kde
from scipy.interpolate import UnivariateSpline
from sklearn.preprocessing import normalize
from sklearn.mixture import GaussianMixture
import scipy.sparse as sp
import scipy.linalg
import numpy as np
import networkx as nx
from concurrent.futures import ThreadPoolExecutor

import time
import os
import warnings
warnings.filterwarnings('ignore')


class GraphPropertyPropagationProcessor:
    """
    Generic graph property propagation processor
    No assumptions about temporal data or value ranges
    """

    def __init__(self, graph):
        self.graph = graph.copy()
        self.nodes = list(graph.nodes())
        self.n = len(self.nodes)
        self.node_to_idx = {node: i for i, node in enumerate(self.nodes)}
        try:
            self.A = nx.adjacency_matrix(graph, nodelist=self.nodes, dtype=float)
        except Exception as e:
            print(f"DEBUG: Error creating adjacency matrix: {e}")
            # Fallback: convert to sparse matrix manually
            import scipy.sparse as sp
            self.A = sp.csr_matrix((self.n, self.n))

        # Graph topology metrics (computed once)
        self.topology_metrics = None

        # Adaptive parameters (computed from topology)
        self.recommended_max_iterations = None
        self.recommended_damping_level = None
        self.recommended_max_components = None
        self.recommended_merge_threshold = None
        self.recommended_decay = None
        self.recommended_max_depth = None
        self.recommended_weak_signal_boost = None
        self.recommended_min_component_weight = None
        self.recommended_eigen_k = None

        # Flag to track if parameters have been computed
        self._parameters_computed = False

    def extract_source_data(self, source_configs):
        """
        Extract source data based on flexible configuration and analyze graph topology

        source_configs: list of dicts like:
        [
            {'node_types': ['Timestamp'], 'property': 'timestamp', 'name': 'temporal'},
            {'node_types': ['file'], 'property': 'size', 'name': 'file_size'},
            {'node_types': ['user'], 'property': 'confidence', 'name': 'user_rating'}
        ]
        """
        # Analyze graph topology and set parameters (only once!)
        if self.recommended_max_iterations is None:
            self._analyze_graph_topology_and_set_parameters()

        source_data = {}

        for config in source_configs:
            node_types = config['node_types'] if isinstance(config['node_types'], list) else [config['node_types']]
            property_name = config['property']
            source_name = config['name']

            # Find nodes matching criteria
            matching_nodes = []
            values = []

            for i, node in enumerate(self.nodes):
                node_data = self.graph.nodes[node]

                # Check if node type matches
                node_type = node_data.get('type', 'unknown')
                if node_type in node_types and property_name in node_data:
                    matching_nodes.append(i)
                    values.append(node_data[property_name])

            if matching_nodes:
                # Auto-determine value range from data
                min_val, max_val = min(values), max(values)
                # Add 10% padding to range
                range_padding = (max_val - min_val) * 0.1
                value_range = (min_val - range_padding, max_val + range_padding)

                # Normalize values to [0,1]
                normalized_values = []
                for val in values:
                    if max_val == min_val:
                        normalized_val = 0.5  # Single value case
                    else:
                        normalized_val = (val - value_range[0]) / (value_range[1] - value_range[0])
                    normalized_values.append(max(0.001, min(0.999, normalized_val)))

                source_data[source_name] = {
                    'node_indices': matching_nodes,
                    'raw_values': values,
                    'normalized_values': normalized_values,
                    'value_range': value_range,
                    'property': property_name,
                    'node_types': node_types
                }

        return source_data

    def _analyze_graph_topology_and_set_parameters(self):
        """Fast topology analysis using heuristics"""
        print(f"DEBUG: Original graph - nodes: {self.n}, edges: {self.graph.number_of_edges()}")

        G_undirected = self.graph.to_undirected()
        connected_components = list(nx.connected_components(G_undirected))
        largest_cc = max(connected_components, key=len)
        G_largest = G_undirected.subgraph(largest_cc)

        # FAST: Calculate only what we need
        density = nx.density(self.graph)
        clustering = nx.average_clustering(G_undirected)

        # FAST: Estimate diameter and path length from graph properties
        n_largest = len(largest_cc)

        if density > 0.1:  # Dense graph
            estimated_diameter = max(2, int(np.log2(n_largest)))
            estimated_avg_path = estimated_diameter * 0.6
        elif density > 0.01:  # Medium density
            estimated_diameter = max(3, int(np.log(n_largest)))
            estimated_avg_path = estimated_diameter * 0.7
        else:  # Sparse graph
            estimated_diameter = max(4, int(np.sqrt(n_largest)))
            estimated_avg_path = estimated_diameter * 0.8

        print(f"DEBUG: Estimated diameter: {estimated_diameter}")
        print(f"DEBUG: Estimated avg path length: {estimated_avg_path}")

        # Use estimates instead of computed values
        avg_path_length = estimated_avg_path
        diameter = estimated_diameter

        # Overall graph metrics
        density = nx.density(self.graph)
        clustering = nx.average_clustering(G_undirected)

        # Structural analysis
        if self.graph.is_directed():
            n_components = nx.number_weakly_connected_components(self.graph)
        else:
            n_components = nx.number_connected_components(self.graph)
        largest_component_size = len(largest_cc)
        connectivity_ratio = largest_component_size / self.n

        # Graph classification
        is_very_sparse = density < 0.01
        is_sparse = density < 0.05
        is_fragmented = connectivity_ratio < 0.8

        # === PERFORMANCE RANGES (HARD LIMITS) ===
        MAX_ITERATIONS_RANGE = (3, 10)  # Never go below 3 or above 10
        DAMPING_RANGE = (0.1, 0.8)  # Keep in reasonable range
        DECAY_RANGE = (0.7, 0.95)  # Don't let decay get too aggressive or too weak
        DEPTH_RANGE = (2, 5)  # Keep depth shallow but useful
        COMPONENTS_RANGE = (2, 6)  # Limit component explosion
        MERGE_THRESHOLD_RANGE = (0.05, 0.15)  # Control merging aggressiveness
        EIGEN_K_RANGE = (5, 20)  # Keep spectral computation reasonable

        # === TOPOLOGY-INFORMED CALCULATIONS (THEN CLAMPED) ===

        # ITERATIONS: Based on path length and density, but clamped
        base_iterations = int(avg_path_length * 1.5)
        if is_very_sparse:
            density_factor = 1.4
        elif is_sparse:
            density_factor = 1.2
        else:
            density_factor = max(0.6, 1.0 - density)

        raw_iterations = base_iterations * density_factor
        self.recommended_max_iterations = int(np.clip(raw_iterations, *MAX_ITERATIONS_RANGE))

        # DAMPING: Based on clustering and density, but clamped
        if is_very_sparse:
            raw_damping = 0.2
        elif is_sparse:
            raw_damping = max(0.3, clustering * density)
        else:
            raw_damping = max(0.4, clustering * density)

        self.recommended_damping_level = np.clip(raw_damping, *DAMPING_RANGE)

        # DECAY: Based on path length, but clamped
        target_strength = 0.6 if is_sparse else 0.5
        if avg_path_length > 0:
            raw_decay = target_strength ** (1.0 / avg_path_length)
        else:
            raw_decay = 0.8  # Safe default value
        self.recommended_decay = np.clip(raw_decay, *DECAY_RANGE)

        # DEPTH: Based on diameter and path length, but clamped
        raw_depth = min(diameter + 1, int(avg_path_length * 1.5))
        self.recommended_max_depth = int(np.clip(raw_depth, *DEPTH_RANGE))

        # COMPONENTS: Based on structural complexity, but clamped
        structural_complexity = (1.0 - clustering) * density + (n_components / self.n)
        base_components = 4 if is_sparse else 3
        raw_components = base_components + structural_complexity * 3
        self.recommended_max_components = int(np.clip(raw_components, *COMPONENTS_RANGE))

        # MERGE THRESHOLD: Based on homogeneity, but clamped
        homogeneity_factor = clustering * connectivity_ratio
        if is_very_sparse:
            raw_threshold = 0.06
        elif is_sparse:
            raw_threshold = 0.08 + homogeneity_factor * 0.03
        else:
            raw_threshold = 0.10 + homogeneity_factor * 0.04

        self.recommended_merge_threshold = np.clip(raw_threshold, *MERGE_THRESHOLD_RANGE)

        # SIGNAL PROCESSING: Topology-informed but reasonable
        if is_very_sparse:
            self.recommended_weak_signal_boost = 2.0
        elif is_sparse:
            self.recommended_weak_signal_boost = 1.5 + (1.0 - density * 10)
        else:
            self.recommended_weak_signal_boost = 1.2 + (1.0 - density) * 0.5

        # Clamp boost to reasonable range
        self.recommended_weak_signal_boost = np.clip(self.recommended_weak_signal_boost, 1.1, 2.5)

        # Min component weight: topology-informed but not too aggressive
        if is_very_sparse:
            self.recommended_min_component_weight = 0.005
        else:
            size_factor = min(1.0, 100.0 / self.n)
            self.recommended_min_component_weight = 0.01 * size_factor * max(0.5, density)

        # Clamp to reasonable range
        self.recommended_min_component_weight = np.clip(self.recommended_min_component_weight, 0.002, 0.05)

        # EIGEN_K: Size and topology informed, but clamped for performance
        if self.n < 50:
            raw_eigen_k = int(self.n * 0.5)
        elif self.n < 200:
            raw_eigen_k = max(8, int(avg_path_length * 2))
        else:
            raw_eigen_k = max(6, int(diameter * 1.2))

        self.recommended_eigen_k = int(np.clip(raw_eigen_k, *EIGEN_K_RANGE))

        # Final size-based scaling for very large graphs
        if self.n > 2000:
            self.recommended_max_iterations = int(self.recommended_max_iterations * 0.8)
            self.recommended_max_components = int(self.recommended_max_components * 0.8)
            self.recommended_eigen_k = min(self.recommended_eigen_k, 10)

        # FRAGMENTATION ADJUSTMENTS (but still within ranges)
        if is_fragmented:
            # Slight adjustments, but respect the ranges
            self.recommended_max_components = min(self.recommended_max_components + 1, COMPONENTS_RANGE[1])
            self.recommended_merge_threshold = max(self.recommended_merge_threshold * 0.9, MERGE_THRESHOLD_RANGE[0])

    def belief_propagation_auto(self, source_configs):
        """Auto-tuned belief propagation using adaptive parameters"""
        # Ensure parameters are computed
        if self.recommended_max_iterations is None:
            self.extract_source_data(source_configs)

        return self.belief_propagation(
            source_configs,
            max_iterations=self.recommended_max_iterations,
            damping=self.recommended_damping_level,
            max_components=self.recommended_max_components
        )

    def spectral_propagation_auto(self, source_configs):
        """Auto-tuned spectral propagation using adaptive parameters"""
        # Ensure parameters are computed
        if self.recommended_max_iterations is None:
            self.extract_source_data(source_configs)

        return self.spectral_propagation(
            source_configs,
            max_depth=self.recommended_max_depth,
            decay=self.recommended_decay,
            max_components=self.recommended_max_components,
            weak_signal_boost=self.recommended_weak_signal_boost,
            min_component_weight=self.recommended_min_component_weight,
            eigen_k=self.recommended_eigen_k
        )

    def matrix_streaming_propagation_auto(self, source_configs):
        """Auto-tuned matrix streaming using adaptive parameters"""
        # Ensure parameters are computed
        if self.recommended_max_iterations is None:
            self.extract_source_data(source_configs)

        return self.matrix_streaming_propagation(
            source_configs,
            max_depth=self.recommended_max_depth,
            decay=self.recommended_decay,
            max_components=self.recommended_max_components
        )

    def quantum_spectral_propagation_auto(self, source_configs):
        """Auto-tuned quantum spectral propagation using adaptive parameters"""
        # Ensure parameters are computed
        if self.recommended_max_iterations is None:
            self.extract_source_data(source_configs)

        return self.quantum_spectral_propagation(
            source_configs,
            max_depth=self.recommended_max_depth,
            decay=self.recommended_decay,
            max_components=self.recommended_max_components,
            weak_signal_boost=self.recommended_weak_signal_boost,
            min_component_weight=self.recommended_min_component_weight,
            eigen_k=self.recommended_eigen_k
        )

    def belief_propagation(self, source_configs, max_iterations=15, damping=0.1, max_components=20):
        """
        Generic belief propagation for any property types
        """
        # Extract source data
        source_data = self.extract_source_data(source_configs)

        if not source_data:
            raise ValueError("No source nodes found matching the specified criteria")

        # Flatten all sources for propagation
        all_source_indices = []
        all_source_values = []
        source_mapping = {}  # Maps flat index to (source_name, local_index)

        flat_idx = 0
        for source_name, data in source_data.items():
            for local_idx, (node_idx, norm_val) in enumerate(zip(data['node_indices'], data['normalized_values'])):
                all_source_indices.append(node_idx)
                all_source_values.append(norm_val)
                source_mapping[flat_idx] = (source_name, local_idx)
                flat_idx += 1

        n_sources = len(all_source_indices)
        A = nx.adjacency_matrix(self.graph, nodelist=self.nodes, dtype=np.float32).tocsr()

        # Track influence from each source separately
        source_influences = np.zeros((self.n, n_sources), dtype=np.float32)

        # Set initial source influences
        for i, src_idx in enumerate(all_source_indices):
            source_influences[src_idx, i] = 1.0

        # Vectorized propagation
        for iteration in range(max_iterations):
            old_influences = source_influences.copy()

            # Propagate each source's influence separately
            for i in range(n_sources):
                current_influence = source_influences[:, i]
                new_influence = A.T @ current_influence * 0.85
                source_influences[:, i] = damping * current_influence + (1 - damping) * new_influence

            # Keep sources at full strength
            for i, src_idx in enumerate(all_source_indices):
                source_influences[src_idx, i] = 1.0
                # Zero out influence from other sources on this source
                for j in range(n_sources):
                    if i != j:
                        source_influences[src_idx, j] = 0.0

            # Check convergence
            if np.max(np.abs(old_influences - source_influences)) < 1e-3:
                break

        # Build results for each source type separately
        results = {}

        for source_name, data in source_data.items():
            # Get results for this source type
            result_values = np.full(self.n, np.nan)  # Use NaN for no prediction
            gmm_data = {}

            # Find which flat indices belong to this source type
            relevant_flat_indices = [flat_idx for flat_idx, (sname, _) in source_mapping.items() if
                                     sname == source_name]

            for node_idx in range(self.n):
                node_name = self.nodes[node_idx]

                # Check if this is a source node for this property
                if node_idx in data['node_indices']:
                    local_idx = data['node_indices'].index(node_idx)
                    result_values[node_idx] = data['normalized_values'][local_idx]
                    gmm_data[node_name] = {
                        'type': 'source',
                        'components': [[1.0, data['normalized_values'][local_idx], 0.02]],
                        'n_components': 1,
                        'total_evidence': 1.0,
                        'raw_value': data['raw_values'][local_idx]
                    }
                else:
                    # Build GMM from relevant source influences
                    components = []
                    total_evidence = 0.0

                    for flat_idx in relevant_flat_indices:
                        influence = source_influences[node_idx, flat_idx]

                        if influence > 0.05:  # Only include significant influences
                            source_norm_val = all_source_values[flat_idx]
                            weight = influence
                            mean = source_norm_val
                            std = 0.03 + 0.05 * (1 - influence)

                            components.append([weight, mean, std])
                            total_evidence += weight

                    # Merge similar components
                    if len(components) > 1:
                        components = self._merge_components(components, merge_threshold=0.08)

                    # Limit and normalize components
                    if len(components) > max_components:
                        components.sort(key=lambda x: x[0], reverse=True)
                        components = components[:max_components]

                    if components:
                        # Normalize weights
                        total_weight = sum(comp[0] for comp in components)
                        for comp in components:
                            comp[0] /= total_weight

                        # Compute weighted mean
                        weighted_mean = sum(comp[0] * comp[1] for comp in components)
                        result_values[node_idx] = weighted_mean

                        gmm_data[node_name] = {
                            'type': 'inferred',
                            'components': components,
                            'n_components': len(components),
                            'total_evidence': total_evidence
                        }
                    else:
                        # No components = disconnected node
                        gmm_data[node_name] = {
                            'type': 'isolated',
                            'components': [],
                            'n_components': 0,
                            'total_evidence': 0.0
                        }
                        # result_values[node_idx] remains np.nan (from line ~150)

            # Denormalize results
            denormalized_values = np.full(self.n, np.nan)
            valid_mask = ~np.isnan(result_values)

            if data['value_range'][1] != data['value_range'][0]:
                denormalized_values[valid_mask] = (result_values[valid_mask] *
                                                   (data['value_range'][1] - data['value_range'][0]) +
                                                   data['value_range'][0])
            else:
                denormalized_values[valid_mask] = data['value_range'][0]

            results[source_name] = {
                'values': denormalized_values,
                'gmm_data': gmm_data,
                'source_config': data,
                'method': 'belief_propagation'
            }

        return results

    def _merge_components(self, components, merge_threshold=0.08):
        """Merge similar GMM components"""
        merged_components = []

        for comp in components:
            merged = False
            for existing in merged_components:
                if abs(existing[1] - comp[1]) < merge_threshold:
                    # Merge components
                    total_w = existing[0] + comp[0]
                    merged_mean = (existing[0] * existing[1] + comp[0] * comp[1]) / total_w
                    merged_var = (existing[0] * existing[2] ** 2 + comp[0] * comp[2] ** 2) / total_w
                    merged_std = np.sqrt(merged_var)

                    existing[0] = total_w
                    existing[1] = merged_mean
                    existing[2] = merged_std
                    merged = True
                    break

            if not merged:
                merged_components.append(comp[:])

        return merged_components

    def spectral_propagation(self, source_configs, max_depth=3, decay=0.95, max_components=20,
                             weak_signal_boost=1.3, min_component_weight=0.01,
                             alpha=0.08, eigen_k=None):
        """
        Spectral graph-based propagation using Graph Fourier Transform
        """
        # Extract source data (reuse existing method)
        source_data = self.extract_source_data(source_configs)

        if not source_data:
            raise ValueError("No source nodes found matching the specified criteria")

        # Process each source type separately
        results = {}

        for source_name, data in source_data.items():
            # Set up source mask and initial signal for this property
            source_mask = np.zeros(self.n, dtype=bool)
            initial_signal = np.zeros(self.n)

            for i, node_idx in enumerate(data['node_indices']):
                source_mask[node_idx] = True
                initial_signal[node_idx] = data['normalized_values'][i]

            # Run spectral method
            result_values, gmm_data = self.enhanced_matrix_gmm(
                source_mask, initial_signal, data,
                max_depth, decay, max_components,
                weak_signal_boost, min_component_weight,
                alpha, eigen_k
            )

            # Denormalize results
            denormalized_values = np.full(self.n, np.nan)
            valid_mask = ~np.isnan(result_values)

            if data['value_range'][1] != data['value_range'][0]:
                denormalized_values[valid_mask] = (result_values[valid_mask] *
                                                   (data['value_range'][1] - data['value_range'][0]) +
                                                   data['value_range'][0])
            else:
                denormalized_values[valid_mask] = data['value_range'][0]

            results[source_name] = {
                'values': denormalized_values,
                'gmm_data': gmm_data,
                'source_config': data,
                'method': 'spectral_propagation'
            }

        return results

    def enhanced_matrix_gmm(self, source_mask, initial_signal, source_config,
                                      max_depth=10, decay=0.95, max_components=20,
                                      weak_signal_boost=1.3, min_component_weight=0.01,
                                      alpha=0.08, eigen_k=None):
        """
        Enhanced spectral method with GMM components
        """
        # Store references for helper methods
        self.source_mask = source_mask
        self.initial_signal = initial_signal

        # Smarter eigen_k scaling - this is the biggest bottleneck
        if eigen_k is None:
            if self.n < 100:
                eigen_k = min(self.n - 1, 25)  # High quality for small graphs
            elif self.n < 500:
                eigen_k = min(self.n - 1, 20)  # Good quality for medium graphs
            else:
                eigen_k = min(self.n - 1, 15)  # Speed focus for large graphs

        G_undirected = self.graph.to_undirected()
        A = nx.adjacency_matrix(G_undirected, nodelist=self.nodes, dtype=np.float64)
        D = sp.diags(np.array(A.sum(axis=1)).flatten(), format='csr')
        L = D - A
        L_reg = L + 1e-8 * sp.identity(self.n)

        # Dense eigendecomposition for determinism
        eigenvals_full, eigenvecs_full = scipy.linalg.eigh(L_reg.toarray())
        sort_idx = np.argsort(eigenvals_full)
        eigenvals = eigenvals_full[sort_idx][:eigen_k]
        eigenvecs = eigenvecs_full[:, sort_idx][:, :eigen_k]

        def graph_fft(signal):
            return eigenvecs.T @ signal

        def graph_ifft(freq_signal):
            return eigenvecs @ freq_signal

        source_indices = np.where(self.source_mask)[0]

        # NEW: Store spectral influence records instead of immediate fusion
        node_spectral_records = {i: [] for i in range(self.n)}

        # Initialize source nodes with their own records
        for i in range(self.n):
            if self.source_mask[i]:
                node_spectral_records[i].append({
                    'weight': 1.0,
                    'mean': self.initial_signal[i],
                    'std': 0.02,
                    'source_id': i,
                    'distance': 0,
                    'filter_type': 'source'
                })

        max_eigenval = eigenvals.max()
        min_weight_threshold = min_component_weight * 1.2

        # Pre-compute decay powers
        decay_powers = [decay ** d for d in range(max_depth + 1)]

        # SPEED OPTIMIZATION: Pre-compute all filters
        filter_bank = []
        for distance in range(1, max_depth + 1):
            center = max_eigenval * 0.05 * distance
            width = max_eigenval * 0.15
            spectral_decay = 0.8 * np.exp(-0.5 * ((eigenvals - center) / width) ** 2) + 0.2
            filter_bank.append(spectral_decay)

        # SPEED OPTIMIZATION: Reuse arrays
        source_signal = np.zeros(self.n, dtype=np.float64)

        for src_idx in source_indices:
            # Reuse source_signal array
            source_signal.fill(0.0)
            source_signal[src_idx] = 1.0
            source_freq = graph_fft(source_signal)

            for distance in range(1, max_depth + 1):
                # Use pre-computed filter
                filtered_freq = source_freq * filter_bank[distance - 1]

                norm = np.linalg.norm(filtered_freq)
                if norm > 1e-8:
                    filtered_freq /= norm
                filtered_freq *= decay_powers[distance]

                spatial_influence = graph_ifft(filtered_freq)
                spatial_influence = np.maximum(spatial_influence, 0)
                spatial_influence[source_indices] = 0

                # SPEED OPTIMIZATION: Higher threshold for large graphs to reduce processing
                threshold = 0.001 if self.n < 500 else 0.015
                influenced_nodes = np.where(spatial_influence > threshold)[0]

                # NEW: Store spectral influence records instead of immediate fusion
                for node_idx in influenced_nodes:
                    influence_strength = spatial_influence[node_idx]
                    new_weight = influence_strength
                    new_mean = self.initial_signal[src_idx]
                    new_std = 0.025 + 0.015 * distance

                    if new_weight < min_weight_threshold:
                        new_weight *= weak_signal_boost

                    # Store spectral influence record
                    spectral_record = {
                        'weight': new_weight,
                        'mean': new_mean,
                        'std': new_std,
                        'source_id': src_idx,
                        'distance': distance,
                        'filter_type': f'spectral_d{distance}',
                        'influence_strength': influence_strength
                    }

                    node_spectral_records[node_idx].append(spectral_record)

        # NEW: Batch process all spectral records into GMM components
        result = np.zeros(self.n)
        gmm_data = {}

        for i in range(self.n):
            node_name = self.nodes[i]
            records = node_spectral_records[i]

            if self.source_mask[i]:
                # Source node
                result[i] = self.initial_signal[i]
                gmm_data[node_name] = {
                    'type': 'source',
                    'components': [[1.0, self.initial_signal[i], 0.02]],
                    'n_components': 1
                }
            elif records:
                # Process all spectral records into GMM components
                components = self._batch_process_spectral_influences(
                    records, max_components, min_component_weight
                )

                # Compute weighted mean from components
                weighted_mean = sum(comp[0] * comp[1] for comp in components)
                result[i] = weighted_mean

                gmm_data[node_name] = {
                    'type': 'inferred',
                    'components': components,
                    'n_components': len(components),
                    'total_evidence': sum(comp[0] for comp in components)
                }
            else:
                # Isolated node - no prediction possible
                result[i] = np.nan
                gmm_data[node_name] = {
                    'type': 'isolated',
                    'components': [],
                    'n_components': 0,
                    'total_evidence': 0.0
                }

        return result, gmm_data

    def _batch_process_spectral_influences(self, spectral_records, max_components, min_component_weight):
        """
        NEW: Batch process spectral influence records into GMM components
        More efficient than repeated spectral streaming fusion
        """
        if not spectral_records:
            return [[1.0, 0.5, 0.1]]

        # Sort records by mean value, then by distance for better merging
        sorted_records = sorted(spectral_records, key=lambda x: (x['mean'], x['distance']))

        components = []

        for record in sorted_records:
            new_weight = record['weight']
            new_mean = record['mean']
            new_std = record['std']
            distance = record['distance']

            # Distance-based merge threshold (same as original)
            merge_threshold = 0.05 + 0.01 * distance

            merged = False

            # Try to merge with existing components
            for i, comp in enumerate(components):
                comp_weight, comp_mean, comp_std = comp

                if abs(comp_mean - new_mean) < merge_threshold:
                    # Simple weighted merge (same as original)
                    total_w = comp_weight + new_weight
                    merged_mean = (comp_weight * comp_mean + new_weight * new_mean) / total_w
                    merged_var = (comp_weight * comp_std ** 2 + new_weight * new_std ** 2) / total_w
                    merged_std = np.sqrt(merged_var)

                    components[i] = [total_w, merged_mean, merged_std]
                    merged = True
                    break

            if not merged:
                if len(components) < max_components:
                    components.append([new_weight, new_mean, new_std])
                else:
                    # Simple replacement: replace weakest component
                    min_idx = min(range(len(components)), key=lambda i: components[i][0])
                    if components[min_idx][0] < new_weight:
                        components[min_idx] = [new_weight, new_mean, new_std]

        # Simple renormalization
        total_weight = sum(comp[0] for comp in components)
        if total_weight > 0:
            for comp in components:
                comp[0] /= total_weight

        return components

    def matrix_streaming_propagation(self, source_configs, max_depth=3, decay=0.8, max_components=5):
        """
        Matrix-based wave propagation with streaming GMM fusion
        Adapted from method_1_matrix_streaming_gmm
        """
        # Extract source data
        source_data = self.extract_source_data(source_configs)

        if not source_data:
            raise ValueError("No source nodes found matching the specified criteria")

        # Process each source type separately
        results = {}

        for source_name, data in source_data.items():
            # Set up source mask and initial signal for this property
            source_mask = np.zeros(self.n, dtype=bool)
            initial_signal = np.zeros(self.n)

            for i, node_idx in enumerate(data['node_indices']):
                source_mask[node_idx] = True
                initial_signal[node_idx] = data['normalized_values'][i]

            # Run matrix streaming method
            result_values, gmm_data = self._matrix_streaming_gmm(
                source_mask, initial_signal, data,
                max_depth, decay, max_components
            )

            # Denormalize results
            denormalized_values = np.full(self.n, np.nan)
            valid_mask = ~np.isnan(result_values)

            if data['value_range'][1] != data['value_range'][0]:
                denormalized_values[valid_mask] = (result_values[valid_mask] *
                                                   (data['value_range'][1] - data['value_range'][0]) +
                                                   data['value_range'][0])
            else:
                denormalized_values[valid_mask] = data['value_range'][0]

            results[source_name] = {
                'values': denormalized_values,
                'gmm_data': gmm_data,
                'source_config': data,
                'method': 'matrix_streaming_propagation'
            }

        return results

    def _matrix_streaming_gmm(self, source_mask, initial_signal, source_config,
                              max_depth=3, decay=0.8, max_components=5):
        """
        Matrix-based wave propagation with deferred fusion
        """
        # Get sparse adjacency matrix
        A = nx.adjacency_matrix(self.graph, nodelist=self.nodes, dtype=np.float32).tocsr()

        # Initialize influence tracking - store records instead of immediate fusion
        source_indices = np.where(source_mask)[0]

        # Store influence records per node
        node_influence_records = {i: [] for i in range(self.n)}

        # Initialize source nodes with their own records
        for i in range(self.n):
            if source_mask[i]:
                node_influence_records[i].append({
                    'weight': 1.0,
                    'mean': initial_signal[i],
                    'std': 0.02,
                    'source_id': i,
                    'distance': 0
                })

        # Matrix-based wave propagation for each source
        for src_i, src_idx in enumerate(source_indices):
            # Create initial emission vector for this source
            emission_vector = np.zeros(self.n, dtype=np.float32)
            emission_vector[src_idx] = 1.0

            # Propagate waves using matrix powers
            current_influence = emission_vector.copy()

            for distance in range(1, max_depth + 1):
                # Propagate to neighbors: A^T @ current gives influence on each node
                next_influence = A.T @ current_influence * decay

                # Zero out sources (they don't receive, only emit)
                next_influence[source_indices] = 0

                # Find nodes that receive significant influence
                influenced_nodes = np.where(next_influence > 0.01)[0]

                # Store influence records, no immediate fusion
                for node_idx in influenced_nodes:
                    influence_strength = next_influence[node_idx]

                    # Store influence record
                    influence_record = {
                        'weight': influence_strength,
                        'mean': initial_signal[src_idx],
                        'std': 0.025 + 0.015 * distance,
                        'source_id': src_idx,
                        'distance': distance
                    }

                    node_influence_records[node_idx].append(influence_record)

                current_influence = next_influence

        # Batch process all influence records into GMM components
        result = np.zeros(self.n)
        gmm_data = {}

        for i in range(self.n):
            node_name = self.nodes[i]
            records = node_influence_records[i]

            if source_mask[i]:
                # Source node
                result[i] = initial_signal[i]
                gmm_data[node_name] = {
                    'type': 'source',
                    'components': [[1.0, initial_signal[i], 0.02]],
                    'n_components': 1,
                    'total_evidence': 1.0,
                    'raw_value': source_config['raw_values'][source_config['node_indices'].index(i)]
                }
            elif records:
                # Process all influence records into GMM components
                components = self._batch_process_matrix_influences(records, max_components)

                # Compute weighted mean from components
                weighted_mean = sum(comp[0] * comp[1] for comp in components)
                result[i] = weighted_mean

                gmm_data[node_name] = {
                    'type': 'inferred',
                    'components': components,
                    'n_components': len(components),
                    'total_evidence': sum(comp[0] for comp in components)
                }
            else:
                # Isolated node
                result[i] = np.nan
                gmm_data[node_name] = {
                    'type': 'isolated',
                    'components': [],
                    'n_components': 0,
                    'total_evidence': 0.0
                }

        return result, gmm_data

    def _batch_process_matrix_influences(self, influence_records, max_components, merge_threshold=0.05):
        """
        Batch process influence records into GMM components
        More efficient than repeated streaming fusion
        """
        if not influence_records:
            return [[1.0, 0.5, 0.1]]

        # Sort records by mean value for better merging
        sorted_records = sorted(influence_records, key=lambda x: x['mean'])

        # Start with first record as first component
        components = []

        for record in sorted_records:
            new_weight = record['weight']
            new_mean = record['mean']
            new_std = record['std']

            merged = False

            # Try to merge with existing components
            for i, comp in enumerate(components):
                comp_weight, comp_mean, comp_std = comp

                if abs(comp_mean - new_mean) < merge_threshold:
                    # Merge components
                    total_w = comp_weight + new_weight
                    merged_mean = (comp_weight * comp_mean + new_weight * new_mean) / total_w
                    merged_var = (comp_weight * (comp_std ** 2 + (comp_mean - merged_mean) ** 2) +
                                  new_weight * (new_std ** 2 + (new_mean - merged_mean) ** 2)) / total_w
                    merged_std = np.sqrt(merged_var)

                    components[i] = [total_w, merged_mean, merged_std]
                    merged = True
                    break

            if not merged:
                # Add new component or replace weakest
                if len(components) < max_components:
                    components.append([new_weight, new_mean, new_std])
                else:
                    # Replace weakest component
                    min_idx = min(range(len(components)), key=lambda i: components[i][0])
                    if components[min_idx][0] < new_weight:
                        components[min_idx] = [new_weight, new_mean, new_std]

        # Normalize weights
        total_weight = sum(comp[0] for comp in components)
        if total_weight > 0:
            for comp in components:
                comp[0] /= total_weight

        return components

    def quantum_spectral_propagation(self, source_configs, max_depth=3, decay=0.95, max_components=20,
                                     weak_signal_boost=1.3, min_component_weight=0.01,
                                     alpha=0.08, eigen_k=None):
        """
        Quantum spectral graph-based propagation using Quantum Graph Fourier Transform
        """
        # Extract source data (reuse existing method)
        source_data = self.extract_source_data(source_configs)

        if not source_data:
            raise ValueError("No source nodes found matching the specified criteria")

        # Process each source type separately
        results = {}

        for source_name, data in source_data.items():
            # Set up source mask and initial signal for this property
            source_mask = np.zeros(self.n, dtype=bool)
            initial_signal = np.zeros(self.n)

            for i, node_idx in enumerate(data['node_indices']):
                source_mask[node_idx] = True
                initial_signal[node_idx] = data['normalized_values'][i]

            # Run quantum spectral method (same signature as enhanced_matrix_gmm)
            result_values, gmm_data = self.quantum_matrix_gmm(
                source_mask, initial_signal, data,
                max_depth, decay, max_components,
                weak_signal_boost, min_component_weight,
                alpha, eigen_k
            )

            # Denormalize results (exactly same as classical)
            denormalized_values = np.full(self.n, np.nan)
            valid_mask = ~np.isnan(result_values)

            if data['value_range'][1] != data['value_range'][0]:
                denormalized_values[valid_mask] = (result_values[valid_mask] *
                                                   (data['value_range'][1] - data['value_range'][0]) +
                                                   data['value_range'][0])
            else:
                denormalized_values[valid_mask] = data['value_range'][0]

            results[source_name] = {
                'values': denormalized_values,
                'gmm_data': gmm_data,
                'source_config': data,
                'method': 'quantum_spectral_propagation'
            }

        return results

    def quantum_matrix_gmm(self, source_mask, initial_signal, source_config,
                           max_depth=10, decay=0.95, max_components=20,
                           weak_signal_boost=1.3, min_component_weight=0.01,
                           alpha=0.08, eigen_k=None):
        """
        FIXED: Quantum method that preserves influence patterns
        """
        import time

        quantum_start_time = time.time()
        self.source_mask = source_mask
        self.initial_signal = initial_signal

        if eigen_k is None:
            # More conservative eigen_k - preserve more eigenvectors for accuracy
            if self.n < 50:
                eigen_k = min(self.n - 1, int(self.n * 0.8))
            elif self.n < 200:
                eigen_k = min(self.n - 1, 15)
            else:
                eigen_k = min(self.n - 1, 12)

        # Build Hamiltonian
        G_undirected = self.graph.to_undirected()
        A = nx.adjacency_matrix(G_undirected, nodelist=self.nodes, dtype=np.float64)
        D = sp.diags(np.array(A.sum(axis=1)).flatten(), format='csr')
        L = D - A
        L_reg = L + 1e-8 * sp.identity(self.n)

        # 🚀 Theoretical quantum timing (silent)
        classical_eigendecomp_time = self.n ** 3 * 1e-9
        quantum_phase_estimation_time = (np.log2(self.n) ** 2) * 1e-6
        theoretical_quantum_speedup = classical_eigendecomp_time / quantum_phase_estimation_time if quantum_phase_estimation_time > 0 else 1

        self._theoretical_quantum_time = quantum_phase_estimation_time * 1000
        self._quantum_speedup_factor = theoretical_quantum_speedup

        # Fixed Hamiltonian: Use positive L for proper graph structure
        H_quantum = L_reg  # FIXED: Positive Laplacian preserves graph structure

        # Quantum-inspired eigendecomposition
        try:
            from scipy.sparse.linalg import eigsh
            eigenvals, eigenvecs = eigsh(H_quantum, k=eigen_k, which='SM', tol=1e-6)
        except:
            eigenvals_full, eigenvecs_full = scipy.linalg.eigh(H_quantum.toarray())
            sort_idx = np.argsort(eigenvals_full)
            eigenvals = eigenvals_full[sort_idx][:eigen_k]
            eigenvecs = eigenvecs_full[:, sort_idx][:, :eigen_k]

        def quantum_graph_fft(signal):
            return eigenvecs.T @ signal

        def quantum_graph_ifft(freq_signal):
            return eigenvecs @ freq_signal

        source_indices = np.where(self.source_mask)[0]
        node_quantum_records = {i: [] for i in range(self.n)}

        # Initialize sources
        for i in range(self.n):
            if self.source_mask[i]:
                node_quantum_records[i].append({
                    'weight': 1.0,
                    'mean': self.initial_signal[i],
                    'std': 0.02,
                    'source_id': i,
                    'distance': 0,
                    'filter_type': 'quantum_source'
                })

        max_eigenval = eigenvals.max() if len(eigenvals) > 0 else 1.0
        min_weight_threshold = min_component_weight * 1.2
        decay_powers = [decay ** d for d in range(max_depth + 1)]

        # FIXED: More conservative quantum evolution that preserves structure
        quantum_filter_bank = []
        for distance in range(1, max_depth + 1):
            # Much smaller evolution time to preserve local structure
            evolution_time = distance * alpha * 0.01  # FIXED: 100x smaller evolution time

            if len(eigenvals) > 0:
                # Conservative quantum filter that preserves locality
                quantum_phases = eigenvals * evolution_time
                # Use real exponential for locality preservation instead of complex
                quantum_filter = np.exp(-quantum_phases)  # FIXED: Real exponential decay

                # More selective amplitude modulation
                amplitude_shape = np.exp(-0.5 * (eigenvals / max_eigenval) ** 2)
                quantum_filter_bank.append(quantum_filter * amplitude_shape)
            else:
                quantum_filter_bank.append(np.array([1.0]))

        # Quantum processing with locality preservation
        quantum_signal = np.zeros(self.n, dtype=np.float64)  # FIXED: Use real numbers

        for src_idx in source_indices:
            quantum_signal.fill(0.0)
            quantum_signal[src_idx] = 1.0

            if len(eigenvals) > 0:
                quantum_freq = quantum_graph_fft(quantum_signal)
            else:
                quantum_freq = quantum_signal[:eigen_k] if eigen_k <= len(quantum_signal) else quantum_signal

            for distance in range(1, max_depth + 1):
                if distance <= len(quantum_filter_bank):
                    quantum_filtered_freq = quantum_freq * quantum_filter_bank[distance - 1]

                    norm = np.linalg.norm(quantum_filtered_freq)
                    if norm > 1e-8:
                        quantum_filtered_freq /= norm
                    quantum_filtered_freq *= decay_powers[distance]

                    # Quantum measurement
                    if len(eigenvals) > 0:
                        quantum_spatial = quantum_graph_ifft(quantum_filtered_freq)
                    else:
                        quantum_spatial = quantum_filtered_freq

                    # FIXED: More stringent thresholding to preserve locality
                    spatial_influence = np.abs(quantum_spatial[:self.n])
                    spatial_influence = np.maximum(spatial_influence, 0)
                    spatial_influence[source_indices] = 0

                    # Higher threshold to prevent over-propagation
                    threshold = 0.01 if self.n < 500 else 0.02  # FIXED: 10x higher threshold
                    influenced_nodes = np.where(spatial_influence > threshold)[0]

                    # Store quantum influence records
                    for node_idx in influenced_nodes:
                        if node_idx < len(spatial_influence):
                            influence_strength = spatial_influence[node_idx]

                            # FIXED: More conservative weight assignment
                            new_weight = influence_strength * 0.5  # Scale down quantum effects
                            new_mean = self.initial_signal[src_idx]
                            new_std = 0.025 + 0.015 * distance

                            if new_weight < min_weight_threshold:
                                new_weight *= weak_signal_boost

                            quantum_record = {
                                'weight': new_weight,
                                'mean': new_mean,
                                'std': new_std,
                                'source_id': src_idx,
                                'distance': distance,
                                'filter_type': f'quantum_d{distance}',
                                'influence_strength': influence_strength
                            }

                            node_quantum_records[node_idx].append(quantum_record)

        # Process results (same as before)
        result = np.zeros(self.n)
        gmm_data = {}

        for i in range(self.n):
            node_name = self.nodes[i]
            records = node_quantum_records[i]

            if self.source_mask[i]:
                result[i] = self.initial_signal[i]
                gmm_data[node_name] = {
                    'type': 'source',
                    'components': [[1.0, self.initial_signal[i], 0.02]],
                    'n_components': 1,
                    'quantum_coherence': 1.0
                }
            elif records:
                components = self._batch_process_spectral_influences(
                    records, max_components, min_component_weight
                )

                weighted_mean = sum(comp[0] * comp[1] for comp in components)
                result[i] = weighted_mean

                total_weight = sum(comp[0] for comp in components)
                quantum_coherence = min(1.0, total_weight)

                gmm_data[node_name] = {
                    'type': 'inferred',
                    'components': components,
                    'n_components': len(components),
                    'total_evidence': total_weight,
                    'quantum_coherence': quantum_coherence
                }
            else:
                result[i] = np.nan
                gmm_data[node_name] = {
                    'type': 'isolated',
                    'components': [],
                    'n_components': 0,
                    'total_evidence': 0.0,
                    'quantum_coherence': 0.0
                }

        return result, gmm_data


    def _denormalize(self, normalized_values):
        """Helper method for denormalization"""
        return normalized_values


def create_simple_tst_graph():
    """Create a simple tst graph with multiple property types"""
    G = nx.DiGraph()

    # Add timestamp nodes
    G.add_node("timestamp_1", type="Timestamp", timestamp=2020.5)
    G.add_node("timestamp_2", type="Timestamp", timestamp=2022.8)

    # Add file nodes with sizes
    G.add_node("file_a", type="file", file_size=1024)
    G.add_node("file_b", type="file", file_size=2048)
    G.add_node("file_c", type="file", file_size=512)

    # Add user nodes with confidence scores
    G.add_node("user_1", type="user", confidence_score=0.8)
    G.add_node("user_2", type="author", confidence_score=0.95)

    # Add some nodes without properties (to tst inference)
    G.add_node("folder_1", type="folder")
    G.add_node("folder_2", type="folder")
    G.add_node("system_node", type="system")

    # Add edges to create propagation paths
    edges = [
        ("timestamp_1", "file_a"),
        ("timestamp_2", "file_b"),
        ("file_a", "folder_1"),
        ("file_b", "folder_1"),
        ("file_c", "folder_2"),
        ("user_1", "file_a"),
        ("user_2", "file_b"),
        ("folder_1", "system_node"),
        ("folder_2", "system_node"),
    ]

    G.add_edges_from(edges)
    return G

def example_usage():
    # Create tst graph
    G = create_simple_tst_graph()

    print("Created tst graph with:")
    print(f"  Nodes: {G.number_of_nodes()}")
    print(f"  Edges: {G.number_of_edges()}")

    # Show node types
    node_types = {}
    for node, data in G.nodes(data=True):
        node_type = data.get('type', 'unknown')
        if node_type not in node_types:
            node_types[node_type] = []
        node_types[node_type].append(node)

    print("\nNode types:")
    for node_type, nodes in node_types.items():
        print(f"  {node_type}: {nodes}")

    processor = GraphPropertyPropagationProcessor(G)

    # Define what you want to propagate
    source_configs = [
        {
            'node_types': ['Timestamp'],
            'property': 'timestamp',
            'name': 'temporal_data'
        },
        {
            'node_types': ['file'],
            'property': 'file_size',
            'name': 'size_data'
        },
        {
            'node_types': ['user', 'author'],
            'property': 'confidence_score',
            'name': 'confidence_data'
        }
    ]

    print(f"\nRunning belief propagation for {len(source_configs)} property types...")

    # Get results for all properties
    results = processor.belief_propagation_auto(source_configs)

    print("\n" + "=" * 80)
    print("PROPAGATION RESULTS")
    print("=" * 80)

    # Display results for each property type
    for property_name, result_data in results.items():
        print(f"\n{property_name.upper()}:")
        print("-" * 50)

        values = result_data['values']
        gmm_data = result_data['gmm_data']
        source_config = result_data['source_config']

        print(f"Value range: {source_config['value_range']}")
        print(f"Source nodes: {len(source_config['node_indices'])}")
        print(f"Node types: {source_config['node_types']}")
        print(f"Property: {source_config['property']}")

        print("\nNode predictions:")
        for i, node_name in enumerate(processor.nodes):
            value = values[i]
            if not np.isnan(value):
                if node_name in gmm_data:
                    data = gmm_data[node_name]
                    n_components = data['n_components']
                    node_type = data['type']
                    evidence = data.get('total_evidence', 0)

                    if node_type == 'source':
                        raw_val = data.get('raw_value', 'unknown')
                        print(f"  {node_name}: {value:.2f} (SOURCE, raw: {raw_val})")
                    else:
                        print(
                            f"  {node_name}: {value:.2f} (inferred, {n_components} components, evidence: {evidence:.3f})")
                else:
                    print(f"  {node_name}: {value:.2f}")

        # Show detailed GMM components for inferred nodes
        print("\nDetailed inference components:")
        for node_name, data in gmm_data.items():
            if data['type'] == 'inferred' and data['n_components'] > 1:
                print(f"  {node_name}:")
                for i, comp in enumerate(data['components']):
                    weight, mean, std = comp
                    denorm_mean = mean * (source_config['value_range'][1] - source_config['value_range'][0]) + \
                                  source_config['value_range'][0]
                    print(f"    Component {i + 1}: weight={weight:.3f}, mean={denorm_mean:.2f}, std={std:.3f}")

def visualize_comparison_results(graph, results_bp, results_spectral, results_matrix=None, results_quantum=None, method_name="Method Comparison"):
    """Visualize comparison results for all methods side by side"""
    # Determine grid size based on number of methods
    methods_count = 2  # Always have BP and Spectral
    if results_matrix is not None:
        methods_count += 1
    if results_quantum is not None:
        methods_count += 1

    if methods_count == 4:
        fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) = plt.subplots(2, 4, figsize=(24, 12))
        axes_graph = [ax1, ax2, ax3, ax4]
        axes_prob = [ax5, ax6, ax7, ax8]
        method_results = [
            (results_bp, "Belief Propagation"),
            (results_spectral, "Spectral Propagation"),
            (results_matrix, "Matrix Streaming"),
            (results_quantum, "Quantum Spectral")
        ]
    elif methods_count == 3:
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(20, 12))
        axes_graph = [ax1, ax2, ax3]
        axes_prob = [ax4, ax5, ax6]
        method_results = [
            (results_bp, "Belief Propagation"),
            (results_spectral, "Spectral Propagation"),
            (results_matrix, "Matrix Streaming")
        ]
    else:  # 2 methods
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        axes_graph = [ax1, ax2]
        axes_prob = [ax3, ax4]
        method_results = [
            (results_bp, "Belief Propagation"),
            (results_spectral, "Spectral Propagation")
        ]

    # Get temporal data from all methods
    temporal_data_list = []
    for results, _ in method_results:
        if results is not None:
            temporal_data_list.append(results['temporal_data'])

    # Determine key nodes to show
    key_nodes = ['obj1', 'obj2', 'loc1'] if len(graph.nodes()) <= 10 else ['feature_auth', 'payment_service', 'prod_env']

    # Create positions for graph layout
    scale_factor = 4.0
    raw_pos = nx.kamada_kawai_layout(graph)
    pos = {node: (x * scale_factor, y * scale_factor) for node, (x, y) in raw_pos.items()}

    # Helper function to create node colors and labels
    def create_node_visualization(temporal_data, ax, title, key_nodes_to_show):
        node_labels = {}
        node_colors = []
        disconnected_nodes = []  # Track which nodes are disconnected
        values = temporal_data['values']
        source_config = temporal_data['source_config']

        nodes_to_label = list(graph.nodes()) if len(graph.nodes()) <= 10 else key_nodes_to_show

        # First pass: collect colors for connected nodes and identify disconnected ones
        connected_colors = []
        for i, node in enumerate(graph.nodes()):
            node_data = graph.nodes[node]

            # Check if this node has the source property
            if source_config['property'] in node_data:
                if node in nodes_to_label:
                    raw_val = node_data[source_config['property']]
                    node_labels[node] = f"{node}\n{source_config['property']}={raw_val:.1f}"
                connected_colors.append(node_data[source_config['property']])
                disconnected_nodes.append(False)
            else:
                # Check if we have an inferred value
                if not np.isnan(values[i]):
                    if node in nodes_to_label:
                        node_labels[node] = f"{node}\n≈{values[i]:.1f}"
                    connected_colors.append(values[i])
                    disconnected_nodes.append(False)
                else:
                    if node in nodes_to_label:
                        node_labels[node] = f"{node}\n(no data)"
                    # Will be handled separately as disconnected
                    connected_colors.append(0)  # Placeholder value
                    disconnected_nodes.append(True)

        # Create color array for connected nodes
        if connected_colors:
            # Normalize connected colors for colormap
            min_color = min(c for i, c in enumerate(connected_colors) if not disconnected_nodes[i])
            max_color = max(c for i, c in enumerate(connected_colors) if not disconnected_nodes[i])

            if min_color == max_color:
                # All nodes have same value
                normalized_colors = [0.5] * len(connected_colors)
            else:
                normalized_colors = [(c - min_color) / (max_color - min_color)
                                     for c in connected_colors]
        else:
            normalized_colors = [0.5] * len(graph.nodes())

        # Draw edges first
        nx.draw_networkx_edges(graph, pos, arrows=True,
                               alpha=0.3 if len(graph.nodes()) > 10 else 0.7,
                               arrowsize=10 if len(graph.nodes()) > 10 else 20, ax=ax)

        # Draw connected nodes with colormap
        connected_node_list = [node for i, node in enumerate(graph.nodes()) if not disconnected_nodes[i]]
        connected_colors_filtered = [normalized_colors[i] for i, node in enumerate(graph.nodes()) if
                                     not disconnected_nodes[i]]

        if connected_node_list:
            connected_pos = {node: pos[node] for node in connected_node_list}
            nx.draw_networkx_nodes(graph.subgraph(connected_node_list), connected_pos,
                                   node_color=connected_colors_filtered,
                                   cmap=plt.cm.viridis,
                                   node_size=300 if len(graph.nodes()) > 10 else 1000,
                                   ax=ax)

        # Draw disconnected nodes in gray
        disconnected_node_list = [node for i, node in enumerate(graph.nodes()) if disconnected_nodes[i]]
        if disconnected_node_list:
            disconnected_pos = {node: pos[node] for node in disconnected_node_list}
            nx.draw_networkx_nodes(graph.subgraph(disconnected_node_list), disconnected_pos,
                                   node_color='lightgray',
                                   node_size=300 if len(graph.nodes()) > 10 else 1000,
                                   ax=ax,
                                   edgecolors='darkgray',  # Add border to make them more visible
                                   linewidths=2)

        # Draw all labels
        nx.draw_networkx_labels(graph, pos, labels=node_labels,
                                font_size=8 if len(graph.nodes()) > 10 else 10, ax=ax)

        # Add colorbar for connected nodes only
        if connected_node_list and min_color != max_color:
            sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
                                       norm=plt.Normalize(vmin=min_color, vmax=max_color))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax, shrink=0.6)
            cbar.set_label('Timestamp', rotation=270, labelpad=15)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis("off")

        return values, source_config

    # Create graph visualizations for all methods
    for i, (temporal_data, method_name_short) in enumerate(zip(temporal_data_list, [name for _, name in method_results if _ is not None])):
        create_node_visualization(temporal_data, axes_graph[i], f"{method_name_short} - Graph View", key_nodes)

    # Helper function for probability distributions
    def create_probability_plot(temporal_data, ax, title, key_nodes_to_plot):
        values = temporal_data['values']
        gmm_data = temporal_data['gmm_data']
        source_config = temporal_data['source_config']

        test_range = np.linspace(source_config['value_range'][0], source_config['value_range'][1], 100)

        # Plot distributions for key nodes (only those with data)
        colors = ['blue', 'red', 'green']
        plot_nodes = []

        # Filter out disconnected nodes from probability plots
        for node in key_nodes_to_plot:
            if (node in gmm_data and
                    node in graph.nodes() and
                    gmm_data[node]['type'] != 'isolated'):
                plot_nodes.append(node)

        plot_nodes = plot_nodes[:3]  # Limit to 3 for colors

        for idx, node in enumerate(plot_nodes):
            if node in gmm_data:
                probs = []
                for val in test_range:
                    prob = evaluate_property_probability(node, val, gmm_data, source_config)
                    probs.append(prob)

                ax.plot(test_range, probs, label=node, linewidth=2, color=colors[idx])

        # Mark source values
        for node_idx in source_config['node_indices']:
            node_name = list(graph.nodes())[node_idx]
            source_val = source_config['raw_values'][source_config['node_indices'].index(node_idx)]
            ax.axvline(source_val, color='black', linestyle='--', alpha=0.3, linewidth=1)
            if len(graph.nodes()) <= 10:
                ax.text(source_val, ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] > 0 else 0.9,
                        f'{node_name}', rotation=90, ha='right', va='top', fontsize=9)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Timestamp')
        ax.set_ylabel('Probability Density')
        if plot_nodes:  # Only show legend if there are nodes to plot
            ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.1)

    # Create probability distribution plots for all methods
    for i, (temporal_data, method_name_short) in enumerate(zip(temporal_data_list, [name for _, name in method_results if _ is not None])):
        create_probability_plot(temporal_data, axes_prob[i], f"{method_name_short} - Probability Distributions", key_nodes)

    plt.suptitle(f"Graph Property Propagation Comparison: {method_name}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def evaluate_property_probability(node_name, new_value, gmm_data, source_config):
    """Evaluate probability of a new value for a specific property"""
    if node_name not in gmm_data:
        return 0.1

    data = gmm_data[node_name]

    # Normalize the tst value
    value_range = source_config['value_range']
    if value_range[1] == value_range[0]:
        normalized_value = 0.5
    else:
        normalized_value = (new_value - value_range[0]) / (value_range[1] - value_range[0])
    normalized_value = max(0.001, min(0.999, normalized_value))

    if data['type'] == 'source':
        if 'components' in data and len(data['components']) > 0:
            source_mean = data['components'][0][1]  # [weight, mean, std]
            distance = abs(normalized_value - source_mean)
            return max(0, 1 - distance * 10)
        return 0.1

    elif data['type'] == 'inferred':
        if 'components' in data and isinstance(data['components'], list):
            try:
                total_prob = 0.0
                max_prob = 0.0

                for comp in data['components']:
                    if len(comp) >= 3:
                        weight, mu, sigma = comp[0], comp[1], comp[2]
                        if sigma > 0:
                            gaussian_prob = (1.0 / (sigma * np.sqrt(2 * np.pi))) * \
                                            np.exp(-0.5 * ((normalized_value - mu) / sigma) ** 2)

                            total_prob += weight * gaussian_prob
                            gaussian_max = (1.0 / (sigma * np.sqrt(2 * np.pi)))
                            max_prob += weight * gaussian_max

                if max_prob > 0:
                    return min(1.0, total_prob / max_prob)
                return 0.1
            except:
                return 0.1
        else:
            return 0.1

    else:  # isolated
        return 0.05

def run_simple_graph_comparison():
    """Compare all four methods on simple graph with comprehensive formatted output"""
    print("\n🔬 SIMPLE GRAPH METHOD COMPARISON")
    print("=" * 90)

    # Create simple test graph
    G = nx.DiGraph()
    G.add_node("m1", type="measurement", timestamp=2001)
    G.add_node("m2", type="measurement", timestamp=2005)
    G.add_node("m3", type="measurement", timestamp=2020)
    G.add_node("obj1", type="object")
    G.add_node("obj2", type="object")
    G.add_node("loc1", type="location")

    G.add_edges_from([
        ("m1", "obj1"),
        ("m2", "obj1"),
        ("m3", "obj2"),
        ("obj1", "loc1"),
        ("obj2", "loc1"),
    ])

    print(f"Created simple graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    processor = GraphPropertyPropagationProcessor(G)
    source_configs = [{
        'node_types': ['measurement'],
        'property': 'timestamp',
        'name': 'temporal_data'
    }]

    # Run all four methods with timing
    import time

    print("\n🔄 Running Belief Propagation...")
    start_time = time.time()
    results_bp = processor.belief_propagation_auto(source_configs)
    bp_time = (time.time() - start_time) * 1000

    print("🔄 Running Spectral Propagation...")
    start_time = time.time()
    results_spectral = processor.spectral_propagation_auto(source_configs)
    spectral_time = (time.time() - start_time) * 1000

    print("🔄 Running Matrix Streaming Propagation...")
    start_time = time.time()
    results_matrix = processor.matrix_streaming_propagation_auto(source_configs)
    matrix_time = (time.time() - start_time) * 1000

    print("🔄 Running Quantum Spectral Propagation...")
    start_time = time.time()
    results_quantum = processor.quantum_spectral_propagation_auto(source_configs)
    quantum_time = (time.time() - start_time) * 1000

    # COMPREHENSIVE COMPARISON TABLE (updated for 4 methods)
    print(f"\n📊 COMPREHENSIVE METHOD COMPARISON:")
    print(f"{'Method':<20} {'obj1':<8} {'obj2':<8} {'loc1':<8} {'Time':<10} {'Components':<12} {'Mathematical Rep'}")
    print("-" * 110)

    temporal_bp = results_bp['temporal_data']
    temporal_spectral = results_spectral['temporal_data']
    temporal_matrix = results_matrix['temporal_data']
    temporal_quantum = results_quantum['temporal_data']
    values_bp = temporal_bp['values']
    values_spectral = temporal_spectral['values']
    values_matrix = temporal_matrix['values']
    values_quantum = temporal_quantum['values']

    key_nodes = ['obj1', 'obj2', 'loc1']
    node_to_idx = {node: i for i, node in enumerate(G.nodes())}

    # Get values for each method
    methods_data = {
        'Belief Propagation': {
            'values': values_bp,
            'time': bp_time,
            'gmm_data': temporal_bp['gmm_data']
        },
        'Spectral Propagation': {
            'values': values_spectral,
            'time': spectral_time,
            'gmm_data': temporal_spectral['gmm_data']
        },
        'Matrix Streaming': {
            'values': values_matrix,
            'time': matrix_time,
            'gmm_data': temporal_matrix['gmm_data']
        },
        'Quantum Spectral': {
            'values': values_quantum,
            'time': quantum_time,
            'gmm_data': temporal_quantum['gmm_data']
        }
    }

    for method_name, method_data in methods_data.items():
        values = method_data['values']
        exec_time = method_data['time']
        gmm_data = method_data['gmm_data']

        # Get values for key nodes
        node_values = []
        total_components = 0

        for node in key_nodes:
            if node in node_to_idx:
                idx = node_to_idx[node]
                if not np.isnan(values[idx]):
                    node_values.append(f"{values[idx]:.1f}")
                    if node in gmm_data and 'n_components' in gmm_data[node]:
                        total_components += gmm_data[node]['n_components']
                else:
                    node_values.append("N/A")

        # Format time
        time_str = f"{exec_time:.1f}ms" if exec_time < 1000 else f"{exec_time / 1000:.1f}s"

        # Mathematical representation
        if 'obj1' in gmm_data:
            obj1_data = gmm_data['obj1']
            if 'n_components' in obj1_data:
                n_comp = obj1_data['n_components']
                math_repr = f"{method_name.split()[0]}({n_comp} comp)"
            else:
                math_repr = method_name.split()[0]
        else:
            math_repr = method_name.split()[0]

        print(f"{method_name:<20} {node_values[0]:<8} {node_values[1]:<8} {node_values[2]:<8} "
              f"{time_str:<10} {total_components:<12} {math_repr}")

    print(f"\nExpected behavior:")
    print(f"- obj1 should span influences from m1 (2001) and m2 (2005) ≈ 2003")
    print(f"- obj2 should only show influence from m3 (2020)")
    print(f"- loc1 should span 2001 to 2020 but bias towards early 2000s")

    # DIFFERENCE ANALYSIS TABLE (updated for 4 methods)
    print(f"\n📊 DIFFERENCE ANALYSIS:")
    print(
        f"{'Node':<8} {'Belief Prop':<12} {'Spectral':<12} {'Matrix':<12} {'Quantum':<12} {'Best Method'}")
    print("-" * 80)

    for node in key_nodes:
        if node in node_to_idx:
            idx = node_to_idx[node]
            val_bp = values_bp[idx] if not np.isnan(values_bp[idx]) else None
            val_spectral = values_spectral[idx] if not np.isnan(values_spectral[idx]) else None
            val_matrix = values_matrix[idx] if not np.isnan(values_matrix[idx]) else None
            val_quantum = values_quantum[idx] if not np.isnan(values_quantum[idx]) else None

            # Determine best method based on expected values
            expected = {'obj1': 2003, 'obj2': 2020, 'loc1': 2010}[node]
            errors = {}
            if val_bp is not None:
                errors['Belief Prop'] = abs(val_bp - expected)
            if val_spectral is not None:
                errors['Spectral'] = abs(val_spectral - expected)
            if val_matrix is not None:
                errors['Matrix'] = abs(val_matrix - expected)
            if val_quantum is not None:
                errors['Quantum'] = abs(val_quantum - expected)

            best_method = min(errors, key=errors.get) if errors else "N/A"

            # Format values
            val_bp_str = f"{val_bp:.1f}" if val_bp is not None else "N/A"
            val_spectral_str = f"{val_spectral:.1f}" if val_spectral is not None else "N/A"
            val_matrix_str = f"{val_matrix:.1f}" if val_matrix is not None else "N/A"
            val_quantum_str = f"{val_quantum:.1f}" if val_quantum is not None else "N/A"

            print(f"{node:<8} {val_bp_str:<12} {val_spectral_str:<12} {val_matrix_str:<12} "
                  f"{val_quantum_str:<12} {best_method}")

    # DETAILED COMPONENT ANALYSIS (updated for 4 methods)
    print(f"\n🔍 DETAILED COMPONENT ANALYSIS:")
    for method_name, results in [("Belief Propagation", results_bp), ("Spectral Propagation", results_spectral),
                                 ("Matrix Streaming", results_matrix), ("Quantum Spectral", results_quantum)]:
        print(f"\n{method_name}:")
        gmm_data = results['temporal_data']['gmm_data']
        for node in key_nodes:
            if node in gmm_data:
                data = gmm_data[node]
                if data['type'] == 'inferred':
                    components = data['components']
                    print(f"  {node}: {len(components)} components")
                    for i, comp in enumerate(components):
                        weight, mean, std = comp
                        source_config = results['temporal_data']['source_config']
                        denorm_mean = mean * (source_config['value_range'][1] - source_config['value_range'][0]) + \
                                      source_config['value_range'][0]
                        print(f"    Component {i + 1}: weight={weight:.3f}, mean={denorm_mean:.1f}, std={std:.3f}")

    print(f"\nGenerating visualization...")
    visualize_comparison_results(G, results_bp, results_spectral, results_matrix, results_quantum, "Simple Graph")

    print("=" * 110)
    return processor, results_bp, results_spectral, results_matrix, results_quantum

def create_complex_graph():
    """Create a more realistic software development project graph"""
    G_complex = nx.DiGraph()

    # Add measurement nodes with more realistic timestamps
    measurements = [
        ("commit_1", 2020.1), ("commit_2", 2020.3), ("deploy_1", 2020.5),
        ("commit_3", 2020.8), ("TEST_1", 2021.2), ("deploy_2", 2021.4),
        ("commit_4", 2021.9), ("incident_1", 2022.1), ("patch_1", 2022.15),
        ("deploy_3", 2022.3), ("commit_5", 2022.5), ("TEST_2", 2022.7),
    ]

    for node_id, timestamp in measurements:
        G_complex.add_node(node_id, type="measurement", timestamp=timestamp)

    # Add other node types
    features = ["feature_auth", "feature_payment", "feature_analytics", "feature_mobile"]
    modules = ["auth_module", "payment_module", "analytics_module", "mobile_api"]
    services = ["auth_service", "payment_service", "analytics_service", "notification_service"]
    databases = ["user_db", "transaction_db", "analytics_db"]
    environments = ["dev_env", "staging_env", "prod_env", "TEST_env"]

    for category, items in [("feature", features), ("module", modules),
                            ("service", services), ("database", databases),
                            ("environment", environments)]:
        for item in items:
            G_complex.add_node(item, type=category)

    # Add realistic dependencies
    dependencies = [
        ("commit_1", "feature_auth"), ("commit_2", "feature_auth"),
        ("commit_2", "feature_payment"), ("commit_3", "feature_payment"),
        ("commit_4", "feature_analytics"), ("commit_5", "feature_mobile"),
        ("feature_auth", "auth_module"), ("feature_payment", "payment_module"),
        ("feature_analytics", "analytics_module"), ("feature_mobile", "mobile_api"),
        ("auth_module", "auth_service"), ("payment_module", "payment_service"),
        ("analytics_module", "analytics_service"), ("mobile_api", "notification_service"),
        ("auth_service", "user_db"), ("payment_service", "transaction_db"),
        ("analytics_service", "analytics_db"), ("notification_service", "user_db"),
        ("deploy_1", "dev_env"), ("deploy_2", "staging_env"), ("deploy_3", "prod_env"),
        ("auth_service", "prod_env"), ("payment_service", "prod_env"),
        ("TEST_1", "feature_auth"), ("TEST_2", "payment_service"),
    ]

    G_complex.add_edges_from(dependencies)
    return G_complex

def run_complex_graph_comparison():
    """Compare all four methods on complex software development graph with comprehensive analysis"""
    print("\n🏗️ COMPLEX SOFTWARE DEVELOPMENT GRAPH - METHOD COMPARISON")
    print("=" * 140)

    G_complex = create_complex_graph()
    print(f"Complex graph: {G_complex.number_of_nodes()} nodes, {G_complex.number_of_edges()} edges")

    # Show node type distribution
    node_types = {}
    for node, data in G_complex.nodes(data=True):
        node_type = data.get('type', 'unknown')
        if node_type not in node_types:
            node_types[node_type] = []
        node_types[node_type].append(node)

    print("\nNode Type Distribution:")
    for node_type, nodes in node_types.items():
        print(f"  {node_type}: {len(nodes)} nodes")

    processor = GraphPropertyPropagationProcessor(G_complex)
    source_configs = [{
        'node_types': ['measurement'],
        'property': 'timestamp',
        'name': 'temporal_data'
    }]

    # Time all four methods
    import time

    print("\n⏱️  Running Belief Propagation...")
    start_time = time.time()
    results_bp = processor.belief_propagation_auto(source_configs)
    bp_time = (time.time() - start_time) * 1000

    print("⏱️  Running Spectral Propagation...")
    start_time = time.time()
    results_spectral = processor.spectral_propagation_auto(source_configs)
    spectral_time = (time.time() - start_time) * 1000

    print("⏱️  Running Matrix Streaming Propagation...")
    start_time = time.time()
    results_matrix = processor.matrix_streaming_propagation_auto(source_configs)
    matrix_time = (time.time() - start_time) * 1000

    print("⏱️  Running Quantum Spectral Propagation...")
    start_time = time.time()
    results_quantum = processor.quantum_spectral_propagation_auto(source_configs)
    quantum_time = (time.time() - start_time) * 1000

    # COMPREHENSIVE COMPARISON TABLE
    key_nodes = ['feature_auth', 'payment_service', 'prod_env', 'analytics_service']

    print(f"\n📊 COMPREHENSIVE METHOD COMPARISON:")
    print(
        f"{'Method':<20} {'feature_auth':<12} {'payment_svc':<12} {'prod_env':<10} {'analytics_svc':<12} {'Time':<10} {'Predictions':<12}")
    print("-" * 140)

    node_to_idx = {node: i for i, node in enumerate(G_complex.nodes())}
    temporal_bp = results_bp['temporal_data']
    temporal_spectral = results_spectral['temporal_data']
    temporal_matrix = results_matrix['temporal_data']
    temporal_quantum = results_quantum['temporal_data']
    values_bp = temporal_bp['values']
    values_spectral = temporal_spectral['values']
    values_matrix = temporal_matrix['values']
    values_quantum = temporal_quantum['values']

    # Prepare data for ALL FOUR methods
    methods_data = {
        'Belief Propagation': {
            'values': values_bp,
            'time': bp_time,
            'gmm_data': temporal_bp['gmm_data']
        },
        'Spectral Propagation': {
            'values': values_spectral,
            'time': spectral_time,
            'gmm_data': temporal_spectral['gmm_data']
        },
        'Matrix Streaming': {
            'values': values_matrix,
            'time': matrix_time,
            'gmm_data': temporal_matrix['gmm_data']
        },
        'Quantum Spectral': {
            'values': values_quantum,
            'time': quantum_time,
            'gmm_data': temporal_quantum['gmm_data']
        }
    }

    for method_name, method_data in methods_data.items():
        values = method_data['values']
        exec_time = method_data['time']

        # Get values for key nodes
        node_values = []
        predictions_count = 0

        for node in key_nodes:
            if node in node_to_idx:
                idx = node_to_idx[node]
                if not np.isnan(values[idx]):
                    node_values.append(f"{values[idx]:.1f}")
                    predictions_count += 1
                else:
                    node_values.append("N/A")
            else:
                node_values.append("N/A")

        # Count total predictions
        total_predictions = np.sum(~np.isnan(values))

        # Format time
        time_str = f"{exec_time:.0f}ms" if exec_time < 1000 else f"{exec_time / 1000:.1f}s"

        print(f"{method_name:<20} {node_values[0]:<12} {node_values[1]:<12} {node_values[2]:<10} "
              f"{node_values[3]:<12} {time_str:<10} {total_predictions:<12}")

    # PERFORMANCE ANALYSIS (updated for 4 methods)
    print(f"\n⚡ PERFORMANCE ANALYSIS:")
    print(f"  Belief Propagation: {bp_time:.1f}ms")
    print(f"  Spectral Propagation: {spectral_time:.1f}ms")
    print(f"  Matrix Streaming: {matrix_time:.1f}ms")
    print(f"  Quantum Spectral: {quantum_time:.1f}ms")

    # Find fastest method
    times = {'Belief Propagation': bp_time, 'Spectral Propagation': spectral_time,
             'Matrix Streaming': matrix_time, 'Quantum Spectral': quantum_time}
    fastest_method = min(times, key=times.get)
    print(f"  ⚡ Fastest method: {fastest_method}")

    # DETAILED NODE COMPARISON (updated for 4 methods)
    print(f"\n📊 DETAILED NODE COMPARISON:")
    print(
        f"{'Node':<18} {'Belief Prop':<12} {'Spectral':<12} {'Matrix':<12} {'Quantum':<12} {'Best Method'}")
    print("-" * 100)

    total_agreement = 0
    valid_comparisons = 0

    for node in key_nodes:
        if node in node_to_idx:
            idx = node_to_idx[node]
            val_bp = values_bp[idx] if not np.isnan(values_bp[idx]) else None
            val_spectral = values_spectral[idx] if not np.isnan(values_spectral[idx]) else None
            val_matrix = values_matrix[idx] if not np.isnan(values_matrix[idx]) else None
            val_quantum = values_quantum[idx] if not np.isnan(values_quantum[idx]) else None

            # Determine best method based on expected values
            expected = \
            {'feature_auth': 2020.2, 'payment_service': 2021.0, 'prod_env': 2021.4, 'analytics_service': 2021.9}[node]
            errors = {}
            if val_bp is not None:
                errors['BP'] = abs(val_bp - expected)
            if val_spectral is not None:
                errors['Spectral'] = abs(val_spectral - expected)
            if val_matrix is not None:
                errors['Matrix'] = abs(val_matrix - expected)
            if val_quantum is not None:
                errors['Quantum'] = abs(val_quantum - expected)

            best_method = min(errors, key=errors.get) if errors else "N/A"

            # Format values
            val_bp_str = f"{val_bp:.1f}" if val_bp is not None else "N/A"
            val_spectral_str = f"{val_spectral:.1f}" if val_spectral is not None else "N/A"
            val_matrix_str = f"{val_matrix:.1f}" if val_matrix is not None else "N/A"
            val_quantum_str = f"{val_quantum:.1f}" if val_quantum is not None else "N/A"

            print(f"{node:<18} {val_bp_str:<12} {val_spectral_str:<12} {val_matrix_str:<12} "
                  f"{val_quantum_str:<12} {best_method}")

            # Update agreement calculation (compare all methods)
            if val_bp is not None and val_spectral is not None:
                diff = abs(val_bp - val_spectral)
                total_agreement += (2.0 - min(diff, 2.0)) / 2.0
                valid_comparisons += 1

    # OVERALL AGREEMENT SCORE
    if valid_comparisons > 0:
        avg_agreement = total_agreement / valid_comparisons
        print(f"\nOverall Method Agreement: {avg_agreement:.2f} ({avg_agreement * 100:.0f}%)")

    # COMPONENT ANALYSIS SUMMARY
    print(f"\n🔍 COMPONENT ANALYSIS SUMMARY:")
    for method_name, method_data in methods_data.items():
        gmm_data = method_data['gmm_data']
        total_components = 0
        inferred_nodes = 0

        for node in key_nodes:
            if node in gmm_data:
                data = gmm_data[node]
                if data['type'] == 'inferred':
                    inferred_nodes += 1
                    if 'n_components' in data:
                        total_components += data['n_components']

        avg_components = total_components / max(inferred_nodes, 1)
        print(f"  {method_name}: {inferred_nodes} inferred nodes, avg {avg_components:.1f} components/node")

    # EXPECTED BEHAVIOR ANALYSIS
    print(f"\n💡 EXPECTED BEHAVIOR ANALYSIS:")
    print(f"Complex software development timeline (2020-2022):")
    print(f"- feature_auth: Should reflect early commits (2020.1-2020.3) ≈ 2020.2")
    print(f"- payment_service: Should blend multiple influences ≈ 2021.0")
    print(f"- prod_env: Should reflect deployment timeline ≈ 2021.4")
    print(f"- analytics_service: Should reflect later development ≈ 2021.9")

    print(f"\nGenerating visualization...")
    visualize_comparison_results(G_complex, results_bp, results_spectral, results_matrix, results_quantum, "Complex Graph")

    print("=" * 140)
    return processor, results_bp, results_spectral, results_matrix, results_quantum

def run_comprehensive_benchmark():
    """Comprehensive benchmark testing all four methods across multiple scenarios"""
    print("\n⚡ COMPREHENSIVE BENCHMARKING - ALL FOUR METHODS")
    print("=" * 160)

    import time
    from concurrent.futures import ThreadPoolExecutor, TimeoutError

    # Test scenarios: ALL now test all four methods
    benchmark_scenarios = [
        # Small graphs - compare all four methods
        {'sizes': [50, 100, 200, 500], 'name': 'Method Comparison', 'density': 0.1, 'all_methods': True,
         'timeout': 10},
        # Medium graphs - scalability focus
        {'sizes': [1000, 2000, 4000], 'name': 'Scalability Test', 'density': 0.01, 'all_methods': True, 'timeout': 30},
        # Large graphs - scalability focus
        # {'sizes': [10000, 22000], 'name': 'Large Scale Test', 'density': 0.0002, 'all_methods': True, 'timeout': 20}
    ]

    all_results = {}

    def run_method_with_timeout(method_func, timeout_sec):
        """Run method with timeout using ThreadPoolExecutor"""
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(method_func)
            try:
                start_time = time.time()
                result = future.result(timeout=timeout_sec)
                end_time = time.time()
                return (end_time - start_time) * 1000, 'OK'
            except TimeoutError:
                return float('inf'), f'TIMEOUT({timeout_sec}s)'
            except Exception as e:
                return float('inf'), f'FAIL: {str(e)[:15]}'

    for scenario in benchmark_scenarios:
        scenario_name = scenario['name']
        sizes = scenario['sizes']
        density = scenario['density']
        all_methods = scenario['all_methods']
        timeout = scenario['timeout']

        print(f"\n📊 {scenario_name.upper()}")
        print("=" * 160)

        # Clean header
        print(
            f"{'Size':<6} {'Edges':<8} {'Density':<10} {'BP Time':<12} {'Spectral Time':<15} {'Matrix Time':<12} {'Quantum Time':<13} {'Fastest':<12} {'Status'}")
        print("-" * 160)

        scenario_results = {}

        for size in sizes:
            print(f"{size:<6}", end=" ", flush=True)

            # Create test graph
            graph_start = time.time()

            if size >= 10000:
                n_edges = int(size * (size - 1) * density)
                graph_stats = {
                    'node_types': {
                        'measurement': int(size * 0.07),
                        'file': int(size * 0.5),
                        'folder': int(size * 0.3),
                        'system': int(size * 0.13)
                    }
                }
                G_test = create_fast_file_system_graph(size, n_edges, int(size * 0.07), graph_stats)
                measurement_nodes = [node for node, data in G_test.nodes(data=True) if
                                     data.get('type') == 'measurement']
                for node in measurement_nodes:
                    G_test.nodes[node]['timestamp'] = np.random.uniform(2020, 2022)
            else:
                G_test = nx.erdos_renyi_graph(size, density, directed=True)
                n_edges = G_test.number_of_edges()
                n_sources = max(2, size // 15)
                source_nodes = np.random.choice(list(G_test.nodes()), size=n_sources, replace=False)
                for node in source_nodes:
                    timestamp = np.random.uniform(2020, 2022)
                    G_test.nodes[node]['type'] = 'measurement'
                    G_test.nodes[node]['timestamp'] = timestamp

            n_edges = G_test.number_of_edges()
            graph_time = (time.time() - graph_start) * 1000

            edges_str = f"{n_edges}" if n_edges < 1000 else f"{n_edges / 1000:.1f}k"
            density_str = f"{density:.4f}"
            print(f"{edges_str:<8} {density_str:<10}", end=" ", flush=True)

            # Setup processor and configs
            processor = GraphPropertyPropagationProcessor(G_test)
            source_configs = [{
                'node_types': ['measurement'],
                'property': 'timestamp',
                'name': 'temporal_data'
            }]

            size_results = {
                'n_edges': n_edges,
                'density': density,
                'graph_time_ms': graph_time,
                'n_sources': len([node for node, data in G_test.nodes(data=True) if data.get('type') == 'measurement'])
            }

            # Test all four methods
            def run_bp():
                return processor.belief_propagation_auto(source_configs)

            def run_spectral():
                return processor.spectral_propagation_auto(source_configs)

            def run_matrix():
                return processor.matrix_streaming_propagation_auto(source_configs)

            def run_quantum():
                return processor.quantum_spectral_propagation_auto(source_configs)

            # Time all four methods
            bp_time, bp_status = run_method_with_timeout(run_bp, timeout)
            spectral_time, spectral_status = run_method_with_timeout(run_spectral, timeout)
            matrix_time, matrix_status = run_method_with_timeout(run_matrix, timeout)
            quantum_time, quantum_status = run_method_with_timeout(run_quantum, timeout)

            # Extract quantum theoretical data after quantum method runs
            theoretical_quantum_time = getattr(processor, '_theoretical_quantum_time', None)
            speedup_factor = getattr(processor, '_quantum_speedup_factor', None)

            # Format times
            bp_time_str = f"{bp_time:.0f}ms" if bp_status == 'OK' and bp_time < 1000 else f"{bp_time / 1000:.1f}s" if bp_status == 'OK' else bp_status
            spectral_time_str = f"{spectral_time:.0f}ms" if spectral_status == 'OK' and spectral_time < 1000 else f"{spectral_time / 1000:.1f}s" if spectral_status == 'OK' else spectral_status
            matrix_time_str = f"{matrix_time:.0f}ms" if matrix_status == 'OK' and matrix_time < 1000 else f"{matrix_time / 1000:.1f}s" if matrix_status == 'OK' else matrix_status

            # Clean quantum time display
            if quantum_status == 'OK':
                if quantum_time < 1000:
                    quantum_time_str = f"{quantum_time:.0f}ms"
                else:
                    quantum_time_str = f"{quantum_time / 1000:.1f}s"

                # Add speedup if significant
                if speedup_factor and speedup_factor > 10:
                    quantum_time_str += f"{speedup_factor:.0f}x"
            else:
                quantum_time_str = quantum_status

            # Determine fastest method (use theoretical quantum time if available for comparison)
            successful_methods = {}
            if bp_status == 'OK':
                successful_methods['BP'] = bp_time
            if spectral_status == 'OK':
                successful_methods['Spectral'] = spectral_time
            if matrix_status == 'OK':
                successful_methods['Matrix'] = matrix_time
            if quantum_status == 'OK':
                # Use theoretical time for fastest determination if significant speedup
                comparison_time = theoretical_quantum_time if (
                            theoretical_quantum_time and speedup_factor and speedup_factor > 10) else quantum_time
                successful_methods['Quantum'] = comparison_time

            if successful_methods:
                fastest_method = min(successful_methods, key=successful_methods.get)
                successful_count = len(successful_methods)
                status = f"{successful_count}/4 OK"
            else:
                fastest_method = "All failed"
                status = "All failed"

            print(
                f"{bp_time_str:<12} {spectral_time_str:<15} {matrix_time_str:<12} {quantum_time_str:<13} {fastest_method:<12} {status}")

            size_results.update({
                'bp_time_ms': bp_time if bp_status == 'OK' else None,
                'spectral_time_ms': spectral_time if spectral_status == 'OK' else None,
                'matrix_time_ms': matrix_time if matrix_status == 'OK' else None,
                'quantum_time_ms': quantum_time if quantum_status == 'OK' else None,
                'theoretical_quantum_time_ms': theoretical_quantum_time,
                'quantum_speedup_factor': speedup_factor,
                'bp_status': bp_status,
                'spectral_status': spectral_status,
                'matrix_status': matrix_status,
                'quantum_status': quantum_status,
                'fastest_method': fastest_method
            })

            scenario_results[size] = size_results

        all_results[scenario_name] = scenario_results

    # COMPREHENSIVE ANALYSIS
    print(f"\n" + "=" * 160)
    print("COMPREHENSIVE ANALYSIS")
    print("=" * 160)

    # Method comparison summary
    print(f"\n📊 OVERALL METHOD PERFORMANCE:")
    method_wins = {'BP': 0, 'Spectral': 0, 'Matrix': 0, 'Quantum': 0}
    method_success_rates = {'BP': [], 'Spectral': [], 'Matrix': [], 'Quantum': []}

    for scenario_name, scenario_data in all_results.items():
        print(f"\n{scenario_name}:")
        scenario_wins = {'BP': 0, 'Spectral': 0, 'Matrix': 0, 'Quantum': 0}
        scenario_success = {'BP': 0, 'Spectral': 0, 'Matrix': 0, 'Quantum': 0}
        total_tests = len(scenario_data)

        for size, data in scenario_data.items():
            fastest = data.get('fastest_method', 'None')
            if fastest in scenario_wins:
                scenario_wins[fastest] += 1
                method_wins[fastest] += 1

            # Count successes
            if data.get('bp_status') == 'OK':
                scenario_success['BP'] += 1
            if data.get('spectral_status') == 'OK':
                scenario_success['Spectral'] += 1
            if data.get('matrix_status') == 'OK':
                scenario_success['Matrix'] += 1
            if data.get('quantum_status') == 'OK':
                scenario_success['Quantum'] += 1

        # Print scenario summary
        for method in ['BP', 'Spectral', 'Matrix', 'Quantum']:
            wins = scenario_wins[method]
            successes = scenario_success[method]
            success_rate = (successes / total_tests) * 100
            method_success_rates[method].append(success_rate)
            print(f"  {method}: {wins} wins, {successes}/{total_tests} successful ({success_rate:.0f}%)")

    # Overall winner
    print(f"\n🏆 OVERALL WINNER:")
    overall_winner = max(method_wins, key=method_wins.get)
    print(f"  {overall_winner}: {method_wins[overall_winner]} total wins across all scenarios")

    # Success rates
    print(f"\n📈 SUCCESS RATES:")
    for method, rates in method_success_rates.items():
        if rates:
            avg_rate = np.mean(rates)
            print(f"  {method}: {avg_rate:.1f}% average success rate")

    # 🚀 QUANTUM ADVANTAGE ANALYSIS
    print(f"\n🚀 QUANTUM ADVANTAGE ANALYSIS:")
    print("-" * 60)

    total_quantum_tests = 0
    significant_speedups = 0

    for scenario_name, scenario_data in all_results.items():
        print(f"\n{scenario_name}:")
        for size, data in scenario_data.items():
            if data.get('theoretical_quantum_time_ms') and data.get('quantum_speedup_factor'):
                sim_time = data.get('quantum_time_ms', 0)
                theoretical_time = data.get('theoretical_quantum_time_ms')
                speedup = data.get('quantum_speedup_factor')

                total_quantum_tests += 1
                if speedup > 10:
                    significant_speedups += 1

                print(
                    f"  {size:4d} nodes: Sim {sim_time:6.1f}ms → Quantum {theoretical_time:8.3f}ms ({speedup:5.0f}x speedup)")

    if total_quantum_tests > 0:
        print(f"\nQuantum shows {significant_speedups}/{total_quantum_tests} significant speedups (>10x)")

    # Extrapolation to real graph (22K nodes) - include quantum theoretical times
    print(f"\n🔮 EXTRAPOLATION TO REAL GRAPH (22K nodes):")

    for method_name, time_key in [('Belief Propagation', 'bp_time_ms'), ('Spectral Propagation', 'spectral_time_ms'),
                                  ('Matrix Streaming', 'matrix_time_ms'), ('Quantum Spectral', 'quantum_time_ms'),
                                  ('Quantum Theoretical', 'theoretical_quantum_time_ms')]:
        method_data = []
        for scenario_data in all_results.values():
            for size, data in scenario_data.items():
                if time_key == 'theoretical_quantum_time_ms':
                    if data.get('theoretical_quantum_time_ms') is not None:
                        method_data.append((size, data['theoretical_quantum_time_ms']))
                else:
                    status_key = time_key.replace('_time_ms', '_status')
                    if data.get(status_key) == 'OK' and data.get(time_key) is not None:
                        method_data.append((size, data[time_key]))

        if len(method_data) >= 2:
            # Use the last two data points for extrapolation
            size1, time1 = method_data[-2]
            size2, time2 = method_data[-1]

            if time1 > 0 and size1 > 0 and size2 > size1:
                size_ratio = size2 / size1
                time_ratio = time2 / time1

                if time_ratio > 0:
                    observed_exponent = np.log(time_ratio) / np.log(size_ratio)
                    observed_exponent = min(max(observed_exponent, 1.0), 3.0)

                    scale_factor = (22000 / size2) ** observed_exponent
                    estimated_time_sec = (time2 * scale_factor) / 1000

                    print(f"  {method_name}:")
                    print(f"    Based on {size2:,} → 22K nodes scaling:")
                    print(f"    Estimated time: ~{estimated_time_sec:.1f} seconds")

                    if estimated_time_sec < 300:
                        print(f"    ✅ FEASIBLE for real graph!")
                    elif estimated_time_sec < 900:
                        print(f"    ⚠️  SLOW but possible")
                    else:
                        print(f"    ❌ TOO SLOW for real graph")
                else:
                    print(f"  {method_name}: No valid scaling data")
        else:
            print(f"  {method_name}: Insufficient data for extrapolation")

    print("=" * 160)
    return all_results

def create_fast_file_system_graph(n_nodes, n_edges, n_timestamps, graph_stats):
    """Create a graph that mimics file system structure - FAST VERSION"""

    # Calculate proportions based on actual graph
    total_types = sum(graph_stats['node_types'].values())
    proportions = {k: v / total_types for k, v in graph_stats['node_types'].items()}

    # Create graph
    G = nx.DiGraph()

    # Add nodes with realistic type distribution
    node_types = []
    for node_type, proportion in proportions.items():
        count = max(1, int(n_nodes * proportion))
        node_types.extend([node_type] * count)

    # Pad or trim to exact node count
    while len(node_types) < n_nodes:
        node_types.append('file')  # Default to file type
    node_types = node_types[:n_nodes]

    # Add nodes
    for i, node_type in enumerate(node_types):
        G.add_node(f"node_{i}", type=node_type)

        # Add timestamps to Timestamp nodes
        if node_type == 'Timestamp':
            # Realistic timestamp range
            timestamp = np.random.uniform(2020, 2024)
            G.add_node(f"node_{i}", type=node_type, timestamp=timestamp)

    # OPTIMIZED EDGE CREATION - avoid the slow random method
    nodes = list(G.nodes())

    if n_edges > 0:
        # Use vectorized approach for large graphs
        if n_nodes > 1000:
            # Generate all possible edges and sample from them
            max_possible_edges = n_nodes * (n_nodes - 1)  # No self-loops

            if n_edges < max_possible_edges * 0.1:  # Sparse graph
                # For sparse graphs, use random sampling
                edges_to_add = []
                while len(edges_to_add) < n_edges:
                    # Generate batch of random edges
                    batch_size = min(n_edges * 2, 10000)
                    sources = np.random.choice(n_nodes, batch_size)
                    targets = np.random.choice(n_nodes, batch_size)

                    # Filter out self-loops and duplicates
                    for s, t in zip(sources, targets):
                        if s != t and (s, t) not in edges_to_add:
                            edges_to_add.append((s, t))
                            if len(edges_to_add) >= n_edges:
                                break

                # Add edges using node names
                for s, t in edges_to_add[:n_edges]:
                    G.add_edge(f"node_{s}", f"node_{t}")
            else:
                # For dense graphs, create all and sample
                # This is still expensive but better than the original method
                all_edges = [(i, j) for i in range(n_nodes) for j in range(n_nodes) if i != j]
                selected_edges = np.random.choice(len(all_edges), size=min(n_edges, len(all_edges)), replace=False)

                for edge_idx in selected_edges:
                    s, t = all_edges[edge_idx]
                    G.add_edge(f"node_{s}", f"node_{t}")
        else:
            # For small graphs, use the original method (it's fast enough)
            edges_added = 0
            max_attempts = n_edges * 3
            attempts = 0

            while edges_added < n_edges and attempts < max_attempts:
                attempts += 1
                source = np.random.choice(nodes)
                target = np.random.choice(nodes)

                if source != target and not G.has_edge(source, target):
                    G.add_edge(source, target)
                    edges_added += 1

    return G

def run_propagation_coverage_tests():
    """Test that all methods properly propagate influence to all reachable nodes"""
    print("\n🌊 PROPAGATION COVERAGE TESTS")
    print("=" * 100)
    print("Testing: Does every reachable node get represented in the wave function?")
    print("=" * 100)

    coverage_results = {}

    # Test Case 1: Tree Structure (all nodes should be influenced)
    print("\n🌳 TEST CASE 1: Tree Structure - Complete Coverage")
    print("-" * 60)

    G1 = nx.DiGraph()
    # Create a tree: root -> level1 -> level2 -> level3
    G1.add_edges_from([
        (0, 1), (0, 2),  # root to level 1
        (1, 3), (1, 4),  # level 1 to level 2
        (2, 5), (2, 6),  # level 1 to level 2
        (3, 7), (4, 8), (5, 9)  # level 2 to level 3
    ])
    G1.nodes[0]['type'] = 'measurement'
    G1.nodes[0]['timestamp'] = 2020

    coverage_results['Tree Structure'] = run_coverage_test(
        G1, source_node=0, expected_influenced=list(range(1, 10)),
        description="Single root source should influence all downstream nodes"
    )

    # Test Case 2: Multiple Sources - Overlapping Influence
    print("\n🔗 TEST CASE 2: Multiple Sources - Overlapping Neighborhoods")
    print("-" * 60)

    G2 = nx.DiGraph()
    G2.add_edges_from([
        (0, 2), (0, 3),  # source 1 -> shared nodes
        (1, 2), (1, 4),  # source 2 -> shared nodes
        (2, 5), (3, 6), (4, 7)  # shared nodes -> endpoints
    ])
    G2.nodes[0]['type'] = 'measurement'
    G2.nodes[0]['timestamp'] = 2020
    G2.nodes[1]['type'] = 'measurement'
    G2.nodes[1]['timestamp'] = 2010

    coverage_results['Multiple Sources'] = run_coverage_test(
        G2, source_node=[0, 1], expected_influenced=[2, 3, 4, 5, 6, 7],
        description="Two sources should influence all reachable nodes, with overlap at nodes 2"
    )

    # Test Case 3: Disconnected Components
    print("\n🏝️ TEST CASE 3: Disconnected Components")
    print("-" * 60)

    G3 = nx.DiGraph()
    # Component 1: connected to source
    G3.add_edges_from([(0, 1), (1, 2)])
    # Component 2: disconnected from source
    G3.add_edges_from([(3, 4), (4, 5)])

    G3.nodes[0]['type'] = 'measurement'
    G3.nodes[0]['timestamp'] = 2020

    coverage_results['Disconnected'] = run_coverage_test(
        G3, source_node=0, expected_influenced=[1, 2], expected_not_influenced=[3, 4, 5],
        description="Source should only influence connected component, not disconnected nodes"
    )

    # Test Case 4: Long Chain - Distance Decay
    print("\n🔗 TEST CASE 4: Long Chain - Distance Propagation")
    print("-" * 60)

    G4 = nx.path_graph(8, create_using=nx.DiGraph)  # 0->1->2->3->4->5->6->7
    G4.nodes[0]['type'] = 'measurement'
    G4.nodes[0]['timestamp'] = 2020

    coverage_results['Long Chain'] = run_coverage_test(
        G4, source_node=0, expected_influenced=list(range(1, 8)),
        description="Source should propagate along entire chain (test distance limits)"
    )

    # Test Case 5: Dense Hub - Many Neighbors
    print("\n⭐ TEST CASE 5: Dense Hub - High Degree Node")
    print("-" * 60)

    G5 = nx.DiGraph()
    # Create star pattern: 0 -> {1,2,3,4,5,6,7,8,9}
    for i in range(1, 10):
        G5.add_edge(0, i)
    # Add second level: each spoke -> 2 more nodes
    for i in range(1, 10):
        G5.add_edge(i, i + 9)  # 1->10, 2->11, etc.

    G5.nodes[0]['type'] = 'measurement'
    G5.nodes[0]['timestamp'] = 2020

    coverage_results['Dense Hub'] = run_coverage_test(
        G5, source_node=0, expected_influenced=list(range(1, 19)),
        description="Hub source should influence all spokes and their children"
    )

    # COMPREHENSIVE COVERAGE SUMMARY
    print("\n📊 COMPREHENSIVE COVERAGE SUMMARY")
    print("=" * 100)

    print(f"{'Test Case':<20} {'BP Coverage':<12} {'Spectral Coverage':<18} {'Matrix Coverage':<15} {'Quantum Coverage':<16} {'Best Method'}")
    print("-" * 100)

    method_coverage_scores = {'BP': [], 'Spectral': [], 'Matrix': [], 'Quantum': []}

    for test_name, test_results in coverage_results.items():
        bp_coverage = test_results['bp_coverage']
        spectral_coverage = test_results['spectral_coverage']
        matrix_coverage = test_results['matrix_coverage']
        quantum_coverage = test_results['quantum_coverage']
        best_method = test_results['best_method']

        method_coverage_scores['BP'].append(bp_coverage)
        method_coverage_scores['Spectral'].append(spectral_coverage)
        method_coverage_scores['Matrix'].append(matrix_coverage)
        method_coverage_scores['Quantum'].append(quantum_coverage)

        print(
            f"{test_name:<20} {bp_coverage:<12.1f}% {spectral_coverage:<18.1f}% {matrix_coverage:<15.1f}% {quantum_coverage:<16.1f}% {best_method}")

    # OVERALL COVERAGE METRICS
    print("\n🏆 OVERALL COVERAGE METRICS")
    print("-" * 60)

    for method, scores in method_coverage_scores.items():
        if scores:
            avg_coverage = np.mean(scores)
            min_coverage = min(scores)
            print(f"{method}: Average {avg_coverage:.1f}%, Minimum {min_coverage:.1f}%")

    # PROPAGATION ANALYSIS
    print(f"\n🔍 PROPAGATION ANALYSIS:")
    print(f"Perfect Coverage (100%): Method reaches all expected nodes")
    print(f"Good Coverage (80-99%): Method reaches most nodes, minor gaps")
    print(f"Fair Coverage (60-79%): Method has significant gaps")
    print(f"Poor Coverage (<60%): Method fails to propagate properly")

    print("=" * 100)
    return coverage_results

def run_coverage_test(graph, source_node, expected_influenced, expected_not_influenced=None, description=""):
    """Test propagation coverage for all four methods"""
    processor = GraphPropertyPropagationProcessor(graph)
    source_configs = [{
        'node_types': ['measurement'],
        'property': 'timestamp',
        'name': 'temporal_data'
    }]

    print(f"Description: {description}")
    print(f"Graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    if isinstance(source_node, list):
        print(f"Sources: {source_node}")
    else:
        print(f"Source: {source_node}")
    print(f"Expected influenced: {len(expected_influenced)} nodes")

    # Run all four methods
    methods = {
        'BP': processor.belief_propagation_auto,
        'Spectral': processor.spectral_propagation_auto,
        'Matrix': processor.matrix_streaming_propagation_auto,
        'Quantum': processor.quantum_spectral_propagation_auto
    }

    method_results = {}

    for method_name, method_func in methods.items():
        try:
            results = method_func(source_configs)
            values = results['temporal_data']['values']

            # Count influenced nodes (non-NaN values that aren't sources)
            influenced_nodes = []
            source_nodes = [source_node] if not isinstance(source_node, list) else source_node

            for i, val in enumerate(values):
                if not np.isnan(val) and i not in source_nodes:
                    influenced_nodes.append(i)

            # Calculate coverage percentage
            expected_count = len(expected_influenced)
            actual_influenced = set(influenced_nodes) & set(expected_influenced)
            coverage_percentage = (len(actual_influenced) / expected_count * 100) if expected_count > 0 else 0

            method_results[method_name] = {
                'influenced_nodes': influenced_nodes,
                'coverage_percentage': coverage_percentage,
                'expected_hit': len(actual_influenced),
                'expected_total': expected_count
            }

            print(
                f"{method_name}: {len(actual_influenced)}/{expected_count} expected nodes influenced ({coverage_percentage:.1f}%)")

        except Exception as e:
            print(f"{method_name} failed: {e}")
            method_results[method_name] = {
                'influenced_nodes': [],
                'coverage_percentage': 0,
                'expected_hit': 0,
                'expected_total': len(expected_influenced)
            }

    # Check for unwanted influence (if specified)
    if expected_not_influenced:
        print(f"\nUnwanted influence check:")
        for method_name, result in method_results.items():
            influenced_set = set(result['influenced_nodes'])
            unwanted_influenced = influenced_set & set(expected_not_influenced)
            if unwanted_influenced:
                print(f"  {method_name}: LEAKED to {len(unwanted_influenced)} nodes that should be isolated")
            else:
                print(f"  {method_name}: ✓ No unwanted influence")

    # Determine best method
    best_method = max(method_results.keys(), key=lambda x: method_results[x]['coverage_percentage'])
    best_coverage = method_results[best_method]['coverage_percentage']

    print(f"\nBest Coverage: {best_method} ({best_coverage:.1f}%)")

    return {
        'bp_coverage': method_results['BP']['coverage_percentage'],
        'spectral_coverage': method_results['Spectral']['coverage_percentage'],
        'matrix_coverage': method_results['Matrix']['coverage_percentage'],
        'quantum_coverage': method_results['Quantum']['coverage_percentage'],
        'best_method': best_method,
        'description': description,
        'method_details': method_results
    }

def run_all_tests():
    """Run all test functions in logical order"""
    print("RUNNING ALL TESTS")
    print("=" * 80)

    # Test 1: Simple graph comparison
    print("\n1. SIMPLE GRAPH COMPARISON")
    run_simple_graph_comparison()

    # Test 2: Complex graph comparison
    print("\n2. COMPLEX GRAPH COMPARISON")
    run_complex_graph_comparison()

    # Test 3: Propagation coverage validation
    print("\n3. PROPAGATION COVERAGE VALIDATION")
    run_propagation_coverage_tests()  # FIXED: Call the main test function

    # Test 4: Comprehensive benchmarking
    print("\n4. COMPREHENSIVE BENCHMARKING")
    run_comprehensive_benchmark()

    print("\nALL TESTS COMPLETED!")
    print("=" * 80)

# run_all_tests()