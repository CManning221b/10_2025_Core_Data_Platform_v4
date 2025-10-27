# backend/services/graph_propagation_service.py

import os
import json
import networkx as nx
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from backend.services.instance_graph_service import InstanceGraphService

# Import the propagation processor
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from backend.property_propogation.GraphPropertyPropagationProcessor import *


class GraphPropagationService:
    def __init__(self):
        self.instance_service = InstanceGraphService()

    def get_graph_metadata(self, graph_id):
        """Extract metadata for dropdowns and configuration"""
        graph_data = self.instance_service.get_graph_data(graph_id)

        if not graph_data:
            return None

        # Extract node types and their properties
        node_types = {}
        edge_types = set()
        properties_by_type = {}

        # Analyze nodes
        for node_id, node_data in graph_data.get('nodes', {}).items():
            node_type = node_data.get('type', 'unknown')

            if node_type not in node_types:
                node_types[node_type] = 0
                properties_by_type[node_type] = set()

            node_types[node_type] += 1

            # Collect properties for this node type
            for prop_name, prop_value in node_data.items():
                if prop_name != 'type' and isinstance(prop_value, (int, float, str)):
                    # Only include numeric properties for propagation
                    if isinstance(prop_value, (int, float)):
                        properties_by_type[node_type].add(prop_name)

        # Analyze edges
        for edge_id, edge_data in graph_data.get('edges', {}).items():
            edge_type = edge_data.get('edge_type', edge_data.get('type', 'connected'))
            edge_types.add(edge_type)

        # Convert sets to lists for JSON serialization
        properties_by_type = {k: list(v) for k, v in properties_by_type.items()}

        return {
            'node_types': node_types,
            'edge_types': list(edge_types),
            'properties_by_type': properties_by_type,
            'total_nodes': len(graph_data.get('nodes', {})),
            'total_edges': len(graph_data.get('edges', {}))
        }

    def propagate_property(self, graph_id, config):
        """Main entry point - handles both propagation methods"""

        # Store config for visualization use
        self._last_config = config  # NEW

        # Load graph data
        graph_data = self.instance_service.get_graph_data(graph_id)
        if not graph_data:
            raise ValueError(f"Graph {graph_id} not found")

        # Convert to NetworkX graph
        nx_graph = self._convert_to_networkx(graph_data, config)

        # Create processor
        processor = GraphPropertyPropagationProcessor(nx_graph)

        # Build source configuration
        source_configs = [{
            'node_types': config['source_node_types'] if isinstance(config['source_node_types'], list) else [
                config['source_node_types']],
            'property': config['source_property'],
            'name': 'propagated_data'
        }]

        # Run propagation
        method = config.get('method', 'matrix')
        if method == 'matrix':
            results = processor.matrix_streaming_propagation_auto(source_configs)
        elif method == 'spectral':
            results = processor.spectral_propagation_auto(source_configs)
        else:
            raise ValueError(f"Unknown propagation method: {method}")

        # Extract the propagated data
        propagated_data = results['propagated_data']

        # Build response with node mapping
        print("DEBUG: Building response data...")

        # Build response incrementally to avoid memory spikes
        response = {
            'method': method,
            'source_config': config,
            'propagated_values': {},
            'gmm_data': {},
            'value_range': propagated_data['source_config']['value_range'],
            'source_nodes': []
        }

        # Process in chunks to manage memory
        node_list = list(nx_graph.nodes())
        values = propagated_data['values']
        full_gmm_data = propagated_data['gmm_data']

        print(f"DEBUG: Processing {len(node_list)} nodes for response...")

        chunk_size = 1000
        for i in range(0, len(node_list), chunk_size):
            chunk_nodes = node_list[i:i + chunk_size]

            for j, node_id in enumerate(chunk_nodes):
                idx = i + j
                if not np.isnan(values[idx]):
                    response['propagated_values'][node_id] = float(values[idx])  # Ensure JSON serializable

                    if node_id in full_gmm_data:
                        # Keep all GMM data - it's important for visualization
                        gmm_data = full_gmm_data[node_id]

                        # Ensure all values are JSON serializable
                        clean_gmm = {
                            'type': gmm_data.get('type', 'unknown'),
                            'n_components': int(gmm_data.get('n_components', 0)),
                            'total_evidence': float(gmm_data.get('total_evidence', 0.0)) if gmm_data.get(
                                'total_evidence') is not None else 0.0
                        }

                        # Clean components data
                        components = gmm_data.get('components', [])
                        clean_components = []
                        for comp in components:
                            if len(comp) >= 3:
                                clean_comp = [
                                    float(comp[0]),  # weight
                                    float(comp[1]),  # mean
                                    float(comp[2])  # std
                                ]
                                clean_components.append(clean_comp)

                        clean_gmm['components'] = clean_components
                        response['gmm_data'][node_id] = clean_gmm

            if i % 5000 == 0:  # Progress every 5000 nodes
                print(f"DEBUG: Processed {i + len(chunk_nodes)}/{len(node_list)} nodes")

        # Identify source nodes
        source_config = propagated_data['source_config']
        for idx in source_config['node_indices']:
            node_id = node_list[idx]
            raw_value = source_config['raw_values'][source_config['node_indices'].index(idx)]
            response['source_nodes'].append({
                'node_id': node_id,
                'raw_value': float(raw_value)  # Ensure JSON serializable
            })

        print(
            f"DEBUG: Response built - propagated_values: {len(response['propagated_values'])}, gmm_data: {len(response['gmm_data'])}")

        return response

    def filter_by_range(self, propagation_results, min_val, max_val):
        """Filter nodes by property range - include nodes with active signals in range"""
        filtered_results = {
            'nodes_in_range': [],
            'nodes_out_of_range': [],
            'range_filter': {'min': min_val, 'max': max_val}
        }

        gmm_data = propagation_results['gmm_data']
        propagated_values = propagation_results['propagated_values']

        for node_id, gmm_info in gmm_data.items():
            has_signal_in_range = False

            if gmm_info['type'] == 'source':
                # For source nodes, check the actual value
                if node_id in propagated_values:
                    value = propagated_values[node_id]
                    has_signal_in_range = min_val <= value <= max_val

            elif gmm_info['type'] == 'inferred':
                # For inferred nodes, check if any GMM component overlaps with range
                components = gmm_info.get('components', [])
                source_config = propagation_results.get('source_config', {})
                value_range = source_config.get('value_range', (min_val, max_val))

                for component in components:
                    if len(component) >= 3:
                        weight, norm_mean, std = component[0], component[1], component[2]

                        # Denormalize the mean
                        if value_range[1] != value_range[0]:
                            actual_mean = norm_mean * (value_range[1] - value_range[0]) + value_range[0]
                        else:
                            actual_mean = value_range[0]

                        # Check if component's confidence interval overlaps with range
                        # Using 2 standard deviations for confidence interval
                        actual_std = std * (value_range[1] - value_range[0]) if value_range[1] != value_range[0] else 0
                        component_min = actual_mean - 2 * actual_std
                        component_max = actual_mean + 2 * actual_std

                        # Check for overlap with filter range
                        if not (component_max < min_val or component_min > max_val):
                            has_signal_in_range = True
                            break

            # Categorize the node
            node_info = {
                'node_id': node_id,
                'gmm_type': gmm_info['type'],
                'value': propagated_values.get(node_id, None),
                'n_components': gmm_info.get('n_components', 0)
            }

            if has_signal_in_range:
                filtered_results['nodes_in_range'].append(node_info)
            else:
                filtered_results['nodes_out_of_range'].append(node_info)

        return filtered_results

    def calculate_similarity_layout(self, propagation_results, config):
        """Calculate similarity-based node positions using polar coordinates"""
        gmm_data = propagation_results['gmm_data']
        propagated_values = propagation_results['propagated_values']

        # Calculate pairwise similarities
        similarities = self._calculate_pairwise_similarities(gmm_data)

        # Generate layout based on similarity
        layout_type = config.get('layout_type', 'polar')

        if layout_type == 'polar':
            positions = self._polar_similarity_layout(similarities, propagated_values, config)
        elif layout_type == 'force':
            positions = self._force_directed_layout(similarities, propagated_values, config)
        else:
            positions = self._hierarchical_layout(similarities, propagated_values, config)

        return {
            'positions': positions,
            'layout_type': layout_type,
            'similarity_matrix': similarities
        }

    def extract_subgraph(self, graph_id, node_list, include_neighbors=True):
        """Extract subgraph containing specified nodes and optionally their neighbors"""
        graph_data = self.instance_service.get_graph_data(graph_id)
        if not graph_data:
            return None

        node_set = set(node_list)

        # Add neighbors if requested
        if include_neighbors:
            for edge_id, edge_data in graph_data.get('edges', {}).items():
                source = edge_data.get('source')
                target = edge_data.get('target')

                if source in node_set and target:
                    node_set.add(target)
                elif target in node_set and source:
                    node_set.add(source)

        # Build subgraph
        subgraph_nodes = {
            node_id: node_data
            for node_id, node_data in graph_data.get('nodes', {}).items()
            if node_id in node_set
        }

        subgraph_edges = {}
        for edge_id, edge_data in graph_data.get('edges', {}).items():
            source = edge_data.get('source')
            target = edge_data.get('target')

            if source in node_set and target in node_set:
                subgraph_edges[edge_id] = edge_data

        return {
            'nodes': subgraph_nodes,
            'edges': subgraph_edges,
            'node_count': len(subgraph_nodes),
            'edge_count': len(subgraph_edges)
        }

    # In GraphPropagationService._convert_to_networkx(), enhance filtering:
    def _convert_to_networkx(self, graph_data, config):
        """Convert graph data to NetworkX graph, with complete node/edge exclusion"""

        graph_direction = config.get('graph_direction', 'undirected')
        if graph_direction == 'undirected':
            G = nx.Graph()
        else:
            G = nx.DiGraph()

        # Get exclusion lists
        exclude_node_types = config.get('exclude_node_types', [])
        exclude_edge_types = config.get('exclude_edge_types', [])
        valid_edge_types = config.get('valid_edge_types', ['all'])

        print(f"DEBUG: Excluding node types: {exclude_node_types}")
        print(f"DEBUG: Excluding edge types: {exclude_edge_types}")

        # Step 1: Add nodes (excluding unwanted types)
        node_count = 0
        excluded_node_count = 0

        for node_id, node_data in graph_data.get('nodes', {}).items():
            node_type = node_data.get('type', 'unknown')

            # Skip excluded node types
            if node_type in exclude_node_types:
                excluded_node_count += 1
                continue

            G.add_node(node_id, **node_data)
            node_count += 1

        print(f"DEBUG: Added {node_count} nodes, excluded {excluded_node_count} nodes")

        # Step 2: Add edges (excluding unwanted types and connections to excluded nodes)
        edge_count = 0
        skipped_edges = 0

        for edge_id, edge_data in graph_data.get('edges', {}).items():

            if '_' in edge_id:
                parts = edge_id.split('_')
                if len(parts) == 2:
                    source, target = parts
                else:
                    skipped_edges += 1
                    continue
            else:
                skipped_edges += 1
                continue

            # Skip if either node was excluded
            if source not in G.nodes or target not in G.nodes:
                skipped_edges += 1
                continue

            edge_type = edge_data.get('edge_type', edge_data.get('type', 'connected'))

            # Skip excluded edge types
            if edge_type in exclude_edge_types:
                print(f"DEBUG: Skipping edge {edge_id} - excluded edge type {edge_type}")
                skipped_edges += 1
                continue

            # Apply existing edge type filter (for propagation control)
            if 'all' not in valid_edge_types and edge_type not in valid_edge_types:
                print(f"DEBUG: Skipping edge {edge_id} - edge type {edge_type} not in valid types")
                skipped_edges += 1
                continue

            G.add_edge(source, target, **edge_data)
            edge_count += 1

        print(f"DEBUG: Final filtered graph - nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")
        print(f"DEBUG: Added {edge_count} edges, skipped {skipped_edges} edges")

        return G

    def _calculate_pairwise_similarities(self, gmm_data):
        """Calculate similarity matrix between nodes based on GMM components"""
        node_ids = list(gmm_data.keys())
        n_nodes = len(node_ids)
        similarities = np.zeros((n_nodes, n_nodes))

        for i, node_i in enumerate(node_ids):
            for j, node_j in enumerate(node_ids):
                if i == j:
                    similarities[i][j] = 1.0
                else:
                    similarities[i][j] = self._calculate_gmm_similarity(
                        gmm_data[node_i], gmm_data[node_j]
                    )

        return {
            'matrix': similarities.tolist(),
            'node_ids': node_ids
        }

    def _calculate_gmm_similarity(self, gmm_a, gmm_b):
        """Calculate similarity between two GMM distributions"""
        # Handle different node types
        if gmm_a['type'] == 'isolated' or gmm_b['type'] == 'isolated':
            return 0.0

        if gmm_a['type'] == 'source' and gmm_b['type'] == 'source':
            # Both sources - compare their values directly
            comp_a = gmm_a['components'][0] if gmm_a['components'] else [1.0, 0.5, 0.1]
            comp_b = gmm_b['components'][0] if gmm_b['components'] else [1.0, 0.5, 0.1]

            mean_diff = abs(comp_a[1] - comp_b[1])
            return max(0.0, 1.0 - mean_diff * 2)  # Scale difference

        # For inferred nodes, compare component overlap
        components_a = gmm_a.get('components', [])
        components_b = gmm_b.get('components', [])

        if not components_a or not components_b:
            return 0.0

        max_overlap = 0.0
        for comp_a in components_a:
            for comp_b in components_b:
                if len(comp_a) >= 3 and len(comp_b) >= 3:
                    # Calculate Gaussian overlap
                    weight_a, mean_a, std_a = comp_a[0], comp_a[1], comp_a[2]
                    weight_b, mean_b, std_b = comp_b[0], comp_b[1], comp_b[2]

                    # Simple overlap metric based on mean distance and std
                    mean_diff = abs(mean_a - mean_b)
                    combined_std = (std_a + std_b) / 2

                    if combined_std > 0:
                        overlap = max(0.0, 1.0 - mean_diff / (2 * combined_std))
                        weighted_overlap = overlap * weight_a * weight_b
                        max_overlap = max(max_overlap, weighted_overlap)

        return max_overlap

    def _polar_similarity_layout(self, similarities, propagated_values, config):
        """Generate polar coordinate layout based on similarity"""
        node_ids = similarities['node_ids']
        sim_matrix = np.array(similarities['matrix'])

        positions = {}
        base_radius = config.get('base_radius', 100)

        # Calculate average similarity for each node (centrality measure)
        avg_similarities = np.mean(sim_matrix, axis=1)

        for i, node_id in enumerate(node_ids):
            # Radius based on inverse of average similarity (less similar = further out)
            centrality = avg_similarities[i]
            radius = base_radius * (1.0 - centrality) * 2  # Scale factor

            # Angle based on propagated value (if available)
            if node_id in propagated_values:
                # Normalize angle based on value
                value = propagated_values[node_id]
                # Map value to angle (0 to 2π)
                all_values = [v for v in propagated_values.values() if v is not None]
                if all_values:
                    min_val, max_val = min(all_values), max(all_values)
                    if max_val != min_val:
                        normalized_val = (value - min_val) / (max_val - min_val)
                    else:
                        normalized_val = 0.5
                else:
                    normalized_val = 0.5

                angle = normalized_val * 2 * np.pi
            else:
                # Random angle for nodes without values
                angle = np.random.uniform(0, 2 * np.pi)

            # Convert to cartesian coordinates
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)

            positions[node_id] = {
                'x': float(x),
                'y': float(y),
                'radius': float(radius),
                'angle': float(angle),
                'centrality': float(centrality)
            }

        return positions

    def _force_directed_layout(self, similarities, propagated_values, config):
        """Generate force-directed layout based on similarity"""
        # This would implement a force-directed algorithm
        # For now, return a simple grid layout
        node_ids = similarities['node_ids']
        positions = {}

        grid_size = int(np.ceil(np.sqrt(len(node_ids))))
        spacing = config.get('spacing', 50)

        for i, node_id in enumerate(node_ids):
            row = i // grid_size
            col = i % grid_size

            positions[node_id] = {
                'x': float(col * spacing),
                'y': float(row * spacing),
                'grid_position': [row, col]
            }

        return positions

    def generate_interactive_propagation_viz(self, graph_id, propagation_results, layout_positions=None):
        """Generate interactive propagation visualization with smart node filtering for large graphs"""
        try:
            print("DEBUG: Starting generate_interactive_propagation_viz with filtering")

            # Get original graph data
            from backend.services.instance_graph_service import InstanceGraphService
            instance_service = InstanceGraphService()
            graph_data = instance_service.get_graph_data(graph_id)

            if not graph_data:
                print("DEBUG: No graph data found")
                return None

            print(f"DEBUG: Original graph has {len(graph_data.get('nodes', {}))} nodes")

            # Get exclusion configuration from stored config
            config = getattr(self, '_last_config', {})
            exclude_node_types = config.get('exclude_node_types', [])
            exclude_edge_types = config.get('exclude_edge_types', [])

            print(f"DEBUG: Excluding node types: {exclude_node_types}")
            print(f"DEBUG: Excluding edge types: {exclude_edge_types}")

            from pyvis.network import Network
            import json
            import os

            # Create network visualization
            net = Network(
                height="600px",
                width="100%",
                bgcolor="#ffffff",
                font_color="black",
                directed=True
            )

            # Get propagation data
            propagated_values = propagation_results.get('propagated_values', {})
            gmm_data = propagation_results.get('gmm_data', {})
            source_nodes = [s['node_id'] for s in propagation_results.get('source_nodes', [])]
            value_range = propagation_results.get('value_range', [0, 1])

            print(f"DEBUG: Propagated values: {len(propagated_values)} nodes")
            print(f"DEBUG: Source nodes: {source_nodes}")
            print(f"DEBUG: Value range: {value_range}")

            # Helper functions
            def get_node_display_name(node_data):
                """Get the best display name for a node"""
                if 'value' in node_data:
                    return str(node_data['value'])
                if 'name' in node_data:
                    return str(node_data['name'])
                if 'label' in node_data:
                    return str(node_data['label'])
                return None

            def get_node_color(node_id, value):
                if value is not None:
                    if value_range[1] > value_range[0]:
                        normalized = (value - value_range[0]) / (value_range[1] - value_range[0])
                    else:
                        normalized = 0.5
                    normalized = max(0, min(1, normalized))

                    # Blue → Green → Yellow → Red gradient
                    if normalized < 0.33:
                        r, g, b = 0, int(255 * (normalized / 0.33)), int(255 * (1 - normalized / 0.33))
                    elif normalized < 0.66:
                        r, g, b = int(255 * ((normalized - 0.33) / 0.33)), 255, 0
                    else:
                        r, g, b = 255, int(255 * (1 - (normalized - 0.66) / 0.34)), 0

                    return f"#{r:02x}{g:02x}{b:02x}"
                else:
                    return "#cccccc"

            # Collect eligible nodes
            eligible_nodes = []
            excluded_viz_nodes = 0

            for node_id, node_data in graph_data.get('nodes', {}).items():
                node_type = node_data.get('type', 'unknown')

                # Skip excluded node types
                if node_type in exclude_node_types:
                    excluded_viz_nodes += 1
                    continue

                # Must have propagation data or be a source
                prop_value = propagated_values.get(node_id)
                if prop_value is None and node_id not in source_nodes:
                    continue

                eligible_nodes.append((node_id, node_data, prop_value))

            print(f"DEBUG: {len(eligible_nodes)} eligible nodes for visualization")

            # Smart node selection for large graphs
            if len(eligible_nodes) <= 25000:
                nodes_to_show = eligible_nodes
                print(f"DEBUG: Showing all {len(nodes_to_show)} nodes (small graph)")
            else:
                print(f"DEBUG: Large graph ({len(eligible_nodes)} nodes) - selecting top aggregated nodes")

                # Always include source nodes
                source_node_data = [(nid, nd, pv) for nid, nd, pv in eligible_nodes if nid in source_nodes]

                # Score and select top non-source nodes
                non_source_nodes = [(nid, nd, pv) for nid, nd, pv in eligible_nodes if nid not in source_nodes]

                scored_nodes = []
                for node_id, node_data, prop_value in non_source_nodes:
                    score = self._calculate_influence_score(node_id, gmm_data)
                    scored_nodes.append((score, node_id, node_data, prop_value))

                scored_nodes.sort(reverse=True, key=lambda x: x[0])
                top_non_sources = [(nid, nd, pv) for score, nid, nd, pv in scored_nodes[:1100]]

                nodes_to_show = source_node_data + top_non_sources
                print(
                    f"DEBUG: Selected {len(source_node_data)} sources + {len(top_non_sources)} aggregated = {len(nodes_to_show)} total")

            # Create visualization nodes
            node_data_for_js = {}
            selected_node_ids = set()

            for node_id, node_data, prop_value in nodes_to_show:
                selected_node_ids.add(node_id)
                node_type = node_data.get('type', 'unknown')

                # Visual properties
                color = get_node_color(node_id, prop_value)
                size = 25 if node_id in source_nodes else (20 if prop_value is not None else 15)

                # Label
                display_name = get_node_display_name(node_data)
                if display_name:
                    label = display_name
                    if prop_value is not None:
                        label += f"\n{prop_value:.2f}"
                else:
                    label = node_id
                    if prop_value is not None:
                        label += f"\n{prop_value:.2f}"

                # Hover title
                title = f"ID: {node_id}\n"
                if display_name and display_name != node_id:
                    title += f"Name: {display_name}\n"
                title += f"Type: {node_type}\n"

                for key, value in node_data.items():
                    if key not in ['type', 'value', 'name', 'label'] and isinstance(value, (str, int, float)):
                        title += f"{key}: {value}\n"

                if node_id in source_nodes:
                    title += "SOURCE NODE\n"
                if prop_value is not None:
                    title += f"Propagated Value: {prop_value:.3f}\n"

                if node_id in gmm_data:
                    gmm = gmm_data[node_id]
                    title += f"Components: {gmm.get('n_components', 0)}\n"
                    title += f"GMM Type: {gmm.get('type', 'unknown')}\n"

                title += "\nUse search below to analyze signal"

                # Add to visualization
                net.add_node(node_id, label=label, title=title, color=color, size=size)

                # Store for JavaScript
                node_data_for_js[node_id] = {
                    'id': node_id,
                    'type': node_type,
                    'propagated_value': prop_value,
                    'is_source': node_id in source_nodes,
                    'gmm_data': gmm_data.get(node_id, {}),
                    'original_data': node_data,
                    'display_name': display_name or node_id
                }

            print(
                f"DEBUG: Visualization nodes: {len(nodes_to_show)} shown, {len(eligible_nodes) - len(nodes_to_show)} filtered out")
            # Add edges between displayed nodes
            filtered_edge_count = 0
            excluded_viz_edges = 0

            for edge_id, edge_data in graph_data.get('edges', {}).items():

                if '_' in edge_id:
                    parts = edge_id.split('_')
                    if len(parts) == 2:
                        source, target = parts

                if not source or not target:
                    excluded_viz_edges += 1
                    continue

                # Only show edges between displayed nodes
                if source not in selected_node_ids or target not in selected_node_ids:
                    excluded_viz_edges += 1
                    continue

                # Apply edge type filters
                edge_type = edge_data.get('edge_type', edge_data.get('type', 'connected'))
                if edge_type in exclude_edge_types:
                    excluded_viz_edges += 1
                    continue

                # Add the edge
                net.add_edge(source, target, title=f"Type: {edge_type}", color="#999999")
                filtered_edge_count += 1

            print(f"DEBUG: Visualization edges: {filtered_edge_count} shown, {excluded_viz_edges} excluded")

            # Generate HTML file
            vis_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                'app', 'static', 'visualizations'
            )
            os.makedirs(vis_dir, exist_ok=True)

            filename = f"{graph_id}_propagation.html"
            filepath = os.path.join(vis_dir, filename)

            print(f"DEBUG: Saving filtered visualization to {filepath}")

            # Save and enhance
            net.save_graph(filepath)
            self._enhance_propagation_visualization(filepath, node_data_for_js, propagation_results)

            print(f"DEBUG: Successfully generated filtered visualization: {filename}")
            return filename

        except Exception as e:
            print(f"Error generating filtered propagation visualization: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _calculate_influence_score(self, node_id, gmm_data):
        """Calculate how 'aggregated/influential' a node is"""
        if node_id not in gmm_data:
            return 0.0

        gmm = gmm_data[node_id]
        score = 0.0

        # Number of components (more = more diverse influences)
        n_components = gmm.get('n_components', 0)
        score += n_components * 10

        # Total evidence (higher = stronger aggregation)
        total_evidence = gmm.get('total_evidence', 0)
        score += total_evidence * 5

        # Signal breadth (spread of components)
        components = gmm.get('components', [])
        if len(components) > 1:
            means = [comp[1] for comp in components if len(comp) >= 2]
            if means:
                signal_spread = max(means) - min(means)
                score += signal_spread * 20

        # Bonus for inferred nodes (we want aggregated nodes, not leaves)
        if gmm.get('type') == 'inferred':
            score += 15

        return score

    def _enhance_propagation_visualization(self, filepath, node_data, propagation_results):
        """Add interactive signal plotting to propagation visualization"""
        try:
            print(f"DEBUG: Enhancing visualization at {filepath}")

            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            import json

            # Enhanced search-based interface with better node display and comparison plots
            enhancement = f"""
            <style>
            #mynetworkid {{
                width: 100% !important;
                height: 600px !important;
                border: 2px solid #ddd;
                border-radius: 8px;
            }}

            body {{
                margin: 0;
                padding: 10px;
                font-family: Arial, sans-serif;
                background: #f8f9fa;
            }}

            .network-container {{
                background: white;
                border-radius: 8px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                padding: 20px;
            }}

            .graph-title {{
                text-align: center;
                margin-bottom: 20px;
                color: #2c3e50;
                font-size: 24px;
                font-weight: bold;
            }}

            .search-panel {{
                background: #f8f9fa;
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 20px;
            }}

            .search-box {{
                position: relative;
                margin-bottom: 15px;
            }}

            .search-input {{
                width: 100%;
                padding: 12px;
                border: 2px solid #ddd;
                border-radius: 6px;
                font-size: 14px;
                box-sizing: border-box;
            }}

            .search-input:focus {{
                outline: none;
                border-color: #007bff;
            }}

            .search-dropdown {{
                position: absolute;
                top: 100%;
                left: 0;
                right: 0;
                background: white;
                border: 1px solid #ddd;
                border-radius: 6px;
                max-height: 300px;
                overflow-y: auto;
                z-index: 1000;
                display: none;
            }}

            .search-item {{
                padding: 12px;
                cursor: pointer;
                border-bottom: 1px solid #eee;
                font-size: 12px;
            }}

            .search-item:hover {{
                background: #f0f8ff;
            }}

            .search-item-main {{
                font-weight: bold;
                font-size: 14px;
                margin-bottom: 4px;
            }}

            .search-item-details {{
                color: #666;
                font-size: 11px;
            }}

            .selected-nodes {{
                margin-bottom: 15px;
            }}

            .selected-node {{
                display: inline-block;
                background: #007bff;
                color: white;
                padding: 8px 12px;
                margin: 3px;
                border-radius: 15px;
                font-size: 12px;
            }}

            .selected-node-name {{
                font-weight: bold;
            }}

            .selected-node-details {{
                opacity: 0.8;
                font-size: 10px;
            }}

            .remove-node {{
                margin-left: 8px;
                cursor: pointer;
                font-weight: bold;
                font-size: 14px;
            }}

            .analyze-btn {{
                background: #28a745;
                color: white;
                border: none;
                padding: 12px 24px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 14px;
                font-weight: bold;
            }}

            .analyze-btn:hover {{
                background: #218838;
            }}

            .analyze-btn:disabled {{
                background: #ccc;
                cursor: not-allowed;
            }}

            .signal-display {{
                margin-top: 20px;
                padding: 20px;
                background: white;
                border: 1px solid #ddd;
                border-radius: 8px;
            }}

            .comparison-info {{
                background: #e7f3ff;
                border: 1px solid #b3d9ff;
                border-radius: 6px;
                padding: 15px;
                margin-bottom: 20px;
            }}

            .node-summary {{
                display: inline-block;
                margin: 5px 10px 5px 0;
                padding: 5px 10px;
                border-radius: 12px;
                font-size: 11px;
                font-weight: bold;
            }}

            .legend {{
                position: absolute;
                top: 20px;
                right: 20px;
                background: rgba(255,255,255,0.95);
                padding: 15px;
                border-radius: 8px;
                border: 1px solid #ddd;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                z-index: 100;
            }}

            .legend h4 {{
                margin: 0 0 10px 0;
                color: #2c3e50;
            }}

            .legend-item {{
                display: flex;
                align-items: center;
                margin-bottom: 8px;
                font-size: 12px;
            }}

            .legend-gradient {{
                width: 80px;
                height: 15px;
                margin-right: 8px;
                border-radius: 3px;
                border: 1px solid #ccc;
            }}

            .legend-color {{
                width: 20px;
                height: 20px;
                border-radius: 50%;
                margin-right: 8px;
                border: 1px solid #ccc;
            }}
            </style>

            <div class="network-container">
                <div class="graph-title">🚀 Propagation Visualization</div>

                <div class="legend">
                    <h4>Value Legend</h4>
                    <div class="legend-item">
                        <div class="legend-gradient" style="background: linear-gradient(to right, #0000ff, #00ff00, #ffff00, #ff0000);"></div>
                        <span id="value-range-text">Value Range</span>
                    </div>
                    <div class="legend-item">
                        <div class="legend-color" style="background-color: #cccccc;"></div>
                        <span>No Data</span>
                    </div>
                </div>

                <div class="search-panel">
                    <h4>🔍 Node Signal Analysis</h4>
                    <div class="search-box">
                        <input type="text" id="node-search" class="search-input" placeholder="Search for nodes by name, type, or properties...">
                        <div id="search-dropdown" class="search-dropdown"></div>
                    </div>

                    <div class="selected-nodes" id="selected-nodes"></div>

                    <button id="analyze-selected" class="analyze-btn" disabled>
                        📊 Compare Selected Signals
                    </button>
                </div>

                <div id="signal-display" class="signal-display" style="display: none;">
                    <h4>Signal Comparison Analysis</h4>
                    <div id="signal-content"></div>
                </div>
            </div>

            <script src="https://cdn.plot.ly/plotly-2.26.0.min.js"></script>
            <script>
            // Store the data globally
            window.nodeData = {json.dumps(node_data)};
            window.propagationResults = {json.dumps(propagation_results)};
            window.selectedNodeIds = new Set();

            console.log('DEBUG: Node data loaded:', Object.keys(window.nodeData).length, 'nodes');

            // Update legend with actual value range
            const valueRange = window.propagationResults.value_range || [0, 1];
            document.getElementById('value-range-text').textContent = 
                valueRange[0].toFixed(2) + ' → ' + valueRange[1].toFixed(2);

            // Search functionality
            const searchInput = document.getElementById('node-search');
            const searchDropdown = document.getElementById('search-dropdown');
            const selectedNodesDiv = document.getElementById('selected-nodes');
            const analyzeBtn = document.getElementById('analyze-selected');

            // Get all node IDs for searching
            const allNodeIds = Object.keys(window.nodeData);

            // Function to get node display name (use 'value' property if available)
            function getNodeDisplayName(nodeId) {{
                const node = window.nodeData[nodeId];

                // Check for common name/value properties
                if (node.original_data && node.original_data.value) {{
                    return node.original_data.value;
                }}
                if (node.original_data && node.original_data.name) {{
                    return node.original_data.name;
                }}
                if (node.original_data && node.original_data.label) {{
                    return node.original_data.label;
                }}

                // Fallback to node ID
                return nodeId;
            }}

            // Function to get searchable text for a node
            function getNodeSearchText(nodeId) {{
                const node = window.nodeData[nodeId];
                const displayName = getNodeDisplayName(nodeId);

                let searchText = `${{nodeId}} ${{displayName}} ${{node.type}}`;

                // Add all original properties to search
                if (node.original_data) {{
                    Object.entries(node.original_data).forEach(([key, value]) => {{
                        if (typeof value === 'string' || typeof value === 'number') {{
                            searchText += ` ${{key}}:${{value}}`;
                        }}
                    }});
                }}

                return searchText.toLowerCase();
            }}

            searchInput.addEventListener('input', function() {{
                const query = this.value.toLowerCase();

                if (query.length < 1) {{
                    searchDropdown.style.display = 'none';
                    return;
                }}

                // Filter nodes based on search
                const matches = allNodeIds.filter(nodeId => {{
                    const searchText = getNodeSearchText(nodeId);
                    return searchText.includes(query);
                }}).slice(0, 15); // Show max 15 results

                if (matches.length > 0) {{
                    searchDropdown.innerHTML = matches.map(nodeId => {{
                        const node = window.nodeData[nodeId];
                        const displayName = getNodeDisplayName(nodeId);
                        const isSelected = window.selectedNodeIds.has(nodeId);

                        // Build property details
                        let details = `Type: ${{node.type}}`;
                        if (node.propagated_value !== null) {{
                            details += ` • Value: ${{node.propagated_value.toFixed(3)}}`;
                        }}
                        if (node.is_source) {{
                            details += ` • SOURCE`;
                        }}

                        return `
                            <div class="search-item" data-node-id="${{nodeId}}" style="${{isSelected ? 'opacity: 0.5;' : ''}}">
                                <div class="search-item-main">${{displayName}} (${{nodeId}})</div>
                                <div class="search-item-details">${{details}}</div>
                            </div>
                        `;
                    }}).join('');
                    searchDropdown.style.display = 'block';
                }} else {{
                    searchDropdown.style.display = 'none';
                }}
            }});

            // Handle search item clicks
            searchDropdown.addEventListener('click', function(e) {{
                const searchItem = e.target.closest('.search-item');
                if (searchItem) {{
                    const nodeId = searchItem.getAttribute('data-node-id');
                    addSelectedNode(nodeId);
                    searchInput.value = '';
                    searchDropdown.style.display = 'none';
                }}
            }});

            // Hide dropdown when clicking elsewhere
            document.addEventListener('click', function(e) {{
                if (!e.target.closest('.search-box')) {{
                    searchDropdown.style.display = 'none';
                }}
            }});

            function addSelectedNode(nodeId) {{
                if (!window.selectedNodeIds.has(nodeId)) {{
                    window.selectedNodeIds.add(nodeId);
                    updateSelectedNodesDisplay();
                    updateAnalyzeButton();
                }}
            }}

            function removeSelectedNode(nodeId) {{
                window.selectedNodeIds.delete(nodeId);
                updateSelectedNodesDisplay();
                updateAnalyzeButton();
            }}

            function updateSelectedNodesDisplay() {{
                const selectedArray = Array.from(window.selectedNodeIds);
                selectedNodesDiv.innerHTML = selectedArray.map(nodeId => {{
                    const displayName = getNodeDisplayName(nodeId);
                    const node = window.nodeData[nodeId];

                    return `
                        <span class="selected-node">
                            <div class="selected-node-name">${{displayName}}</div>
                            <div class="selected-node-details">${{node.type}} • ${{nodeId}}</div>
                            <span class="remove-node" onclick="removeSelectedNode('${{nodeId}}')">&times;</span>
                        </span>
                    `;
                }}).join('');
            }}

            function updateAnalyzeButton() {{
                analyzeBtn.disabled = window.selectedNodeIds.size === 0;
                if (window.selectedNodeIds.size > 1) {{
                    analyzeBtn.textContent = `📊 Compare ${{window.selectedNodeIds.size}} Signals`;
                }} else {{
                    analyzeBtn.textContent = '📊 Analyze Signal';
                }}
            }}

            // Analyze selected nodes
            analyzeBtn.addEventListener('click', function() {{
                analyzeSelectedNodes();
            }});

            function analyzeSelectedNodes() {{
                const selectedArray = Array.from(window.selectedNodeIds);
                if (selectedArray.length === 0) return;

                const signalDisplay = document.getElementById('signal-display');
                const signalContent = document.getElementById('signal-content');

                signalDisplay.style.display = 'block';
                signalContent.innerHTML = '<div style="text-align: center; padding: 20px;">Generating comparison plot...</div>';

                // Create comparison plot
                setTimeout(() => {{
                    generateComparisonPlot(selectedArray);
                }}, 100);
            }}

            function generateComparisonPlot(nodeIds) {{
                const signalContent = document.getElementById('signal-content');

                // Color palette for different nodes
                const nodeColors = [
                    '#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4', '#ffeaa7', 
                    '#dda0dd', '#98d8c8', '#ff9f43', '#a29bfe', '#fd79a8',
                    '#00b894', '#e17055', '#0984e3', '#6c5ce7', '#fdcb6e'
                ];

                // Build comparison info
                let comparisonHtml = '<div class="comparison-info">';
                comparisonHtml += '<h5>📊 Comparing Signals:</h5>';

                nodeIds.forEach((nodeId, index) => {{
                    const node = window.nodeData[nodeId];
                    const displayName = getNodeDisplayName(nodeId);
                    const color = nodeColors[index % nodeColors.length];

                    comparisonHtml += `
                        <div class="node-summary" style="background-color: ${{color}}20; border: 2px solid ${{color}};">
                            <strong>${{displayName}}</strong> (${{node.type}})
                            ${{node.propagated_value !== null ? ' • ' + node.propagated_value.toFixed(3) : ''}}
                            ${{node.is_source ? ' • SOURCE' : ''}}
                        </div>
                    `;
                }});

                comparisonHtml += '</div>';
                comparisonHtml += '<div id="comparison-plot" style="width: 100%; height: 600px;"></div>';

                signalContent.innerHTML = comparisonHtml;

                // Generate the comparison plot
                const plotDiv = document.getElementById('comparison-plot');

                // Create x-range for plotting
                const xRange = [];
                const step = (valueRange[1] - valueRange[0]) / 300;
                for (let x = valueRange[0]; x <= valueRange[1]; x += step) {{
                    xRange.push(x);
                }}

                const traces = [];
                let hasValidData = false;

                // Plot each node's signal
                nodeIds.forEach((nodeId, nodeIndex) => {{
                    const node = window.nodeData[nodeId];
                    const displayName = getNodeDisplayName(nodeId);
                    const baseColor = nodeColors[nodeIndex % nodeColors.length];

                    if (node.gmm_data && node.gmm_data.components && node.gmm_data.components.length > 0) {{
                        hasValidData = true;

                        // Plot individual components for this node
                        node.gmm_data.components.forEach((comp, compIndex) => {{
                            const weight = comp[0];
                            const mean = comp[1];
                            const std = comp[2];

                            // Denormalize for actual values
                            const actualMean = mean * (valueRange[1] - valueRange[0]) + valueRange[0];
                            const actualStd = std * (valueRange[1] - valueRange[0]);

                            const yValues = xRange.map(x => {{
                                const gaussian = weight * (1 / (actualStd * Math.sqrt(2 * Math.PI))) * 
                                               Math.exp(-0.5 * Math.pow((x - actualMean) / actualStd, 2));
                                return gaussian;
                            }});

                            traces.push({{
                                x: xRange,
                                y: yValues,
                                type: 'scatter',
                                mode: 'lines',
                                name: `${{displayName}} C${{compIndex + 1}}`,
                                line: {{ color: baseColor, width: 1, dash: 'dot' }},
                                opacity: 0.6,
                                showlegend: false
                            }});
                        }});

                        // Plot combined signal for this node
                        const combinedY = xRange.map(x => {{
                            let total = 0;
                            node.gmm_data.components.forEach(comp => {{
                                const weight = comp[0];
                                const mean = comp[1];
                                const std = comp[2];
                                const actualMean = mean * (valueRange[1] - valueRange[0]) + valueRange[0];
                                const actualStd = std * (valueRange[1] - valueRange[0]);
                                const gaussian = weight * (1 / (actualStd * Math.sqrt(2 * Math.PI))) * 
                                               Math.exp(-0.5 * Math.pow((x - actualMean) / actualStd, 2));
                                total += gaussian;
                            }});
                            return total;
                        }});

                        traces.push({{
                            x: xRange,
                            y: combinedY,
                            type: 'scatter',
                            mode: 'lines',
                            name: `${{displayName}} (Combined)`,
                            line: {{ color: baseColor, width: 3 }},
                            opacity: 1.0
                        }});

                        // Add vertical line for propagated value
                        if (node.propagated_value !== null) {{
                            const maxY = Math.max.apply(Math, combinedY);
                            traces.push({{
                                x: [node.propagated_value, node.propagated_value],
                                y: [0, maxY * 1.1],
                                type: 'scatter',
                                mode: 'lines',
                                name: `${{displayName}} Final`,
                                line: {{ color: baseColor, dash: 'dash', width: 2 }},
                                opacity: 0.8
                            }});
                        }}
                    }}
                }});

                if (!hasValidData) {{
                    plotDiv.innerHTML = '<div style="text-align: center; padding: 50px; color: #666;"><h3>No Signal Data Available</h3><p>None of the selected nodes have signal components to display.</p></div>';
                    return;
                }}

                const layout = {{
                    title: {{ 
                        text: `Signal Comparison: ${{nodeIds.map(id => getNodeDisplayName(id)).join(' vs ')}}`,
                        font: {{ size: 18 }}
                    }},
                    xaxis: {{ title: 'Value', gridcolor: '#e1e5e9', showgrid: true }},
                    yaxis: {{ title: 'Probability Density', gridcolor: '#e1e5e9', showgrid: true }},
                    showlegend: true,
                    legend: {{ x: 0.02, y: 0.98 }},
                    margin: {{ t: 60, b: 60, l: 80, r: 60 }},
                    plot_bgcolor: '#fafafa',
                    paper_bgcolor: 'white'
                }};

                const config = {{ responsive: true, displayModeBar: true, 'displaylogo': false }};

                Plotly.newPlot(plotDiv, traces, layout, config);
            }}

            // Make functions global so they can be called from HTML
            window.removeSelectedNode = removeSelectedNode;
            window.getNodeDisplayName = getNodeDisplayName;
            </script>
            """

            # Insert enhancement before closing body tag
            enhanced_content = content.replace('</body>', enhancement + '</body>')

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(enhanced_content)

            print("DEBUG: Enhancement complete")

        except Exception as e:
            print(f"Error enhancing propagation visualization: {e}")
            import traceback
            traceback.print_exc()

