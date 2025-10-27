# backend/methods_application/methods/file_histogram_method.py
from backend.methods_application.method_implementation import MethodImplementation
import json
from collections import Counter
from datetime import datetime


class FileHistogramMethod(MethodImplementation):
    method_id = "HistogramMethod"
    method_name = "File Histogram Generator"
    description = "Creates or updates histogram data nodes with file system distribution analysis"

    def execute(self, graph_manager, parameters):
        changes = {
            "nodes_added": 0,
            "edges_added": 0,
            "histograms_updated": 0,
            "method_instances_created": 0,
            "property": parameters.get('property', 'extension'),
            "histogram_data": {},
            "analysis_summary": {}
        }

        property_type = parameters.get('property', 'extension')

        # Create method instance node for traceability
        method_instance_id = self._create_method_instance(graph_manager, property_type, changes)

        # Find or create histogram data node
        histogram_node_id = self._find_or_create_histogram_node(
            graph_manager, property_type, changes
        )

        # Generate histogram data based on property type
        if property_type == 'extension':
            histogram_data, source_data_info = self._generate_extension_histogram(graph_manager)
        elif property_type == 'file_size':
            histogram_data, source_data_info = self._generate_file_size_histogram(graph_manager)
        elif property_type == 'file_type':
            histogram_data, source_data_info = self._generate_file_type_histogram(graph_manager)
        elif property_type == 'folder_contents':
            histogram_data, source_data_info = self._generate_folder_contents_histogram(graph_manager)
        else:
            histogram_data, source_data_info = {}, {}

        # Always connect method to histogram (even if empty)
        self._connect_method_to_output(graph_manager, method_instance_id, histogram_node_id, changes)

        # Update histogram node with data
        self._update_histogram_node(graph_manager, histogram_node_id, histogram_data, changes)

        # Only create provenance edges if we have actual data
        if histogram_data and source_data_info:
            # Connect method instance to input data sources
            self._connect_method_to_inputs(graph_manager, method_instance_id, source_data_info, changes)

            # Connect histogram to the graph structure with provenance
            self._connect_histogram_to_graph(
                graph_manager, histogram_node_id, property_type, source_data_info, changes
            )

        print(f" Created method_instance {method_instance_id} -> histogram_data {histogram_node_id}")
        print(f" Total edges added: {changes['edges_added']}")

        return changes

    def _create_method_instance(self, graph_manager, property_type, changes):
        """Create a method instance node for execution traceability"""
        # Get next available ID to maintain consistency
        method_instance_id = self._get_next_numeric_id(graph_manager)

        # Use GraphManager's add_node method instead of direct manipulation
        try:
            graph_manager.add_node(
                node_id=method_instance_id,
                value=f"HistogramMethod execution for {property_type}",
                type='method_instance',
                hierarchy='analysis',
                attributes={
                    'method_type': 'HistogramMethod',
                    'property_analyzed': property_type,
                    'execution_time': datetime.now().isoformat(),
                    'method_id': self.method_id,
                    'method_name': self.method_name
                }
            )
            changes["nodes_added"] += 1
            changes["method_instances_created"] += 1
            print(f" Created method_instance node {method_instance_id}")

        except KeyError as e:
            print(f" Method instance node {method_instance_id} already exists")

        return method_instance_id

    def _connect_method_to_output(self, graph_manager, method_instance_id, histogram_node_id, changes):
        """Connect method instance to its output histogram data"""
        print(f" Attempting to create edge {method_instance_id} -> {histogram_node_id}")

        # Use GraphManager's add_edge method
        try:
            graph_manager.add_edge(
                source=method_instance_id,
                target=histogram_node_id,
                attributes={
                    'edge_type': 'produces',
                    'direction': 'out',
                    'provenance_type': 'generated_by',
                    'created': datetime.now().isoformat()
                }
            )
            changes["edges_added"] += 1
            print(f" Successfully created edge {method_instance_id} -> {histogram_node_id}")

        except (KeyError, ValueError) as e:
            print(f" Error creating edge {method_instance_id} -> {histogram_node_id}: {e}")

    def _connect_method_to_inputs(self, graph_manager, method_instance_id, source_data_info, changes):
        """Connect method instance to its input data sources"""
        print(f" Connecting method {method_instance_id} to inputs: {source_data_info}")

        for source_type, source_nodes in source_data_info.items():
            for source_node_id in source_nodes:
                print(f" Creating input edge {method_instance_id} -> {source_node_id}")

                try:
                    graph_manager.add_edge(
                        source=method_instance_id,
                        target=source_node_id,
                        attributes={
                            'edge_type': 'uses',
                            'direction': 'out',
                            'provenance_type': 'derived_from',
                            'created': datetime.now().isoformat()
                        }
                    )
                    changes["edges_added"] += 1
                    print(f" Created input edge {method_instance_id} -> {source_node_id}")

                except (KeyError, ValueError) as e:
                    print(f" Error creating input edge: {e}")

    def _generate_extension_histogram(self, graph_manager):
        """Generate histogram data for file extensions with source tracking"""
        extension_counts = Counter()
        source_data_info = {
            'extension': [],
            'file': [],
            'folder': []
        }

        print(" Looking for extension data...")

        # Find all extension nodes
        extension_nodes = []
        for node_id, node_data in graph_manager.node_data.items():
            if node_data.get('type') == 'extension':
                extension_nodes.append((node_id, node_data))
                source_data_info['extension'].append(node_id)

        print(f" Found {len(extension_nodes)} extension nodes: {[n[0] for n in extension_nodes]}")

        for ext_id, ext_data in extension_nodes:
            extension = ext_data.get('value', 'unknown')
            if extension.startswith('.'):
                extension = extension[1:]

            # Count files and track source data
            file_count = 0
            # Look through edge_data which uses tuple keys internally
            for (source, target), edge_attrs in graph_manager.edge_data.items():
                if (target == ext_id and edge_attrs.get('edge_type') == 'has_extension'):
                    file_count += 1
                    source_data_info['file'].append(source)
                    print(f" Found file {source} with extension {ext_id}")

                    # Find containing folders - look for contains edges
                    for (folder_src, folder_tgt), folder_edge_attrs in graph_manager.edge_data.items():
                        if (folder_tgt == source and
                                folder_edge_attrs.get('edge_type') in ['contains_file', 'contains']):
                            source_data_info['folder'].append(folder_src)
                            print(f" Found containing folder {folder_src}")

            print(f" Extension '{extension}' used by {file_count} files")
            if file_count > 0:
                extension_counts[extension] = file_count

        print(f" Final extension counts: {dict(extension_counts)}")
        print(f" Source data info: {source_data_info}")

        return dict(extension_counts), source_data_info

    def _connect_histogram_to_graph(self, graph_manager, histogram_node_id, property_type, source_data_info, changes):
        """Enhanced connection method that handles all property types and creates provenance edges"""

        print(f" Connecting histogram {histogram_node_id} to graph for {property_type}")

        if property_type == 'extension':
            # Connect to contributing folders - remove duplicates
            contributing_folders = list(set(source_data_info.get('folder', [])))

            print(f" Contributing folders: {contributing_folders}")

            for folder_id in contributing_folders:
                print(f" Creating analysis edge {folder_id} -> {histogram_node_id}")

                try:
                    graph_manager.add_edge(
                        source=folder_id,
                        target=histogram_node_id,
                        attributes={
                            'edge_type': 'has_analysis',
                            'direction': 'out',
                            'analysis_type': 'extension_histogram',
                            'provenance_type': 'contributes_to',
                            'created': datetime.now().isoformat()
                        }
                    )
                    changes["edges_added"] += 1
                    print(f" Created analysis edge {folder_id} -> {histogram_node_id}")

                except (KeyError, ValueError) as e:
                    print(f" Error creating analysis edge: {e}")

    def _get_next_numeric_id(self, graph_manager):
        """Generate next available numeric ID"""
        existing_ids = set()
        for node_id in graph_manager.node_data.keys():
            try:
                existing_ids.add(int(node_id))
            except (ValueError, TypeError):
                pass

        next_id = 0
        while next_id in existing_ids:
            next_id += 1

        result = str(next_id)
        print(f" Generated next ID: {result}")
        return result

    def _find_or_create_histogram_node(self, graph_manager, property_type, changes):
        """Find existing histogram node or create new one"""
        # Search for existing histogram nodes directly
        for node_id, node_data in graph_manager.node_data.items():
            if (node_data.get('type') == 'histogram_data' and
                    node_data.get('property_type') == property_type):
                print(f" Found existing histogram node {node_id}")
                return node_id

        # Create new histogram node with consistent numeric ID
        hist_id = self._get_next_numeric_id(graph_manager)

        # Use GraphManager's add_node method
        try:
            graph_manager.add_node(
                node_id=hist_id,
                value=f"{property_type} histogram",
                type='histogram_data',
                hierarchy='analysis',
                attributes={
                    'property_type': property_type,
                    'histogram_values': '[]',
                    'histogram_labels': '[]',
                    'created': datetime.now().isoformat()
                }
            )
            changes["nodes_added"] += 1
            print(f" Created new histogram node {hist_id}")

        except KeyError as e:
            print(f" Histogram node {hist_id} already exists")

        return hist_id

    def _update_histogram_node(self, graph_manager, hist_id, histogram_data, changes):
        """Update histogram node with new data"""
        labels = list(histogram_data.keys())
        values = list(histogram_data.values())

        # Use GraphManager's update method
        update_attrs = {
            'histogram_labels': json.dumps(labels),
            'histogram_values': json.dumps(values),
            'last_updated': datetime.now().isoformat(),
            'total_items': sum(values) if values else 0,
            'categories': len(labels)
        }

        try:
            graph_manager.update_node_attributes(hist_id, update_attrs)
            changes["histograms_updated"] += 1

        except KeyError as e:
            print(f" Error updating histogram node {hist_id}: {e}")

        changes["histogram_data"] = histogram_data
        changes["analysis_summary"] = {
            'total_items': sum(values) if values else 0,
            'categories': len(labels),
            'most_common': max(histogram_data.items(), key=lambda x: x[1]) if histogram_data else None
        }

    def _format_size(self, size_bytes):
        """Format size in human readable format"""
        if size_bytes < 1024:
            return f"{int(size_bytes)}B"
        elif size_bytes < 1024 ** 2:
            return f"{size_bytes / 1024:.1f}KB"
        elif size_bytes < 1024 ** 3:
            return f"{size_bytes / (1024 ** 2):.1f}MB"
        else:
            return f"{size_bytes / (1024 ** 3):.1f}GB"