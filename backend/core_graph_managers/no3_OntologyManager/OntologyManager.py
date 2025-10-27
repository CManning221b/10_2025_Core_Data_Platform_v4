from collections import defaultdict
import re
import statistics
import datetime
from typing import Dict, List, Set, Tuple, Optional, Any


class OntologyManager:
    """
    GraphOntologyAnalyzer: A straightforward tool for extracting and analyzing type information from graphs.

    This class examines a graph to identify node types, edge types, and their relationships. It focuses
    on practical pattern discovery rather than complex ontological reasoning.

    Core Functions:
        1. Type Identification - Extract node and edge types with their frequency distribution
        2. Property Analysis - Analyze properties of each type (ranges, patterns, statistics)
        3. Relationship Mapping - Identify valid node-edge-node patterns and their cardinality
        4. Pattern Discovery - Find common patterns in data values (min/max, regex, length)
        5. Visualization - Export findings as a visual graph or in standard formats (OWL, JSON)
        6. Comparison - Compare patterns across different graphs to identify similarities
        7. Type Inference - Suggest types for untyped nodes based on property signatures

    The analyzer builds a clear, statistical understanding of your graph structure that helps identify
    anomalies, suggest improvements, and define schemas for data validation.

    Usage Examples:
        - Extract a complete type map from a graph of files and folders
        - Analyze how entities change over time by comparing versions
        - Recognize unknown nodes by matching their patterns to known types
        - Generate data validation schemas based on observed patterns
        - Find inconsistencies in data structures across multiple sources
    """

    def __init__(self, graph_manager=None):
        """Initialize with an optional GraphManager instance_graph"""
        # The connected graph manager
        self.graph_manager = graph_manager

        # Core ontology structures
        self.node_types = {}  # Dict[type_name] = {"count": int, "properties": set(), "nodes": set()}
        self.edge_types = {}  # Dict[type_name] = {"count": int, "properties": set(), "edges": set()}

        # Relationship patterns
        self.relationships = defaultdict(int)  # Dict[(source_type, edge_type, target_type)] = count

        # Property statistics for each type
        self.node_property_stats = {}  # Dict[type_name][property_name] = {"type": type, "stats": {...}}
        self.edge_property_stats = {}  # Dict[type_name][property_name] = {"type": type, "stats": {...}}

        # Ontology extracted flag
        self.ontology_extracted = False

        # If a graph manager was provided, automatically load it
        if graph_manager:
            self.load_graph(graph_manager)

    def load_graph(self, graph_manager):
        """
        Connect to a graph for analysis

        Args:
            graph_manager: An instance_graph of GraphManager containing the graph to analyze

        Raises:
            ValueError: If the provided graph_manager is invalid
        """
        if graph_manager is None:
            raise ValueError("A valid GraphManager instance_graph must be provided")

        # Store the graph manager
        self.graph_manager = graph_manager

        # Reset ontology state
        self.node_types = {}
        self.edge_types = {}
        self.relationships = defaultdict(int)
        self.node_property_stats = {}
        self.edge_property_stats = {}
        self.ontology_extracted = False

        # Log successful connection
        print(f"Connected to graph with {len(graph_manager.node_data)} nodes and {len(graph_manager.edge_data)} edges")

    def extract_ontology(self):
        """
        Extract the full ontology from the connected graph

        This method analyzes the graph to identify:
        - Node types and their properties
        - Edge types and their properties
        - Valid relationship patterns (source_type -> edge_type -> target_type)
        - Property statistics and patterns

        Returns:
            Dict: A summary of the extracted ontology

        Raises:
            ValueError: If no graph is loaded
        """
        if self.graph_manager is None:
            raise ValueError("No graph loaded. Call load_graph() first.")

        # Extract node types and properties
        self._extract_node_types()

        # Extract edge types and properties
        self._extract_edge_types()

        # Extract relationship patterns
        self._extract_relationships()

        # Analyze property patterns
        self._analyze_property_patterns()

        # Mark ontology as extracted
        self.ontology_extracted = True

        # Return a summary of the extraction
        return {
            "node_types": len(self.node_types),
            "edge_types": len(self.edge_types),
            "relationship_patterns": len(self.relationships),
            "total_nodes": sum(type_info["count"] for type_info in self.node_types.values()),
            "total_edges": sum(type_info["count"] for type_info in self.edge_types.values())
        }

    def get_node_types(self):
        """
        Return a dictionary of all node types with frequency counts

        Returns:
            Dict: A dictionary where keys are node type names and values are dictionaries
                 containing count, property list, and sample nodes

        Raises:
            ValueError: If ontology has not been extracted yet
        """
        if not self.ontology_extracted:
            self.extract_ontology()

        # Create a simplified view of node types with counts
        result = {}
        for type_name, type_info in self.node_types.items():
            result[type_name] = {
                "count": type_info["count"],
                "properties": list(type_info["properties"]),
                "coverage": type_info["count"] / len(
                    self.graph_manager.node_data) if self.graph_manager.node_data else 0,
                "sample_nodes": list(type_info["nodes"])[:5] if "nodes" in type_info else []
            }

        return result

    def get_edge_types(self):
        """
        Return a dictionary of all edge types with frequency counts

        Returns:
            Dict: A dictionary where keys are edge type names and values are dictionaries
                 containing count, property list, and sample edges

        Raises:
            ValueError: If ontology has not been extracted yet
        """
        if not self.ontology_extracted:
            self.extract_ontology()

        # Create a simplified view of edge types with counts
        result = {}
        for type_name, type_info in self.edge_types.items():
            result[type_name] = {
                "count": type_info["count"],
                "properties": list(type_info["properties"]),
                "coverage": type_info["count"] / len(
                    self.graph_manager.edge_data) if self.graph_manager.edge_data else 0,
                "sample_edges": list(type_info["edges"])[:5] if "edges" in type_info else []
            }

        return result

    # Helper methods_application

    def _extract_node_types(self):
        """Extract node types and their properties from the graph"""
        # Reset node types
        self.node_types = {}

        # Process each node in the graph
        for node_id, node_data in self.graph_manager.node_data.items():
            # Get the node type (defaulting to "Unknown" if not present)
            node_type = node_data.get("type", "Unknown")

            # Initialize entry for this type if it doesn't exist
            if node_type not in self.node_types:
                self.node_types[node_type] = {
                    "count": 0,
                    "properties": set(),
                    "nodes": set()
                }

            # Update the count and add this node to the samples
            self.node_types[node_type]["count"] += 1
            self.node_types[node_type]["nodes"].add(node_id)

            # Add all property names from this node
            for prop_name in node_data.keys():
                self.node_types[node_type]["properties"].add(prop_name)

    def _extract_edge_types(self):
        """Extract edge types and their properties from the graph"""
        # Reset edge types
        self.edge_types = {}

        # Process each edge in the graph
        for edge_key, edge_data in self.graph_manager.edge_data.items():
            # Get the edge type (defaulting to "Unknown" if not present)
            edge_type = edge_data.get("edge_type", "Unknown")

            # Initialize entry for this type if it doesn't exist
            if edge_type not in self.edge_types:
                self.edge_types[edge_type] = {
                    "count": 0,
                    "properties": set(),
                    "edges": set()
                }

            # Update the count and add this edge to the samples
            self.edge_types[edge_type]["count"] += 1
            self.edge_types[edge_type]["edges"].add(edge_key)

            # Add all property names from this edge
            for prop_name in edge_data.keys():
                self.edge_types[edge_type]["properties"].add(prop_name)

    def _extract_relationships(self):
        """Identify valid relationship patterns between node types"""
        # Reset relationships
        self.relationships = defaultdict(int)

        # Process each edge to identify relationship patterns
        for (source_id, target_id), edge_data in self.graph_manager.edge_data.items():
            # Get types
            source_type = self.graph_manager.node_data.get(source_id, {}).get("type", "Unknown")
            edge_type = edge_data.get("edge_type", "Unknown")
            target_type = self.graph_manager.node_data.get(target_id, {}).get("type", "Unknown")

            # Record this relationship pattern
            relationship_key = (source_type, edge_type, target_type)
            self.relationships[relationship_key] += 1

    def _analyze_property_patterns(self):
        """Analyze patterns in property values for each type"""
        # Reset property statistics
        self.node_property_stats = {}
        self.edge_property_stats = {}

        # Analyze node property patterns
        for node_type, type_info in self.node_types.items():
            self.node_property_stats[node_type] = {}

            # For each property of this type
            for prop_name in type_info["properties"]:
                # Collect all values of this property
                values = []
                for node_id in type_info["nodes"]:
                    node_data = self.graph_manager.node_data.get(node_id, {})
                    if prop_name in node_data:
                        values.append(node_data[prop_name])

                # Skip empty properties
                if not values:
                    continue

                # Analyze this property
                self.node_property_stats[node_type][prop_name] = self._analyze_property_values(values)

        # Analyze edge property patterns
        for edge_type, type_info in self.edge_types.items():
            self.edge_property_stats[edge_type] = {}

            # For each property of this type
            for prop_name in type_info["properties"]:
                # Collect all values of this property
                values = []
                for edge_key in type_info["edges"]:
                    edge_data = self.graph_manager.edge_data.get(edge_key, {})
                    if prop_name in edge_data:
                        values.append(edge_data[prop_name])

                # Skip empty properties
                if not values:
                    continue

                # Analyze this property
                self.edge_property_stats[edge_type][prop_name] = self._analyze_property_values(values)

    def _analyze_property_values(self, values):
        """
        Analyze a list of property values to determine patterns

        Args:
            values: List of values to analyze

        Returns:
            Dict: Statistics and patterns for these values
        """
        # Determine value type
        value_types = set(type(val).__name__ for val in values)
        primary_type = max(value_types, key=lambda t: sum(1 for v in values if type(v).__name__ == t))

        # Count presence
        present_count = len(values)

        # Basic result with type info
        result = {
            "value_types": list(value_types),
            "primary_type": primary_type,
            "present_count": present_count,
            "unique_count": len(set(str(v) for v in values))  # Convert to strings for comparison
        }

        # Analyze based on primary type
        if primary_type in ("str", "string"):
            # String analysis
            lengths = [len(str(v)) for v in values if isinstance(v, str)]

            if lengths:
                result.update({
                    "min_length": min(lengths),
                    "max_length": max(lengths),
                    "avg_length": sum(lengths) / len(lengths),
                    "sample_values": [str(v) for v in values[:5] if isinstance(v, str)]
                })

                # Try to determine pattern
                if len(set(values)) <= 10:
                    # Looks like an enumeration
                    result["pattern_type"] = "enumeration"
                    result["enum_values"] = list(set(str(v) for v in values))
                else:
                    # Try to infer a regex pattern
                    result["pattern_type"] = "string"

                    # Check if it's a date
                    date_pattern = r'\d{4}-\d{2}-\d{2}'
                    if all(re.match(date_pattern, str(v)) for v in values[:10] if isinstance(v, str)):
                        result["pattern_type"] = "date"

                    # Check if it's an email
                    email_pattern = r'[\w\.-]+@[\w\.-]+'
                    if all(re.match(email_pattern, str(v)) for v in values[:10] if isinstance(v, str)):
                        result["pattern_type"] = "email"

                    # Check if it's a URL
                    url_pattern = r'https?://\S+'
                    if all(re.match(url_pattern, str(v)) for v in values[:10] if isinstance(v, str)):
                        result["pattern_type"] = "url"

        elif primary_type in ("int", "float", "decimal"):
            # Numeric analysis
            numeric_values = [float(v) for v in values if isinstance(v, (int, float))]

            if numeric_values:
                result.update({
                    "min_value": min(numeric_values),
                    "max_value": max(numeric_values),
                    "avg_value": sum(numeric_values) / len(numeric_values),
                    "median_value": statistics.median(numeric_values) if len(numeric_values) > 0 else None,
                    "sample_values": numeric_values[:5]
                })

                # If all values are integers, specify int pattern
                if all(float(v).is_integer() for v in numeric_values):
                    result["pattern_type"] = "integer"
                else:
                    result["pattern_type"] = "float"

        elif primary_type == "bool":
            # Boolean analysis
            true_count = sum(1 for v in values if v)
            result.update({
                "true_count": true_count,
                "false_count": present_count - true_count,
                "true_percentage": (true_count / present_count) * 100 if present_count > 0 else 0,
                "pattern_type": "boolean"
            })

        elif primary_type in ("list", "set", "tuple", "array"):
            # Collection analysis
            result.update({
                "min_items": min(len(v) for v in values if hasattr(v, "__len__")),
                "max_items": max(len(v) for v in values if hasattr(v, "__len__")),
                "pattern_type": "array"
            })

            # Try to determine item types
            item_types = set()
            for v in values:
                if hasattr(v, "__iter__"):
                    for item in v:
                        item_types.add(type(item).__name__)

            result["item_types"] = list(item_types)

        elif primary_type == "dict":
            # Dictionary analysis
            result.update({
                "min_keys": min(len(v.keys()) for v in values if hasattr(v, "keys")),
                "max_keys": max(len(v.keys()) for v in values if hasattr(v, "keys")),
                "pattern_type": "object"
            })

            # Collect common keys
            key_counts = defaultdict(int)
            for v in values:
                if hasattr(v, "keys"):
                    for k in v.keys():
                        key_counts[k] += 1

            result["common_keys"] = {k: count for k, count in key_counts.items()
                                     if count >= present_count * 0.5}

        else:
            # Other types
            result["pattern_type"] = "unknown"

        return result

    def analyze_node_type(self, type_name):
        """
        Analyze properties and patterns for a specific node type

        This method provides detailed information about a node type, including:
        - Property distribution and statistics
        - Common value patterns
        - Relationships with other node types
        - Structural characteristics (centrality, hierarchy position)

        Args:
            type_name: The name of the node type to analyze

        Returns:
            Dict: A comprehensive analysis of the node type

        Raises:
            ValueError: If the ontology hasn't been extracted or the type doesn't exist
        """
        if not self.ontology_extracted:
            self.extract_ontology()

        if type_name not in self.node_types:
            raise ValueError(f"Node type '{type_name}' not found in the ontology")

        # Get basic information about this node type
        type_info = self.node_types[type_name]
        node_ids = type_info["nodes"]

        # Prepare the analysis result
        analysis = {
            "type_name": type_name,
            "count": type_info["count"],
            "percentage_of_graph": type_info["count"] / len(self.graph_manager.node_data) * 100,
            "properties": {},
            "relationships": {
                "outgoing": defaultdict(list),
                "incoming": defaultdict(list)
            },
            "structural_metrics": {}
        }

        # Analyze each property
        for prop_name in type_info["properties"]:
            if type_name in self.node_property_stats and prop_name in self.node_property_stats[type_name]:
                property_stats = self.node_property_stats[type_name][prop_name]
                property_analysis = {
                    "presence_percentage": property_stats["present_count"] / type_info["count"] * 100,
                    "value_type": property_stats["primary_type"],
                    "stats": property_stats
                }

                # Determine if this property is required or optional
                if property_stats["present_count"] == type_info["count"]:
                    property_analysis["requirement"] = "required"
                elif property_stats["present_count"] > type_info["count"] * 0.9:
                    property_analysis["requirement"] = "typically_present"
                elif property_stats["present_count"] < type_info["count"] * 0.3:
                    property_analysis["requirement"] = "rarely_present"
                else:
                    property_analysis["requirement"] = "optional"

                # Determine property uniqueness
                if property_stats["unique_count"] == property_stats["present_count"]:
                    property_analysis["uniqueness"] = "unique"
                elif property_stats["unique_count"] == 1:
                    property_analysis["uniqueness"] = "constant"
                elif property_stats["unique_count"] < property_stats["present_count"] * 0.1:
                    property_analysis["uniqueness"] = "low_cardinality"
                else:
                    property_analysis["uniqueness"] = "variable"

                analysis["properties"][prop_name] = property_analysis

        # Analyze relationships with other node types
        for (src_type, edge_type, tgt_type), count in self.relationships.items():
            # Outgoing relationships (this type is the source)
            if src_type == type_name:
                analysis["relationships"]["outgoing"][edge_type].append({
                    "target_type": tgt_type,
                    "count": count,
                    "percentage": count / type_info["count"] * 100
                })

            # Incoming relationships (this type is the target)
            if tgt_type == type_name:
                analysis["relationships"]["incoming"][edge_type].append({
                    "source_type": src_type,
                    "count": count,
                    "percentage": count / type_info["count"] * 100
                })

        # Calculate structural metrics
        total_outgoing = sum(rel["count"] for edge_rels in analysis["relationships"]["outgoing"].values()
                             for rel in edge_rels)
        total_incoming = sum(rel["count"] for edge_rels in analysis["relationships"]["incoming"].values()
                             for rel in edge_rels)

        analysis["structural_metrics"] = {
            "avg_outgoing_edges": total_outgoing / type_info["count"] if type_info["count"] > 0 else 0,
            "avg_incoming_edges": total_incoming / type_info["count"] if type_info["count"] > 0 else 0,
            "connectivity_ratio": (total_outgoing + total_incoming) / type_info["count"] if type_info[
                                                                                                "count"] > 0 else 0
        }

        # Calculate degree distribution (if the graph is not too large)
        if type_info["count"] <= 1000:
            out_degrees = []
            in_degrees = []

            for node_id in node_ids:
                # Use graph_manager to get node degrees
                if hasattr(self.graph_manager, "get_node_degree"):
                    degree_info = self.graph_manager.get_node_degree(node_id)
                    out_degrees.append(degree_info.get("out_degree", 0))
                    in_degrees.append(degree_info.get("in_degree", 0))
                else:
                    # Fallback method if get_node_degree is not available
                    outgoing = sum(1 for edge_key in self.graph_manager.edge_data
                                   if edge_key[0] == node_id)
                    incoming = sum(1 for edge_key in self.graph_manager.edge_data
                                   if edge_key[1] == node_id)
                    out_degrees.append(outgoing)
                    in_degrees.append(incoming)

            # Add degree distribution metrics
            analysis["structural_metrics"]["out_degree_distribution"] = {
                "min": min(out_degrees) if out_degrees else 0,
                "max": max(out_degrees) if out_degrees else 0,
                "median": statistics.median(out_degrees) if out_degrees else 0,
                "isolated_nodes": sum(1 for d in out_degrees if d == 0),
                "histogram": self._generate_histogram(out_degrees, 5)  # 5 bins
            }

            analysis["structural_metrics"]["in_degree_distribution"] = {
                "min": min(in_degrees) if in_degrees else 0,
                "max": max(in_degrees) if in_degrees else 0,
                "median": statistics.median(in_degrees) if in_degrees else 0,
                "terminal_nodes": sum(1 for d in in_degrees if d == 0),
                "histogram": self._generate_histogram(in_degrees, 5)  # 5 bins
            }

        # Look for patterns in node position within the graph
        hierarchical_position = self._analyze_hierarchical_position(type_name)
        if hierarchical_position:
            analysis["structural_metrics"]["hierarchical_position"] = hierarchical_position

        # Add a sample of node IDs for reference
        analysis["sample_nodes"] = list(node_ids)[:10]

        return analysis

    def analyze_edge_type(self, type_name):
        """
        Analyze properties and patterns for a specific edge type

        This method provides detailed information about an edge type, including:
        - Property distribution and statistics
        - Common value patterns
        - Node types it connects
        - Structural characteristics (directionality, cardinality)

        Args:
            type_name: The name of the edge type to analyze

        Returns:
            Dict: A comprehensive analysis of the edge type

        Raises:
            ValueError: If the ontology hasn't been extracted or the type doesn't exist
        """
        if not self.ontology_extracted:
            self.extract_ontology()

        if type_name not in self.edge_types:
            raise ValueError(f"Edge type '{type_name}' not found in the ontology")

        # Get basic information about this edge type
        type_info = self.edge_types[type_name]
        edge_keys = type_info["edges"]

        # Prepare the analysis result
        analysis = {
            "type_name": type_name,
            "count": type_info["count"],
            "percentage_of_graph": type_info["count"] / len(self.graph_manager.edge_data) * 100,
            "properties": {},
            "connectivity": {
                "source_types": defaultdict(int),
                "target_types": defaultdict(int),
                "node_type_pairs": defaultdict(int)
            },
            "cardinality": {}
        }

        # Analyze each property
        for prop_name in type_info["properties"]:
            if type_name in self.edge_property_stats and prop_name in self.edge_property_stats[type_name]:
                property_stats = self.edge_property_stats[type_name][prop_name]
                property_analysis = {
                    "presence_percentage": property_stats["present_count"] / type_info["count"] * 100,
                    "value_type": property_stats["primary_type"],
                    "stats": property_stats
                }

                # Determine if this property is required or optional
                if property_stats["present_count"] == type_info["count"]:
                    property_analysis["requirement"] = "required"
                elif property_stats["present_count"] > type_info["count"] * 0.9:
                    property_analysis["requirement"] = "typically_present"
                elif property_stats["present_count"] < type_info["count"] * 0.3:
                    property_analysis["requirement"] = "rarely_present"
                else:
                    property_analysis["requirement"] = "optional"

                # Determine property uniqueness
                if property_stats["unique_count"] == property_stats["present_count"]:
                    property_analysis["uniqueness"] = "unique"
                elif property_stats["unique_count"] == 1:
                    property_analysis["uniqueness"] = "constant"
                elif property_stats["unique_count"] < property_stats["present_count"] * 0.1:
                    property_analysis["uniqueness"] = "low_cardinality"
                else:
                    property_analysis["uniqueness"] = "variable"

                analysis["properties"][prop_name] = property_analysis

        # Analyze the node types this edge connects
        for (src_type, edge_type, tgt_type), count in self.relationships.items():
            if edge_type == type_name:
                # Count source and target node types
                analysis["connectivity"]["source_types"][src_type] += count
                analysis["connectivity"]["target_types"][tgt_type] += count

                # Count type pairs
                type_pair = (src_type, tgt_type)
                analysis["connectivity"]["node_type_pairs"][type_pair] += count

        # Calculate the most common source and target types
        analysis["connectivity"]["primary_source_type"] = max(
            analysis["connectivity"]["source_types"].items(),
            key=lambda x: x[1],
            default=(None, 0)
        )[0]

        analysis["connectivity"]["primary_target_type"] = max(
            analysis["connectivity"]["target_types"].items(),
            key=lambda x: x[1],
            default=(None, 0)
        )[0]

        # Analyze cardinality patterns
        self._analyze_cardinality_patterns(type_name, analysis)

        # Convert defaultdicts to regular dicts for easier serialization
        analysis["connectivity"]["source_types"] = dict(analysis["connectivity"]["source_types"])
        analysis["connectivity"]["target_types"] = dict(analysis["connectivity"]["target_types"])
        analysis["connectivity"]["node_type_pairs"] = {
            f"{src}→{tgt}": count for (src, tgt), count in analysis["connectivity"]["node_type_pairs"].items()
        }

        # Add edge length statistics if the type is used for connecting nodes of the same type
        self._analyze_edge_cycles(type_name, analysis)

        # Add a sample of edge keys for reference
        analysis["sample_edges"] = list(edge_keys)[:10]

        return analysis

    # Additional helper methods_application

    def _analyze_hierarchical_position(self, type_name):
        """Analyze where nodes of this type tend to sit in the graph hierarchy"""
        nodes_of_type = self.node_types[type_name]["nodes"]

        # Skip if there are too many nodes to analyze efficiently
        if len(nodes_of_type) > 1000:
            return {
                "analysis_status": "skipped",
                "reason": "Too many nodes to analyze efficiently"
            }

        # Look for patterns in the graph structure
        leaf_nodes = 0
        root_nodes = 0
        intermediate_nodes = 0

        for node_id in nodes_of_type:
            # Count incoming and outgoing edges
            outgoing = 0
            incoming = 0

            # Check if the graph manager has neighbor info methods_application
            if hasattr(self.graph_manager, "get_node_neighbors"):
                neighbors = self.graph_manager.get_node_neighbors(node_id)
                outgoing = len(neighbors.get("outgoing", []))
                incoming = len(neighbors.get("incoming", []))
            else:
                # Fallback method
                for edge_key in self.graph_manager.edge_data:
                    if edge_key[0] == node_id:
                        outgoing += 1
                    if edge_key[1] == node_id:
                        incoming += 1

            # Categorize node based on connections
            if outgoing == 0 and incoming > 0:
                leaf_nodes += 1
            elif incoming == 0 and outgoing > 0:
                root_nodes += 1
            elif outgoing > 0 and incoming > 0:
                intermediate_nodes += 1

        # Calculate percentages
        total_nodes = len(nodes_of_type)

        return {
            "leaf_percentage": (leaf_nodes / total_nodes) * 100 if total_nodes > 0 else 0,
            "root_percentage": (root_nodes / total_nodes) * 100 if total_nodes > 0 else 0,
            "intermediate_percentage": (intermediate_nodes / total_nodes) * 100 if total_nodes > 0 else 0,
            "isolated_percentage": ((total_nodes - leaf_nodes - root_nodes - intermediate_nodes) / total_nodes) * 100
            if total_nodes > 0 else 0
        }

    def _analyze_cardinality_patterns(self, edge_type, analysis):
        """Analyze cardinality patterns for an edge type"""
        # Need the edges of this type for analysis
        edge_keys = self.edge_types[edge_type]["edges"]

        # Track cardinality stats
        source_targets = defaultdict(set)  # source -> set of targets
        target_sources = defaultdict(set)  # target -> set of sources

        # Build the mappings
        for edge_key in edge_keys:
            source, target = edge_key
            source_targets[source].add(target)
            target_sources[target].add(source)

        # Analyze outgoing cardinality (source -> targets)
        out_counts = [len(targets) for targets in source_targets.values()]

        # Analyze incoming cardinality (target <- sources)
        in_counts = [len(sources) for sources in target_sources.values()]

        # Determine cardinality types
        if not out_counts or not in_counts:
            return

        # Outgoing cardinality
        out_max = max(out_counts) if out_counts else 0
        out_min = min(out_counts) if out_counts else 0
        out_median = statistics.median(out_counts) if out_counts else 0
        out_mean = sum(out_counts) / len(out_counts) if out_counts else 0

        # Incoming cardinality
        in_max = max(in_counts) if in_counts else 0
        in_min = min(in_counts) if in_counts else 0
        in_median = statistics.median(in_counts) if in_counts else 0
        in_mean = sum(in_counts) / len(in_counts) if in_counts else 0

        # Determine overall cardinality patterns
        if out_max <= 1 and in_max <= 1:
            cardinality_type = "one-to-one"
        elif out_max <= 1 and in_max > 1:
            cardinality_type = "one-to-many"
        elif out_max > 1 and in_max <= 1:
            cardinality_type = "many-to-one"
        else:
            cardinality_type = "many-to-many"

        # Populate cardinality analysis
        analysis["cardinality"] = {
            "pattern": cardinality_type,
            "outgoing": {
                "min": out_min,
                "max": out_max,
                "median": out_median,
                "mean": out_mean,
                "histogram": self._generate_histogram(out_counts, 5)
            },
            "incoming": {
                "min": in_min,
                "max": in_max,
                "median": in_median,
                "mean": in_mean,
                "histogram": self._generate_histogram(in_counts, 5)
            }
        }

    def _analyze_edge_cycles(self, edge_type, analysis):
        """Analyze if this edge type forms cycles (connecting same-type nodes)"""
        type_pairs = analysis["connectivity"]["node_type_pairs"]

        # Check for edges connecting the same node type
        cycles = [(src, tgt) for (src, tgt) in [tuple(pair.split('→')) for pair in type_pairs.keys()]
                  if src == tgt]

        if cycles:
            # Get relevant edges
            cycle_edges = []
            for edge_key in self.edge_types[edge_type]["edges"]:
                source, target = edge_key
                source_type = self.graph_manager.node_data.get(source, {}).get("type", "Unknown")
                target_type = self.graph_manager.node_data.get(target, {}).get("type", "Unknown")

                if source_type == target_type and (source_type, target_type) in cycles:
                    cycle_edges.append(edge_key)

            # Analyze type of cycles
            if cycle_edges:
                # Check if any edges form direct cycles
                direct_cycles = sum(1 for (src, tgt) in cycle_edges if (tgt, src) in cycle_edges)

                analysis["cycles"] = {
                    "same_type_connections": len(cycle_edges),
                    "direct_cycles": direct_cycles > 0,
                    "self_references": sum(1 for (src, tgt) in cycle_edges if src == tgt)
                }

    def _generate_histogram(self, values, num_bins=5):
        """Generate a histogram for a list of values"""
        if not values:
            return []

        # Determine bin edges
        min_val = min(values)
        max_val = max(values)

        # If all values are the same, return a single bin
        if min_val == max_val:
            return [{"range": f"{min_val}", "count": len(values)}]

        # Calculate bin width
        bin_width = (max_val - min_val) / num_bins

        # Initialize bins
        bins = [{"range": f"{min_val + i * bin_width:.1f}-{min_val + (i + 1) * bin_width:.1f}",
                 "count": 0} for i in range(num_bins)]

        # Count values in bins
        for val in values:
            # Handle edge case for maximum value
            if val == max_val:
                bin_idx = num_bins - 1
            else:
                bin_idx = min(int((val - min_val) / bin_width), num_bins - 1)
            bins[bin_idx]["count"] += 1

        return bins

    def get_type_property_statistics(self, type_name, is_node_type=True):
        """
        Get statistical information about properties for a given type

        This method provides consolidated statistics about all properties
        for a specific node or edge type, including:
        - Property presence and frequency
        - Value type distributions
        - Pattern recognition findings
        - Validation suggestions

        Args:
            type_name: The name of the node or edge type to analyze
            is_node_type: Boolean flag indicating if this is a node type (True) or edge type (False)

        Returns:
            Dict: A dictionary of property statistics for the specified type

        Raises:
            ValueError: If the ontology hasn't been extracted or the type doesn't exist
        """
        if not self.ontology_extracted:
            self.extract_ontology()

        # Verify type exists and get the right property stats collection
        if is_node_type:
            if type_name not in self.node_types:
                raise ValueError(f"Node type '{type_name}' not found in the ontology")
            type_info = self.node_types[type_name]
            property_stats = self.node_property_stats.get(type_name, {})
        else:
            if type_name not in self.edge_types:
                raise ValueError(f"Edge type '{type_name}' not found in the ontology")
            type_info = self.edge_types[type_name]
            property_stats = self.edge_property_stats.get(type_name, {})

        # Count of this type
        type_count = type_info["count"]

        # Prepare result structure
        result = {
            "type_name": type_name,
            "is_node_type": is_node_type,
            "instance_count": type_count,
            "property_count": len(type_info["properties"]),
            "properties": {},
            "validation_suggestions": []
        }

        # Analyze each property
        for prop_name in type_info["properties"]:
            if prop_name in property_stats:
                stats = property_stats[prop_name]
                prop_presence = stats["present_count"] / type_count if type_count > 0 else 0

                # Classify property requirements
                requirement = "optional"
                if prop_presence == 1.0:
                    requirement = "required"
                elif prop_presence > 0.9:
                    requirement = "typically_present"
                elif prop_presence < 0.3:
                    requirement = "rarely_present"

                # Classify property cardinality
                uniqueness = "variable"
                if stats["unique_count"] == stats["present_count"]:
                    uniqueness = "unique"
                elif stats["unique_count"] == 1:
                    uniqueness = "constant"
                elif stats["unique_count"] < stats["present_count"] * 0.1:
                    uniqueness = "low_cardinality"

                # Create property summary
                property_summary = {
                    "presence": {
                        "count": stats["present_count"],
                        "percentage": prop_presence * 100,
                        "requirement": requirement
                    },
                    "type_info": {
                        "primary_type": stats["primary_type"],
                        "all_types": stats["value_types"],
                        "pattern_type": stats.get("pattern_type", "unknown")
                    },
                    "cardinality": {
                        "unique_values": stats["unique_count"],
                        "uniqueness": uniqueness
                    },
                    "statistics": {}
                }

                # Add type-specific statistics
                if stats["primary_type"] in ("str", "string"):
                    if "min_length" in stats:
                        property_summary["statistics"].update({
                            "min_length": stats["min_length"],
                            "max_length": stats["max_length"],
                            "avg_length": stats["avg_length"]
                        })
                    if "enum_values" in stats:
                        property_summary["statistics"]["allowed_values"] = stats["enum_values"]
                    if "sample_values" in stats:
                        property_summary["statistics"]["sample_values"] = stats["sample_values"]

                elif stats["primary_type"] in ("int", "float", "decimal"):
                    if "min_value" in stats:
                        property_summary["statistics"].update({
                            "min_value": stats["min_value"],
                            "max_value": stats["max_value"],
                            "avg_value": stats["avg_value"],
                            "median_value": stats["median_value"]
                        })
                    if "sample_values" in stats:
                        property_summary["statistics"]["sample_values"] = stats["sample_values"]

                elif stats["primary_type"] == "bool":
                    if "true_count" in stats:
                        property_summary["statistics"].update({
                            "true_count": stats["true_count"],
                            "false_count": stats["false_count"],
                            "true_percentage": stats["true_percentage"]
                        })

                elif stats["primary_type"] in ("list", "set", "tuple", "array"):
                    if "min_items" in stats:
                        property_summary["statistics"].update({
                            "min_items": stats["min_items"],
                            "max_items": stats["max_items"],
                            "item_types": stats.get("item_types", [])
                        })

                elif stats["primary_type"] == "dict":
                    if "min_keys" in stats:
                        property_summary["statistics"].update({
                            "min_keys": stats["min_keys"],
                            "max_keys": stats["max_keys"],
                            "common_keys": stats.get("common_keys", {})
                        })

                # Generate validation suggestions
                property_summary["validation"] = self._generate_property_validation(prop_name, stats, requirement)

                # Add property to result
                result["properties"][prop_name] = property_summary

                # Add any property-specific validation suggestions to the overall list
                if requirement == "required" and prop_presence < 1.0:
                    result["validation_suggestions"].append(
                        f"Property '{prop_name}' appears to be required but is missing in some instances"
                    )
                if uniqueness == "unique" and "id" in prop_name.lower():
                    result["validation_suggestions"].append(
                        f"Property '{prop_name}' contains unique values and could be used as an identifier"
                    )

        # Generate overall type validation suggestions
        if not result["validation_suggestions"]:
            if is_node_type:
                result["validation_suggestions"].append(
                    f"Node type '{type_name}' has consistent property patterns"
                )
            else:
                result["validation_suggestions"].append(
                    f"Edge type '{type_name}' has consistent property patterns"
                )

        return result

    def extract_relationship_patterns(self):
        """
        Find all valid node-edge-node type patterns in the graph

        This method analyzes the relationship patterns between node types,
        identifying valid combinations and their statistical significance.

        Returns:
            Dict: A structured representation of all relationship patterns

        Raises:
            ValueError: If the ontology hasn't been extracted
        """
        if not self.ontology_extracted:
            self.extract_ontology()

        # Structure to store the relationship pattern analysis
        patterns = {
            "counts": {
                "total_relationships": len(self.relationships),
                "source_types": len(set(src for (src, _, _) in self.relationships.keys())),
                "edge_types": len(set(edge for (_, edge, _) in self.relationships.keys())),
                "target_types": len(set(tgt for (_, _, tgt) in self.relationships.keys()))
            },
            "by_source_type": defaultdict(dict),
            "by_edge_type": defaultdict(dict),
            "by_target_type": defaultdict(dict),
            "full_patterns": [],
            "common_patterns": []
        }

        # Calculate total edges for percentage calculations
        total_edges = sum(self.relationships.values())

        # Process each relationship pattern
        for (src_type, edge_type, tgt_type), count in self.relationships.items():
            # Calculate percentage of total edges
            percentage = (count / total_edges) * 100 if total_edges > 0 else 0

            # Build the pattern details
            pattern_details = {
                "source_type": src_type,
                "edge_type": edge_type,
                "target_type": tgt_type,
                "count": count,
                "percentage": percentage
            }

            # Add to full patterns list
            patterns["full_patterns"].append(pattern_details)

            # Organize by source type
            if src_type not in patterns["by_source_type"]:
                patterns["by_source_type"][src_type] = {
                    "outgoing_edge_types": set(),
                    "target_types": set(),
                    "patterns": []
                }

            patterns["by_source_type"][src_type]["outgoing_edge_types"].add(edge_type)
            patterns["by_source_type"][src_type]["target_types"].add(tgt_type)
            patterns["by_source_type"][src_type]["patterns"].append(pattern_details)

            # Organize by edge type
            if edge_type not in patterns["by_edge_type"]:
                patterns["by_edge_type"][edge_type] = {
                    "source_types": set(),
                    "target_types": set(),
                    "patterns": []
                }

            patterns["by_edge_type"][edge_type]["source_types"].add(src_type)
            patterns["by_edge_type"][edge_type]["target_types"].add(tgt_type)
            patterns["by_edge_type"][edge_type]["patterns"].append(pattern_details)

            # Organize by target type
            if tgt_type not in patterns["by_target_type"]:
                patterns["by_target_type"][tgt_type] = {
                    "incoming_edge_types": set(),
                    "source_types": set(),
                    "patterns": []
                }

            patterns["by_target_type"][tgt_type]["incoming_edge_types"].add(edge_type)
            patterns["by_target_type"][tgt_type]["source_types"].add(src_type)
            patterns["by_target_type"][tgt_type]["patterns"].append(pattern_details)

        # Sort patterns by count (descending)
        patterns["full_patterns"].sort(key=lambda x: x["count"], reverse=True)

        # Identify common patterns (those representing at least 5% of edges)
        patterns["common_patterns"] = [p for p in patterns["full_patterns"] if p["percentage"] >= 5.0]

        # Convert sets to lists for better serialization
        for src_type in patterns["by_source_type"]:
            patterns["by_source_type"][src_type]["outgoing_edge_types"] = list(
                patterns["by_source_type"][src_type]["outgoing_edge_types"])
            patterns["by_source_type"][src_type]["target_types"] = list(
                patterns["by_source_type"][src_type]["target_types"])
            patterns["by_source_type"][src_type]["patterns"].sort(key=lambda x: x["count"], reverse=True)

        for edge_type in patterns["by_edge_type"]:
            patterns["by_edge_type"][edge_type]["source_types"] = list(
                patterns["by_edge_type"][edge_type]["source_types"])
            patterns["by_edge_type"][edge_type]["target_types"] = list(
                patterns["by_edge_type"][edge_type]["target_types"])
            patterns["by_edge_type"][edge_type]["patterns"].sort(key=lambda x: x["count"], reverse=True)

        for tgt_type in patterns["by_target_type"]:
            patterns["by_target_type"][tgt_type]["incoming_edge_types"] = list(
                patterns["by_target_type"][tgt_type]["incoming_edge_types"])
            patterns["by_target_type"][tgt_type]["source_types"] = list(
                patterns["by_target_type"][tgt_type]["source_types"])
            patterns["by_target_type"][tgt_type]["patterns"].sort(key=lambda x: x["count"], reverse=True)

        # Identify potential hierarchical relationships (tree-like structures)
        self._identify_hierarchical_relationships(patterns)

        # Identify potential semantic subgraphs (clusters of related types)
        self._identify_semantic_subgraphs(patterns)

        # Convert defaultdicts to regular dicts for easier serialization
        patterns["by_source_type"] = dict(patterns["by_source_type"])
        patterns["by_edge_type"] = dict(patterns["by_edge_type"])
        patterns["by_target_type"] = dict(patterns["by_target_type"])

        return patterns

    # Additional helper method needed for get_type_property_statistics
    def _generate_property_validation(self, prop_name, stats, requirement):
        """Generate validation rules for a property based on its statistics"""
        validation = {
            "rules": []
        }

        # Add basic presence rule if required
        if requirement == "required":
            validation["rules"].append({
                "type": "required",
                "message": f"'{prop_name}' is required"
            })

        # Add type-specific validation rules
        primary_type = stats["primary_type"]

        if primary_type in ("str", "string"):
            # String validation
            validation["rules"].append({
                "type": "string",
                "message": f"'{prop_name}' must be a string"
            })

            if "min_length" in stats and "max_length" in stats:
                # If lengths are very consistent, add length validation
                if stats["min_length"] == stats["max_length"]:
                    validation["rules"].append({
                        "type": "length",
                        "exact": stats["min_length"],
                        "message": f"'{prop_name}' must be exactly {stats['min_length']} characters"
                    })
                else:
                    validation["rules"].append({
                        "type": "length",
                        "min": stats["min_length"],
                        "max": stats["max_length"],
                        "message": f"'{prop_name}' must be between {stats['min_length']} and {stats['max_length']} characters"
                    })

            # If it looks like an enumeration, suggest allowed values
            if "pattern_type" in stats and stats["pattern_type"] == "enumeration" and "enum_values" in stats:
                validation["rules"].append({
                    "type": "enum",
                    "values": stats["enum_values"],
                    "message": f"'{prop_name}' must be one of the allowed values"
                })

            # If it looks like a specific format, suggest format validation
            elif "pattern_type" in stats:
                if stats["pattern_type"] == "email":
                    validation["rules"].append({
                        "type": "format",
                        "format": "email",
                        "message": f"'{prop_name}' must be a valid email address"
                    })
                elif stats["pattern_type"] == "url":
                    validation["rules"].append({
                        "type": "format",
                        "format": "url",
                        "message": f"'{prop_name}' must be a valid URL"
                    })
                elif stats["pattern_type"] == "date":
                    validation["rules"].append({
                        "type": "format",
                        "format": "date",
                        "message": f"'{prop_name}' must be a valid date in YYYY-MM-DD format"
                    })

        elif primary_type in ("int", "float", "decimal"):
            # Numeric validation
            if primary_type == "int" or (
                    "pattern_type" in stats and stats["pattern_type"] == "integer"):
                validation["rules"].append({
                    "type": "integer",
                    "message": f"'{prop_name}' must be an integer"
                })
            else:
                validation["rules"].append({
                    "type": "number",
                    "message": f"'{prop_name}' must be a number"
                })

            # Add range validation if we have min/max values
            if "min_value" in stats and "max_value" in stats:
                validation["rules"].append({
                    "type": "range",
                    "min": stats["min_value"],
                    "max": stats["max_value"],
                    "message": f"'{prop_name}' must be between {stats['min_value']} and {stats['max_value']}"
                })

        elif primary_type == "bool":
            # Boolean validation
            validation["rules"].append({
                "type": "boolean",
                "message": f"'{prop_name}' must be a boolean"
            })

        elif primary_type in ("list", "set", "tuple", "array"):
            # Array validation
            validation["rules"].append({
                "type": "array",
                "message": f"'{prop_name}' must be an array"
            })

            # Add length validation if we have min/max items
            if "min_items" in stats and "max_items" in stats:
                validation["rules"].append({
                    "type": "array_length",
                    "min": stats["min_items"],
                    "max": stats["max_items"],
                    "message": f"'{prop_name}' must have between {stats['min_items']} and {stats['max_items']} items"
                })

            # Add item type validation if we have item types
            if "item_types" in stats and len(stats["item_types"]) == 1:
                item_type = stats["item_types"][0]
                validation["rules"].append({
                    "type": "array_items",
                    "items_type": item_type,
                    "message": f"All items in '{prop_name}' must be of type {item_type}"
                })

        elif primary_type == "dict":
            # Object validation
            validation["rules"].append({
                "type": "object",
                "message": f"'{prop_name}' must be an object"
            })

            # Add required keys if we have common keys
            if "common_keys" in stats and stats["common_keys"]:
                required_keys = [k for k, count in stats["common_keys"].items()
                                 if count == stats["present_count"]]
                if required_keys:
                    validation["rules"].append({
                        "type": "required_keys",
                        "keys": required_keys,
                        "message": f"'{prop_name}' must have these required keys: {', '.join(required_keys)}"
                    })

        return validation

    # Additional helper methods_application needed for extract_relationship_patterns
    def _identify_hierarchical_relationships(self, patterns):
        """Identify potential hierarchical (tree-like) relationships in the graph"""
        hierarchical_patterns = []

        # Look for edge types that typically connect a parent type to child type
        for edge_type, edge_info in patterns["by_edge_type"].items():
            # Skip if this edge type is used between many different node types
            if len(edge_info["source_types"]) > 3 or len(edge_info["target_types"]) > 3:
                continue

            # Check if this edge type creates hierarchical patterns
            hierarchy_indicators = [
                "parent", "child", "contains", "has", "owns", "belongs_to",
                "part_of", "member_of", "subclass", "is_a"
            ]

            is_hierarchical = False

            # Check if the edge type name suggests hierarchy
            for indicator in hierarchy_indicators:
                if indicator in edge_type.lower():
                    is_hierarchical = True
                    break

            # Check for self-referential patterns (same source and target type)
            for pattern in edge_info["patterns"]:
                if pattern["source_type"] == pattern["target_type"]:
                    is_hierarchical = True

                    # Check for cycles (children pointing to parents)
                    for (src, edge, tgt), count in self.relationships.items():
                        if edge == edge_type and src == pattern["target_type"] and tgt == pattern["source_type"]:
                            # Found a potential cycle, likely hierarchical
                            is_hierarchical = True

            if is_hierarchical:
                hierarchical_patterns.append({
                    "edge_type": edge_type,
                    "source_types": edge_info["source_types"],
                    "target_types": edge_info["target_types"],
                    "count": sum(p["count"] for p in edge_info["patterns"]),
                    "likely_direction": "parent_to_child"  # Could be refined with more analysis
                })

        if hierarchical_patterns:
            patterns["hierarchical_relationships"] = hierarchical_patterns

    def _identify_semantic_subgraphs(self, patterns):
        """Identify potential semantic subgraphs (clusters of related types)"""
        # Create a graph of node types connected by edge types
        type_connections = defaultdict(set)

        for (src, edge, tgt), count in self.relationships.items():
            type_connections[src].add(tgt)
            type_connections[tgt].add(src)

        # Identify connected components (basic clustering)
        subgraphs = []
        visited = set()

        for node_type in type_connections.keys():
            if node_type in visited:
                continue

            # Start a new component
            component = set()
            queue = [node_type]

            while queue:
                current = queue.pop(0)
                if current in visited:
                    continue

                visited.add(current)
                component.add(current)

                # Add connected types
                for connected in type_connections[current]:
                    if connected not in visited:
                        queue.append(connected)

            # Add this component if it has multiple node types
            if len(component) > 1:
                # Count connections within this component
                internal_connections = 0
                for (src, edge, tgt), count in self.relationships.items():
                    if src in component and tgt in component:
                        internal_connections += count

                subgraphs.append({
                    "node_types": list(component),
                    "size": len(component),
                    "internal_connections": internal_connections
                })

        if subgraphs:
            # Sort by size
            subgraphs.sort(key=lambda x: x["size"], reverse=True)
            patterns["semantic_subgraphs"] = subgraphs

    def get_valid_relationships(self, source_type=None, edge_type=None, target_type=None):
        """
        Get relationship patterns matching specified type criteria

        This method finds all relationship patterns that match the provided criteria.
        It can filter by source type, edge type, target type, or any combination.

        Args:
            source_type: Optional filter for the source node type
            edge_type: Optional filter for the edge type
            target_type: Optional filter for the target node type

        Returns:
            Dict: Matching relationship patterns with statistics

        Raises:
            ValueError: If the ontology hasn't been extracted
        """
        if not self.ontology_extracted:
            self.extract_ontology()

        # Structure to store results
        results = {
            "matches": [],
            "total_matches": 0,
            "total_instances": 0,
            "query": {
                "source_type": source_type,
                "edge_type": edge_type,
                "target_type": target_type
            }
        }

        # Filter relationships based on criteria
        for (src, edge, tgt), count in self.relationships.items():
            # Apply filters
            if source_type is not None and src != source_type:
                continue
            if edge_type is not None and edge != edge_type:
                continue
            if target_type is not None and tgt != target_type:
                continue

            # This relationship matches the criteria
            results["matches"].append({
                "source_type": src,
                "edge_type": edge,
                "target_type": tgt,
                "count": count,
                "examples": self._get_relationship_examples(src, edge, tgt, limit=3)
            })

            results["total_instances"] += count

        # Sort matches by count (descending)
        results["matches"].sort(key=lambda x: x["count"], reverse=True)
        results["total_matches"] = len(results["matches"])

        # Add analysis summary
        if results["total_matches"] > 0:
            # Determine if this is a complete or partial query
            if source_type and edge_type and target_type:
                # Complete pattern query
                if results["total_matches"] > 0:
                    results[
                        "summary"] = f"Found {results['total_instances']} instances of the {source_type}→{edge_type}→{target_type} relationship."
                else:
                    results[
                        "summary"] = f"The {source_type}→{edge_type}→{target_type} relationship does not exist in the graph."
            else:
                # Partial query
                filter_parts = []
                if source_type:
                    filter_parts.append(f"source type '{source_type}'")
                if edge_type:
                    filter_parts.append(f"edge type '{edge_type}'")
                if target_type:
                    filter_parts.append(f"target type '{target_type}'")

                filter_desc = " and ".join(filter_parts)
                results[
                    "summary"] = f"Found {results['total_matches']} distinct relationship patterns with {filter_desc}, involving {results['total_instances']} total edges."
        else:
            results["summary"] = "No relationships match the specified criteria."

        return results

    def analyze_relationship_cardinality(self, source_type, edge_type, target_type):
        """
        Analyze the cardinality of a relationship (1-1, 1-many, etc.)

        This method provides detailed cardinality analysis for a specific
        relationship pattern, including distribution of connections and
        specific examples of different cardinality scenarios.

        Args:
            source_type: The source node type
            edge_type: The edge type
            target_type: The target node type

        Returns:
            Dict: Detailed cardinality analysis

        Raises:
            ValueError: If the ontology hasn't been extracted or the relationship doesn't exist
        """
        if not self.ontology_extracted:
            self.extract_ontology()

        # Verify the relationship exists
        relationship_key = (source_type, edge_type, target_type)
        if relationship_key not in self.relationships:
            raise ValueError(f"Relationship {source_type}→{edge_type}→{target_type} not found in the ontology")

        # Get count of this relationship
        relationship_count = self.relationships[relationship_key]

        # Find all edges of this pattern
        pattern_edges = []
        for (src_id, tgt_id), edge_data in self.graph_manager.edge_data.items():
            src_type = self.graph_manager.node_data.get(src_id, {}).get("type", "Unknown")
            edge_type_value = edge_data.get("edge_type", "Unknown")
            tgt_type = self.graph_manager.node_data.get(tgt_id, {}).get("type", "Unknown")

            if (src_type, edge_type_value, tgt_type) == relationship_key:
                pattern_edges.append((src_id, tgt_id))

        # Track cardinality stats
        source_targets = defaultdict(set)  # source -> set of targets
        target_sources = defaultdict(set)  # target -> set of sources

        # Build the mappings
        for source, target in pattern_edges:
            source_targets[source].add(target)
            target_sources[target].add(source)

        # Analyze outgoing cardinality (source -> targets)
        out_counts = [len(targets) for targets in source_targets.values()]

        # Analyze incoming cardinality (target <- sources)
        in_counts = [len(sources) for sources in target_sources.values()]

        # Determine cardinality characteristics
        out_max = max(out_counts) if out_counts else 0
        out_min = min(out_counts) if out_counts else 0
        out_median = statistics.median(out_counts) if out_counts else 0
        out_mean = sum(out_counts) / len(out_counts) if out_counts else 0

        in_max = max(in_counts) if in_counts else 0
        in_min = min(in_counts) if in_counts else 0
        in_median = statistics.median(in_counts) if in_counts else 0
        in_mean = sum(in_counts) / len(in_counts) if in_counts else 0

        # Determine overall cardinality pattern
        if out_max <= 1 and in_max <= 1:
            cardinality_type = "one-to-one"
            description = f"Each {source_type} connects to at most one {target_type}, and each {target_type} connects to at most one {source_type}"
        elif out_max <= 1 and in_max > 1:
            cardinality_type = "one-to-many"
            description = f"Each {source_type} connects to at most one {target_type}, but a {target_type} may connect to multiple {source_type}s"
        elif out_max > 1 and in_max <= 1:
            cardinality_type = "many-to-one"
            description = f"A {source_type} may connect to multiple {target_type}s, but each {target_type} connects to at most one {source_type}"
        else:
            cardinality_type = "many-to-many"
            description = f"A {source_type} may connect to multiple {target_type}s, and a {target_type} may connect to multiple {source_type}s"

        # Create result structure
        result = {
            "relationship": {
                "source_type": source_type,
                "edge_type": edge_type,
                "target_type": target_type,
                "total_edges": relationship_count
            },
            "cardinality": {
                "type": cardinality_type,
                "description": description,
                "outgoing": {  # source → targets
                    "min": out_min,
                    "max": out_max,
                    "median": out_median,
                    "mean": out_mean,
                    "distribution": self._generate_histogram(out_counts, 5),
                    "distinct_sources": len(source_targets),
                    "sources_with_multiple_targets": sum(1 for count in out_counts if count > 1)
                },
                "incoming": {  # target ← sources
                    "min": in_min,
                    "max": in_max,
                    "median": in_median,
                    "mean": in_mean,
                    "distribution": self._generate_histogram(in_counts, 5),
                    "distinct_targets": len(target_sources),
                    "targets_with_multiple_sources": sum(1 for count in in_counts if count > 1)
                }
            },
            "examples": {}
        }

        # Add examples for different cardinality scenarios
        if cardinality_type == "one-to-one":
            # Find an example one-to-one pair
            for src, targets in source_targets.items():
                if len(targets) == 1:
                    tgt = next(iter(targets))
                    if len(target_sources[tgt]) == 1:
                        result["examples"]["one_to_one"] = {
                            "source": src,
                            "target": tgt,
                            "source_value": self.graph_manager.node_data.get(src, {}).get("value", src),
                            "target_value": self.graph_manager.node_data.get(tgt, {}).get("value", tgt)
                        }
                        break

        elif cardinality_type == "one-to-many":
            # Find an example where multiple sources point to the same target
            for tgt, sources in sorted(target_sources.items(), key=lambda x: len(x[1]), reverse=True):
                if len(sources) > 1:
                    result["examples"]["many_sources_one_target"] = {
                        "target": tgt,
                        "target_value": self.graph_manager.node_data.get(tgt, {}).get("value", tgt),
                        "sources": list(sources)[:5],
                        "total_sources": len(sources)
                    }
                    break

        elif cardinality_type == "many-to-one":
            # Find an example where one source points to multiple targets
            for src, targets in sorted(source_targets.items(), key=lambda x: len(x[1]), reverse=True):
                if len(targets) > 1:
                    result["examples"]["one_source_many_targets"] = {
                        "source": src,
                        "source_value": self.graph_manager.node_data.get(src, {}).get("value", src),
                        "targets": list(targets)[:5],
                        "total_targets": len(targets)
                    }
                    break

        elif cardinality_type == "many-to-many":
            # Find both types of examples

            # Example where multiple sources point to the same target
            for tgt, sources in sorted(target_sources.items(), key=lambda x: len(x[1]), reverse=True):
                if len(sources) > 1:
                    result["examples"]["many_sources_one_target"] = {
                        "target": tgt,
                        "target_value": self.graph_manager.node_data.get(tgt, {}).get("value", tgt),
                        "sources": list(sources)[:5],
                        "total_sources": len(sources)
                    }
                    break

            # Example where one source points to multiple targets
            for src, targets in sorted(source_targets.items(), key=lambda x: len(x[1]), reverse=True):
                if len(targets) > 1:
                    result["examples"]["one_source_many_targets"] = {
                        "source": src,
                        "source_value": self.graph_manager.node_data.get(src, {}).get("value", src),
                        "targets": list(targets)[:5],
                        "total_targets": len(targets)
                    }
                    break

        # Calculate constraint metrics (how restricted the relationship is)
        source_type_count = self.node_types.get(source_type, {}).get("count", 0)
        target_type_count = self.node_types.get(target_type, {}).get("count", 0)

        if source_type_count > 0 and target_type_count > 0:
            result["cardinality"]["constraint_metrics"] = {
                "source_coverage": len(source_targets) / source_type_count * 100,
                # % of all source-type nodes participating
                "target_coverage": len(target_sources) / target_type_count * 100,
                # % of all target-type nodes participating
                "edge_density": relationship_count / (source_type_count * target_type_count) * 100
                # % of possible edges that exist
            }

        # Provide recommendations based on cardinality
        result["recommendations"] = []

        if cardinality_type == "one-to-one":
            result["recommendations"].append(
                f"This is a one-to-one relationship. Consider using validation rules to ensure that each {source_type} connects to exactly one {target_type}."
            )
        elif cardinality_type == "one-to-many":
            result["recommendations"].append(
                f"This is a one-to-many relationship from {target_type} to {source_type}. Consider using validation rules to ensure that each {source_type} connects to at most one {target_type}."
            )
        elif cardinality_type == "many-to-one":
            result["recommendations"].append(
                f"This is a many-to-one relationship from {source_type} to {target_type}. Consider using validation rules to ensure that each {target_type} connects to at most one {source_type}."
            )
        else:  # many-to-many
            result["recommendations"].append(
                f"This is a many-to-many relationship. No additional cardinality constraints are needed."
            )

        return result

    # Additional helper method needed for get_valid_relationships
    def _get_relationship_examples(self, source_type, edge_type, target_type, limit=3):
        """Get example edges for a specific relationship pattern"""
        examples = []

        # Find matching edges
        for (src_id, tgt_id), edge_data in self.graph_manager.edge_data.items():
            src_type = self.graph_manager.node_data.get(src_id, {}).get("type", "Unknown")
            edge_type_value = edge_data.get("edge_type", "Unknown")
            tgt_type = self.graph_manager.node_data.get(tgt_id, {}).get("type", "Unknown")

            if (src_type, edge_type_value, tgt_type) == (source_type, edge_type, target_type):
                # Get display values for source and target nodes
                src_value = self.graph_manager.node_data.get(src_id, {}).get("value", src_id)
                tgt_value = self.graph_manager.node_data.get(tgt_id, {}).get("value", tgt_id)

                examples.append({
                    "source_id": src_id,
                    "source_value": src_value,
                    "target_id": tgt_id,
                    "target_value": tgt_value,
                    "edge_properties": {k: v for k, v in edge_data.items() if k != "edge_type"}
                })

                if len(examples) >= limit:
                    break

        return examples

    def get_relationship_statistics(self):
        """
        Generate statistics about all relationship patterns

        This method analyzes all relationship patterns in the graph to provide
        a comprehensive overview of the graph's connectivity structure.

        Returns:
            Dict: Statistical overview of relationships in the graph

        Raises:
            ValueError: If the ontology hasn't been extracted
        """
        if not self.ontology_extracted:
            self.extract_ontology()

        # Calculate basic counts
        total_node_types = len(self.node_types)
        total_edge_types = len(self.edge_types)
        total_relationship_patterns = len(self.relationships)
        total_edges = sum(self.relationships.values())

        # Structure for stats
        stats = {
            "counts": {
                "node_types": total_node_types,
                "edge_types": total_edge_types,
                "relationship_patterns": total_relationship_patterns,
                "total_edges": total_edges,
                "total_nodes": sum(type_info["count"] for type_info in self.node_types.values())
            },
            "connectivity": {
                "avg_patterns_per_edge_type": total_relationship_patterns / total_edge_types if total_edge_types > 0 else 0,
                "avg_edges_per_pattern": total_edges / total_relationship_patterns if total_relationship_patterns > 0 else 0,
                "avg_edges_per_node": total_edges / stats["counts"]["total_nodes"] if stats["counts"][
                                                                                          "total_nodes"] > 0 else 0
            },
            "top_patterns": [],
            "edge_type_diversity": {},
            "node_type_connectivity": {},
            "cardinality_distribution": {
                "one_to_one": 0,
                "one_to_many": 0,
                "many_to_one": 0,
                "many_to_many": 0
            }
        }

        # Analyze each relationship pattern
        for (src_type, edge_type, tgt_type), count in sorted(
                self.relationships.items(), key=lambda x: x[1], reverse=True):

            # Add to top patterns if in the top 10
            if len(stats["top_patterns"]) < 10:
                stats["top_patterns"].append({
                    "pattern": f"{src_type}→{edge_type}→{tgt_type}",
                    "count": count,
                    "percentage": (count / total_edges) * 100 if total_edges > 0 else 0
                })

            # Track edge type diversity
            if edge_type not in stats["edge_type_diversity"]:
                stats["edge_type_diversity"][edge_type] = {
                    "count": 0,
                    "patterns": 0,
                    "source_types": set(),
                    "target_types": set()
                }

            stats["edge_type_diversity"][edge_type]["count"] += count
            stats["edge_type_diversity"][edge_type]["patterns"] += 1
            stats["edge_type_diversity"][edge_type]["source_types"].add(src_type)
            stats["edge_type_diversity"][edge_type]["target_types"].add(tgt_type)

            # Track node type connectivity
            for node_type, role in [(src_type, "source"), (tgt_type, "target")]:
                if node_type not in stats["node_type_connectivity"]:
                    stats["node_type_connectivity"][node_type] = {
                        "as_source": {
                            "edge_count": 0,
                            "edge_types": set(),
                            "target_types": set()
                        },
                        "as_target": {
                            "edge_count": 0,
                            "edge_types": set(),
                            "source_types": set()
                        }
                    }

                if role == "source":
                    stats["node_type_connectivity"][node_type]["as_source"]["edge_count"] += count
                    stats["node_type_connectivity"][node_type]["as_source"]["edge_types"].add(edge_type)
                    stats["node_type_connectivity"][node_type]["as_source"]["target_types"].add(tgt_type)
                else:  # target
                    stats["node_type_connectivity"][node_type]["as_target"]["edge_count"] += count
                    stats["node_type_connectivity"][node_type]["as_target"]["edge_types"].add(edge_type)
                    stats["node_type_connectivity"][node_type]["as_target"]["source_types"].add(src_type)

            # Analyze cardinality
            try:
                cardinality = self.analyze_relationship_cardinality(src_type, edge_type, tgt_type)
                cardinality_type = cardinality["cardinality"]["type"]
                stats["cardinality_distribution"][cardinality_type] += 1
            except Exception:
                # Skip if there's an error analyzing cardinality
                pass

        # Convert sets to lists for serialization
        for edge_type in stats["edge_type_diversity"]:
            stats["edge_type_diversity"][edge_type]["source_types"] = list(
                stats["edge_type_diversity"][edge_type]["source_types"])
            stats["edge_type_diversity"][edge_type]["target_types"] = list(
                stats["edge_type_diversity"][edge_type]["target_types"])

            # Add diversity metrics
            source_count = len(stats["edge_type_diversity"][edge_type]["source_types"])
            target_count = len(stats["edge_type_diversity"][edge_type]["target_types"])
            pattern_count = stats["edge_type_diversity"][edge_type]["patterns"]

            stats["edge_type_diversity"][edge_type]["diversity_index"] = pattern_count / (
                        source_count * target_count) if source_count * target_count > 0 else 0

        for node_type in stats["node_type_connectivity"]:
            for role in ["as_source", "as_target"]:
                for set_key in ["edge_types", "source_types", "target_types"]:
                    if set_key in stats["node_type_connectivity"][node_type][role]:
                        stats["node_type_connectivity"][node_type][role][set_key] = list(
                            stats["node_type_connectivity"][node_type][role][set_key]
                        )

        # Calculate node type connectivity metrics
        if stats["counts"]["node_types"] > 1:
            node_type_connections = 0
            for node_type, conn_data in stats["node_type_connectivity"].items():
                # Count distinct node types this one connects to (either as source or target)
                connected_types = set(conn_data["as_source"]["target_types"])
                connected_types.update(conn_data["as_target"]["source_types"])
                node_type_connections += len(connected_types)

            # Average number of other node types each node type connects to
            stats["connectivity"]["avg_connected_node_types"] = node_type_connections / stats["counts"]["node_types"]

            # Graph density at the type level (percentage of possible connections that exist)
            max_type_connections = stats["counts"]["node_types"] * (stats["counts"]["node_types"] - 1)
            if max_type_connections > 0:
                stats["connectivity"]["type_level_density"] = (total_relationship_patterns / max_type_connections) * 100

        # Identify most central node types (highest connectivity)
        node_type_centrality = []
        for node_type, conn_data in stats["node_type_connectivity"].items():
            total_connections = (
                    len(conn_data["as_source"]["target_types"]) +
                    len(conn_data["as_source"]["edge_types"]) +
                    len(conn_data["as_target"]["source_types"]) +
                    len(conn_data["as_target"]["edge_types"])
            )
            total_edges = conn_data["as_source"]["edge_count"] + conn_data["as_target"]["edge_count"]
            node_type_centrality.append({
                "node_type": node_type,
                "connection_score": total_connections,
                "edge_count": total_edges
            })

        # Sort by connection score and include top 5
        node_type_centrality.sort(key=lambda x: x["connection_score"], reverse=True)
        stats["most_connected_types"] = node_type_centrality[:5]

        # Generate insights based on the statistics
        stats["insights"] = []

        # Check for isolated node types
        isolated_types = []
        for node_type, type_info in self.node_types.items():
            if (node_type not in stats["node_type_connectivity"] or
                    (stats["node_type_connectivity"][node_type]["as_source"]["edge_count"] == 0 and
                     stats["node_type_connectivity"][node_type]["as_target"]["edge_count"] == 0)):
                isolated_types.append(node_type)

        if isolated_types:
            stats["insights"].append({
                "type": "isolated_node_types",
                "description": f"Found {len(isolated_types)} node types that are not connected to any other types: {', '.join(isolated_types)}",
                "suggestion": "Consider connecting these isolated types to the main graph or removing them if they're not needed."
            })

        # Check for hub node types (connect to many others)
        for node_type_data in node_type_centrality[:3]:  # Top 3 most connected
            node_type = node_type_data["node_type"]
            conn_data = stats["node_type_connectivity"][node_type]
            source_types = len(conn_data["as_target"]["source_types"])
            target_types = len(conn_data["as_source"]["target_types"])

            if source_types > 3 or target_types > 3:
                stats["insights"].append({
                    "type": "hub_node_type",
                    "description": f"'{node_type}' is a hub type connecting to {source_types} source types and {target_types} target types",
                    "suggestion": "Hub types often represent central concepts in your domain and should have robust validation rules."
                })

        # Look for dominant edge types
        sorted_edge_types = sorted(
            [(edge_type, data["count"]) for edge_type, data in stats["edge_type_diversity"].items()],
            key=lambda x: x[1], reverse=True
        )

        if sorted_edge_types and total_edges > 0:
            top_edge_type, top_count = sorted_edge_types[0]
            percentage = (top_count / total_edges) * 100

            if percentage > 50:
                stats["insights"].append({
                    "type": "dominant_edge_type",
                    "description": f"'{top_edge_type}' is the dominant edge type, representing {percentage:.1f}% of all edges",
                    "suggestion": "Consider subdividing this edge type into more specific categories if appropriate for your domain."
                })

        # Comment on cardinality distribution
        if stats["cardinality_distribution"]["one_to_one"] > 0:
            one_to_one_pct = (stats["cardinality_distribution"]["one_to_one"] / total_relationship_patterns) * 100
            if one_to_one_pct > 50:
                stats["insights"].append({
                    "type": "cardinality_pattern",
                    "description": f"{one_to_one_pct:.1f}% of relationship patterns are one-to-one",
                    "suggestion": "Your graph has many one-to-one relationships, suggesting a structured, normalized data model."
                })

        if stats["cardinality_distribution"]["many_to_many"] > 0:
            many_to_many_pct = (stats["cardinality_distribution"]["many_to_many"] / total_relationship_patterns) * 100
            if many_to_many_pct > 50:
                stats["insights"].append({
                    "type": "cardinality_pattern",
                    "description": f"{many_to_many_pct:.1f}% of relationship patterns are many-to-many",
                    "suggestion": "Your graph has many many-to-many relationships, suggesting a highly interconnected data model."
                })

        return stats

    def discover_property_patterns(self, type_name, property_name):
        """
        Discover patterns in property values (ranges, formats, etc.)

        This method performs an in-depth analysis of a specific property,
        identifying patterns, formats, and potential validation rules.

        Args:
            type_name: The name of the node or edge type
            property_name: The name of the property to analyze

        Returns:
            Dict: Detailed pattern analysis of the property

        Raises:
            ValueError: If the ontology hasn't been extracted or the property doesn't exist
        """
        if not self.ontology_extracted:
            self.extract_ontology()

        # Check if this is a node type or edge type
        is_node_type = type_name in self.node_types
        is_edge_type = type_name in self.edge_types

        if not is_node_type and not is_edge_type:
            raise ValueError(f"Type '{type_name}' not found in the ontology")

        # Determine the appropriate stats collection
        if is_node_type:
            type_info = self.node_types[type_name]
            property_stats = self.node_property_stats.get(type_name, {}).get(property_name)
            elements = {node_id: self.graph_manager.node_data.get(node_id, {})
                        for node_id in type_info["nodes"]}
        else:  # Edge type
            type_info = self.edge_types[type_name]
            property_stats = self.edge_property_stats.get(type_name, {}).get(property_name)
            elements = {edge_key: self.graph_manager.edge_data.get(edge_key, {})
                        for edge_key in type_info["edges"]}

        # Check if the property exists
        if property_name not in type_info["properties"]:
            raise ValueError(f"Property '{property_name}' not found in type '{type_name}'")

        if not property_stats:
            # Property exists but no stats available, re-analyze
            values = []
            for element_data in elements.values():
                if property_name in element_data:
                    values.append(element_data[property_name])

            # Skip if still no values found
            if not values:
                raise ValueError(f"No values found for property '{property_name}' in type '{type_name}'")

            # Analyze the property
            property_stats = self._analyze_property_values(values)

        # Basic result structure
        result = {
            "type_name": type_name,
            "is_node_type": is_node_type,
            "property_name": property_name,
            "presence": {
                "count": property_stats["present_count"],
                "percentage": property_stats["present_count"] / len(elements) * 100 if elements else 0
            },
            "type_info": {
                "primary_type": property_stats["primary_type"],
                "all_types": property_stats["value_types"],
            },
            "unique_values": {
                "count": property_stats["unique_count"],
                "percentage": property_stats["unique_count"] / property_stats["present_count"] * 100
                if property_stats["present_count"] > 0 else 0
            },
            "pattern_details": {},
            "value_samples": self._get_diverse_samples(elements, property_name, 10),
            "value_distribution": {},
            "advanced_patterns": {},
            "validation_suggestions": []
        }

        # Generate basic validation suggestions
        if result["presence"]["percentage"] == 100:
            result["validation_suggestions"].append({
                "type": "required",
                "description": f"'{property_name}' should be required for all {type_name} instances"
            })
        elif result["presence"]["percentage"] > 90:
            result["validation_suggestions"].append({
                "type": "recommended",
                "description": f"'{property_name}' should be present in most {type_name} instances"
            })

        # Type-specific analysis
        primary_type = property_stats["primary_type"]

        if primary_type in ("str", "string"):
            # String-specific analysis
            if "min_length" in property_stats and "max_length" in property_stats:
                result["pattern_details"]["length"] = {
                    "min": property_stats["min_length"],
                    "max": property_stats["max_length"],
                    "avg": property_stats["avg_length"]
                }

                # Generate length validation suggestions
                if property_stats["min_length"] == property_stats["max_length"]:
                    result["validation_suggestions"].append({
                        "type": "exact_length",
                        "description": f"All values have exactly {property_stats['min_length']} characters"
                    })
                else:
                    result["validation_suggestions"].append({
                        "type": "length_range",
                        "description": f"Values should be between {property_stats['min_length']} and {property_stats['max_length']} characters"
                    })

            # Analyze value patterns
            pattern_type = property_stats.get("pattern_type", "string")
            result["pattern_details"]["pattern_type"] = pattern_type

            if pattern_type == "enumeration" and "enum_values" in property_stats:
                result["pattern_details"]["allowed_values"] = property_stats["enum_values"]
                result["validation_suggestions"].append({
                    "type": "enumeration",
                    "description": f"Values should be one of: {', '.join(map(str, property_stats['enum_values']))}"
                })
            elif pattern_type == "email":
                result["validation_suggestions"].append({
                    "type": "format",
                    "description": "Values should be valid email addresses"
                })
            elif pattern_type == "url":
                result["validation_suggestions"].append({
                    "type": "format",
                    "description": "Values should be valid URLs"
                })
            elif pattern_type == "date":
                result["validation_suggestions"].append({
                    "type": "format",
                    "description": "Values should be valid dates in YYYY-MM-DD format"
                })
            else:
                # Perform more advanced string pattern analysis
                self._analyze_string_patterns(result, elements, property_name)

        elif primary_type in ("int", "float", "decimal"):
            # Numeric-specific analysis
            if "min_value" in property_stats and "max_value" in property_stats:
                result["pattern_details"]["range"] = {
                    "min": property_stats["min_value"],
                    "max": property_stats["max_value"],
                    "avg": property_stats["avg_value"],
                    "median": property_stats["median_value"]
                }

                # Generate range validation suggestions
                result["validation_suggestions"].append({
                    "type": "range",
                    "description": f"Values should be between {property_stats['min_value']} and {property_stats['max_value']}"
                })

                # Check if values are likely to be identifiers or sequential
                if primary_type == "int" or property_stats.get("pattern_type") == "integer":
                    values = [element_data.get(property_name) for element_data in elements.values()
                              if
                              property_name in element_data and isinstance(element_data[property_name], (int, float))]

                    if values:
                        values.sort()
                        is_sequential = True
                        for i in range(1, len(values)):
                            if values[i] - values[i - 1] != 1:
                                is_sequential = False
                                break

                        if is_sequential:
                            result["advanced_patterns"]["is_sequential"] = True
                            result["validation_suggestions"].append({
                                "type": "sequential",
                                "description": "Values appear to be sequential integers, possibly auto-incremented IDs"
                            })

            # Add histogram for distribution
            values = [element_data.get(property_name) for element_data in elements.values()
                      if property_name in element_data and isinstance(element_data[property_name], (int, float))]

            if values:
                result["value_distribution"]["histogram"] = self._generate_histogram(values, 10)

        elif primary_type == "bool":
            # Boolean-specific analysis
            if "true_count" in property_stats and "false_count" in property_stats:
                result["pattern_details"]["distribution"] = {
                    "true_count": property_stats["true_count"],
                    "false_count": property_stats["false_count"],
                    "true_percentage": property_stats["true_percentage"]
                }

                # Check if values are heavily skewed
                if property_stats["true_percentage"] > 95:
                    result["validation_suggestions"].append({
                        "type": "default_value",
                        "description": "Values are almost always true, consider making true the default"
                    })
                elif property_stats["true_percentage"] < 5:
                    result["validation_suggestions"].append({
                        "type": "default_value",
                        "description": "Values are almost always false, consider making false the default"
                    })

        elif primary_type in ("list", "set", "tuple", "array"):
            # Collection-specific analysis
            if "min_items" in property_stats and "max_items" in property_stats:
                result["pattern_details"]["size"] = {
                    "min_items": property_stats["min_items"],
                    "max_items": property_stats["max_items"]
                }

                # Generate size validation suggestions
                if property_stats["min_items"] == property_stats["max_items"]:
                    result["validation_suggestions"].append({
                        "type": "exact_size",
                        "description": f"Collections should have exactly {property_stats['min_items']} items"
                    })
                else:
                    result["validation_suggestions"].append({
                        "type": "size_range",
                        "description": f"Collections should have between {property_stats['min_items']} and {property_stats['max_items']} items"
                    })

            # Analyze item types
            if "item_types" in property_stats:
                result["pattern_details"]["item_types"] = property_stats["item_types"]

                if len(property_stats["item_types"]) == 1:
                    item_type = property_stats["item_types"][0]
                    result["validation_suggestions"].append({
                        "type": "homogeneous",
                        "description": f"All items in the collection should be of type {item_type}"
                    })
                else:
                    result["validation_suggestions"].append({
                        "type": "heterogeneous",
                        "description": f"Collections contain mixed types: {', '.join(property_stats['item_types'])}"
                    })

        elif primary_type == "dict":
            # Dictionary-specific analysis
            if "min_keys" in property_stats and "max_keys" in property_stats:
                result["pattern_details"]["keys"] = {
                    "min_keys": property_stats["min_keys"],
                    "max_keys": property_stats["max_keys"]
                }

            # Analyze common keys
            if "common_keys" in property_stats:
                result["pattern_details"]["common_keys"] = property_stats["common_keys"]

                # Extract keys that appear in all or most dictionaries
                always_present = []
                usually_present = []

                for key, count in property_stats["common_keys"].items():
                    if count == property_stats["present_count"]:
                        always_present.append(key)
                    elif count >= property_stats["present_count"] * 0.8:
                        usually_present.append(key)

                if always_present:
                    result["validation_suggestions"].append({
                        "type": "required_keys",
                        "description": f"Objects should always have these keys: {', '.join(always_present)}"
                    })

                if usually_present:
                    result["validation_suggestions"].append({
                        "type": "recommended_keys",
                        "description": f"Objects should usually have these keys: {', '.join(usually_present)}"
                    })

        # Look for correlations with other properties
        if is_node_type and len(type_info["properties"]) > 1:
            correlated_properties = self._find_correlated_properties(
                type_name, property_name, elements, result["pattern_details"].get("pattern_type", "unknown")
            )

            if correlated_properties:
                result["advanced_patterns"]["correlated_properties"] = correlated_properties

                # Add validation suggestions based on correlations
                for corr in correlated_properties[:2]:  # Top 2 correlations
                    if corr["correlation_type"] == "presence":
                        result["validation_suggestions"].append({
                            "type": "dependency",
                            "description": f"When '{property_name}' is present, '{corr['property']}' should also be present"
                        })
                    elif corr["correlation_type"] == "value":
                        result["validation_suggestions"].append({
                            "type": "co_constraint",
                            "description": f"The value of '{property_name}' appears to be related to the value of '{corr['property']}'"
                        })

        # Generate regex pattern if appropriate
        if primary_type in ("str", "string") and result["unique_values"]["count"] > 3:
            # Don't generate regex for enumeration types or standard formats
            pattern_type = result["pattern_details"].get("pattern_type", "unknown")
            if pattern_type not in ["enumeration", "email", "url", "date"]:
                regex_pattern = self._generate_regex_pattern(elements, property_name)
                if regex_pattern:
                    result["advanced_patterns"]["regex"] = regex_pattern
                    result["validation_suggestions"].append({
                        "type": "regex",
                        "description": f"Values should match pattern: {regex_pattern}"
                    })

        return result

    # Additional helper methods_application for discover_property_patterns

    def _get_diverse_samples(self, elements, property_name, limit=10):
        """Get a diverse sample of property values"""
        samples = []
        seen_values = set()

        # First collect all unique values
        all_values = []
        for element_data in elements.values():
            if property_name in element_data:
                value = element_data[property_name]
                try:
                    # Convert to string representation for set comparison
                    value_str = str(value)
                    if value_str not in seen_values:
                        seen_values.add(value_str)
                        all_values.append(value)
                except:
                    # Skip values that can't be converted to strings
                    pass

        # If we have fewer unique values than limit, return all of them
        if len(all_values) <= limit:
            return all_values

        # Otherwise, select a diverse subset
        # Start with the first and last values (if they can be sorted)
        try:
            all_values.sort()
            samples.append(all_values[0])
            samples.append(all_values[-1])
        except:
            # If they can't be sorted, just take the first two
            samples.append(all_values[0])
            if len(all_values) > 1:
                samples.append(all_values[1])

        # Add samples at regular intervals
        step = len(all_values) / (limit - len(samples))
        i = step
        while len(samples) < limit and i < len(all_values):
            index = int(i)
            if all_values[index] not in samples:
                samples.append(all_values[index])
            i += step

        # If we still need more samples, add randomly
        remaining = limit - len(samples)
        if remaining > 0:
            import random
            random_samples = random.sample([v for v in all_values if v not in samples],
                                           min(remaining, len(all_values) - len(samples)))
            samples.extend(random_samples)

        return samples

    def _analyze_string_patterns(self, result, elements, property_name):
        """Analyze patterns in string values"""
        # Collect string values
        string_values = [str(element_data.get(property_name)) for element_data in elements.values()
                         if property_name in element_data and
                         isinstance(element_data[property_name], (str, int, float, bool))]

        if not string_values:
            return

        # Check for common prefixes and suffixes
        prefix_candidates = {}
        suffix_candidates = {}

        for value in string_values:
            # Check prefixes up to half the string length
            max_prefix_len = min(len(value) // 2 + 1, 10)
            for i in range(1, max_prefix_len):
                prefix = value[:i]
                prefix_candidates[prefix] = prefix_candidates.get(prefix, 0) + 1

            # Check suffixes up to half the string length
            max_suffix_len = min(len(value) // 2 + 1, 10)
            for i in range(1, max_suffix_len):
                suffix = value[-i:]
                suffix_candidates[suffix] = suffix_candidates.get(suffix, 0) + 1

        # Find the most common prefix and suffix
        common_prefix = None
        common_prefix_count = 0
        for prefix, count in prefix_candidates.items():
            if count > common_prefix_count and count >= len(string_values) * 0.8:
                common_prefix = prefix
                common_prefix_count = count

        common_suffix = None
        common_suffix_count = 0
        for suffix, count in suffix_candidates.items():
            if count > common_suffix_count and count >= len(string_values) * 0.8:
                common_suffix = suffix
                common_suffix_count = count

        # Add to result if found
        if common_prefix:
            result["advanced_patterns"]["common_prefix"] = {
                "prefix": common_prefix,
                "count": common_prefix_count,
                "percentage": (common_prefix_count / len(string_values)) * 100
            }
            result["validation_suggestions"].append({
                "type": "prefix",
                "description": f"Values should start with '{common_prefix}'"
            })

        if common_suffix:
            result["advanced_patterns"]["common_suffix"] = {
                "suffix": common_suffix,
                "count": common_suffix_count,
                "percentage": (common_suffix_count / len(string_values)) * 100
            }
            result["validation_suggestions"].append({
                "type": "suffix",
                "description": f"Values should end with '{common_suffix}'"
            })

        # Check for capitalization patterns
        capitalization_patterns = {
            "all_lowercase": 0,
            "all_uppercase": 0,
            "title_case": 0,
            "sentence_case": 0,
            "mixed_case": 0
        }

        for value in string_values:
            if value.islower():
                capitalization_patterns["all_lowercase"] += 1
            elif value.isupper():
                capitalization_patterns["all_uppercase"] += 1
            elif value == value.title():
                capitalization_patterns["title_case"] += 1
            elif value[0].isupper() and value[1:].islower():
                capitalization_patterns["sentence_case"] += 1
            else:
                capitalization_patterns["mixed_case"] += 1

        # Find the dominant pattern (if any)
        dominant_pattern = max(capitalization_patterns.items(), key=lambda x: x[1])
        if dominant_pattern[1] >= len(string_values) * 0.8:
            result["advanced_patterns"]["capitalization"] = {
                "pattern": dominant_pattern[0],
                "count": dominant_pattern[1],
                "percentage": (dominant_pattern[1] / len(string_values)) * 100
            }

            # Add validation suggestion based on capitalization pattern
            if dominant_pattern[0] == "all_lowercase":
                result["validation_suggestions"].append({
                    "type": "capitalization",
                    "description": "Values should be all lowercase"
                })
            elif dominant_pattern[0] == "all_uppercase":
                result["validation_suggestions"].append({
                    "type": "capitalization",
                    "description": "Values should be all uppercase"
                })
            elif dominant_pattern[0] == "title_case":
                result["validation_suggestions"].append({
                    "type": "capitalization",
                    "description": "Values should be in Title Case"
                })
            elif dominant_pattern[0] == "sentence_case":
                result["validation_suggestions"].append({
                    "type": "capitalization",
                    "description": "Values should be in Sentence case"
                })

        # Check for character set patterns
        char_sets = {
            "alphanumeric": 0,
            "alpha_only": 0,
            "numeric_only": 0,
            "contains_spaces": 0,
            "contains_special": 0
        }

        for value in string_values:
            if value.isalnum():
                char_sets["alphanumeric"] += 1
            if value.isalpha():
                char_sets["alpha_only"] += 1
            if value.isdigit():
                char_sets["numeric_only"] += 1
            if " " in value:
                char_sets["contains_spaces"] += 1
            if any(c for c in value if not c.isalnum() and c != " "):
                char_sets["contains_special"] += 1

        # Add character set insights
        for char_set, count in char_sets.items():
            if count >= len(string_values) * 0.8:
                result["advanced_patterns"]["character_set"] = {
                    "pattern": char_set,
                    "count": count,
                    "percentage": (count / len(string_values)) * 100
                }

                if char_set == "alphanumeric":
                    result["validation_suggestions"].append({
                        "type": "character_set",
                        "description": "Values should only contain letters and numbers"
                    })
                elif char_set == "alpha_only":
                    result["validation_suggestions"].append({
                        "type": "character_set",
                        "description": "Values should only contain letters"
                    })
                elif char_set == "numeric_only":
                    result["validation_suggestions"].append({
                        "type": "character_set",
                        "description": "Values should only contain numbers"
                    })
                break  # Only use the most dominant pattern

    def _find_correlated_properties(self, type_name, property_name, elements, pattern_type):
        """Find properties that correlate with the given property"""
        correlations = []

        # Get all properties for this type
        if type_name in self.node_types:
            all_properties = self.node_types[type_name]["properties"]
        else:  # Edge type
            all_properties = self.edge_types[type_name]["properties"]

        # Skip self-correlation
        other_properties = [p for p in all_properties if p != property_name]

        # Calculate presence correlation (when one property is present, another is too)
        property_presence = {}
        co_presence = {}

        for element_id, element_data in elements.items():
            # Skip if target property isn't present
            if property_name not in element_data:
                continue

            # Count co-presence with other properties
            for other_prop in other_properties:
                # Initialize counters if needed
                if other_prop not in property_presence:
                    property_presence[other_prop] = 0
                    co_presence[other_prop] = 0

                # Count presence of this property
                if other_prop in element_data:
                    property_presence[other_prop] += 1
                    co_presence[other_prop] += 1

        # Calculate correlation coefficients
        for other_prop in other_properties:
            if other_prop not in property_presence or property_presence[other_prop] == 0:
                continue

            # Calculate simple correlation ratio
            presence_ratio = co_presence[other_prop] / property_presence[other_prop]

            # Add to correlations if strong enough
            if presence_ratio > 0.8:
                correlations.append({
                    "property": other_prop,
                    "correlation_type": "presence",
                    "correlation_strength": presence_ratio
                })

        # For string properties, check for value correlations
        if pattern_type in ["string", "enumeration", "email", "url", "date"]:
            for other_prop in other_properties:
                # Count cases where properties match in some way
                matches = 0
                total = 0

                for element_id, element_data in elements.items():
                    # Skip if either property isn't present
                    if property_name not in element_data or other_prop not in element_data:
                        continue

                    prop_value = str(element_data[property_name])
                    other_value = str(element_data[other_prop])

                    total += 1

                    # Check for exact match
                    if prop_value == other_value:
                        matches += 1
                        continue

                    # Check if one is a prefix of the other
                    if prop_value.startswith(other_value) or other_value.startswith(prop_value):
                        matches += 0.5  # Partial match
                        continue

                    # Check for case-insensitive match
                    if prop_value.lower() == other_value.lower():
                        matches += 0.8  # Strong match but not exact
                        continue

                    # Check for substring relationship
                    if prop_value in other_value or other_value in prop_value:
                        matches += 0.3  # Weak match

                # Calculate correlation if enough samples
                if total >= 5 and matches / total > 0.6:
                    correlations.append({
                        "property": other_prop,
                        "correlation_type": "value",
                        "correlation_strength": matches / total
                    })

        # Sort by correlation strength
        correlations.sort(key=lambda x: x["correlation_strength"], reverse=True)

        return correlations

    def _generate_regex_pattern(self, elements, property_name):
        """Generate a regex pattern that matches the structure of property values"""
        # Collect string values
        string_values = [str(element_data.get(property_name)) for element_data in elements.values()
                         if property_name in element_data and
                         isinstance(element_data[property_name], (str, int, float, bool))]

        if not string_values or len(string_values) < 3:
            return None

        # Analyze structure to detect patterns
        common_length = None
        if len(set(len(s) for s in string_values)) == 1:
            common_length = len(string_values[0])

        # Check for common character classes at each position
        if common_length and common_length <= 20:  # Only for reasonably short strings
            position_classes = []

            for i in range(common_length):
                position_chars = set(s[i] for s in string_values)

                # Determine character class
                if all(c.isdigit() for c in position_chars):
                    position_classes.append("\\d")
                elif all(c.isalpha() for c in position_chars):
                    if all(c.isupper() for c in position_chars):
                        position_classes.append("[A-Z]")
                    elif all(c.islower() for c in position_chars):
                        position_classes.append("[a-z]")
                    else:
                        position_classes.append("[A-Za-z]")
                elif position_chars == {'-'}:
                    position_classes.append("\\-")
                elif position_chars == {'/'}:
                    position_classes.append("\\/")
                elif position_chars == {':'}:
                    position_classes.append("\\:")
                elif position_chars == {'.'}:
                    position_classes.append("\\.")
                elif len(position_chars) == 1:
                    char = next(iter(position_chars))
                    if char in ".$^*+?()[]{}\|":
                        position_classes.append(f"\\{char}")
                    else:
                        position_classes.append(char)
                else:
                    position_classes.append(".")

            # Combine into regex pattern
            return "^" + "".join(position_classes) + "$"

        # Check for common patterns like dates, emails, etc.
        if all(re.match(r'\d{4}-\d{2}-\d{2}', s) for s in string_values[:10]):
            return r"^\d{4}-\d{2}-\d{2}$"

        if all(re.match(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', s) for s in string_values[:10]):
            return r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"

        if all(re.match(r'https?://\S+', s) for s in string_values[:10]):
            return r"^https?://\S+$"

        # If no specific pattern found, return None
        return None

    def generate_property_regex(self, type_name, property_name):
        """
        Generate a regex pattern that matches observed property values

        This method is a convenience wrapper around the pattern discovery
        functionality already provided by discover_property_patterns.

        Args:
            type_name: The name of the node or edge type
            property_name: The name of the property to analyze

        Returns:
            str: A regular expression pattern that matches the observed values,
                 or None if no pattern could be generated
        """
        # Get full pattern analysis first
        pattern_analysis = self.discover_property_patterns(type_name, property_name)

        # Extract regex if available
        return pattern_analysis.get("advanced_patterns", {}).get("regex")

    def detect_anomalies(self):
        """
        Detect nodes or edges that deviate from established patterns

        This method applies the validation rules discovered through property
        and relationship analysis to identify elements that violate the
        expected patterns.

        Returns:
            Dict: A collection of anomalies found in the graph, organized by type
        """
        if not self.ontology_extracted:
            self.extract_ontology()

        anomalies = {
            "node_anomalies": [],
            "edge_anomalies": [],
            "relationship_anomalies": []
        }

        # Analyze each node type for property anomalies
        for node_type, type_info in self.node_types.items():
            # Get property statistics for validation rules
            type_stats = self.get_type_property_statistics(node_type, is_node_type=True)

            # Check each node of this type
            for node_id in type_info["nodes"]:
                node_data = self.graph_manager.node_data.get(node_id, {})
                node_anomalies = []

                # Check each property
                for prop_name, prop_info in type_stats["properties"].items():
                    validation = prop_info.get("validation", {}).get("rules", [])

                    # Apply each validation rule
                    for rule in validation:
                        rule_type = rule.get("type")

                        # Required property missing
                        if rule_type == "required" and prop_name not in node_data:
                            node_anomalies.append({
                                "type": "missing_required_property",
                                "property": prop_name,
                                "message": f"Required property '{prop_name}' is missing"
                            })
                            continue

                        # Skip further validation if property is not present
                        if prop_name not in node_data:
                            continue

                        value = node_data[prop_name]

                        # Type validation
                        if rule_type == "string" and not isinstance(value, str):
                            node_anomalies.append({
                                "type": "type_mismatch",
                                "property": prop_name,
                                "expected": "string",
                                "actual": type(value).__name__
                            })

                        elif rule_type == "integer" and not (isinstance(value, int) or
                                                             (isinstance(value, float) and value.is_integer())):
                            node_anomalies.append({
                                "type": "type_mismatch",
                                "property": prop_name,
                                "expected": "integer",
                                "actual": type(value).__name__
                            })

                        elif rule_type == "number" and not isinstance(value, (int, float)):
                            node_anomalies.append({
                                "type": "type_mismatch",
                                "property": prop_name,
                                "expected": "number",
                                "actual": type(value).__name__
                            })

                        # Other validations would follow...

                # Add to anomalies if any found
                if node_anomalies:
                    anomalies["node_anomalies"].append({
                        "node_id": node_id,
                        "node_type": node_type,
                        "anomalies": node_anomalies
                    })

        # Similar pattern for edge anomalies...

        # Check relationship anomalies
        # This would look for relationship patterns that violate cardinality constraints

        return anomalies

    def discover_motifs(self):
        """
        Find repeated structural patterns (motifs) in the graph

        This method leverages relationship patterns and the GraphManager's
        motif detection capabilities to identify recurring structural patterns
        in the graph.

        Returns:
            Dict: A collection of motifs found in the graph
        """
        if not self.ontology_extracted:
            self.extract_ontology()

        motifs = {
            "relationship_motifs": [],
            "structural_motifs": []
        }

        # First, get relationship patterns (these are type-level motifs)
        relationship_patterns = self.extract_relationship_patterns()

        # Identify chains of relationships that form meaningful patterns
        if "common_patterns" in relationship_patterns:
            # Start with the most common patterns
            for pattern in relationship_patterns["common_patterns"]:
                src_type = pattern["source_type"]
                edge_type = pattern["edge_type"]
                tgt_type = pattern["target_type"]

                # Look for extensions of this pattern (A→B→C chains)
                for next_pattern in relationship_patterns["full_patterns"]:
                    if next_pattern["source_type"] == tgt_type:
                        # This forms a chain
                        chain = {
                            "pattern": f"{src_type}→{edge_type}→{tgt_type}→{next_pattern['edge_type']}→{next_pattern['target_type']}",
                            "types": [src_type, tgt_type, next_pattern["target_type"]],
                            "edges": [edge_type, next_pattern["edge_type"]],
                            "count": min(pattern["count"], next_pattern["count"])  # Conservative estimate
                        }
                        motifs["relationship_motifs"].append(chain)

        # Use GraphManager's motif capabilities if available
        if hasattr(self.graph_manager, "motif_cache"):
            # If the GraphManager has a motif cache, use it
            for motif_sig, instances in self.graph_manager.motif_cache.items():
                if len(instances) > 1:  # Only include repeated motifs
                    motifs["structural_motifs"].append({
                        "signature": motif_sig,
                        "count": len(instances),
                        "examples": instances[:3]  # Include a few examples
                    })

        return motifs

    def export_ontology_schema_graph(self, include_statistics=True):
        """
        Export ontology as a simplified schema graph

        This method creates a graph where:
        - Each node represents a node type from the original graph
        - Each edge represents an edge type from the original graph
        - Statistics and distribution information are attached as properties

        Args:
            include_statistics: Whether to include detailed statistical data

        Returns:
            GraphManager: A new GraphManager instance_graph containing the schema graph

        Raises:
            ValueError: If the ontology hasn't been extracted
        """
        if not self.ontology_extracted:
            self.extract_ontology()

        # Create a new GraphManager instance_graph
        from copy import deepcopy
        import json

        # Import or create GraphManager class (assuming it's available)
        try:
            # First try to import it from wherever it might be defined
            from no2_graphManager import GraphManager
        except ImportError:
            # If not available, use the same class as our own graph_manager
            if not hasattr(self.graph_manager, "__class__"):
                raise ImportError("GraphManager class not found and current graph_manager is invalid")
            GraphManager = self.graph_manager.__class__

        # Create a new instance_graph
        schema_graph = GraphManager(preload=False)

        # Generate regex patterns for properties
        property_regex_patterns = {}
        for node_type in self.node_types:
            if node_type in self.node_property_stats:
                for prop_name in self.node_property_stats[node_type]:
                    try:
                        regex = self.generate_property_regex(node_type, prop_name)
                        if regex:
                            if node_type not in property_regex_patterns:
                                property_regex_patterns[node_type] = {}
                            property_regex_patterns[node_type][prop_name] = regex
                    except:
                        pass  # Skip if regex generation fails

        # Add node types as nodes
        for node_type, type_info in self.node_types.items():
            # Collect statistics for this node type
            node_stats = {
                "count": type_info["count"],
                "percentage_of_graph": (type_info["count"] /
                                        sum(info["count"] for info in self.node_types.values()) * 100)
                if self.node_types else 0,
                "property_count": len(type_info["properties"]),
                "properties": list(type_info["properties"]),
            }

            # Add property statistics
            if include_statistics and node_type in self.node_property_stats:
                property_stats = {}
                for prop_name, stats in self.node_property_stats[node_type].items():
                    prop_stats = {
                        "type": stats.get("primary_type", "unknown"),
                        "occurrence_percentage": (stats.get("present_count", 0) /
                                                  type_info["count"] * 100)
                        if type_info["count"] > 0 else 0,
                        "unique_value_percentage": (stats.get("unique_count", 0) /
                                                    stats.get("present_count", 1) * 100)
                        if stats.get("present_count", 0) > 0 else 0,
                    }

                    # Add pattern type if available
                    if "pattern_type" in stats:
                        prop_stats["pattern_type"] = stats["pattern_type"]

                    # Add value ranges if available
                    if "min_value" in stats and "max_value" in stats:
                        prop_stats["value_range"] = {
                            "min": stats["min_value"],
                            "max": stats["max_value"],
                            "avg": stats.get("avg_value", None),
                            "median": stats.get("median_value", None)
                        }

                    # Add length ranges for strings
                    if "min_length" in stats and "max_length" in stats:
                        prop_stats["length_range"] = {
                            "min": stats["min_length"],
                            "max": stats["max_length"],
                            "avg": stats.get("avg_length", None)
                        }

                    # Add regex pattern if available
                    if node_type in property_regex_patterns and prop_name in property_regex_patterns[node_type]:
                        prop_stats["regex_pattern"] = property_regex_patterns[node_type][prop_name]

                    # Add enumeration values if available
                    if "enum_values" in stats:
                        prop_stats["enum_values"] = stats["enum_values"]

                    property_stats[prop_name] = prop_stats

                node_stats["property_statistics"] = property_stats

                # Add structural role information
                try:
                    node_analysis = self.analyze_node_type(node_type)
                    if "structural_metrics" in node_analysis:
                        metrics = node_analysis["structural_metrics"]
                        node_stats["structural_role"] = {
                            "avg_outgoing_edges": metrics.get("avg_outgoing_edges", 0),
                            "avg_incoming_edges": metrics.get("avg_incoming_edges", 0),
                            "connectivity_ratio": metrics.get("connectivity_ratio", 0)
                        }

                        # Add hierarchical position if available
                        if "hierarchical_position" in metrics:
                            hier_pos = metrics["hierarchical_position"]
                            node_stats["hierarchical_position"] = {
                                "leaf_percentage": hier_pos.get("leaf_percentage", 0),
                                "root_percentage": hier_pos.get("root_percentage", 0),
                                "intermediate_percentage": hier_pos.get("intermediate_percentage", 0)
                            }
                except:
                    pass  # Skip if analysis fails

            # Add the node to the graph
            schema_graph.add_node(
                node_id=node_type,  # Use the node type name directly as the ID
                value=node_type,
                type=node_type,  # Set the type to itself
                hierarchy="NodeType",
                attributes=node_stats
            )

        # Add relationships as edges
        for (src_type, edge_type, tgt_type), count in self.relationships.items():
            # Skip if source or target doesn't exist
            if src_type not in schema_graph.node_data or tgt_type not in schema_graph.node_data:
                continue

            # Collect statistics for this edge type
            edge_stats = {
                "edge_type": edge_type,
                "count": count,
                "percentage_of_type": (count /
                                       self.edge_types.get(edge_type, {}).get("count", 1) * 100)
                if edge_type in self.edge_types else 0
            }

            # Add cardinality information
            try:
                cardinality = self.analyze_relationship_cardinality(src_type, edge_type, tgt_type)
                edge_stats["cardinality"] = {
                    "type": cardinality["cardinality"]["type"],
                    "description": cardinality["cardinality"]["description"],
                    "outgoing": {
                        "min": cardinality["cardinality"]["outgoing"]["min"],
                        "max": cardinality["cardinality"]["outgoing"]["max"],
                        "median": cardinality["cardinality"]["outgoing"]["median"],
                        "sources_with_multiple_targets": cardinality["cardinality"]["outgoing"][
                            "sources_with_multiple_targets"]
                    },
                    "incoming": {
                        "min": cardinality["cardinality"]["incoming"]["min"],
                        "max": cardinality["cardinality"]["incoming"]["max"],
                        "median": cardinality["cardinality"]["incoming"]["median"],
                        "targets_with_multiple_sources": cardinality["cardinality"]["incoming"][
                            "targets_with_multiple_sources"]
                    }
                }

                # Add constraint metrics if available
                if "constraint_metrics" in cardinality["cardinality"]:
                    metrics = cardinality["cardinality"]["constraint_metrics"]
                    edge_stats["cardinality"]["constraints"] = {
                        "source_coverage": metrics.get("source_coverage", 0),
                        "target_coverage": metrics.get("target_coverage", 0),
                        "edge_density": metrics.get("edge_density", 0)
                    }
            except:
                edge_stats["cardinality"] = {"type": "unknown"}

            # Add property statistics for this edge type
            if include_statistics and edge_type in self.edge_property_stats:
                property_stats = {}
                for prop_name, stats in self.edge_property_stats[edge_type].items():
                    prop_stats = {
                        "type": stats.get("primary_type", "unknown"),
                        "occurrence_percentage": (stats.get("present_count", 0) /
                                                  self.edge_types[edge_type]["count"] * 100)
                        if edge_type in self.edge_types and
                           self.edge_types[edge_type]["count"] > 0 else 0
                    }

                    # Add other statistics as with node properties
                    if "pattern_type" in stats:
                        prop_stats["pattern_type"] = stats["pattern_type"]

                    if "min_value" in stats and "max_value" in stats:
                        prop_stats["value_range"] = {
                            "min": stats["min_value"],
                            "max": stats["max_value"]
                        }

                    property_stats[prop_name] = prop_stats

                edge_stats["property_statistics"] = property_stats

            # Check if edge already exists (to handle multiple edge types between the same nodes)
            edge_key = (src_type, tgt_type)
            edge_id = f"{src_type}_{edge_type}_{tgt_type}"

            # Add the edge to the graph
            try:
                schema_graph.add_edge(
                    source=src_type,
                    target=tgt_type,
                    attributes=edge_stats
                )
            except ValueError:
                # If edge already exists, create a different edge representation
                # Add a virtual node for this specific relationship
                virtual_node_id = f"Rel_{edge_id}"
                schema_graph.add_node(
                    node_id=virtual_node_id,
                    value=edge_type,
                    type="RelationshipType",
                    hierarchy="EdgeType",
                    attributes=edge_stats
                )

                # Connect the virtual node to source and target
                schema_graph.add_edge(
                    source=src_type,
                    target=virtual_node_id,
                    attributes={"edge_type": "SOURCE_OF", "count": count}
                )

                schema_graph.add_edge(
                    source=virtual_node_id,
                    target=tgt_type,
                    attributes={"edge_type": "TARGET_OF", "count": count}
                )

        # Add metadata node
        schema_graph.add_node(
            node_id="OntologyMeta",
            value="Ontology Schema Metadata",
            type="Metadata",
            hierarchy="OntologyElement",
            attributes={
                "node_type_count": len(self.node_types),
                "edge_type_count": len(self.edge_types),
                "relationship_pattern_count": len(self.relationships),
                "total_nodes_in_source": sum(type_info["count"] for type_info in self.node_types.values()),
                "total_edges_in_source": sum(type_info["count"] for type_info in self.edge_types.values()),
                "extraction_timestamp": datetime.datetime.now().isoformat()
            }
        )

        return schema_graph

    def export_to_graph_manager(self, include_statistics=True):
        """
        Export ontology to a new GraphManager instance_graph as a graph

        This method transforms the ontology metadata into a graph structure
        where node types and edge types become nodes, and relationships
        between them become edges.

        Args:
            include_statistics: Whether to include statistical data as attributes

        Returns:
            GraphManager: A new GraphManager instance_graph containing the ontology graph

        Raises:
            ValueError: If the ontology hasn't been extracted
        """
        if not self.ontology_extracted:
            self.extract_ontology()

        # Create a new GraphManager instance_graph
        from copy import deepcopy
        import json

        # Import or create GraphManager class (assuming it's available)
        try:
            # First try to import it from wherever it might be defined
            from no2_graphManager import GraphManager
        except ImportError:
            # If not available, use the same class as our own graph_manager
            if not hasattr(self.graph_manager, "__class__"):
                raise ImportError("GraphManager class not found and current graph_manager is invalid")
            GraphManager = self.graph_manager.__class__

        # Create a new instance_graph
        ontology_graph = GraphManager(preload=False)

        # Add node type nodes
        for node_type, type_info in self.node_types.items():
            # Create attributes including statistics
            attrs = {
                "count": type_info["count"],
                "property_count": len(type_info["properties"]),
                "properties": json.dumps(list(type_info["properties"])),
            }

            if include_statistics and node_type in self.node_property_stats:
                # Add summary statistics for properties
                property_stats = {}
                for prop_name, stats in self.node_property_stats[node_type].items():
                    property_stats[prop_name] = {
                        "type": stats.get("primary_type", "unknown"),
                        "present_count": stats.get("present_count", 0),
                        "unique_count": stats.get("unique_count", 0)
                    }
                attrs["property_stats"] = json.dumps(property_stats)

            # Add the node type as a node
            node_id = f"NodeType:{node_type}"
            ontology_graph.add_node(
                node_id=node_id,
                value=node_type,
                type="NodeType",
                hierarchy="OntologyElement",
                attributes=attrs
            )

            # Add individual properties as nodes
            if node_type in self.node_property_stats:
                for prop_name, stats in self.node_property_stats[node_type].items():
                    prop_id = f"Property:{node_type}.{prop_name}"

                    # Create property attributes
                    prop_attrs = {
                        "owner_type": node_type,
                        "data_type": stats.get("primary_type", "unknown"),
                        "presence_count": stats.get("present_count", 0),
                        "unique_count": stats.get("unique_count", 0)
                    }

                    # Add pattern type if available
                    if "pattern_type" in stats:
                        prop_attrs["pattern_type"] = stats["pattern_type"]

                    # Add the property as a node
                    ontology_graph.add_node(
                        node_id=prop_id,
                        value=prop_name,
                        type="Property",
                        hierarchy="OntologyElement",
                        attributes=prop_attrs
                    )

                    # Connect property to its node type
                    ontology_graph.add_edge(
                        source=node_id,
                        target=prop_id,
                        attributes={"edge_type": "HAS_PROPERTY"}
                    )

        # Add edge type nodes
        for edge_type, type_info in self.edge_types.items():
            # Create attributes including statistics
            attrs = {
                "count": type_info["count"],
                "property_count": len(type_info["properties"]),
                "properties": json.dumps(list(type_info["properties"])),
            }

            if include_statistics and edge_type in self.edge_property_stats:
                # Add summary statistics for properties
                property_stats = {}
                for prop_name, stats in self.edge_property_stats[edge_type].items():
                    property_stats[prop_name] = {
                        "type": stats.get("primary_type", "unknown"),
                        "present_count": stats.get("present_count", 0),
                        "unique_count": stats.get("unique_count", 0)
                    }
                attrs["property_stats"] = json.dumps(property_stats)

            # Add the edge type as a node
            node_id = f"EdgeType:{edge_type}"
            ontology_graph.add_node(
                node_id=node_id,
                value=edge_type,
                type="EdgeType",
                hierarchy="OntologyElement",
                attributes=attrs
            )

            # Add individual properties as nodes
            if edge_type in self.edge_property_stats:
                for prop_name, stats in self.edge_property_stats[edge_type].items():
                    prop_id = f"Property:{edge_type}.{prop_name}"

                    # Create property attributes
                    prop_attrs = {
                        "owner_type": edge_type,
                        "data_type": stats.get("primary_type", "unknown"),
                        "presence_count": stats.get("present_count", 0),
                        "unique_count": stats.get("unique_count", 0)
                    }

                    # Add pattern type if available
                    if "pattern_type" in stats:
                        prop_attrs["pattern_type"] = stats["pattern_type"]

                    # Add the property as a node
                    ontology_graph.add_node(
                        node_id=prop_id,
                        value=prop_name,
                        type="Property",
                        hierarchy="OntologyElement",
                        attributes=prop_attrs
                    )

                    # Connect property to its edge type
                    ontology_graph.add_edge(
                        source=node_id,
                        target=prop_id,
                        attributes={"edge_type": "HAS_PROPERTY"}
                    )

        # Track relationship counts to handle duplicates
        relationship_counts = {}

        # Add relationship connections
        for (src_type, edge_type, tgt_type), count in self.relationships.items():
            # Get node IDs
            src_node_id = f"NodeType:{src_type}"
            edge_type_id = f"EdgeType:{edge_type}"
            tgt_node_id = f"NodeType:{tgt_type}"

            # Connect source node type to edge type - handle potential duplicates
            if src_node_id in ontology_graph.node_data and edge_type_id in ontology_graph.node_data:
                # Check if this edge already exists
                edge_key = (src_node_id, edge_type_id)
                if edge_key in ontology_graph.edge_data:
                    # Update the count instead of creating a duplicate edge
                    current_count = ontology_graph.edge_data[edge_key].get("count", 0)
                    ontology_graph.edge_data[edge_key]["count"] = current_count + count
                else:
                    # Add a new edge
                    ontology_graph.add_edge(
                        source=src_node_id,
                        target=edge_type_id,
                        attributes={
                            "edge_type": "SOURCE_OF",
                            "count": count
                        }
                    )

            # Connect edge type to target node type - handle potential duplicates
            if edge_type_id in ontology_graph.node_data and tgt_node_id in ontology_graph.node_data:
                # Check if this edge already exists
                edge_key = (edge_type_id, tgt_node_id)
                if edge_key in ontology_graph.edge_data:
                    # Update the count instead of creating a duplicate edge
                    current_count = ontology_graph.edge_data[edge_key].get("count", 0)
                    ontology_graph.edge_data[edge_key]["count"] = current_count + count
                else:
                    # Add a new edge
                    ontology_graph.add_edge(
                        source=edge_type_id,
                        target=tgt_node_id,
                        attributes={
                            "edge_type": "TARGET_OF",
                            "count": count
                        }
                    )

        # Add meta-information node
        ontology_graph.add_node(
            node_id="OntologyMeta",
            value="Ontology Metadata",
            type="OntologyMeta",
            hierarchy="OntologyElement",
            attributes={
                "node_type_count": len(self.node_types),
                "edge_type_count": len(self.edge_types),
                "relationship_pattern_count": len(self.relationships),
                "source_graph_nodes": sum(type_info["count"] for type_info in self.node_types.values()),
                "source_graph_edges": sum(type_info["count"] for type_info in self.edge_types.values()),
                "extraction_timestamp": datetime.datetime.now().isoformat()
            }
        )

        return ontology_graph

    def export_to_owl(self, filepath, format='ttl'):
        """
        Export the ontology to OWL format with a focus on TTL compatibility

        Args:
            filepath: Path where the OWL file will be saved
            format: Output format - 'ttl' (Turtle, default), 'xml' (OWL/XML), or 'rdf' (RDF/XML)

        Returns:
            str: The filepath where the ontology was saved
        """
        if not self.ontology_extracted:
            self.extract_ontology()

        try:
            # For TTL format (preferred), use direct TTL syntax
            if format.lower() == 'ttl':
                # Create TTL content directly
                ttl_content = self._generate_ttl_content()

                # Write to file
                with open(filepath, "w", encoding='utf-8') as f:
                    f.write(ttl_content)

                return filepath

            # For other formats, use RDFLib if available
            try:
                from rdflib import Graph, Namespace, RDF, RDFS, OWL, XSD, URIRef, Literal, BNode

                # Create a graph
                g = Graph()

                # Define namespaces
                ex = Namespace(f"http://example.org/ontology/{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}#")
                g.bind("", ex)
                g.bind("owl", OWL)
                g.bind("rdf", RDF)
                g.bind("rdfs", RDFS)
                g.bind("xsd", XSD)

                # Add ontology declaration
                ontology = URIRef(ex)
                g.add((ontology, RDF.type, OWL.Ontology))

                # Add classes for node types
                for node_type in self.node_types:
                    # Create a class URI
                    class_uri = URIRef(ex[node_type.replace(" ", "_").replace("-", "_")])

                    # Add class declaration
                    g.add((class_uri, RDF.type, OWL.Class))
                    g.add((class_uri, RDFS.label, Literal(node_type)))

                # Add object properties for edge types
                for edge_type in self.edge_types:
                    # Create a property URI
                    prop_uri = URIRef(ex[edge_type.replace(" ", "_").replace("-", "_")])

                    # Add property declaration
                    g.add((prop_uri, RDF.type, OWL.ObjectProperty))
                    g.add((prop_uri, RDFS.label, Literal(edge_type)))

                # Add relationship constraints
                for (src_type, edge_type, tgt_type), count in self.relationships.items():
                    # Get URIs
                    src_uri = URIRef(ex[src_type.replace(" ", "_").replace("-", "_")])
                    edge_uri = URIRef(ex[edge_type.replace(" ", "_").replace("-", "_")])
                    tgt_uri = URIRef(ex[tgt_type.replace(" ", "_").replace("-", "_")])

                    # Add domain and range constraints
                    g.add((edge_uri, RDFS.domain, src_uri))
                    g.add((edge_uri, RDFS.range, tgt_uri))

                # Serialize in requested format
                if format.lower() == 'rdf':
                    content = g.serialize(format='xml')
                else:  # Default to owl/xml
                    content = g.serialize(format='xml')

                # Write to file
                with open(filepath, "w", encoding='utf-8') as f:
                    f.write(content)

                return filepath

            except ImportError:
                # Fallback to direct TTL generation if RDFLib not available
                ttl_content = self._generate_ttl_content()

                # Write to file
                with open(filepath, "w", encoding='utf-8') as f:
                    f.write(ttl_content)

                return filepath

        except Exception as e:
            print(f"Error exporting ontology to OWL: {str(e)}")
            raise

    def _generate_ttl_content(self):
        """Generate TTL content directly without requiring RDFLib"""
        # Generate a timestamp for a unique namespace
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        base_uri = f"http://example.org/ontology/{timestamp}#"

        # Build TTL content with proper prefixes
        ttl_content = f"""@prefix : <{base_uri}> .
    @prefix owl: <http://www.w3.org/2002/07/owl#> .
    @prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
    @prefix xml: <http://www.w3.org/XML/1998/namespace> .
    @prefix xsd: <http://www.w3.org/2001/XMLSchema#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @base <{base_uri}> .

    # Ontology Declaration
    <{base_uri}> rdf:type owl:Ontology ;
        rdfs:comment "Automatically inferred ontology from graph data" ;
        rdfs:label "Inferred Graph Ontology" ;
        owl:versionInfo "{timestamp}" .

    # Node Type Classes
    """

        # Add classes for node types
        for node_type in self.node_types:
            # Clean name for TTL
            clean_type = node_type.replace(" ", "_").replace("-", "_")
            type_info = self.node_types[node_type]

            ttl_content += f"""
    :{clean_type} rdf:type owl:Class ;
        rdfs:label "{node_type}" ;
        rdfs:comment "Node type with {type_info['count']} instances" .
    """

        # Add object properties for edge types
        ttl_content += "\n# Edge Type Properties\n"

        for edge_type in self.edge_types:
            # Clean name for TTL
            clean_type = edge_type.replace(" ", "_").replace("-", "_")
            type_info = self.edge_types[edge_type]

            ttl_content += f"""
    :{clean_type} rdf:type owl:ObjectProperty ;
        rdfs:label "{edge_type}" ;
        rdfs:comment "Edge type with {type_info['count']} instances" .
    """

        # Add property constraints
        ttl_content += "\n# Relationship Constraints\n"

        for (src_type, edge_type, tgt_type), count in self.relationships.items():
            # Clean names for TTL
            clean_src = src_type.replace(" ", "_").replace("-", "_")
            clean_edge = edge_type.replace(" ", "_").replace("-", "_")
            clean_tgt = tgt_type.replace(" ", "_").replace("-", "_")

            ttl_content += f"""
    # {src_type} → {edge_type} → {tgt_type} ({count} instances)
    :{clean_edge} rdfs:domain :{clean_src} ;
        rdfs:range :{clean_tgt} .
    """

        return ttl_content


    def export_to_json_schema(self, filepath):
        """
        Export the ontology as a JSON Schema for validation

        This method creates a JSON Schema document that can be used to validate
        graph data against the detected patterns and rules.

        Args:
            filepath: Path where the JSON Schema file will be saved

        Returns:
            str: The filepath where the schema was saved

        Raises:
            ValueError: If the ontology hasn't been extracted
        """
        if not self.ontology_extracted:
            self.extract_ontology()

        import json

        # Build schema structure
        schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "title": "Graph Schema",
            "description": "Schema for graph data based on extracted ontology",
            "type": "object",
            "properties": {
                "nodes": {
                    "type": "object",
                    "propertyNames": {
                        "description": "Node IDs can be any string"
                    },
                    "additionalProperties": {
                        "type": "object",
                        "required": ["type"],
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": list(self.node_types.keys())
                            }
                        }
                    }
                },
                "edges": {
                    "type": "object",
                    "propertyNames": {
                        "description": "Edge IDs are in the format 'source_target'"
                    },
                    "additionalProperties": {
                        "type": "object",
                        "required": ["edge_type"],
                        "properties": {
                            "edge_type": {
                                "type": "string",
                                "enum": list(self.edge_types.keys())
                            }
                        }
                    }
                }
            },
            "required": ["nodes", "edges"]
        }

        # Define node type schemas
        node_type_schemas = {}

        for node_type, type_info in self.node_types.items():
            # Basic schema for this node type
            type_schema = {
                "type": "object",
                "properties": {
                    "type": {"type": "string", "enum": [node_type]}
                },
                "required": ["type"]
            }

            # Add property constraints based on stats
            if node_type in self.node_property_stats:
                properties = {}
                required_props = []

                for prop_name, stats in self.node_property_stats[node_type].items():
                    # Only include properties with meaningful patterns
                    if "present_count" not in stats or stats["present_count"] == 0:
                        continue

                    # Determine property schema
                    prop_schema = {}
                    prop_type = stats.get("primary_type", "string")

                    # Basic type mapping
                    if prop_type in ("str", "string"):
                        prop_schema["type"] = "string"

                        # Add string-specific constraints
                        if "min_length" in stats and "max_length" in stats:
                            if stats["min_length"] == stats["max_length"]:
                                prop_schema["minLength"] = stats["min_length"]
                                prop_schema["maxLength"] = stats["min_length"]
                            else:
                                prop_schema["minLength"] = stats["min_length"]
                                prop_schema["maxLength"] = stats["max_length"]

                        # Add pattern type constraints
                        pattern_type = stats.get("pattern_type")
                        if pattern_type == "enumeration" and "enum_values" in stats:
                            prop_schema["enum"] = stats["enum_values"]
                        elif pattern_type == "email":
                            prop_schema["format"] = "email"
                        elif pattern_type == "date":
                            prop_schema["format"] = "date"
                        elif pattern_type == "url":
                            prop_schema["format"] = "uri"

                    elif prop_type in ("int", "integer"):
                        prop_schema["type"] = "integer"

                        # Add numeric constraints
                        if "min_value" in stats and "max_value" in stats:
                            prop_schema["minimum"] = int(stats["min_value"])
                            prop_schema["maximum"] = int(stats["max_value"])

                    elif prop_type in ("float", "decimal"):
                        prop_schema["type"] = "number"

                        # Add numeric constraints
                        if "min_value" in stats and "max_value" in stats:
                            prop_schema["minimum"] = stats["min_value"]
                            prop_schema["maximum"] = stats["max_value"]

                    elif prop_type == "bool":
                        prop_schema["type"] = "boolean"

                    elif prop_type in ("list", "set", "tuple", "array"):
                        prop_schema["type"] = "array"

                        # Add array constraints
                        if "min_items" in stats and "max_items" in stats:
                            prop_schema["minItems"] = stats["min_items"]
                            prop_schema["maxItems"] = stats["max_items"]

                        # Add item type if consistent
                        if "item_types" in stats and len(stats["item_types"]) == 1:
                            item_type = stats["item_types"][0]
                            if item_type in ("str", "string"):
                                prop_schema["items"] = {"type": "string"}
                            elif item_type in ("int", "integer"):
                                prop_schema["items"] = {"type": "integer"}
                            elif item_type in ("float", "decimal"):
                                prop_schema["items"] = {"type": "number"}
                            elif item_type == "bool":
                                prop_schema["items"] = {"type": "boolean"}
                            else:
                                prop_schema["items"] = {}

                    elif prop_type == "dict":
                        prop_schema["type"] = "object"

                        # Add object constraints
                        if "common_keys" in stats:
                            required_object_keys = []
                            for key, count in stats["common_keys"].items():
                                if count == stats["present_count"]:
                                    required_object_keys.append(key)

                            if required_object_keys:
                                prop_schema["required"] = required_object_keys

                    # Add property to schema
                    properties[prop_name] = prop_schema

                    # Check if property should be required
                    if stats["present_count"] == type_info["count"]:
                        required_props.append(prop_name)

                # Add properties and required list to type schema
                if properties:
                    type_schema["properties"].update(properties)

                if required_props:
                    type_schema["required"].extend(required_props)

            # Add this type schema to the collection
            node_type_schemas[node_type] = type_schema

        # Define edge type schemas
        edge_type_schemas = {}

        for edge_type, type_info in self.edge_types.items():
            # Basic schema for this edge type
            type_schema = {
                "type": "object",
                "properties": {
                    "edge_type": {"type": "string", "enum": [edge_type]}
                },
                "required": ["edge_type"]
            }

            # Add property constraints based on stats
            if edge_type in self.edge_property_stats:
                properties = {}
                required_props = []

                for prop_name, stats in self.edge_property_stats[edge_type].items():
                    # Skip edge_type as it's already handled
                    if prop_name == "edge_type":
                        continue

                    # Only include properties with meaningful patterns
                    if "present_count" not in stats or stats["present_count"] == 0:
                        continue

                    # Determine property schema (similar to node properties)
                    prop_schema = {}
                    prop_type = stats.get("primary_type", "string")

                    # Basic type mapping (same as for nodes)
                    if prop_type in ("str", "string"):
                        prop_schema["type"] = "string"
                        # Additional constraints as above...
                    elif prop_type in ("int", "integer"):
                        prop_schema["type"] = "integer"
                        # Additional constraints as above...
                    elif prop_type in ("float", "decimal"):
                        prop_schema["type"] = "number"
                        # Additional constraints as above...
                    # More type mappings...

                    # Add property to schema
                    properties[prop_name] = prop_schema

                    # Check if property should be required
                    if stats["present_count"] == type_info["count"]:
                        required_props.append(prop_name)

                # Add properties and required list to type schema
                if properties:
                    type_schema["properties"].update(properties)

                if required_props:
                    type_schema["required"].extend(required_props)

            # Add this type schema to the collection
            edge_type_schemas[edge_type] = type_schema

        # Add node and edge type definitions to schema
        node_schemas = {
            "oneOf": list(node_type_schemas.values())
        }

        edge_schemas = {
            "oneOf": list(edge_type_schemas.values())
        }

        # Update the main schema
        schema["properties"]["nodes"]["additionalProperties"] = node_schemas
        schema["properties"]["edges"]["additionalProperties"] = edge_schemas

        # Add relationship constraints
        valid_relationships = []
        for (src_type, edge_type, tgt_type), count in self.relationships.items():
            valid_relationships.append({
                "source_type": src_type,
                "edge_type": edge_type,
                "target_type": tgt_type
            })

        schema["relationships"] = valid_relationships

        # Write to file
        with open(filepath, "w") as f:
            json.dump(schema, f, indent=2)

        return filepath

    def load_ontology(self, filepath):
        """
        Load ontology from a file - OWL or JSON

        This method loads an ontology from a file, supporting both OWL
        (Web Ontology Language) and JSON formats.

        Args:
            filepath: Path to the ontology file (.owl or .json)

        Returns:
            Dict: A summary of the loaded ontology

        Raises:
            ValueError: If the file format is not supported
            FileNotFoundError: If the file doesn't exist
        """
        import os
        import json

        # Check if file exists
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        # Determine file format from extension
        file_ext = os.path.splitext(filepath)[1].lower()

        if file_ext == '.json':
            # Load JSON format
            return self._load_ontology_from_json(filepath)
        elif file_ext == '.owl':
            # Load OWL format
            return self._load_ontology_from_owl(filepath)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}. Supported formats are .json and .owl")

    def _load_ontology_from_json(self, filepath):
        """Helper method to load ontology from JSON format"""
        import json

        with open(filepath, 'r') as f:
            data = json.load(f)

        # Reset current ontology state
        self.node_types = {}
        self.edge_types = {}
        self.relationships = defaultdict(int)
        self.node_property_stats = {}
        self.edge_property_stats = {}

        # Check if this is a JSON Schema format
        if "$schema" in data and "properties" in data:
            # Process JSON Schema format

            # Extract node types
            if "nodes" in data["properties"] and "additionalProperties" in data["properties"]["nodes"]:
                node_schemas = data["properties"]["nodes"]["additionalProperties"]

                if "oneOf" in node_schemas:
                    # Process each node type schema
                    for type_schema in node_schemas["oneOf"]:
                        # Get node type name
                        type_name = None
                        if "properties" in type_schema and "type" in type_schema["properties"]:
                            if "enum" in type_schema["properties"]["type"]:
                                type_name = type_schema["properties"]["type"]["enum"][0]

                        if type_name:
                            # Create node type entry
                            self.node_types[type_name] = {
                                "count": 0,  # Unknown from schema
                                "properties": set(),
                                "nodes": set()
                            }

                            # Extract properties
                            if "properties" in type_schema:
                                for prop_name, prop_schema in type_schema["properties"].items():
                                    if prop_name != "type":  # Skip the type property
                                        self.node_types[type_name]["properties"].add(prop_name)

                                        # Extract property statistics
                                        if type_name not in self.node_property_stats:
                                            self.node_property_stats[type_name] = {}

                                        prop_stats = {
                                            "present_count": 0,  # Unknown from schema
                                            "unique_count": 0  # Unknown from schema
                                        }

                                        # Determine property type
                                        if "type" in prop_schema:
                                            if prop_schema["type"] == "string":
                                                prop_stats["primary_type"] = "str"
                                            elif prop_schema["type"] == "integer":
                                                prop_stats["primary_type"] = "int"
                                            elif prop_schema["type"] == "number":
                                                prop_stats["primary_type"] = "float"
                                            elif prop_schema["type"] == "boolean":
                                                prop_stats["primary_type"] = "bool"
                                            elif prop_schema["type"] == "array":
                                                prop_stats["primary_type"] = "array"
                                            elif prop_schema["type"] == "object":
                                                prop_stats["primary_type"] = "dict"

                                        # Extract constraints
                                        if "minLength" in prop_schema:
                                            prop_stats["min_length"] = prop_schema["minLength"]
                                        if "maxLength" in prop_schema:
                                            prop_stats["max_length"] = prop_schema["maxLength"]
                                        if "minimum" in prop_schema:
                                            prop_stats["min_value"] = prop_schema["minimum"]
                                        if "maximum" in prop_schema:
                                            prop_stats["max_value"] = prop_schema["maximum"]
                                        if "enum" in prop_schema:
                                            prop_stats["pattern_type"] = "enumeration"
                                            prop_stats["enum_values"] = prop_schema["enum"]
                                        elif "format" in prop_schema:
                                            if prop_schema["format"] == "email":
                                                prop_stats["pattern_type"] = "email"
                                            elif prop_schema["format"] == "date":
                                                prop_stats["pattern_type"] = "date"
                                            elif prop_schema["format"] == "uri":
                                                prop_stats["pattern_type"] = "url"

                                        self.node_property_stats[type_name][prop_name] = prop_stats

            # Extract edge types
            if "edges" in data["properties"] and "additionalProperties" in data["properties"]["edges"]:
                edge_schemas = data["properties"]["edges"]["additionalProperties"]

                if "oneOf" in edge_schemas:
                    # Process each edge type schema
                    for type_schema in edge_schemas["oneOf"]:
                        # Get edge type name
                        type_name = None
                        if "properties" in type_schema and "edge_type" in type_schema["properties"]:
                            if "enum" in type_schema["properties"]["edge_type"]:
                                type_name = type_schema["properties"]["edge_type"]["enum"][0]

                        if type_name:
                            # Create edge type entry
                            self.edge_types[type_name] = {
                                "count": 0,  # Unknown from schema
                                "properties": set(),
                                "edges": set()
                            }

                            # Extract properties
                            if "properties" in type_schema:
                                for prop_name, prop_schema in type_schema["properties"].items():
                                    if prop_name != "edge_type":  # Skip the edge_type property
                                        self.edge_types[type_name]["properties"].add(prop_name)

                                        # Extract property statistics (similar to node properties)
                                        if type_name not in self.edge_property_stats:
                                            self.edge_property_stats[type_name] = {}

                                        prop_stats = {
                                            "present_count": 0,  # Unknown from schema
                                            "unique_count": 0  # Unknown from schema
                                        }

                                        # Determine property type (similar to node properties)
                                        if "type" in prop_schema:
                                            # Type mapping, same as for nodes
                                            if prop_schema["type"] == "string":
                                                prop_stats["primary_type"] = "str"
                                            # Other type mappings...

                                        # Extract constraints (similar to node properties)
                                        # ...

                                        self.edge_property_stats[type_name][prop_name] = prop_stats

            # Extract relationships
            if "relationships" in data:
                for rel in data["relationships"]:
                    if "source_type" in rel and "edge_type" in rel and "target_type" in rel:
                        relationship_key = (rel["source_type"], rel["edge_type"], rel["target_type"])
                        self.relationships[relationship_key] = 0  # Count unknown from schema

        else:
            # Process direct ontology JSON format (exported by this class)
            if "node_types" in data:
                for type_name, type_info in data["node_types"].items():
                    self.node_types[type_name] = {
                        "count": type_info.get("count", 0),
                        "properties": set(type_info.get("properties", [])),
                        "nodes": set()  # No actual nodes
                    }

            if "edge_types" in data:
                for type_name, type_info in data["edge_types"].items():
                    self.edge_types[type_name] = {
                        "count": type_info.get("count", 0),
                        "properties": set(type_info.get("properties", [])),
                        "edges": set()  # No actual edges
                    }

            if "relationships" in data:
                for rel_info in data["relationships"]:
                    if "source_type" in rel_info and "edge_type" in rel_info and "target_type" in rel_info:
                        rel_key = (rel_info["source_type"], rel_info["edge_type"], rel_info["target_type"])
                        self.relationships[rel_key] = rel_info.get("count", 0)

            if "node_property_stats" in data:
                self.node_property_stats = data["node_property_stats"]

            if "edge_property_stats" in data:
                self.edge_property_stats = data["edge_property_stats"]

        # Mark as extracted
        self.ontology_extracted = True

        return {
            "node_types": len(self.node_types),
            "edge_types": len(self.edge_types),
            "relationship_patterns": len(self.relationships),
            "source": filepath
        }

    def _load_ontology_from_owl(self, filepath):
        """Helper method to load ontology from OWL format"""
        try:
            import xml.etree.ElementTree as ET
        except ImportError:
            raise ImportError("XML parsing requires the 'xml' module")

        # Define XML namespace mapping
        ns = {
            'owl': 'http://www.w3.org/2002/07/owl#',
            'rdf': 'http://www.w3.org/1999/02/22-rdf-syntax-ns#',
            'rdfs': 'http://www.w3.org/2000/01/rdf-schema#',
            'xsd': 'http://www.w3.org/2001/XMLSchema#'
        }

        # Parse OWL file
        tree = ET.parse(filepath)
        root = tree.getroot()

        # Reset current ontology state
        self.node_types = {}
        self.edge_types = {}
        self.relationships = defaultdict(int)
        self.node_property_stats = {}
        self.edge_property_stats = {}

        # Extract classes (node types)
        for class_elem in root.findall('.//owl:Class', ns):
            class_iri = class_elem.attrib.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about')
            if class_iri:
                # Extract class name from IRI
                type_name = class_iri.split('#')[-1].replace('_', ' ')

                # Create node type entry
                self.node_types[type_name] = {
                    "count": 0,  # Unknown from OWL
                    "properties": set(),
                    "nodes": set()
                }

        # Extract object properties (edge types)
        for prop_elem in root.findall('.//owl:ObjectProperty', ns):
            prop_iri = prop_elem.attrib.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about')
            if prop_iri:
                # Extract property name from IRI
                type_name = prop_iri.split('#')[-1].replace('_', ' ')

                # Create edge type entry
                self.edge_types[type_name] = {
                    "count": 0,  # Unknown from OWL
                    "properties": set(),
                    "edges": set()
                }

        # Extract data properties (node and edge properties)
        for data_prop in root.findall('.//owl:DataProperty', ns):
            prop_iri = data_prop.attrib.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about')
            if prop_iri:
                # Extract property name
                full_name = prop_iri.split('#')[-1].replace('_', ' ')

                # Check if this is a class property (format: Class_property)
                parts = full_name.split('_', 1)
                if len(parts) == 2 and parts[0] in self.node_types:
                    class_name = parts[0]
                    prop_name = parts[1]

                    # Add to node type properties
                    self.node_types[class_name]["properties"].add(prop_name)

                    # Add property statistics
                    if class_name not in self.node_property_stats:
                        self.node_property_stats[class_name] = {}

                    # Extract property type from range
                    prop_stats = {
                        "present_count": 0,  # Unknown from OWL
                        "unique_count": 0  # Unknown from OWL
                    }

                    # Find property range
                    for range_elem in root.findall(
                            f'.//owl:DataPropertyRange[./owl:DataProperty/@rdf:about="{prop_iri}"]', ns):
                        datatype = range_elem.find('./owl:Datatype', ns)
                        if datatype is not None:
                            datatype_iri = datatype.attrib.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource')
                            if datatype_iri:
                                if 'string' in datatype_iri:
                                    prop_stats["primary_type"] = "str"
                                elif 'integer' in datatype_iri:
                                    prop_stats["primary_type"] = "int"
                                elif 'decimal' in datatype_iri:
                                    prop_stats["primary_type"] = "float"
                                elif 'boolean' in datatype_iri:
                                    prop_stats["primary_type"] = "bool"
                                elif 'date' in datatype_iri:
                                    prop_stats["primary_type"] = "date"
                                    prop_stats["pattern_type"] = "date"

                    self.node_property_stats[class_name][prop_name] = prop_stats

        # Extract relationships from domains and ranges
        for obj_prop in root.findall('.//owl:ObjectProperty', ns):
            prop_iri = obj_prop.attrib.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about')
            if prop_iri:
                edge_type = prop_iri.split('#')[-1].replace('_', ' ')

                # Find domain (source type)
                domain_elem = root.find(f'.//owl:ObjectPropertyDomain[./owl:ObjectProperty/@rdf:about="{prop_iri}"]',
                                        ns)
                source_type = None
                if domain_elem is not None:
                    class_elem = domain_elem.find('./owl:Class', ns)
                    if class_elem is not None:
                        class_iri = class_elem.attrib.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about')
                        if class_iri:
                            source_type = class_iri.split('#')[-1].replace('_', ' ')

                # Find range (target type)
                range_elem = root.find(f'.//owl:ObjectPropertyRange[./owl:ObjectProperty/@rdf:about="{prop_iri}"]', ns)
                target_type = None
                if range_elem is not None:
                    class_elem = range_elem.find('./owl:Class', ns)
                    if class_elem is not None:
                        class_iri = class_elem.attrib.get('{http://www.w3.org/1999/02/22-rdf-syntax-ns#}about')
                        if class_iri:
                            target_type = class_iri.split('#')[-1].replace('_', ' ')

                # Add relationship
                if source_type and target_type:
                    rel_key = (source_type, edge_type, target_type)
                    self.relationships[rel_key] = 0  # Count unknown from OWL

        # Mark as extracted
        self.ontology_extracted = True

        return {
            "node_types": len(self.node_types),
            "edge_types": len(self.edge_types),
            "relationship_patterns": len(self.relationships),
            "source": filepath
        }

    def compare_ontologies(self, other_ontology):
        """
        Compare this ontology with another to find differences

        This method compares two ontologies and identifies differences in
        node types, edge types, properties, and relationships.

        Args:
            other_ontology: Another no3_OntologyManager instance_graph to compare with

        Returns:
            Dict: A comprehensive comparison of the two ontologies

        Raises:
            ValueError: If either ontology hasn't been extracted
        """
        if not self.ontology_extracted:
            self.extract_ontology()

        if not other_ontology.ontology_extracted:
            other_ontology.extract_ontology()

        # Build comparison structure
        comparison = {
            "summary": {
                "this_ontology": {
                    "node_types": len(self.node_types),
                    "edge_types": len(self.edge_types),
                    "relationships": len(self.relationships)
                },
                "other_ontology": {
                    "node_types": len(other_ontology.node_types),
                    "edge_types": len(other_ontology.edge_types),
                    "relationships": len(other_ontology.relationships)
                }
            },
            "node_types": {
                "common": [],
                "only_in_this": [],
                "only_in_other": [],
                "property_differences": {}
            },
            "edge_types": {
                "common": [],
                "only_in_this": [],
                "only_in_other": [],
                "property_differences": {}
            },
            "relationships": {
                "common": [],
                "only_in_this": [],
                "only_in_other": []
            }
        }

        # Compare node types
        this_node_types = set(self.node_types.keys())
        other_node_types = set(other_ontology.node_types.keys())

        comparison["node_types"]["common"] = sorted(list(this_node_types.intersection(other_node_types)))
        comparison["node_types"]["only_in_this"] = sorted(list(this_node_types - other_node_types))
        comparison["node_types"]["only_in_other"] = sorted(list(other_node_types - this_node_types))

        # Compare node type properties
        for node_type in comparison["node_types"]["common"]:
            this_props = self.node_types[node_type]["properties"]
            other_props = other_ontology.node_types[node_type]["properties"]

            if this_props != other_props:
                comparison["node_types"]["property_differences"][node_type] = {
                    "common": sorted(list(this_props.intersection(other_props))),
                    "only_in_this": sorted(list(this_props - other_props)),
                    "only_in_other": sorted(list(other_props - this_props))
                }

        # Compare edge types
        this_edge_types = set(self.edge_types.keys())
        other_edge_types = set(other_ontology.edge_types.keys())

        comparison["edge_types"]["common"] = sorted(list(this_edge_types.intersection(other_edge_types)))
        comparison["edge_types"]["only_in_this"] = sorted(list(this_edge_types - other_edge_types))
        comparison["edge_types"]["only_in_other"] = sorted(list(other_edge_types - this_edge_types))

        # Compare edge type properties
        for edge_type in comparison["edge_types"]["common"]:
            this_props = self.edge_types[edge_type]["properties"]
            other_props = other_ontology.edge_types[edge_type]["properties"]

            if this_props != other_props:
                comparison["edge_types"]["property_differences"][edge_type] = {
                    "common": sorted(list(this_props.intersection(other_props))),
                    "only_in_this": sorted(list(this_props - other_props)),
                    "only_in_other": sorted(list(other_props - this_props))
                }

        # Compare relationships
        this_rels = set(self.relationships.keys())
        other_rels = set(other_ontology.relationships.keys())

        comparison["relationships"]["common"] = sorted([
            {
                "source_type": src,
                "edge_type": edge,
                "target_type": tgt
            } for src, edge, tgt in this_rels.intersection(other_rels)
        ], key=lambda x: (x["source_type"], x["edge_type"], x["target_type"]))

        comparison["relationships"]["only_in_this"] = sorted([
            {
                "source_type": src,
                "edge_type": edge,
                "target_type": tgt
            } for src, edge, tgt in this_rels - other_rels
        ], key=lambda x: (x["source_type"], x["edge_type"], x["target_type"]))

        comparison["relationships"]["only_in_other"] = sorted([
            {
                "source_type": src,
                "edge_type": edge,
                "target_type": tgt
            } for src, edge, tgt in other_rels - this_rels
        ], key=lambda x: (x["source_type"], x["edge_type"], x["target_type"]))

        # Calculate similarity scores
        node_type_similarity = len(comparison["node_types"]["common"]) / (
                    len(this_node_types.union(other_node_types)) or 1)
        edge_type_similarity = len(comparison["edge_types"]["common"]) / (
                    len(this_edge_types.union(other_edge_types)) or 1)
        relationship_similarity = len(comparison["relationships"]["common"]) / (len(this_rels.union(other_rels)) or 1)

        comparison["similarity"] = {
            "node_types": node_type_similarity,
            "edge_types": edge_type_similarity,
            "relationships": relationship_similarity,
            "overall": (node_type_similarity + edge_type_similarity + relationship_similarity) / 3
        }

        # Generate insights
        comparison["insights"] = []

        # Check for significant schema evolution
        if node_type_similarity < 0.7:
            comparison["insights"].append({
                "type": "significant_change",
                "description": "The node type schemas have changed significantly between the two ontologies"
            })

        # Check for schema expansion
        if (len(comparison["node_types"]["only_in_other"]) > len(comparison["node_types"]["only_in_this"])) and \
                (len(comparison["edge_types"]["only_in_other"]) > len(comparison["edge_types"]["only_in_this"])):
            comparison["insights"].append({
                "type": "schema_expansion",
                "description": "The newer ontology appears to be an expansion of this ontology with additional types"
            })

        # Check for schema refinement
        property_differences = len(comparison["node_types"]["property_differences"]) + len(
            comparison["edge_types"]["property_differences"])
        common_types = len(comparison["node_types"]["common"]) + len(comparison["edge_types"]["common"])

        if property_differences > 0 and property_differences / (common_types or 1) > 0.3:
            comparison["insights"].append({
                "type": "property_refinement",
                "description": "Many common types have different property sets, suggesting schema refinement"
            })

        return comparison

    def merge_ontologies(self, other_ontology):
        """
        Merge two ontologies, preserving all type definitions

        This method combines two ontologies, creating a unified ontology that
        preserves all node types, edge types, properties, and relationships
        from both sources.

        Args:
            other_ontology: Another no3_OntologyManager instance_graph to merge with

        Returns:
            OntologyManager: A new no3_OntologyManager instance_graph containing the merged ontology

        Raises:
            ValueError: If either ontology hasn't been extracted
        """
        if not self.ontology_extracted:
            self.extract_ontology()

        if not other_ontology.ontology_extracted:
            other_ontology.extract_ontology()

        # Create a new no3_OntologyManager for the merged result
        merged = OntologyManager()

        # Initialize merged ontology structures
        merged.node_types = {}
        merged.edge_types = {}
        merged.relationships = defaultdict(int)
        merged.node_property_stats = {}
        merged.edge_property_stats = {}

        # Merge node types
        for node_type, type_info in self.node_types.items():
            merged.node_types[node_type] = {
                "count": type_info["count"],
                "properties": set(type_info["properties"]),
                "nodes": set()  # No actual nodes in the merged ontology
            }

        for node_type, type_info in other_ontology.node_types.items():
            if node_type in merged.node_types:
                # Merge properties for existing node type
                merged.node_types[node_type]["count"] += type_info["count"]
                merged.node_types[node_type]["properties"].update(type_info["properties"])
            else:
                # Add new node type
                merged.node_types[node_type] = {
                    "count": type_info["count"],
                    "properties": set(type_info["properties"]),
                    "nodes": set()  # No actual nodes in the merged ontology
                }

        # Merge edge types
        for edge_type, type_info in self.edge_types.items():
            merged.edge_types[edge_type] = {
                "count": type_info["count"],
                "properties": set(type_info["properties"]),
                "edges": set()  # No actual edges in the merged ontology
            }

        for edge_type, type_info in other_ontology.edge_types.items():
            if edge_type in merged.edge_types:
                # Merge properties for existing edge type
                merged.edge_types[edge_type]["count"] += type_info["count"]
                merged.edge_types[edge_type]["properties"].update(type_info["properties"])
            else:
                # Add new edge type
                merged.edge_types[edge_type] = {
                    "count": type_info["count"],
                    "properties": set(type_info["properties"]),
                    "edges": set()  # No actual edges in the merged ontology
                }

        # Merge relationships
        for rel_key, count in self.relationships.items():
            merged.relationships[rel_key] = count

        for rel_key, count in other_ontology.relationships.items():
            if rel_key in merged.relationships:
                merged.relationships[rel_key] += count
            else:
                merged.relationships[rel_key] = count

        # Merge node property statistics
        for node_type, props in self.node_property_stats.items():
            merged.node_property_stats[node_type] = {}
            for prop_name, stats in props.items():
                merged.node_property_stats[node_type][prop_name] = stats.copy()

        for node_type, props in other_ontology.node_property_stats.items():
            if node_type not in merged.node_property_stats:
                merged.node_property_stats[node_type] = {}

            for prop_name, stats in props.items():
                if prop_name in merged.node_property_stats[node_type]:
                    # Merge stats for existing property
                    merged_stats = merged.node_property_stats[node_type][prop_name]
                    other_stats = stats

                    # Update counts
                    merged_stats["present_count"] += other_stats.get("present_count", 0)
                    merged_stats["unique_count"] = max(merged_stats.get("unique_count", 0),
                                                       other_stats.get("unique_count", 0))

                    # Merge ranges
                    if "min_length" in merged_stats and "min_length" in other_stats:
                        merged_stats["min_length"] = min(merged_stats["min_length"], other_stats["min_length"])
                        merged_stats["max_length"] = max(merged_stats["max_length"], other_stats["max_length"])

                    if "min_value" in merged_stats and "min_value" in other_stats:
                        merged_stats["min_value"] = min(merged_stats["min_value"], other_stats["min_value"])
                        merged_stats["max_value"] = max(merged_stats["max_value"], other_stats["max_value"])

                    # Merge enums
                    if "enum_values" in merged_stats and "enum_values" in other_stats:
                        merged_stats["enum_values"] = list(
                            set(merged_stats["enum_values"]).union(other_stats["enum_values"]))
                else:
                    # Add new property stats
                    merged.node_property_stats[node_type][prop_name] = stats.copy()

        # Merge edge property statistics (similar to node property stats)
        for edge_type, props in self.edge_property_stats.items():
            merged.edge_property_stats[edge_type] = {}
            for prop_name, stats in props.items():
                merged.edge_property_stats[edge_type][prop_name] = stats.copy()

        for edge_type, props in other_ontology.edge_property_stats.items():
            if edge_type not in merged.edge_property_stats:
                merged.edge_property_stats[edge_type] = {}

            for prop_name, stats in props.items():
                if prop_name in merged.edge_property_stats[edge_type]:
                    # Merge stats (similar to node property stats)
                    # ...
                    pass
                else:
                    # Add new property stats
                    merged.edge_property_stats[edge_type][prop_name] = stats.copy()

        # Mark as extracted
        merged.ontology_extracted = True

        return merged

    def track_ontology_changes(self, other_ontology):
        """
        Identify changes between two versions of an ontology

        This method analyzes the differences between two ontologies and
        constructs a detailed change log, useful for tracking schema evolution.

        Args:
            other_ontology: Another no3_OntologyManager instance_graph to compare with
                           (typically a newer version)

        Returns:
            Dict: A structured log of changes between the ontologies

        Raises:
            ValueError: If either ontology hasn't been extracted
        """
        if not self.ontology_extracted:
            self.extract_ontology()

        if not other_ontology.ontology_extracted:
            other_ontology.extract_ontology()

        # Create a change log structure
        import datetime

        change_log = {
            "timestamp": datetime.datetime.now().isoformat(),
            "summary": {
                "added_node_types": [],
                "removed_node_types": [],
                "modified_node_types": [],
                "added_edge_types": [],
                "removed_edge_types": [],
                "modified_edge_types": [],
                "added_relationships": [],
                "removed_relationships": []
            },
            "changes": {
                "node_types": [],
                "edge_types": [],
                "relationships": []
            },
            "property_changes": {
                "node_types": {},
                "edge_types": {}
            },
            "statistics": {
                "total_changes": 0,
                "change_types": {}
            }
        }

        # Track all changes
        changes_count = 0
        change_types = defaultdict(int)

        # Check node type changes
        this_node_types = set(self.node_types.keys())
        other_node_types = set(other_ontology.node_types.keys())

        # Added node types
        for node_type in other_node_types - this_node_types:
            changes_count += 1
            change_types["add_node_type"] += 1
            change_log["summary"]["added_node_types"].append(node_type)

            change_log["changes"]["node_types"].append({
                "type": "add_node_type",
                "node_type": node_type,
                "properties": list(other_ontology.node_types[node_type]["properties"]),
                "count": other_ontology.node_types[node_type]["count"]
            })

        # Removed node types
        for node_type in this_node_types - other_node_types:
            changes_count += 1
            change_types["remove_node_type"] += 1
            change_log["summary"]["removed_node_types"].append(node_type)

            change_log["changes"]["node_types"].append({
                "type": "remove_node_type",
                "node_type": node_type,
                "properties": list(self.node_types[node_type]["properties"]),
                "count": self.node_types[node_type]["count"]
            })

        # Modified node types (property changes)
        for node_type in this_node_types.intersection(other_node_types):
            this_props = set(self.node_types[node_type]["properties"])
            other_props = set(other_ontology.node_types[node_type]["properties"])

            # Check for property changes
            if this_props != other_props:
                changes_count += 1
                change_types["modify_node_type"] += 1
                change_log["summary"]["modified_node_types"].append(node_type)

                added_props = list(other_props - this_props)
                removed_props = list(this_props - other_props)

                change_log["changes"]["node_types"].append({
                    "type": "modify_node_type",
                    "node_type": node_type,
                    "added_properties": added_props,
                    "removed_properties": removed_props
                })

                change_log["property_changes"]["node_types"][node_type] = {
                    "added": added_props,
                    "removed": removed_props
                }

                # Track property value changes for common properties
                if node_type in self.node_property_stats and node_type in other_ontology.node_property_stats:
                    common_props = this_props.intersection(other_props)
                    property_value_changes = []

                    for prop_name in common_props:
                        if prop_name in self.node_property_stats[node_type] and prop_name in \
                                other_ontology.node_property_stats[node_type]:
                            this_stats = self.node_property_stats[node_type][prop_name]
                            other_stats = other_ontology.node_property_stats[node_type][prop_name]

                            # Check for type changes
                            if this_stats.get("primary_type") != other_stats.get("primary_type"):
                                property_value_changes.append({
                                    "property": prop_name,
                                    "change_type": "type_change",
                                    "old_type": this_stats.get("primary_type"),
                                    "new_type": other_stats.get("primary_type")
                                })
                                changes_count += 1
                                change_types["property_type_change"] += 1

                            # Check for range changes
                            if "min_value" in this_stats and "min_value" in other_stats:
                                if this_stats["min_value"] != other_stats["min_value"] or this_stats["max_value"] != \
                                        other_stats["max_value"]:
                                    property_value_changes.append({
                                        "property": prop_name,
                                        "change_type": "range_change",
                                        "old_range": [this_stats["min_value"], this_stats["max_value"]],
                                        "new_range": [other_stats["min_value"], other_stats["max_value"]]
                                    })
                                    changes_count += 1
                                    change_types["property_range_change"] += 1

                            # Check for enum value changes
                            if "enum_values" in this_stats and "enum_values" in other_stats:
                                this_enum = set(this_stats["enum_values"])
                                other_enum = set(other_stats["enum_values"])

                                if this_enum != other_enum:
                                    property_value_changes.append({
                                        "property": prop_name,
                                        "change_type": "enum_change",
                                        "added_values": list(other_enum - this_enum),
                                        "removed_values": list(this_enum - other_enum)
                                    })
                                    changes_count += 1
                                    change_types["property_enum_change"] += 1

                    if property_value_changes:
                        if node_type not in change_log["property_changes"]["node_types"]:
                            change_log["property_changes"]["node_types"][node_type] = {}

                        change_log["property_changes"]["node_types"][node_type][
                            "value_changes"] = property_value_changes

        # Check edge type changes (similar pattern to node types)
        this_edge_types = set(self.edge_types.keys())
        other_edge_types = set(other_ontology.edge_types.keys())

        # Added edge types
        for edge_type in other_edge_types - this_edge_types:
            changes_count += 1
            change_types["add_edge_type"] += 1
            change_log["summary"]["added_edge_types"].append(edge_type)

            change_log["changes"]["edge_types"].append({
                "type": "add_edge_type",
                "edge_type": edge_type,
                "properties": list(other_ontology.edge_types[edge_type]["properties"]),
                "count": other_ontology.edge_types[edge_type]["count"]
            })

        # Removed edge types
        for edge_type in this_edge_types - other_edge_types:
            changes_count += 1
            change_types["remove_edge_type"] += 1
            change_log["summary"]["removed_edge_types"].append(edge_type)

            change_log["changes"]["edge_types"].append({
                "type": "remove_edge_type",
                "edge_type": edge_type,
                "properties": list(self.edge_types[edge_type]["properties"]),
                "count": self.edge_types[edge_type]["count"]
            })

        # Modified edge types (property changes)
        for edge_type in this_edge_types.intersection(other_edge_types):
            this_props = set(self.edge_types[edge_type]["properties"])
            other_props = set(other_ontology.edge_types[edge_type]["properties"])

            # Check for property changes
            if this_props != other_props:
                changes_count += 1
                change_types["modify_edge_type"] += 1
                change_log["summary"]["modified_edge_types"].append(edge_type)

                added_props = list(other_props - this_props)
                removed_props = list(this_props - other_props)

                change_log["changes"]["edge_types"].append({
                    "type": "modify_edge_type",
                    "edge_type": edge_type,
                    "added_properties": added_props,
                    "removed_properties": removed_props
                })

                change_log["property_changes"]["edge_types"][edge_type] = {
                    "added": added_props,
                    "removed": removed_props
                }

                # Track property value changes for common properties
                # (Similar to node type property changes)
                if edge_type in self.edge_property_stats and edge_type in other_ontology.edge_property_stats:
                    common_props = this_props.intersection(other_props)
                    property_value_changes = []

                    # Similar analysis to node type properties
                    # ...

                    if property_value_changes:
                        if edge_type not in change_log["property_changes"]["edge_types"]:
                            change_log["property_changes"]["edge_types"][edge_type] = {}

                        change_log["property_changes"]["edge_types"][edge_type][
                            "value_changes"] = property_value_changes

        # Check relationship changes
        this_rels = set(self.relationships.keys())
        other_rels = set(other_ontology.relationships.keys())

        # Added relationships
        for rel_key in other_rels - this_rels:
            changes_count += 1
            change_types["add_relationship"] += 1

            src_type, edge_type, tgt_type = rel_key
            relationship = {
                "source_type": src_type,
                "edge_type": edge_type,
                "target_type": tgt_type
            }

            change_log["summary"]["added_relationships"].append(relationship)

            change_log["changes"]["relationships"].append({
                "type": "add_relationship",
                "relationship": relationship,
                "count": other_ontology.relationships[rel_key]
            })

        # Removed relationships
        for rel_key in this_rels - other_rels:
            changes_count += 1
            change_types["remove_relationship"] += 1

            src_type, edge_type, tgt_type = rel_key
            relationship = {
                "source_type": src_type,
                "edge_type": edge_type,
                "target_type": tgt_type
            }

            change_log["summary"]["removed_relationships"].append(relationship)

            change_log["changes"]["relationships"].append({
                "type": "remove_relationship",
                "relationship": relationship,
                "count": self.relationships[rel_key]
            })

        # Update statistics
        change_log["statistics"]["total_changes"] = changes_count
        change_log["statistics"]["change_types"] = dict(change_types)

        # Generate change impact assessment
        impact_assessment = []

        # Check for major schema changes
        if len(change_log["summary"]["added_node_types"]) > 0 or len(change_log["summary"]["removed_node_types"]) > 0:
            impact_assessment.append({
                "type": "schema_evolution",
                "impact": "high" if len(change_log["summary"]["removed_node_types"]) > 0 else "medium",
                "description": "The ontology schema has evolved with new node types"
            })

        # Check for property changes
        if (len(change_log["summary"]["modified_node_types"]) > 0 or
                len(change_log["summary"]["modified_edge_types"]) > 0):
            impact_assessment.append({
                "type": "property_refinement",
                "impact": "medium",
                "description": "Properties have been modified in existing types"
            })

        # Check for relationship changes
        if (len(change_log["summary"]["added_relationships"]) > 0 or
                len(change_log["summary"]["removed_relationships"]) > 0):
            impact_assessment.append({
                "type": "connectivity_changes",
                "impact": "medium",
                "description": "The relationships between types have changed"
            })

        # Add impact assessment to change log
        change_log["impact_assessment"] = impact_assessment

        return change_log

    def get_ontology_summary(self):
        """
        Generate a human-readable summary of the ontology

        This method creates a comprehensive, human-readable overview
        of the ontology structure, highlighting key statistics and patterns.

        Returns:
            Dict: A structured summary of the ontology

        Raises:
            ValueError: If the ontology hasn't been extracted
        """
        if not self.ontology_extracted:
            self.extract_ontology()

        # Basic counts
        total_node_types = len(self.node_types)
        total_edge_types = len(self.edge_types)
        total_relationship_patterns = len(self.relationships)
        total_nodes = sum(type_info["count"] for type_info in self.node_types.values())
        total_edges = sum(type_info["count"] for type_info in self.edge_types.values())

        # Build summary structure
        summary = {
            "overview": {
                "title": "Graph Ontology Summary",
                "description": f"Analysis of a graph with {total_nodes} nodes across {total_node_types} types and {total_edges} edges across {total_edge_types} types.",
                "counts": {
                    "node_types": total_node_types,
                    "edge_types": total_edge_types,
                    "relationship_patterns": total_relationship_patterns,
                    "nodes": total_nodes,
                    "edges": total_edges
                }
            },
            "node_types": {
                "summary": [],
                "most_frequent": [],
                "property_rich": []
            },
            "edge_types": {
                "summary": [],
                "most_frequent": [],
                "property_rich": []
            },
            "relationships": {
                "common_patterns": [],
                "type_connections": {}
            },
            "property_patterns": {
                "common_properties": [],
                "unique_identifiers": [],
                "enumerations": []
            },
            "structure": {
                "connectivity": {},
                "hierarchy_indicators": [],
                "isolated_components": []
            }
        }

        # Add node type summaries
        node_type_list = [(name, info) for name, info in self.node_types.items()]
        # Sort by count (descending)
        node_type_list.sort(key=lambda x: x[1]["count"], reverse=True)

        # Top node types by frequency
        for name, info in node_type_list[:5]:  # Top 5
            prop_count = len(info["properties"])
            percentage = (info["count"] / total_nodes) * 100 if total_nodes > 0 else 0

            summary["node_types"]["most_frequent"].append({
                "type": name,
                "count": info["count"],
                "percentage": percentage,
                "property_count": prop_count
            })

        # Node types with most properties
        prop_rich_nodes = sorted(node_type_list, key=lambda x: len(x[1]["properties"]), reverse=True)
        for name, info in prop_rich_nodes[:5]:  # Top 5
            prop_count = len(info["properties"])
            percentage = (info["count"] / total_nodes) * 100 if total_nodes > 0 else 0

            summary["node_types"]["property_rich"].append({
                "type": name,
                "property_count": prop_count,
                "node_count": info["count"],
                "percentage": percentage
            })

        # Overall node type summary
        for name, info in self.node_types.items():
            prop_count = len(info["properties"])
            node_count = info["count"]
            percentage = (node_count / total_nodes) * 100 if total_nodes > 0 else 0

            summary["node_types"]["summary"].append({
                "type": name,
                "count": node_count,
                "percentage": percentage,
                "property_count": prop_count
            })

        # Sort summary by count
        summary["node_types"]["summary"].sort(key=lambda x: x["count"], reverse=True)

        # Similar analysis for edge types
        edge_type_list = [(name, info) for name, info in self.edge_types.items()]
        # Sort by count (descending)
        edge_type_list.sort(key=lambda x: x[1]["count"], reverse=True)

        # Top edge types by frequency
        for name, info in edge_type_list[:5]:  # Top 5
            prop_count = len(info["properties"])
            percentage = (info["count"] / total_edges) * 100 if total_edges > 0 else 0

            summary["edge_types"]["most_frequent"].append({
                "type": name,
                "count": info["count"],
                "percentage": percentage,
                "property_count": prop_count
            })

        # Edge types with most properties
        prop_rich_edges = sorted(edge_type_list, key=lambda x: len(x[1]["properties"]), reverse=True)
        for name, info in prop_rich_edges[:5]:  # Top 5
            prop_count = len(info["properties"])
            percentage = (info["count"] / total_edges) * 100 if total_edges > 0 else 0

            summary["edge_types"]["property_rich"].append({
                "type": name,
                "property_count": prop_count,
                "edge_count": info["count"],
                "percentage": percentage
            })

        # Overall edge type summary
        for name, info in self.edge_types.items():
            prop_count = len(info["properties"])
            edge_count = info["count"]
            percentage = (edge_count / total_edges) * 100 if total_edges > 0 else 0

            summary["edge_types"]["summary"].append({
                "type": name,
                "count": edge_count,
                "percentage": percentage,
                "property_count": prop_count
            })

        # Sort summary by count
        summary["edge_types"]["summary"].sort(key=lambda x: x["count"], reverse=True)

        # Analyze relationships
        rel_list = [(k, v) for k, v in self.relationships.items()]
        rel_list.sort(key=lambda x: x[1], reverse=True)

        # Common relationship patterns
        for (src, edge, tgt), count in rel_list[:10]:  # Top 10
            percentage = (count / total_edges) * 100 if total_edges > 0 else 0

            summary["relationships"]["common_patterns"].append({
                "pattern": f"{src} → {edge} → {tgt}",
                "source_type": src,
                "edge_type": edge,
                "target_type": tgt,
                "count": count,
                "percentage": percentage
            })

        # Create type connection graph
        node_connections = {}
        for (src, edge, tgt), count in self.relationships.items():
            if src not in node_connections:
                node_connections[src] = {"outgoing": set(), "incoming": set()}
            if tgt not in node_connections:
                node_connections[tgt] = {"outgoing": set(), "incoming": set()}

            node_connections[src]["outgoing"].add(tgt)
            node_connections[tgt]["incoming"].add(src)

        # Calculate connectivity stats
        for node_type, connections in node_connections.items():
            outgoing = len(connections["outgoing"])
            incoming = len(connections["incoming"])
            total = len(connections["outgoing"].union(connections["incoming"]))

            summary["structure"]["connectivity"][node_type] = {
                "outgoing_connections": outgoing,
                "incoming_connections": incoming,
                "total_connected_types": total
            }

        # Find most central node types
        centrality = [(node_type, len(connections["outgoing"]) + len(connections["incoming"]))
                      for node_type, connections in node_connections.items()]
        centrality.sort(key=lambda x: x[1], reverse=True)

        central_types = [{"type": t, "connections": c} for t, c in centrality[:5]]
        summary["structure"]["central_types"] = central_types

        # Identify potential hierarchical patterns
        hierarchy_indicators = []
        for (src, edge, tgt), count in self.relationships.items():
            # Check for self-references (same source and target type)
            if src == tgt:
                # Check for typical hierarchical edge names
                hierarchical_terms = ["parent", "child", "contains", "part_of", "belongs_to", "member_of"]
                if any(term in edge.lower() for term in hierarchical_terms):
                    hierarchy_indicators.append({
                        "source_type": src,
                        "edge_type": edge,
                        "target_type": tgt,
                        "count": count,
                        "hierarchical_term": next(term for term in hierarchical_terms if term in edge.lower())
                    })

        summary["structure"]["hierarchy_indicators"] = hierarchy_indicators

        # Analyze property patterns
        all_node_properties = {}
        for node_type, type_info in self.node_types.items():
            for prop in type_info["properties"]:
                if prop not in all_node_properties:
                    all_node_properties[prop] = 0
                all_node_properties[prop] += 1

        # Common properties across node types
        common_props = [(prop, count) for prop, count in all_node_properties.items()]
        common_props.sort(key=lambda x: x[1], reverse=True)

        for prop, count in common_props[:10]:  # Top 10
            percentage = (count / total_node_types) * 100 if total_node_types > 0 else 0

            summary["property_patterns"]["common_properties"].append({
                "property": prop,
                "occurrence": count,
                "percentage_of_types": percentage
            })

        # Identify potential unique identifiers
        for node_type, prop_stats in self.node_property_stats.items():
            for prop_name, stats in prop_stats.items():
                # Check if property has unique values across all instances
                if stats.get("unique_count", 0) == stats.get("present_count", 0) and stats.get("present_count", 0) > 1:
                    summary["property_patterns"]["unique_identifiers"].append({
                        "node_type": node_type,
                        "property": prop_name,
                        "count": stats.get("present_count", 0)
                    })

        # Identify enumeration properties
        for node_type, prop_stats in self.node_property_stats.items():
            for prop_name, stats in prop_stats.items():
                if stats.get("pattern_type") == "enumeration" and "enum_values" in stats:
                    enum_values = stats["enum_values"]
                    if len(enum_values) <= 10:  # Reasonable size for an enum
                        summary["property_patterns"]["enumerations"].append({
                            "node_type": node_type,
                            "property": prop_name,
                            "values": enum_values,
                            "count": stats.get("present_count", 0)
                        })

        # Generate natural language description
        description = [
            f"Graph Ontology Analysis Summary:",
            f"This graph contains {total_nodes} nodes of {total_node_types} different types, connected by {total_edges} edges of {total_edge_types} different types.",
            f"There are {total_relationship_patterns} distinct relationship patterns connecting these types."
        ]

        if summary["node_types"]["most_frequent"]:
            top_type = summary["node_types"]["most_frequent"][0]
            description.append(
                f"The most common node type is '{top_type['type']}', which represents {top_type['percentage']:.1f}% of all nodes."
            )

        if summary["edge_types"]["most_frequent"]:
            top_edge = summary["edge_types"]["most_frequent"][0]
            description.append(
                f"The most common edge type is '{top_edge['type']}', which represents {top_edge['percentage']:.1f}% of all edges."
            )

        if summary["relationships"]["common_patterns"]:
            top_pattern = summary["relationships"]["common_patterns"][0]
            description.append(
                f"The most common relationship pattern is '{top_pattern['pattern']}', occurring {top_pattern['count']} times."
            )

        if summary["structure"]["central_types"]:
            central_type = summary["structure"]["central_types"][0]
            description.append(
                f"The most central type in the graph is '{central_type['type']}' with {central_type['connections']} connections to other types."
            )

        if summary["structure"]["hierarchy_indicators"]:
            description.append(
                f"The graph contains hierarchical structures, with {len(summary['structure']['hierarchy_indicators'])} potential hierarchy patterns."
            )

        summary["overview"]["description"] = " ".join(description)

        return summary

    def suggest_ontology_improvements(self):
        """
        Suggest improvements to the current ontology structure

        This method analyzes the ontology and identifies potential improvements
        in structure, naming, properties, and relationships.

        Returns:
            Dict: A collection of suggested improvements

        Raises:
            ValueError: If the ontology hasn't been extracted
        """
        if not self.ontology_extracted:
            self.extract_ontology()

        # Create structure for suggestions
        suggestions = {
            "summary": "Ontology Improvement Suggestions",
            "improvements": [],
            "categories": {
                "naming": [],
                "types": [],
                "properties": [],
                "relationships": [],
                "structure": []
            }
        }

        # Check for naming inconsistencies
        naming_patterns = {
            "snake_case": 0,
            "camelCase": 0,
            "PascalCase": 0,
            "lowercase": 0,
            "UPPERCASE": 0,
            "unknown": 0
        }

        # Analyze type naming patterns
        for node_type in self.node_types:
            if node_type.islower() and "_" in node_type:
                naming_patterns["snake_case"] += 1
            elif node_type.islower() and " " not in node_type and "_" not in node_type:
                naming_patterns["lowercase"] += 1
            elif node_type.isupper():
                naming_patterns["UPPERCASE"] += 1
            elif node_type[0].isupper() and "_" not in node_type and " " not in node_type:
                if any(c.isupper() for c in node_type[1:]):
                    naming_patterns["PascalCase"] += 1
                else:
                    naming_patterns["unknown"] += 1
            elif not node_type[0].isupper() and any(c.isupper() for c in node_type):
                naming_patterns["camelCase"] += 1
            else:
                naming_patterns["unknown"] += 1

        # Check edge type naming patterns as well
        for edge_type in self.edge_types:
            if edge_type.islower() and "_" in edge_type:
                naming_patterns["snake_case"] += 1
            elif edge_type.islower() and " " not in edge_type and "_" not in edge_type:
                naming_patterns["lowercase"] += 1
            elif edge_type.isupper():
                naming_patterns["UPPERCASE"] += 1
            elif edge_type[0].isupper() and "_" not in edge_type and " " not in edge_type:
                if any(c.isupper() for c in edge_type[1:]):
                    naming_patterns["PascalCase"] += 1
                else:
                    naming_patterns["unknown"] += 1
            elif not edge_type[0].isupper() and any(c.isupper() for c in edge_type):
                naming_patterns["camelCase"] += 1
            else:
                naming_patterns["unknown"] += 1

        # Check if naming is inconsistent
        total_types = len(self.node_types) + len(self.edge_types)
        most_common_style = max(naming_patterns.items(), key=lambda x: x[1])

        if most_common_style[1] < total_types * 0.7:  # Less than 70% consistency
            suggestions["categories"]["naming"].append({
                "type": "naming_inconsistency",
                "severity": "medium",
                "description": "Type naming is inconsistent across the ontology",
                "details": f"Multiple naming styles detected; most common is {most_common_style[0]} ({most_common_style[1]} types)",
                "recommendation": f"Standardize on {most_common_style[0]} for all type names"
            })

            suggestions["improvements"].append({
                "category": "naming",
                "description": f"Standardize type naming using {most_common_style[0]} style",
                "impact": "Improves readability and consistency"
            })

        # Check for sparse node types (very few instances)
        total_nodes = sum(info["count"] for info in self.node_types.values())

        for node_type, info in self.node_types.items():
            # If type has very few instances relative to the graph (<0.5%)
            if info["count"] < max(3, total_nodes * 0.005):
                suggestions["categories"]["types"].append({
                    "type": "sparse_node_type",
                    "severity": "low",
                    "description": f"Node type '{node_type}' has very few instances ({info['count']})",
                    "details": f"Only {info['count']} nodes of type {node_type} in a graph with {total_nodes} nodes",
                    "recommendation": f"Consider merging '{node_type}' with a related type or removing it if not essential"
                })

                suggestions["improvements"].append({
                    "category": "types",
                    "description": f"Consider consolidating sparse node type '{node_type}'",
                    "impact": "Simplifies the ontology model"
                })

        # Check for property naming inconsistencies
        for node_type, info in self.node_types.items():
            property_naming = {}

            # Collect property names for this type
            for prop in info["properties"]:
                style = "unknown"

                if prop.islower() and "_" in prop:
                    style = "snake_case"
                elif prop.islower() and " " not in prop and "_" not in prop:
                    style = "lowercase"
                elif prop.isupper():
                    style = "UPPERCASE"
                elif prop[0].isupper() and "_" not in prop and " " not in prop:
                    style = "PascalCase"
                elif not prop[0].isupper() and any(c.isupper() for c in prop):
                    style = "camelCase"

                if style not in property_naming:
                    property_naming[style] = []

                property_naming[style].append(prop)

            # Check if this type uses multiple naming styles
            if len(property_naming) > 1 and len(info["properties"]) > 3:
                # Find the most common style
                most_common = max(property_naming.items(), key=lambda x: len(x[1]))

                suggestions["categories"]["properties"].append({
                    "type": "inconsistent_property_naming",
                    "severity": "low",
                    "description": f"Type '{node_type}' uses inconsistent property naming styles",
                    "details": f"Properties use {len(property_naming)} different naming styles",
                    "recommendation": f"Standardize on {most_common[0]} style for all properties in '{node_type}'"
                })

        # Look for missing required properties
        for node_type, prop_stats in self.node_property_stats.items():
            for prop_name, stats in prop_stats.items():
                # If property is almost always present but not always
                if 0.9 <= stats.get("present_count", 0) / self.node_types[node_type]["count"] < 1.0:
                    suggestions["categories"]["properties"].append({
                        "type": "incomplete_required_property",
                        "severity": "medium",
                        "description": f"Property '{prop_name}' is usually present in '{node_type}' but missing in some instances",
                        "details": f"Present in {stats.get('present_count', 0)} of {self.node_types[node_type]['count']} instances",
                        "recommendation": f"Make '{prop_name}' a required property for all '{node_type}' nodes"
                    })

                    suggestions["improvements"].append({
                        "category": "properties",
                        "description": f"Ensure required property '{prop_name}' is present in all '{node_type}' nodes",
                        "impact": "Improves data completeness and consistency"
                    })

        # Check for unnamed edge types
        if "unspecified" in self.edge_types:
            suggestions["categories"]["types"].append({
                "type": "unnamed_edge_type",
                "severity": "high",
                "description": "Some edges have no specific type ('unspecified')",
                "details": f"{self.edge_types['unspecified']['count']} edges have no specific type",
                "recommendation": "Assign meaningful edge types to all edges"
            })

            suggestions["improvements"].append({
                "category": "types",
                "description": "Replace generic 'unspecified' edge types with meaningful types",
                "impact": "Significantly improves graph semantics and queryability"
            })

        # Check for redundant relationship patterns
        # (Similar edge types connecting the same node types)
        node_type_connections = {}

        for (src, edge, tgt), count in self.relationships.items():
            key = (src, tgt)
            if key not in node_type_connections:
                node_type_connections[key] = []

            node_type_connections[key].append((edge, count))

        # Check for multiple edge types between the same node types
        for (src, tgt), edges in node_type_connections.items():
            if len(edges) > 1:
                # Check if edge names are similar
                edge_names = [edge for edge, _ in edges]

                if all(e1.lower() in e2.lower() or e2.lower() in e1.lower()
                       for i, e1 in enumerate(edge_names)
                       for j, e2 in enumerate(edge_names) if i < j):
                    suggestions["categories"]["relationships"].append({
                        "type": "similar_edge_types",
                        "severity": "medium",
                        "description": f"Multiple similar edge types connect '{src}' to '{tgt}'",
                        "details": f"Edge types: {', '.join(edge_names)}",
                        "recommendation": f"Consider consolidating these edge types"
                    })

                    suggestions["improvements"].append({
                        "category": "relationships",
                        "description": f"Consolidate similar edge types between '{src}' and '{tgt}'",
                        "impact": "Simplifies the relationship model"
                    })

        # Check for disconnected node types
        connected_types = set()

        for src, edge, tgt in self.relationships:
            connected_types.add(src)
            connected_types.add(tgt)

        # Find node types that aren't connected to anything
        for node_type in self.node_types:
            if node_type not in connected_types and self.node_types[node_type]["count"] > 0:
                suggestions["categories"]["structure"].append({
                    "type": "disconnected_type",
                    "severity": "high",
                    "description": f"Node type '{node_type}' is disconnected from the rest of the graph",
                    "details": f"{self.node_types[node_type]['count']} nodes of type '{node_type}' have no connections",
                    "recommendation": "Connect this type to the rest of the graph or reconsider its necessity"
                })

                suggestions["improvements"].append({
                    "category": "structure",
                    "description": f"Integrate disconnected node type '{node_type}' into the graph structure",
                    "impact": "Creates a more cohesive and complete ontology"
                })

        # Suggest hierarchy improvements
        has_hierarchies = False

        # Look for hierarchical relationship names
        hierarchy_terms = ["parent", "child", "contains", "part_of", "belongs_to", "member_of"]
        for edge_type in self.edge_types:
            if any(term in edge_type.lower() for term in hierarchy_terms):
                has_hierarchies = True
                break

        # If no explicit hierarchies but we have node "hierarchy" attributes
        has_hierarchy_attribute = False
        for node_type, info in self.node_types.items():
            if "hierarchy" in info["properties"]:
                has_hierarchy_attribute = True
                break

        if not has_hierarchies and has_hierarchy_attribute:
            suggestions["categories"]["structure"].append({
                "type": "implicit_hierarchy",
                "severity": "medium",
                "description": "Graph uses hierarchy attributes instead of explicit hierarchical relationships",
                "details": "Some node types have 'hierarchy' attributes but no hierarchical edge types exist",
                "recommendation": "Convert hierarchy attributes to explicit parent-child relationships"
            })

            suggestions["improvements"].append({
                "category": "structure",
                "description": "Convert implicit hierarchies (attributes) to explicit relationships",
                "impact": "Improves graph navigability and enables hierarchical queries"
            })

        # Suggest property type normalization
        for node_type, prop_stats in self.node_property_stats.items():
            for prop_name, stats in prop_stats.items():
                # If a property has mixed types but one dominant type
                if len(stats.get("value_types", [])) > 1:
                    primary_type = stats.get("primary_type", "unknown")

                    suggestions["categories"]["properties"].append({
                        "type": "mixed_property_types",
                        "severity": "medium",
                        "description": f"Property '{prop_name}' in '{node_type}' has mixed data types",
                        "details": f"Types: {', '.join(stats.get('value_types', []))}; Primary: {primary_type}",
                        "recommendation": f"Normalize all values to {primary_type} type"
                    })

        # Sort all improvements by category
        for category in suggestions["categories"]:
            suggestions["categories"][category].sort(key=lambda x: {
                "high": 0,
                "medium": 1,
                "low": 2
            }.get(x.get("severity", "low"), 3))

        # Update overall summary
        improvement_count = sum(len(items) for items in suggestions["categories"].values())

        severity_counts = {
            "high": sum(1 for items in suggestions["categories"].values()
                        for item in items if item.get("severity") == "high"),
            "medium": sum(1 for items in suggestions["categories"].values()
                          for item in items if item.get("severity") == "medium"),
            "low": sum(1 for items in suggestions["categories"].values()
                       for item in items if item.get("severity") == "low")
        }

        suggestions["summary"] = (
            f"Ontology Improvement Suggestions: {improvement_count} possible improvements identified "
            f"({severity_counts['high']} high, {severity_counts['medium']} medium, {severity_counts['low']} low priority)"
        )

        return suggestions

    def generate_validation_rules(self):
        """
        Generate validation rules based on observed patterns

        This method analyzes the ontology to create validation rules that
        can be used to ensure data quality and consistency.

        Returns:
            Dict: A collection of validation rules organized by type

        Raises:
            ValueError: If the ontology hasn't been extracted
        """
        if not self.ontology_extracted:
            self.extract_ontology()

        # Structure to store validation rules
        validation_rules = {
            "summary": "Validation Rules Generated from Ontology Patterns",
            "node_type_rules": {},
            "edge_type_rules": {},
            "relationship_rules": [],
            "cross_property_rules": [],
            "global_rules": []
        }

        # Generate node type rules
        for node_type, type_info in self.node_types.items():
            # Skip if no instances or properties
            if type_info["count"] == 0 or not type_info["properties"]:
                continue

            # Create rule set for this node type
            node_rules = {
                "type": node_type,
                "required_properties": [],
                "property_formats": {},
                "property_ranges": {},
                "property_enums": {},
                "property_patterns": {}
            }

            # Get property stats for this type
            if node_type in self.node_property_stats:
                prop_stats = self.node_property_stats[node_type]

                for prop_name, stats in prop_stats.items():
                    # Skip type property (it's inherently required)
                    if prop_name == "type":
                        continue

                    # Check if property is required
                    if stats.get("present_count", 0) == type_info["count"]:
                        node_rules["required_properties"].append(prop_name)
                    elif stats.get("present_count", 0) > type_info["count"] * 0.9:
                        # Usually present but not always
                        node_rules["required_properties"].append({
                            "property": prop_name,
                            "required": "preferred",  # Not strictly required but preferred
                            "presence_rate": stats.get("present_count", 0) / type_info["count"] * 100
                        })

                    # Add property format patterns
                    prop_type = stats.get("primary_type", "unknown")
                    pattern_type = stats.get("pattern_type", "unknown")

                    # String formats
                    if prop_type in ("str", "string"):
                        if pattern_type == "email":
                            node_rules["property_formats"][prop_name] = "email"
                        elif pattern_type == "url":
                            node_rules["property_formats"][prop_name] = "url"
                        elif pattern_type == "date":
                            node_rules["property_formats"][prop_name] = "date"
                        elif "min_length" in stats and "max_length" in stats:
                            # String length constraints
                            if stats["min_length"] == stats["max_length"]:
                                node_rules["property_patterns"][prop_name] = {
                                    "exact_length": stats["min_length"]
                                }
                            else:
                                node_rules["property_patterns"][prop_name] = {
                                    "min_length": stats["min_length"],
                                    "max_length": stats["max_length"]
                                }

                    # Numeric ranges
                    elif prop_type in ("int", "float", "integer"):
                        if "min_value" in stats and "max_value" in stats:
                            node_rules["property_ranges"][prop_name] = {
                                "min": stats["min_value"],
                                "max": stats["max_value"],
                                "type": "integer" if pattern_type == "integer" else "number"
                            }

                    # Enum values
                    if pattern_type == "enumeration" and "enum_values" in stats:
                        node_rules["property_enums"][prop_name] = stats["enum_values"]

            # Add rules for this node type if we have any
            if (node_rules["required_properties"] or
                    node_rules["property_formats"] or
                    node_rules["property_ranges"] or
                    node_rules["property_enums"] or
                    node_rules["property_patterns"]):
                validation_rules["node_type_rules"][node_type] = node_rules

        # Generate edge type rules
        for edge_type, type_info in self.edge_types.items():
            # Skip if no instances or properties
            if type_info["count"] == 0 or not type_info["properties"]:
                continue

            # Create rule set for this edge type
            edge_rules = {
                "type": edge_type,
                "required_properties": [],
                "property_formats": {},
                "property_ranges": {},
                "property_enums": {},
                "property_patterns": {}
            }

            # Get property stats for this type
            if edge_type in self.edge_property_stats:
                prop_stats = self.edge_property_stats[edge_type]

                for prop_name, stats in prop_stats.items():
                    # Skip edge_type property (it's inherently required)
                    if prop_name == "edge_type":
                        continue

                    # Apply the same property rules as for node types
                    # (Similar code to node type property rules)
                    # Check if property is required
                    if stats.get("present_count", 0) == type_info["count"]:
                        edge_rules["required_properties"].append(prop_name)
                    elif stats.get("present_count", 0) > type_info["count"] * 0.9:
                        # Usually present but not always
                        edge_rules["required_properties"].append({
                            "property": prop_name,
                            "required": "preferred",
                            "presence_rate": stats.get("present_count", 0) / type_info["count"] * 100
                        })

                    # Add other property rules (formats, ranges, enums, patterns)
                    # Following the same pattern as node type properties
                    prop_type = stats.get("primary_type", "unknown")
                    pattern_type = stats.get("pattern_type", "unknown")

                    # String formats
                    if prop_type in ("str", "string"):
                        # Same string format rules as for node types
                        # ...
                        if pattern_type == "email":
                            edge_rules["property_formats"][prop_name] = "email"
                        elif pattern_type == "url":
                            edge_rules["property_formats"][prop_name] = "url"
                        elif pattern_type == "date":
                            edge_rules["property_formats"][prop_name] = "date"
                        elif "min_length" in stats and "max_length" in stats:
                            # String length constraints
                            if stats["min_length"] == stats["max_length"]:
                                edge_rules["property_patterns"][prop_name] = {
                                    "exact_length": stats["min_length"]
                                }
                            else:
                                edge_rules["property_patterns"][prop_name] = {
                                    "min_length": stats["min_length"],
                                    "max_length": stats["max_length"]
                                }

                    # Numeric ranges
                    elif prop_type in ("int", "float", "integer"):
                        # Same numeric range rules as for node types
                        # ...
                        if "min_value" in stats and "max_value" in stats:
                            edge_rules["property_ranges"][prop_name] = {
                                "min": stats["min_value"],
                                "max": stats["max_value"],
                                "type": "integer" if pattern_type == "integer" else "number"
                            }

                    # Enum values
                    if pattern_type == "enumeration" and "enum_values" in stats:
                        edge_rules["property_enums"][prop_name] = stats["enum_values"]

            # Add rules for this edge type if we have any
            if (edge_rules["required_properties"] or
                    edge_rules["property_formats"] or
                    edge_rules["property_ranges"] or
                    edge_rules["property_enums"] or
                    edge_rules["property_patterns"]):
                validation_rules["edge_type_rules"][edge_type] = edge_rules

        # Generate relationship rules (valid node type -> edge type -> node type patterns)
        valid_relationships = []

        for (src_type, edge_type, tgt_type), count in self.relationships.items():
            # Only include relationships with significant occurrence
            if count >= 3:  # Arbitrary threshold for significance
                valid_relationships.append({
                    "source_type": src_type,
                    "edge_type": edge_type,
                    "target_type": tgt_type,
                    "count": count
                })

        # Add relationship rules if any
        if valid_relationships:
            validation_rules["relationship_rules"].append({
                "rule_type": "valid_relationships",
                "description": "Edges must connect valid node type combinations",
                "valid_combinations": valid_relationships
            })

        # Generate cardinality rules
        for (src_type, edge_type, tgt_type), count in self.relationships.items():
            # Skip if too few instances
            if count < 5:
                continue

            try:
                # Try to analyze cardinality
                cardinality = self.analyze_relationship_cardinality(src_type, edge_type, tgt_type)

                # Get cardinality type
                cardinality_type = cardinality["cardinality"]["type"]

                # Add cardinality constraint if it's not many-to-many
                if cardinality_type != "many-to-many":
                    validation_rules["relationship_rules"].append({
                        "rule_type": "cardinality_constraint",
                        "description": f"Relationship {src_type}→{edge_type}→{tgt_type} has {cardinality_type} cardinality",
                        "source_type": src_type,
                        "edge_type": edge_type,
                        "target_type": tgt_type,
                        "cardinality": cardinality_type
                    })
            except:
                # Skip if cardinality analysis fails
                pass

        # Find cross-property constraints (correlations between properties)
        for node_type, type_info in self.node_types.items():
            # Skip if too few instances
            if type_info["count"] < 10:
                continue

            # Get all node IDs of this type
            node_ids = type_info["nodes"]

            # Get all property names for this type
            properties = list(type_info["properties"])

            # Skip if too few properties
            if len(properties) < 2:
                continue

            # Check for co-occurrence patterns between properties
            for i, prop1 in enumerate(properties):
                for j, prop2 in enumerate(properties):
                    if i >= j:  # Skip self-comparison and duplicate comparisons
                        continue

                    # Count co-occurrences
                    both_present = 0
                    only_prop1 = 0
                    only_prop2 = 0

                    for node_id in node_ids:
                        node_data = self.graph_manager.node_data.get(node_id, {})
                        has_prop1 = prop1 in node_data
                        has_prop2 = prop2 in node_data

                        if has_prop1 and has_prop2:
                            both_present += 1
                        elif has_prop1:
                            only_prop1 += 1
                        elif has_prop2:
                            only_prop2 += 1

                    # Check for strong co-occurrence patterns
                    total_with_either = both_present + only_prop1 + only_prop2

                    if total_with_either > 0:
                        # If they almost always occur together
                        if both_present > 0.9 * total_with_either:
                            validation_rules["cross_property_rules"].append({
                                "rule_type": "property_co_occurrence",
                                "description": f"Properties '{prop1}' and '{prop2}' in '{node_type}' tend to appear together",
                                "node_type": node_type,
                                "properties": [prop1, prop2],
                                "co_occurrence_rate": both_present / total_with_either * 100
                            })

                        # If one implies the other but not vice versa
                        elif only_prop1 == 0 and only_prop2 > 0:
                            # prop1 implies prop2
                            validation_rules["cross_property_rules"].append({
                                "rule_type": "property_implication",
                                "description": f"When '{prop1}' is present in '{node_type}', '{prop2}' must also be present",
                                "node_type": node_type,
                                "implying_property": prop1,
                                "implied_property": prop2
                            })
                        elif only_prop2 == 0 and only_prop1 > 0:
                            # prop2 implies prop1
                            validation_rules["cross_property_rules"].append({
                                "rule_type": "property_implication",
                                "description": f"When '{prop2}' is present in '{node_type}', '{prop1}' must also be present",
                                "node_type": node_type,
                                "implying_property": prop2,
                                "implied_property": prop1
                            })

        # Add global rules
        validation_rules["global_rules"].append({
            "rule_type": "valid_node_types",
            "description": "Nodes must have a valid type",
            "valid_types": list(self.node_types.keys())
        })

        validation_rules["global_rules"].append({
            "rule_type": "valid_edge_types",
            "description": "Edges must have a valid type",
            "valid_types": list(self.edge_types.keys())
        })

        validation_rules["global_rules"].append({
            "rule_type": "edge_endpoints",
            "description": "Edge endpoints must be valid node IDs",
            "validation": "All edge source and target IDs must exist as nodes"
        })

        return validation_rules

    def find_inconsistencies(self):
        """
        Find inconsistencies in the graph based on ontology patterns

        This method analyzes the graph data against the patterns identified
        in the ontology and highlights violations of expected patterns.

        Returns:
            Dict: A collection of identified inconsistencies

        Raises:
            ValueError: If the ontology hasn't been extracted
        """
        if not self.ontology_extracted:
            self.extract_ontology()

        # Generate validation rules
        validation_rules = self.generate_validation_rules()

        # Structure to store inconsistencies
        inconsistencies = {
            "summary": "Graph Inconsistencies Based on Ontology Patterns",
            "node_inconsistencies": [],
            "edge_inconsistencies": [],
            "relationship_inconsistencies": [],
            "property_inconsistencies": [],
            "severity_counts": {
                "high": 0,
                "medium": 0,
                "low": 0
            }
        }

        # Check node type inconsistencies
        for node_id, node_data in self.graph_manager.node_data.items():
            node_type = node_data.get("type", "unknown")

            # Check if node has a valid type
            if node_type not in self.node_types:
                inconsistencies["node_inconsistencies"].append({
                    "inconsistency_type": "invalid_node_type",
                    "severity": "high",
                    "node_id": node_id,
                    "node_type": node_type,
                    "description": f"Node has an invalid type: '{node_type}'",
                    "valid_types": list(self.node_types.keys())
                })
                inconsistencies["severity_counts"]["high"] += 1
                continue

            # Check node property inconsistencies against rules
            if node_type in validation_rules["node_type_rules"]:
                rule_set = validation_rules["node_type_rules"][node_type]

                # Check required properties
                for prop_req in rule_set["required_properties"]:
                    # Handle both string property names and property objects with required flag
                    if isinstance(prop_req, str):
                        prop_name = prop_req
                        required = "required"
                    else:
                        prop_name = prop_req["property"]
                        required = prop_req["required"]

                    if prop_name not in node_data:
                        severity = "high" if required == "required" else "medium"
                        inconsistencies["property_inconsistencies"].append({
                            "inconsistency_type": "missing_required_property",
                            "severity": severity,
                            "node_id": node_id,
                            "node_type": node_type,
                            "property": prop_name,
                            "requirement": required,
                            "description": f"Node is missing {required} property: '{prop_name}'"
                        })
                        inconsistencies["severity_counts"][severity] += 1

                # Check property formats
                for prop_name, format_type in rule_set["property_formats"].items():
                    if prop_name in node_data:
                        value = node_data[prop_name]

                        # Skip if not a string
                        if not isinstance(value, str):
                            continue

                        # Check format
                        valid = True

                        if format_type == "email":
                            import re
                            email_pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'
                            valid = bool(re.match(email_pattern, value))
                        elif format_type == "url":
                            import re
                            url_pattern = r'^https?://\S+$'
                            valid = bool(re.match(url_pattern, value))
                        elif format_type == "date":
                            import re
                            date_pattern = r'^\d{4}-\d{2}-\d{2}$'
                            valid = bool(re.match(date_pattern, value))

                        if not valid:
                            inconsistencies["property_inconsistencies"].append({
                                "inconsistency_type": "invalid_property_format",
                                "severity": "medium",
                                "node_id": node_id,
                                "node_type": node_type,
                                "property": prop_name,
                                "value": str(value),
                                "expected_format": format_type,
                                "description": f"Property '{prop_name}' value does not match expected format: {format_type}"
                            })
                            inconsistencies["severity_counts"]["medium"] += 1

                # Check property ranges
                for prop_name, range_info in rule_set["property_ranges"].items():
                    if prop_name in node_data:
                        value = node_data[prop_name]

                        # Try to convert to number if needed
                        try:
                            if isinstance(value, str):
                                value = float(value)
                        except:
                            # Skip if conversion fails
                            continue

                        # Check range
                        min_val = range_info.get("min")
                        max_val = range_info.get("max")

                        if (min_val is not None and value < min_val) or (max_val is not None and value > max_val):
                            inconsistencies["property_inconsistencies"].append({
                                "inconsistency_type": "out_of_range_property",
                                "severity": "medium",
                                "node_id": node_id,
                                "node_type": node_type,
                                "property": prop_name,
                                "value": value,
                                "expected_range": f"{min_val} to {max_val}",
                                "description": f"Property '{prop_name}' value {value} is outside expected range: {min_val} to {max_val}"
                            })
                            inconsistencies["severity_counts"]["medium"] += 1

                # Check property enums
                for prop_name, enum_values in rule_set["property_enums"].items():
                    if prop_name in node_data:
                        value = node_data[prop_name]

                        # Convert value to string for comparison
                        str_value = str(value)

                        if str_value not in [str(v) for v in enum_values]:
                            inconsistencies["property_inconsistencies"].append({
                                "inconsistency_type": "invalid_enum_value",
                                "severity": "medium",
                                "node_id": node_id,
                                "node_type": node_type,
                                "property": prop_name,
                                "value": str_value,
                                "allowed_values": [str(v) for v in enum_values],
                                "description": f"Property '{prop_name}' has value '{str_value}' which is not in allowed set: {', '.join(str(v) for v in enum_values)}"
                            })
                            inconsistencies["severity_counts"]["medium"] += 1

                # Check property patterns
                for prop_name, pattern_info in rule_set["property_patterns"].items():
                    if prop_name in node_data:
                        value = node_data[prop_name]

                        # Skip if not a string
                        if not isinstance(value, str):
                            continue

                        # Check length constraints
                        if "exact_length" in pattern_info:
                            expected_len = pattern_info["exact_length"]
                            if len(value) != expected_len:
                                inconsistencies["property_inconsistencies"].append({
                                    "inconsistency_type": "invalid_property_length",
                                    "severity": "low",
                                    "node_id": node_id,
                                    "node_type": node_type,
                                    "property": prop_name,
                                    "value": value,
                                    "actual_length": len(value),
                                    "expected_length": expected_len,
                                    "description": f"Property '{prop_name}' value has length {len(value)}, expected exactly {expected_len}"
                                })
                                inconsistencies["severity_counts"]["low"] += 1
                        elif "min_length" in pattern_info and "max_length" in pattern_info:
                            min_len = pattern_info["min_length"]
                            max_len = pattern_info["max_length"]

                            if len(value) < min_len or len(value) > max_len:
                                inconsistencies["property_inconsistencies"].append({
                                    "inconsistency_type": "invalid_property_length",
                                    "severity": "low",
                                    "node_id": node_id,
                                    "node_type": node_type,
                                    "property": prop_name,
                                    "value": value,
                                    "actual_length": len(value),
                                    "expected_range": f"{min_len} to {max_len}",
                                    "description": f"Property '{prop_name}' value has length {len(value)}, expected between {min_len} and {max_len}"
                                })
                                inconsistencies["severity_counts"]["low"] += 1

        # Check edge type inconsistencies
        for edge_key, edge_data in self.graph_manager.edge_data.items():
            source, target = edge_key
            edge_type = edge_data.get("edge_type", "unknown")

            # Check if edge has valid endpoints
            if source not in self.graph_manager.node_data:
                inconsistencies["edge_inconsistencies"].append({
                    "inconsistency_type": "invalid_edge_source",
                    "severity": "high",
                    "edge_key": edge_key,
                    "edge_type": edge_type,
                    "source": source,
                    "description": f"Edge has a source node that doesn't exist: '{source}'"
                })
                inconsistencies["severity_counts"]["high"] += 1

            if target not in self.graph_manager.node_data:
                inconsistencies["edge_inconsistencies"].append({
                    "inconsistency_type": "invalid_edge_target",
                    "severity": "high",
                    "edge_key": edge_key,
                    "edge_type": edge_type,
                    "target": target,
                    "description": f"Edge has a target node that doesn't exist: '{target}'"
                })
                inconsistencies["severity_counts"]["high"] += 1

            # Skip further checks if endpoints invalid
            if source not in self.graph_manager.node_data or target not in self.graph_manager.node_data:
                continue

            # Check if edge has a valid type
            if edge_type not in self.edge_types:
                inconsistencies["edge_inconsistencies"].append({
                    "inconsistency_type": "invalid_edge_type",
                    "severity": "high",
                    "edge_key": edge_key,
                    "edge_type": edge_type,
                    "description": f"Edge has an invalid type: '{edge_type}'",
                    "valid_types": list(self.edge_types.keys())
                })
                inconsistencies["severity_counts"]["high"] += 1
                continue

            # Check relationship pattern validity
            source_type = self.graph_manager.node_data[source].get("type", "unknown")
            target_type = self.graph_manager.node_data[target].get("type", "unknown")
            relationship_key = (source_type, edge_type, target_type)

            if relationship_key not in self.relationships:
                # Check if this relationship pattern is observed in the ontology
                inconsistencies["relationship_inconsistencies"].append({
                    "inconsistency_type": "invalid_relationship_pattern",
                    "severity": "medium",
                    "edge_key": edge_key,
                    "edge_type": edge_type,
                    "source_type": source_type,
                    "target_type": target_type,
                    "description": f"Relationship pattern {source_type}→{edge_type}→{target_type} is not observed in the ontology"
                })
                inconsistencies["severity_counts"]["medium"] += 1

            # Check edge property inconsistencies against rules
            if edge_type in validation_rules["edge_type_rules"]:
                rule_set = validation_rules["edge_type_rules"][edge_type]

                # Similar property checks as for nodes
                # Check required properties
                for prop_req in rule_set["required_properties"]:
                    if isinstance(prop_req, str):
                        prop_name = prop_req
                        required = "required"
                    else:
                        prop_name = prop_req["property"]
                        required = prop_req["required"]

                    if prop_name not in edge_data:
                        severity = "high" if required == "required" else "medium"
                        inconsistencies["property_inconsistencies"].append({
                            "inconsistency_type": "missing_required_property",
                            "severity": severity,
                            "edge_key": edge_key,
                            "edge_type": edge_type,
                            "property": prop_name,
                            "requirement": required,
                            "description": f"Edge is missing {required} property: '{prop_name}'"
                        })
                        inconsistencies["severity_counts"][severity] += 1

                # Check other property rules for edges
                # (Similar to node property checks)

        # Check relationship cardinality inconsistencies
        for rule in validation_rules.get("relationship_rules", []):
            if rule.get("rule_type") == "cardinality_constraint":
                src_type = rule["source_type"]
                edge_type = rule["edge_type"]
                tgt_type = rule["target_type"]
                cardinality_type = rule["cardinality"]

                # Get all instances of this relationship
                src_nodes = [node_id for node_id, data in self.graph_manager.node_data.items()
                             if data.get("type") == src_type]
                tgt_nodes = [node_id for node_id, data in self.graph_manager.node_data.items()
                             if data.get("type") == tgt_type]

                if cardinality_type == "one-to-one":
                    # Check for violations (a source connected to multiple targets)
                    for src_node in src_nodes:
                        targets = []
                        for (s, t), data in self.graph_manager.edge_data.items():
                            if (s == src_node and
                                    data.get("edge_type") == edge_type and
                                    self.graph_manager.node_data.get(t, {}).get("type") == tgt_type):
                                targets.append(t)

                        if len(targets) > 1:
                            inconsistencies["relationship_inconsistencies"].append({
                                "inconsistency_type": "cardinality_violation",
                                "severity": "medium",
                                "node_id": src_node,
                                "node_type": src_type,
                                "edge_type": edge_type,
                                "target_type": tgt_type,
                                "cardinality_type": cardinality_type,
                                "violation": "one-to-many",
                                "targets": targets,
                                "description": f"Node '{src_node}' has multiple '{edge_type}' edges to '{tgt_type}' nodes, violating one-to-one cardinality"
                            })
                            inconsistencies["severity_counts"]["medium"] += 1

                    # Check in the other direction (a target connected to multiple sources)
                    for tgt_node in tgt_nodes:
                        sources = []
                        for (s, t), data in self.graph_manager.edge_data.items():
                            if (t == tgt_node and
                                    data.get("edge_type") == edge_type and
                                    self.graph_manager.node_data.get(s, {}).get("type") == src_type):
                                sources.append(s)

                        if len(sources) > 1:
                            inconsistencies["relationship_inconsistencies"].append({
                                "inconsistency_type": "cardinality_violation",
                                "severity": "medium",
                                "node_id": tgt_node,
                                "node_type": tgt_type,
                                "edge_type": edge_type,
                                "source_type": src_type,
                                "cardinality_type": cardinality_type,
                                "violation": "many-to-one",
                                "sources": sources,
                                "description": f"Node '{tgt_node}' has multiple incoming '{edge_type}' edges from '{src_type}' nodes, violating one-to-one cardinality"
                            })
                            inconsistencies["severity_counts"]["medium"] += 1

                elif cardinality_type == "one-to-many":
                    # Check for violations (a source connected to multiple targets)
                    # In this case, one source can connect to many targets, so no violation here

                    # Check for violations (a target connected to multiple sources)
                    for tgt_node in tgt_nodes:
                        sources = []
                        for (s, t), data in self.graph_manager.edge_data.items():
                            if (t == tgt_node and
                                    data.get("edge_type") == edge_type and
                                    self.graph_manager.node_data.get(s, {}).get("type") == src_type):
                                sources.append(s)

                        if len(sources) > 1:
                            inconsistencies["relationship_inconsistencies"].append({
                                "inconsistency_type": "cardinality_violation",
                                "severity": "medium",
                                "node_id": tgt_node,
                                "node_type": tgt_type,
                                "edge_type": edge_type,
                                "source_type": src_type,
                                "cardinality_type": cardinality_type,
                                "violation": "many-to-one",
                                "sources": sources,
                                "description": f"Node '{tgt_node}' has multiple incoming '{edge_type}' edges from '{src_type}' nodes, violating one-to-many cardinality"
                            })
                            inconsistencies["severity_counts"]["medium"] += 1

                elif cardinality_type == "many-to-one":
                    # Check for violations (a source connected to multiple targets)
                    for src_node in src_nodes:
                        targets = []
                        for (s, t), data in self.graph_manager.edge_data.items():
                            if (s == src_node and
                                    data.get("edge_type") == edge_type and
                                    self.graph_manager.node_data.get(t, {}).get("type") == tgt_type):
                                targets.append(t)

                        if len(targets) > 1:
                            inconsistencies["relationship_inconsistencies"].append({
                                "inconsistency_type": "cardinality_violation",
                                "severity": "medium",
                                "node_id": src_node,
                                "node_type": src_type,
                                "edge_type": edge_type,
                                "target_type": tgt_type,
                                "cardinality_type": cardinality_type,
                                "violation": "one-to-many",
                                "targets": targets,
                                "description": f"Node '{src_node}' has multiple '{edge_type}' edges to '{tgt_type}' nodes, violating many-to-one cardinality"
                            })
                            inconsistencies["severity_counts"]["medium"] += 1

        # Check cross-property consistency rules
        for rule in validation_rules.get("cross_property_rules", []):
            if rule["rule_type"] == "property_implication":
                node_type = rule["node_type"]
                implying_property = rule["implying_property"]
                implied_property = rule["implied_property"]

                # Find all nodes of this type
                for node_id, node_data in self.graph_manager.node_data.items():
                    if node_data.get("type") == node_type:
                        # Check if rule is violated (implying property exists but implied doesn't)
                        if implying_property in node_data and implied_property not in node_data:
                            inconsistencies["property_inconsistencies"].append({
                                "inconsistency_type": "property_dependency_violation",
                                "severity": "medium",
                                "node_id": node_id,
                                "node_type": node_type,
                                "implying_property": implying_property,
                                "implied_property": implied_property,
                                "description": f"Node has property '{implying_property}' but is missing dependent property '{implied_property}'"
                            })
                            inconsistencies["severity_counts"]["medium"] += 1

        # Generate overall summary
        total_inconsistencies = (
                len(inconsistencies["node_inconsistencies"]) +
                len(inconsistencies["edge_inconsistencies"]) +
                len(inconsistencies["relationship_inconsistencies"]) +
                len(inconsistencies["property_inconsistencies"])
        )

        severity_summary = (
                f"{inconsistencies['severity_counts']['high']} high, " +
                f"{inconsistencies['severity_counts']['medium']} medium, " +
                f"{inconsistencies['severity_counts']['low']} low severity"
        )

        inconsistencies["summary"] = (
                f"Found {total_inconsistencies} inconsistencies in the graph " +
                f"({severity_summary})"
        )

        return inconsistencies