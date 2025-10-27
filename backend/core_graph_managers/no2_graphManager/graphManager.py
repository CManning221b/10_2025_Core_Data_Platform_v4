import json

import networkit as nk

from typing import Optional, List, Any

from backend.core_graph_managers.no1_graphDataIngestor.graphDataIngestion import *


class GraphManager():
    """
    GraphManager: A Unified Interface for Graph Construction, Caching, and Visualization.

    This class ingests heterogeneous data sources and transforms them into a centralized graph structure.
    It supports efficient traversal, rule-based mutation, multi-format caching, and visualization.
    Rule application will be handled externally by a higher-level class that interfaces with GraphManager.

    Core Responsibilities:
     - JSON-based graph as the canonical representation (editable and attribute-rich)
     - Derived in-memory graph structures (e.g., Networkit, igraph) for high-performance analytics
     - Cache generation for fast lookup and traversal:
         - Outgoing and incoming edge maps
         - Node neighborhoods
         - Tree and subtree hierarchies
         - Node and edge types
         - Frequently accessed nodes
         - Structural motifs and repeated sub-graphs
     - Graph update and sync routines to maintain consistency across caches and representations

    Supported Inputs:
     - Node and edge dictionaries (from GraphIngestor)
     - Pre-structured JSON graph (from GraphIngestor or external sources)

    Representation Backends:
     - PyVis (interactive HTML rendering)
     - Dash Cytoscape (React-based frontend for web integration)
     - Networkit (high-performance graph analytics)
     - Igraph (large-scale graph analytics and layout algorithms)
     - Netgraph, Jaal (alternative rendering and inspection options)

    Graph Transformations:
     - Add or remove nodes
     - Edit, add, or remove node attributes
     - Collapse edges into attributes (e.g., triplestore → property graph)
     - Expand attributes back into edges
     - Query nodes, edges, attributes, and motif-based subgraphs
     - Detect specific structures such as trees and cycles

    Optional Integrations:
     - Neo4j sync for persistent storage and Cypher-based querying

    Outputs:
     - JSON representation: {node_id: {...}, (source_id, target_id): {...}}
     - Computation-ready graph structures (e.g., Networkit, igraph)
     - Multiple visualization backends
     - Exportable subgraphs and substructures for further manipulation

    Design Considerations:
     - Caches are built on initialization or upon structural modification
     - Selected cache layers may use `lru_cache` or memoization strategies
     - Rules and transformations are applied via in-memory data structures for efficiency

    Intended Use:
     - Large-scale graph ingestion, traversal, and querying
     - Multi-view visualization and interactive exploration
     - Rule-based mutation, enrichment, and export of graph data
    """

    def __init__(self, preload: bool = True):
        self.node_data = {}  # Dict[node_id] = {attr}
        self.edge_data = {}  # Dict[(src, tgt)] = {attr}
        self.graph_json_address = '' # Address of JSON file that has node_data, edge_data.

        self.outgoing_edges_cache = {} # Fast cache for nodeId -> Outgoing edges
        self.incoming_edges_cache = {} # Fast cache for nodeId -> Incoming edges
        self.neighborhood_cache = {} # Fast cache for nodeId -> Local Neighbourhoods
        self.node_type_cache = {} # Fast cache for nodeType -> List of Nodes
        self.edge_type_cache = {} # Fast cache for edgeType -> List of Edges
        self.motif_cache = {} # Fast Cache - Motif and Structures (Groups of types, values, cycles, trees) -> Examples of those structures

        self.nk_graph = None
        self.ig_igraph = None

        self.preload = preload
        self.consistent = False


    '''
    Core Graph Structure
    '''
    def load_from_json(self, path: str):
        '''
        Load node and edge data from a JSON file.
        '''
        with open(path, "r") as f:
            data = json.load(f)

        self.node_data = data.get("nodes", {})
        # Reverse the key transformation done during export
        self.edge_data = {
            tuple(k.split("_", 1)): v for k, v in data.get("edges", {}).items()
        }

        self.graph_json_address = path
        self._mark_inconsistent()

    def save_to_json(self,return_dict: bool = True, filepath: [str] = None):
        '''
        Convert node and edge dictionaries to serializable json files for export
        '''
        data = {
            "nodes": self.node_data,
            "edges": {f"{k[0]}_{k[1]}": v for k, v in self.edge_data.items()}  # for valid JSON keys
        }
        if return_dict:
            return data
        elif filepath:
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)

    def add_node(self, node_id: str, value, type, hierarchy, attributes: dict):
        '''
        Add a new node with standard metadata and any additional attributes.
        If the node already exists, no action is taken.
        '''
        if node_id in self.node_data:
            raise KeyError(f"Node '{node_id}' already exist.")

        node_entry = {
            "value": value,
            "type": type,
            "hierarchy": hierarchy,
            **attributes  # merge any additional metadata
        }

        self.node_data[node_id] = node_entry
        self._mark_inconsistent()

    def remove_node(self, node_id: str):
        # Remove the node
        if node_id not in self.node_data:
            raise KeyError(f"Node '{node_id}' does not exist.")

        del self.node_data[node_id]

        # Remove all outgoing edges
        outgoing = self.outgoing_edges_cache.get(node_id, [])
        for target, _ in outgoing:
            self.edge_data.pop((node_id, target), None)

        # Remove all incoming edges
        incoming = self.incoming_edges_cache.get(node_id, [])
        for source, _ in incoming:
            self.edge_data.pop((source, node_id), None)

        # Mark state as inconsistent since caches are now outdated
        self._mark_inconsistent()

    def add_edge(self, source: str, target: str, attributes: dict):
        if (source, target) in self.edge_data:
            raise ValueError(f"Edge ({source} → {target}) already exists.")

        if source not in self.node_data or target not in self.node_data:
            raise KeyError("Source or target node does not exist.")

        # Base edge attributes
        edge_attrs = {
            "edge_type": attributes.get("edge_type", "unspecified"),
            "direction": attributes.get("direction", "forward"),
            "source_type": self.node_data[source].get("type", "unknown"),
            "target_type": self.node_data[target].get("type", "unknown"),
        }

        # Merge user attributes (overriding defaults if specified)
        edge_attrs.update(attributes)

        self.edge_data[(source, target)] = edge_attrs
        self._mark_inconsistent()

    def remove_edge(self, source: str, target: str):
        edge_key = (source, target)

        if edge_key not in self.edge_data:
            raise KeyError(f"Edge ({source} → {target}) does not exist.")

        del self.edge_data[edge_key]
        self._mark_inconsistent()

    def update_node_attributes(self, node_id: str, attributes: dict):
        if node_id not in self.node_data:
            raise KeyError(f"Node '{node_id}' does not exist.")

        self.node_data[node_id].update(attributes)
        self._mark_inconsistent()

    def update_edge_attributes(self, source: str, target: str, attributes: dict):
        key = (source, target)
        if key not in self.edge_data:
            raise KeyError(f"Edge from '{source}' to '{target}' does not exist.")

        self.edge_data[key].update(attributes)
        self._mark_inconsistent()

    '''
    Eager Cache Methods
    '''

    def build_caches(self):
        """
        Build all relevant graph caches to support efficient traversal, querying, and motif detection.
        """
        self.build_outgoing_edge_cache()
        self.build_incoming_edge_cache()
        self.build_neighborhood_cache()
        self.build_node_attribute_cache()
        self.build_edge_attribute_cache()
        self.build_tree_cache()
        self.build_motif_cache()

        self._mark_consistent()

    def build_outgoing_edge_cache(self):
        self.outgoing_edges_cache = defaultdict(list)  # Changed from set to list
        for (src, tgt), attrs in self.edge_data.items():
            self.outgoing_edges_cache[src].append((tgt, attrs))  # Changed from add to append
        self._mark_inconsistent()

    def build_incoming_edge_cache(self):
        self.incoming_edges_cache = defaultdict(list)  # Changed from set to list
        for (src, tgt), attrs in self.edge_data.items():
            self.incoming_edges_cache[tgt].append((src, attrs))  # Changed from add to append
        self._mark_inconsistent()

    def build_node_attribute_cache(self):
        """
        Build a cache of node attributes where the attributes are either strings or integers with reasonable cardinality.
        """
        self.node_attribute_index = defaultdict(lambda: defaultdict(set))

        for node_id, attributes in self.node_data.items():
            for key, value in attributes.items():
                # Only index valid attributes (string or integer, reasonable cardinality)
                if self._is_valid_attribute(value):
                    # For each valid attribute, index it
                    if self._is_valid_cardinality(self.node_attribute_index[key][value], max_cardinality=100):
                        self.node_attribute_index[key][value].add(node_id)

        # Purge any keys with only one unique value
        for key in list(self.node_attribute_index.keys()):
            if len(self.node_attribute_index[key]) == 1:
                del self.node_attribute_index[key]

        self._mark_inconsistent()

    def build_edge_attribute_cache(self):
        """
        Build a cache of edge attributes where the attributes are either strings or integers with reasonable cardinality.
        """
        self.edge_attribute_index = defaultdict(lambda: defaultdict(set))

        for (source, target), attributes in self.edge_data.items():
            for key, value in attributes.items():
                # Only index valid attributes (string or integer, reasonable cardinality)
                if self._is_valid_attribute(value):
                    # For each valid attribute, index it
                    if self._is_valid_cardinality(self.edge_attribute_index[key][value], max_cardinality=100):
                        self.edge_attribute_index[key][value].add((source, target))

        # Purge any keys with only one unique value
        for key in list(self.edge_attribute_index.keys()):
            if len(self.edge_attribute_index[key]) == 1:
                del self.edge_attribute_index[key]

        self._mark_inconsistent()

    def build_neighborhood_cache(self):
        self.neighborhood_cache = {}

        # Ensure we have up-to-date caches
        if not self.outgoing_edges_cache or not self.incoming_edges_cache:
            self.build_outgoing_edge_cache()
            self.build_incoming_edge_cache()

        for node_id in self.node_data:
            neighbors = set()
            edges = set()

            # Outgoing neighbors
            for target in self.outgoing_edges_cache.get(node_id, []):
                neighbors.add(target)
                edges.add((node_id, target))

            # Incoming neighbors
            for source in self.incoming_edges_cache.get(node_id, []):
                neighbors.add(source)
                edges.add((source, node_id))

            # Induced edges between neighbors
            for src in neighbors:
                for tgt in neighbors:
                    if (src, tgt) in self.edge_data:
                        edges.add((src, tgt))

            self.neighborhood_cache[node_id] = {
                'neighbors': neighbors,
                'edges': edges
            }
        self._mark_inconsistent()

    def build_tree_cache(self):
        self.tree_cache = {}
        visited = set()

        for node in self.node_data:
            if node in visited:
                continue

            # Try to extract a tree from this root
            tree_nodes = set()
            tree_edges = set()
            stack = [(node, None)]  # (current, parent)
            local_parents = {}

            is_tree = True

            while stack:
                current, parent = stack.pop()
                if current in tree_nodes:
                    is_tree = False  # cycle detected
                    break
                tree_nodes.add(current)
                if parent:
                    tree_edges.add((parent, current))
                    if current in local_parents:
                        is_tree = False  # multiple parents
                        break
                    local_parents[current] = parent

                for child in self.outgoing_edges_cache.get(current, []):
                    if child != parent:
                        stack.append((child, current))

            if is_tree:
                for n in tree_nodes:
                    self.tree_cache[n] = {
                        'root': node,
                        'nodes': tree_nodes.copy(),
                        'edges': tree_edges.copy()
                    }

                visited.update(tree_nodes)
        self._mark_inconsistent()

    def build_motif_cache(self):
        semantic_motifs = defaultdict(list)

        for node_a in self.node_data:
            type_a = self.node_data[node_a].get('type')
            for node_b in self.outgoing_edges_cache.get(node_a, []):
                edge_ab = self.edge_data.get((node_a, node_b), {})
                type_b = self.node_data.get(node_b, {}).get('type')
                edge_type1 = edge_ab.get('edge_type')

                if type_a is None or type_b is None or edge_type1 is None:
                    continue

                for node_c in self.outgoing_edges_cache.get(node_b, []):
                    edge_bc = self.edge_data.get((node_b, node_c), {})
                    type_c = self.node_data.get(node_c, {}).get('type')
                    edge_type2 = edge_bc.get('edge_type')

                    if type_c is None or edge_type2 is None:
                        continue

                    # Signature of the motif
                    signature = (type_a, edge_type1, type_b, edge_type2, type_c)
                    instance = (node_a, node_b, node_c)
                    semantic_motifs[signature].append(instance)

        # Filter: keep only repeated motifs
        self.motif_cache = {
            sig: instances for sig, instances in semantic_motifs.items()
            if len(instances) > 1
        }
        self._mark_inconsistent()

    '''
    Lazy Cache Methods
    '''

    def lazy_build_caches(self, node_id: str):
        """
        Lazily builds all caches for the given node.
        This method updates the caches related to the node, such as:
        - Outgoing edges
        - Incoming edges
        - Neighborhood
        - Node attributes
        - Edge attributes
        - Tree structure (if applicable)
        - Motifs (if applicable)
        """
        # Build or update the outgoing edges cache for the node
        self.lazy_outgoing_edge_cache(node_id)

        # Build or update the incoming edges cache for the node
        self.lazy_incoming_edge_cache(node_id)

        # Build or update the neighborhood cache for the node
        self.lazy_neighborhood_cache(node_id)

        # Build or update the node attribute cache for the node
        self.lazy_node_attribute_cache(node_id)

        # Build or update the edge attribute cache for the node (for all its edges)
        for target, _ in self.lazy_outgoing_edge_cache(node_id):
            self.lazy_edge_attribute_cache(node_id, target)
        for source, _ in self.lazy_incoming_edge_cache(node_id):
            self.lazy_edge_attribute_cache(source, node_id)

        # Build or update the tree cache for the node
        self.lazy_tree_cache(node_id)

        # Build or update the motif cache for the node
        self.lazy_motif_cache(node_id)

        # Mark the system as inconsistent after all cache updates (if necessary)
        self._mark_inconsistent()

    def lazy_outgoing_edge_cache(self, node_id: str):
        """
        Lazily builds and returns the outgoing edges for a given node.
        Updates the outgoing edge cache for that node only.
        """
        if node_id not in self.outgoing_edges_cache:
            self.outgoing_edges_cache[node_id] = []  # Use a list instead of a set
            for (src, tgt), attrs in self.edge_data.items():
                if src == node_id:
                    self.outgoing_edges_cache[node_id].append((tgt, attrs))
        self._mark_inconsistent()
        return self.outgoing_edges_cache[node_id]

    def lazy_incoming_edge_cache(self, node_id: str):
        """
        Lazily builds and returns the incoming edges for a given node.
        Updates the incoming edge cache for that node only.
        """
        if node_id not in self.incoming_edges_cache:
            self.incoming_edges_cache[node_id] = []  # Use a list instead of a set
            for (src, tgt), attrs in self.edge_data.items():
                if tgt == node_id:
                    self.incoming_edges_cache[node_id].append((src, attrs))
        self._mark_inconsistent()
        return self.incoming_edges_cache[node_id]
    def lazy_neighborhood_cache(self, node_id: str):
        """
        Lazily builds and returns the neighborhood for a given node.
        Updates the neighborhood cache for that node only.
        """
        if node_id in self.neighborhood_cache:
            return self.neighborhood_cache[node_id]

        # Ensure edge caches are available
        outgoing = self.lazy_outgoing_edge_cache(node_id)
        incoming = self.lazy_incoming_edge_cache(node_id)

        neighbors = set()
        edges = set()

        for target, _ in outgoing:
            neighbors.add(target)
            edges.add((node_id, target))

        for source, _ in incoming:
            neighbors.add(source)
            edges.add((source, node_id))

        # Add induced edges among neighbors
        for src in neighbors:
            for tgt in neighbors:
                if (src, tgt) in self.edge_data:
                    edges.add((src, tgt))

        result = {
            'neighbors': neighbors,
            'edges': edges
        }

        self.neighborhood_cache[node_id] = result
        self._mark_inconsistent()
        return result

    def lazy_node_attribute_cache(self, node_id: str):
        """
        Lazily builds and returns the attribute cache for a given node.
        """
        if node_id in self.node_attribute_index:
            return self.node_attribute_index[node_id]

        # Create a cache entry for this node if it doesn't exist
        attributes = self.node_data.get(node_id, {})
        valid_attributes = defaultdict(set)

        for key, value in attributes.items():
            if self._is_valid_attribute(value):
                if self._is_valid_cardinality(valid_attributes[key][value], max_cardinality=100):
                    valid_attributes[key][value].add(node_id)

        # If there's only one unique value, don't cache it
        for key in list(valid_attributes.keys()):
            if len(valid_attributes[key]) == 1:
                del valid_attributes[key]

        # Store this node's attributes in the cache
        self.node_attribute_index[node_id] = valid_attributes
        self._mark_inconsistent()
        return valid_attributes

    def lazy_edge_attribute_cache(self, source: str, target: str):
        """
        Lazily builds and returns the attribute cache for the edge (source, target).
        """
        edge_key = (source, target)

        if edge_key in self.edge_attribute_index:
            return self.edge_attribute_index[edge_key]

        # Create a cache entry for this edge if it doesn't exist
        attributes = self.edge_data.get(edge_key, {})
        valid_attributes = defaultdict(set)

        for key, value in attributes.items():
            if self._is_valid_attribute(value):
                if self._is_valid_cardinality(valid_attributes[key][value], max_cardinality=100):
                    valid_attributes[key][value].add(edge_key)

        # If there's only one unique value, don't cache it
        for key in list(valid_attributes.keys()):
            if len(valid_attributes[key]) == 1:
                del valid_attributes[key]

        # Store this edge's attributes in the cache
        self.edge_attribute_index[edge_key] = valid_attributes
        self._mark_inconsistent()
        return valid_attributes

    def lazy_tree_cache(self, node_id: str):
        """
        Lazily builds and returns the tree that the given node is a part of.
        """
        if node_id in self.tree_cache:
            return self.tree_cache[node_id]

        # If the tree is not in the cache, we will build it lazily for this node.
        tree_nodes = set()
        tree_edges = set()
        stack = [(node_id, None)]  # (current, parent)
        local_parents = {}

        is_tree = True

        while stack:
            current, parent = stack.pop()
            if current in tree_nodes:
                is_tree = False  # cycle detected
                break
            tree_nodes.add(current)
            if parent:
                tree_edges.add((parent, current))
                if current in local_parents:
                    is_tree = False  # multiple parents
                    break
                local_parents[current] = parent

            for child in self.outgoing_edges_cache.get(current, []):
                if child != parent:
                    stack.append((child, current))

        if is_tree:
            tree_data = {
                'root': node_id,
                'nodes': tree_nodes,
                'edges': tree_edges
            }
            # Cache the tree for this node
            self.tree_cache[node_id] = tree_data
            self._mark_inconsistent()
            return tree_data
        else:
            # If the node doesn't form a tree, return an empty structure or None
            return None

    def lazy_motif_cache(self, node_id: str):
        """
        Lazily builds and returns the repeated motifs that involve the given node.
        Motifs are defined as type-edge-type chains.
        """
        # Check if the motifs for this node are already cached
        if node_id in self.motif_cache:
            return self.motif_cache[node_id]

        # Otherwise, build motifs lazily for this node
        semantic_motifs = defaultdict(list)

        type_a = self.node_data.get(node_id, {}).get('type')
        if type_a is None:
            return None  # No motifs if the node doesn't have a type

        for node_b in self.outgoing_edges_cache.get(node_id, []):
            edge_ab = self.edge_data.get((node_id, node_b), {})
            type_b = self.node_data.get(node_b, {}).get('type')
            edge_type1 = edge_ab.get('edge_type')

            if type_b is None or edge_type1 is None:
                continue

            for node_c in self.outgoing_edges_cache.get(node_b, []):
                edge_bc = self.edge_data.get((node_b, node_c), {})
                type_c = self.node_data.get(node_c, {}).get('type')
                edge_type2 = edge_bc.get('edge_type')

                if type_c is None or edge_type2 is None:
                    continue

                # Signature of the motif
                signature = (type_a, edge_type1, type_b, edge_type2, type_c)
                instance = (node_id, node_b, node_c)
                semantic_motifs[signature].append(instance)

        # Filter: keep only repeated motifs
        motif_data = {
            sig: instances for sig, instances in semantic_motifs.items()
            if len(instances) > 1
        }

        # Cache the motifs for this node
        self.motif_cache[node_id] = motif_data
        self._mark_inconsistent()
        return motif_data

    def batch_operation(self):
        """
        Returns a context manager for batching graph operations.

        All graph modifications within this context will defer cache updates
        until the end of the block, improving performance for bulk operations.

        Example:
            with graph.batch_operation():
                graph.add_node("A", "Value A", "Type A", "H1", {})
                graph.add_node("B", "Value B", "Type B", "H1", {})
                graph.add_edge("A", "B", {"edge_type": "CONNECTS"})
                # Caches are only invalidated once when the context exits

        Returns:
            A context manager for batched operations
        """
        return _BatchContext(self)

    '''
    Backend Graph Representations
    '''

    def build_networkit_graph(self):
        """
        Builds and returns a Networkit graph from the nodes and edges in the graph.
        The graph is constructed based on the existing node and edge data, including node and edge attributes.
        """
        # Create an empty Networkit graph
        net_graph = nk.graph.Graph(directed=True)

        # Add nodes to the graph
        for node_id, node_entry in self.node_data.items():
            # Add node to Networkit graph (nodes are indexed from 0)
            node_index = net_graph.addNode()

        # Index the edges (required for edge attributes)
        net_graph.indexEdges()

        # Create node attributes
        value_att = net_graph.attachNodeAttribute("value", str)
        type_att = net_graph.attachNodeAttribute("type", str)
        hierarchy_att = net_graph.attachNodeAttribute("hierarchy", str)

        # Add node attribute values
        for node_id, node_entry in self.node_data.items():
            # Get node index
            node_index = self._get_node_index(node_id)

            # Extract node attributes and ensure they're strings
            value = str(node_entry.get("value", "default_value"))
            type_ = str(node_entry.get("type", "default_type"))
            hierarchy = str(node_entry.get("hierarchy", "default_hierarchy"))

            # Set attribute values
            value_att[node_index] = value
            type_att[node_index] = type_
            hierarchy_att[node_index] = hierarchy

        # Add edges to the graph
        for (src, tgt), attrs in self.edge_data.items():
            # Networkit uses zero-based indices, so you'll need to map your nodes to indices
            src_index = self._get_node_index(src)
            tgt_index = self._get_node_index(tgt)

            # Add the edge between the two nodes in the Networkit graph
            net_graph.addEdge(src_index, tgt_index)

        # Create edge attributes
        edge_type_att = net_graph.attachEdgeAttribute("edge_type", str)
        direction_att = net_graph.attachEdgeAttribute("direction", str)
        source_type_att = net_graph.attachEdgeAttribute("source_type", str)
        target_type_att = net_graph.attachEdgeAttribute("target_type", str)

        # Add edge attribute values
        for (src, tgt), attrs in self.edge_data.items():
            src_index = self._get_node_index(src)
            tgt_index = self._get_node_index(tgt)

            # Extract edge attributes and ensure they're strings
            edge_type = str(attrs.get("edge_type", "unspecified"))
            direction = str(attrs.get("direction", "forward"))
            source_type = str(self.node_data[src].get("type", "unknown"))
            target_type = str(self.node_data[tgt].get("type", "unknown"))

            # Set attribute values
            edge_type_att[src_index, tgt_index] = edge_type
            direction_att[src_index, tgt_index] = direction
            source_type_att[src_index, tgt_index] = source_type
            target_type_att[src_index, tgt_index] = target_type

        # Once the graph is built, mark it as inconsistent
        self._mark_inconsistent()

        self.nk_graph = net_graph

        return net_graph

    def build_igraph_graph(self):
        """
        Builds and returns an igraph Graph from the nodes and edges in the graph.
        The graph is constructed based on the existing node and edge data, including node and edge attributes.
        """
        import igraph as ig

        # Create an empty undirected graph with the correct number of vertices
        ig_graph = ig.Graph(
            n=len(self.node_data),  # Number of vertices
            directed=True  # Make it directed
        )

        # Create a mapping of node IDs to indices
        node_id_to_index = {}
        for i, node_id in enumerate(self.node_data.keys()):
            node_id_to_index[node_id] = i

        # Add vertex attributes
        ig_graph.vs["name"] = [str(node_id) for node_id in self.node_data.keys()]
        ig_graph.vs["value"] = [str(node_entry.get("value", "default_value"))
                                for node_entry in self.node_data.values()]
        ig_graph.vs["type"] = [str(node_entry.get("type", "default_type"))
                               for node_entry in self.node_data.values()]
        ig_graph.vs["hierarchy"] = [str(node_entry.get("hierarchy", "default_hierarchy"))
                                    for node_entry in self.node_data.values()]

        # Create edges with their indices
        edge_list = []
        edge_attrs = {"edge_type": [], "direction": [], "source_type": [], "target_type": []}

        for (src, tgt), attrs in self.edge_data.items():
            # Get indices for source and target
            src_index = node_id_to_index[src]
            tgt_index = node_id_to_index[tgt]

            # Add edge to the list
            edge_list.append((src_index, tgt_index))

            # Gather edge attributes
            edge_attrs["edge_type"].append(str(attrs.get("edge_type", "unspecified")))
            edge_attrs["direction"].append(str(attrs.get("direction", "forward")))
            edge_attrs["source_type"].append(str(self.node_data[src].get("type", "unknown")))
            edge_attrs["target_type"].append(str(self.node_data[tgt].get("type", "unknown")))

        # Add all edges at once
        ig_graph.add_edges(edge_list)

        # Add edge attributes
        for attr_name, attr_values in edge_attrs.items():
            ig_graph.es[attr_name] = attr_values

        # Once the graph is built, mark it as inconsistent
        self._mark_inconsistent()

        self.ig_igraph = ig_graph

        return ig_graph


    def sync_backends(self):
        self.build_networkit_graph()
        self.build_igraph_graph()

    '''
    Query + Search
    '''

    def find_nodes_by_attribute(self, key: str, value, limit=None, offset=0) -> list:
        """
        Find all nodes that have the specified attribute key with the given value.

        Args:
            key: The attribute key to search for
            value: The attribute value to match

        Returns:
            A list of node IDs that match the criteria
        """
        # Check if we have a cache for this attribute
        if key in self.node_attribute_index and value in self.node_attribute_index[key]:
            # Return the cached result (convert set to list)
            return list(self.node_attribute_index[key][value])

        # If not cached, search directly
        matching_nodes = []

        for node_id, attributes in self.node_data.items():
            if key in attributes and attributes[key] == value:
                matching_nodes.append(node_id)

        # Update the cache if the attribute is valid
        if self._is_valid_attribute(value):
            if key not in self.node_attribute_index:
                self.node_attribute_index[key] = {}
            if value not in self.node_attribute_index[key]:
                self.node_attribute_index[key][value] = set()

            self.node_attribute_index[key][value].update(matching_nodes)

        if limit is not None:
            return matching_nodes[offset:offset + limit]
        return matching_nodes


    def find_edges_by_attribute(self, key: str, value) -> list:
        """
        Find all edges that have the specified attribute key with the given value.

        Args:
            key: The attribute key to search for
            value: The attribute value to match

        Returns:
            A list of edge tuples (source, target) that match the criteria
        """
        # Check if we have a cache for this attribute
        if key in self.edge_attribute_index and value in self.edge_attribute_index[key]:
            # Return the cached result (convert set to list)
            return list(self.edge_attribute_index[key][value])

        # If not cached, search directly
        matching_edges = []

        for edge_key, attributes in self.edge_data.items():
            if key in attributes and attributes[key] == value:
                matching_edges.append(edge_key)  # edge_key is (source, target)

        # Update the cache if the attribute is valid
        if self._is_valid_attribute(value):
            if key not in self.edge_attribute_index:
                self.edge_attribute_index[key] = {}
            if value not in self.edge_attribute_index[key]:
                self.edge_attribute_index[key][value] = set()

            self.edge_attribute_index[key][value].update(matching_edges)

        return matching_edges

    def find_subgraph_by_motif(self, motif_spec: dict) -> dict:
        """
        Find subgraphs matching a specified motif pattern.

        Args:
            motif_spec: A dictionary specifying the motif pattern to search for.
                       Example: {
                           'node_types': ['Person', 'Company', 'Location'],
                           'edge_types': ['WORKS_AT', 'LOCATED_IN'],
                           'pattern': 'chain'  # 'chain', 'cycle', 'star', etc.
                       }

        Returns:
            A dictionary mapping motif instance_graph IDs to the list of nodes and edges in that instance_graph
        """
        results = {}
        instance_id = 0  # Initialize instance_graph counter for all patterns

        # Step 1: Get all relevant nodes by type
        nodes_by_type = {}
        for node_type in motif_spec.get('node_types', []):
            nodes_by_type[node_type] = self.find_nodes_by_attribute('type', node_type)

        # Step 2: Get all relevant edges by type
        edges_by_type = {}
        for edge_type in motif_spec.get('edge_types', []):
            edges_by_type[edge_type] = self.find_edges_by_attribute('edge_type', edge_type)

        # Step 3: Implement pattern matching based on the pattern type
        pattern = motif_spec.get('pattern', 'chain')
        node_types = motif_spec.get('node_types', [])
        edge_types = motif_spec.get('edge_types', [])

        if pattern == 'chain':
            # For a chain pattern, we're looking for A->B->C where the types match
            # and the edge types match the specified pattern

            # We need at least 2 nodes and 1 edge type for a chain
            if len(node_types) < 2 or len(edge_types) < 1:
                return results

            # For each potential starting node of the right type
            for start_node in nodes_by_type.get(node_types[0], []):
                # Perform a DFS to find matching chains
                stack = [(start_node, [start_node], [])]  # (current, path_nodes, path_edges)

                while stack:
                    current, path_nodes, path_edges = stack.pop()

                    # If we've reached a complete chain
                    if len(path_nodes) == len(node_types):
                        # We've found a match
                        results[instance_id] = {
                            'nodes': path_nodes.copy(),
                            'edges': path_edges.copy()
                        }
                        instance_id += 1
                        continue

                    # Get the expected next node type and edge type
                    current_pos = len(path_nodes) - 1
                    next_node_type = node_types[current_pos + 1]
                    connecting_edge_type = edge_types[current_pos]

                    # Look for outgoing edges of the right type
                    for target, attrs in self.lazy_outgoing_edge_cache(current):
                        edge_key = (current, target)

                        # Check if this edge has the right type
                        if self.edge_data.get(edge_key, {}).get('edge_type') == connecting_edge_type:
                            # Check if the target node has the right type
                            if self.node_data.get(target, {}).get('type') == next_node_type:
                                # Check if this node isn't already in our path (to avoid cycles)
                                if target not in path_nodes:
                                    # Add to the search
                                    new_path_nodes = path_nodes + [target]
                                    new_path_edges = path_edges + [edge_key]
                                    stack.append((target, new_path_nodes, new_path_edges))

        elif pattern == 'star':
            # Validate inputs for star pattern
            if len(node_types) < 2 or len(edge_types) < 1:
                return results

            # Star pattern: A center node connected to multiple nodes
            center_type = node_types[0]
            leaf_type = node_types[1]
            edge_type = edge_types[0]

            # Look for center nodes
            for center_node in nodes_by_type.get(center_type, []):
                star_nodes = [center_node]
                star_edges = []

                # Find all leaves connected to this center
                for leaf, attrs in self.lazy_outgoing_edge_cache(center_node):
                    edge_key = (center_node, leaf)

                    if (self.edge_data.get(edge_key, {}).get('edge_type') == edge_type and
                            self.node_data.get(leaf, {}).get('type') == leaf_type):
                        star_nodes.append(leaf)
                        star_edges.append(edge_key)

                # If we found a valid star
                if len(star_nodes) > 2:  # Center + at least 2 leaves
                    results[instance_id] = {
                        'nodes': star_nodes,
                        'edges': star_edges
                    }
                    instance_id += 1

        elif pattern == 'cycle':
            # We need at least 3 nodes and the same number of edge types for a cycle
            if len(node_types) < 3 or len(edge_types) < len(node_types):
                return results

            # For each potential starting node of the right type
            for start_node in nodes_by_type.get(node_types[0], []):
                # Perform a DFS to find cycles that match our pattern
                stack = [(start_node, [start_node], [], 0)]  # (current, path_nodes, path_edges, type_index)

                while stack:
                    current, path_nodes, path_edges, type_index = stack.pop()

                    # Check if we can close the cycle back to the start
                    if len(path_nodes) > 2 and type_index == len(node_types) - 1:
                        # Check if there's an edge from current back to start_node of the right type
                        edge_key = (current, start_node)
                        if edge_key in self.edge_data and self.edge_data[edge_key].get('edge_type') == edge_types[
                            type_index]:
                            # We found a cycle
                            cycle_edges = path_edges + [edge_key]
                            results[instance_id] = {
                                'nodes': path_nodes.copy(),
                                'edges': cycle_edges
                            }
                            instance_id += 1
                            continue

                    # Continue building the path if we haven't reached the end
                    if type_index < len(node_types) - 1:
                        next_type_index = (type_index + 1) % len(node_types)
                        next_node_type = node_types[next_type_index]
                        connecting_edge_type = edge_types[type_index]

                        # Look for outgoing edges of the right type
                        for target, attrs in self.lazy_outgoing_edge_cache(current):
                            edge_key = (current, target)

                            # Skip if we'd create a premature cycle
                            if target in path_nodes and target != start_node:
                                continue

                            # Check if this edge and target have the right types
                            if (self.edge_data.get(edge_key, {}).get('edge_type') == connecting_edge_type and
                                    self.node_data.get(target, {}).get('type') == next_node_type):

                                # Add to the search only if target not in path or is the start node (to close cycle)
                                if target not in path_nodes or (target == start_node and next_type_index == 0):
                                    new_path_nodes = path_nodes + ([target] if target != start_node else [])
                                    new_path_edges = path_edges + [edge_key]
                                    stack.append((target, new_path_nodes, new_path_edges, next_type_index))

        # Add more pattern types as needed...

        return results

    def get_node_neighbors(self, node_id: str) -> dict:
        """
        Get all neighboring nodes for a specific node, categorized by direction.
        Uses the cached neighborhood data if available.

        Args:
            node_id: The ID of the node to find neighbors for

        Returns:
            A dictionary with 'incoming', 'outgoing', and 'all' neighbor lists
        """
        if node_id not in self.node_data:
            raise KeyError(f"Node '{node_id}' does not exist.")

        # Ensure we have the necessary caches
        outgoing = self.lazy_outgoing_edge_cache(node_id)
        incoming = self.lazy_incoming_edge_cache(node_id)

        # Extract just the node IDs (not the attributes)
        outgoing_neighbors = [target for target, _ in outgoing]
        incoming_neighbors = [source for source, _ in incoming]

        # Get the union of all neighbors
        all_neighbors = list(set(outgoing_neighbors + incoming_neighbors))

        return {
            'outgoing': outgoing_neighbors,
            'incoming': incoming_neighbors,
            'all': all_neighbors
        }

    def get_node_degree(self, node_id: str) -> dict:
        """
        Get the degree (number of connections) for a specific node.

        Args:
            node_id: The ID of the node

        Returns:
            A dictionary with 'in_degree', 'out_degree', and 'total_degree'
        """
        if node_id not in self.node_data:
            raise KeyError(f"Node '{node_id}' does not exist.")

        # Get neighbors
        neighbors = self.get_node_neighbors(node_id)

        # Calculate degrees
        in_degree = len(neighbors['incoming'])
        out_degree = len(neighbors['outgoing'])
        total_degree = len(neighbors['all'])

        return {
            'in_degree': in_degree,
            'out_degree': out_degree,
            'total_degree': total_degree
        }

    def is_tree(self) -> bool:
        """
        Check if the entire graph is a tree.

        A graph is a tree if it is connected and has no cycles.

        Returns:
            True if the graph is a tree, False otherwise
        """
        # If the graph is empty or has only one node, it's a tree
        if len(self.node_data) <= 1:
            return True

        # A tree has exactly n-1 edges, where n is the number of nodes
        if len(self.edge_data) != len(self.node_data) - 1:
            return False

        # Check connectivity and absence of cycles using BFS
        visited = set()
        start_node = next(iter(self.node_data.keys()))

        queue = [start_node]
        visited.add(start_node)

        while queue:
            current = queue.pop(0)

            # Check outgoing edges
            for target, _ in self.lazy_outgoing_edge_cache(current):
                if target not in visited:
                    visited.add(target)
                    queue.append(target)
                else:
                    # If we've already visited this node and it's not the one we came from,
                    # we have a cycle
                    return False

            # Check incoming edges
            for source, _ in self.lazy_incoming_edge_cache(current):
                if source not in visited:
                    visited.add(source)
                    queue.append(source)
                else:
                    # If we've already visited this node and it's not the one we came from,
                    # we have a cycle
                    return False

        # If we've visited all nodes, the graph is connected
        return len(visited) == len(self.node_data)

    def has_cycle(self) -> bool:
        """
        Check if the graph contains any cycles.

        Returns:
            True if the graph has at least one cycle, False otherwise
        """
        # Use DFS to detect cycles
        visited = set()
        recursion_stack = set()

        def dfs_cycle_check(node):
            visited.add(node)
            recursion_stack.add(node)

            # Check all neighbors
            for neighbor, _ in self.lazy_outgoing_edge_cache(node):
                if neighbor not in visited:
                    if dfs_cycle_check(neighbor):
                        return True
                elif neighbor in recursion_stack:
                    # If the neighbor is in the recursion stack, we found a cycle
                    return True

            # Remove the node from the recursion stack
            recursion_stack.remove(node)
            return False

        # Check from each unvisited node
        for node in self.node_data:
            if node not in visited:
                if dfs_cycle_check(node):
                    return True

        return False

    def has_cycle_using_backend(self) -> bool:
        """
        Check for cycles using the igraph backend for better performance
        on large graphs.
        """
        # Ensure igraph backend is up to date
        if not self.ig_igraph or not self.consistent:
            self.build_igraph_graph()

        # Use igraph's built-in cycle detection
        return not self.ig_igraph.is_dag()


    def find_nodes_by_compound_query(self, query_dict: dict) -> list:
        """
        Find nodes matching multiple attribute criteria.

        Args:
            query_dict: Dictionary of attribute constraints, e.g.,
                       {'type': 'Person', 'age': 30, 'status': 'active'}

        Returns:
            List of matching node IDs
        """
        if not query_dict:
            return list(self.node_data.keys())

        result_sets = []
        for key, value in query_dict.items():
            nodes = set(self.find_nodes_by_attribute(key, value))
            result_sets.append(nodes)

        # Intersection of all result sets
        if result_sets:
            return list(set.intersection(*result_sets))
        return []

    '''
    Transformations
    '''

    def collapse_edges_to_attributes(self, edge_types=None, target_to_source=False, remove_orphaned_nodes=False):
        """
        Collapses specified edge types into node attributes, reducing the number of edges
        while preserving the semantic relationships as node attributes.

        This is particularly useful for converting property graphs to more compact representations,
        or for transforming triplestores (subject-predicate-object) into property graphs.

        Args:
            edge_types: List of edge types to collapse. If None, all edges are considered.
            target_to_source: If True, attributes are added to source nodes from target nodes.
                             If False (default), attributes are added to target nodes from source nodes.
            remove_orphaned_nodes: If True, removes nodes that become disconnected after edge removal.

        Returns:
            dict: Statistics about the collapse operation
        """
        if edge_types is None:
            # Consider all edge types if none specified
            edge_types = set()
            for _, attrs in self.edge_data.items():
                edge_types.add(attrs.get('edge_type', 'unspecified'))
            edge_types = list(edge_types)

        # Identify edges to collapse
        edges_to_collapse = []
        for edge_key, attrs in self.edge_data.items():
            if attrs.get('edge_type', 'unspecified') in edge_types:
                edges_to_collapse.append(edge_key)

        # Track nodes that will become orphaned (no remaining connections)
        potentially_orphaned = defaultdict(int)
        for src, tgt in edges_to_collapse:
            potentially_orphaned[src] += 1
            potentially_orphaned[tgt] += 1

        # Process each edge to collapse
        collapsed_count = 0
        nodes_removed = 0
        orphaned_nodes = set()

        for src, tgt in edges_to_collapse:
            edge_attrs = self.edge_data.get((src, tgt), {})
            edge_type = edge_attrs.get('edge_type', 'unspecified')

            # Determine which node gets the attribute based on direction
            if target_to_source:
                target_node = src
                source_node = tgt
            else:
                target_node = tgt
                source_node = src

            # Add a new attribute to the target node based on the edge type
            if source_node in self.node_data and target_node in self.node_data:
                source_value = self.node_data[source_node].get('value', source_node)

                # Update the target node's attributes
                attr_name = f"{edge_type}_value"

                # If the attribute already exists, make it a list
                if attr_name in self.node_data[target_node]:
                    current_value = self.node_data[target_node][attr_name]
                    if isinstance(current_value, list):
                        self.node_data[target_node][attr_name].append(source_value)
                    else:
                        self.node_data[target_node][attr_name] = [current_value, source_value]
                else:
                    self.node_data[target_node][attr_name] = source_value

                # Also store the original node ID for possible future expansion
                attr_id_name = f"{edge_type}_node_id"
                if attr_id_name in self.node_data[target_node]:
                    current_id = self.node_data[target_node][attr_id_name]
                    if isinstance(current_id, list):
                        self.node_data[target_node][attr_id_name].append(source_node)
                    else:
                        self.node_data[target_node][attr_id_name] = [current_id, source_node]
                else:
                    self.node_data[target_node][attr_id_name] = source_node

                # Remove the edge
                self.edge_data.pop((src, tgt))
                collapsed_count += 1

                # Track potential orphan nodes
                potentially_orphaned[src] -= 1
                potentially_orphaned[tgt] -= 1

                if potentially_orphaned[source_node] == 0:
                    # Check if source node is now completely disconnected
                    if len(self.edge_data.keys()) > 0:
                        is_orphaned = True
                        for edge_key in self.edge_data.keys():
                            if source_node in edge_key:
                                is_orphaned = False
                                break
                        if is_orphaned:
                            orphaned_nodes.add(source_node)

        # Remove orphaned nodes if requested
        if remove_orphaned_nodes:
            for node_id in orphaned_nodes:
                # Double-check this node is truly disconnected before removing
                connected = False
                for edge_key in self.edge_data:
                    if node_id in edge_key:
                        connected = True
                        break

                if not connected:
                    del self.node_data[node_id]
                    nodes_removed += 1

        # Mark caches as inconsistent after structural changes
        self._mark_inconsistent()

        return {
            "edges_collapsed": collapsed_count,
            "nodes_removed": nodes_removed,
            "orphaned_nodes": len(orphaned_nodes)
        }

    def expand_attributes_to_edges(self, attribute_patterns=None, create_nodes=False):
        """
        Expands specific node attributes back into edge relationships.
        This is the reverse operation of collapse_edges_to_attributes.

        Args:
            attribute_patterns: List of attribute name patterns to expand (e.g., ['has_*', 'owns_*']).
                               If None, looks for attributes ending with '_value' and '_node_id'.
            create_nodes: If True, creates new nodes for values that don't correspond to existing nodes.

        Returns:
            dict: Statistics about the expansion operation
        """
        # Keep track of the edges and nodes we create
        created_edges = 0
        created_nodes = 0

        # If no patterns provided, look for attributes with standard naming convention
        if attribute_patterns is None:
            attribute_patterns = ['*_value']

        # Process each node in the graph
        for node_id, attributes in self.node_data.items():
            # Find attributes that match the patterns
            expandable_attrs = {}
            for attr_name, attr_value in attributes.items():
                for pattern in attribute_patterns:
                    # Check if the attribute matches any of the patterns
                    if pattern.endswith('*') and attr_name.startswith(pattern[:-1]):
                        expandable_attrs[attr_name] = attr_value
                    elif pattern.startswith('*') and attr_name.endswith(pattern[1:]):
                        expandable_attrs[attr_name] = attr_value
                    elif pattern == attr_name:
                        expandable_attrs[attr_name] = attr_value

            # Process each expandable attribute
            for attr_name, attr_value in expandable_attrs.items():
                # Extract the edge type from the attribute name (remove '_value' suffix)
                if attr_name.endswith('_value'):
                    edge_type = attr_name[:-6]  # Remove '_value'
                else:
                    edge_type = attr_name

                # Check if we have stored node IDs for this relationship
                node_id_attr = f"{edge_type}_node_id"
                if node_id_attr in attributes:
                    # We have the original node IDs, use them
                    target_ids = attributes[node_id_attr]
                    if not isinstance(target_ids, list):
                        target_ids = [target_ids]

                    for target_id in target_ids:
                        # If the target node exists or we're allowed to create it
                        if target_id in self.node_data or create_nodes:
                            if target_id not in self.node_data and create_nodes:
                                # Create the target node with the attribute value as its value
                                self.add_node(
                                    node_id=target_id,
                                    value=attr_value,
                                    type="auto_created",
                                    hierarchy="auto_created",
                                    attributes={}
                                )
                                created_nodes += 1

                            # Create the edge
                            self.add_edge(
                                source=node_id,
                                target=target_id,
                                attributes={"edge_type": edge_type}
                            )
                            created_edges += 1
                else:
                    # No stored node IDs, we need to create nodes from the values
                    if create_nodes:
                        values = attr_value if isinstance(attr_value, list) else [attr_value]

                        for value in values:
                            # Generate a unique ID for the new node
                            target_id = f"{edge_type}_{value}_{hash(str(value)) % 10000}"

                            # Create the node
                            self.add_node(
                                node_id=target_id,
                                value=value,
                                type="auto_created",
                                hierarchy="auto_created",
                                attributes={}
                            )
                            created_nodes += 1

                            # Create the edge
                            self.add_edge(
                                source=node_id,
                                target=target_id,
                                attributes={"edge_type": edge_type}
                            )
                            created_edges += 1

                # Remove the expanded attributes from the node
                if attr_name in self.node_data[node_id]:
                    del self.node_data[node_id][attr_name]
                if node_id_attr in self.node_data[node_id]:
                    del self.node_data[node_id][node_id_attr]

        # Mark caches as inconsistent after structural changes
        self._mark_inconsistent()

        return {
            "edges_created": created_edges,
            "nodes_created": created_nodes
        }

    def subsume_subgraph(self, subgraph_nodes: list, new_node_id=None, merge_attributes=True):
        """
        Collapses a subgraph defined by the given nodes into a single representative node.
        This preserves connections to nodes outside the subgraph while removing internal connections.

        Args:
            subgraph_nodes: List of node IDs to collapse into a single node
            new_node_id: ID for the new node. If None, generates an ID based on the first node
            merge_attributes: If True, combines attributes from all subsumed nodes

        Returns:
            The ID of the new representative node
        """
        if not subgraph_nodes or len(subgraph_nodes) < 2:
            return None  # Nothing to subsume

        # Verify all nodes exist
        for node in subgraph_nodes:
            if node not in self.node_data:
                raise KeyError(f"Node '{node}' does not exist in the graph.")

        # Generate a new node ID if not provided
        if new_node_id is None:
            new_node_id = f"subgraph_{subgraph_nodes[0]}"

        # Ensure new node ID doesn't conflict with existing nodes
        if new_node_id in self.node_data and new_node_id not in subgraph_nodes:
            raise ValueError(f"New node ID '{new_node_id}' already exists in the graph.")

        # Identify edges crossing the boundary of the subgraph
        external_incoming_edges = []
        external_outgoing_edges = []

        for node_id in subgraph_nodes:
            # Find incoming edges from outside the subgraph
            for source, attrs in self.lazy_incoming_edge_cache(node_id):
                if source not in subgraph_nodes:
                    external_incoming_edges.append(((source, node_id), attrs))

            # Find outgoing edges to outside the subgraph
            for target, attrs in self.lazy_outgoing_edge_cache(node_id):
                if target not in subgraph_nodes:
                    external_outgoing_edges.append(((node_id, target), attrs))

        # Create a new representative node with merged attributes
        merged_attrs = {}
        if merge_attributes:
            # Track what nodes were merged
            merged_attrs["subsumed_nodes"] = subgraph_nodes

            # Create aggregate attributes
            type_values = set()
            hierarchy_values = set()
            all_values = set()

            for node_id in subgraph_nodes:
                node_attrs = self.node_data.get(node_id, {})

                # Collect values for common attributes
                if "type" in node_attrs:
                    type_values.add(node_attrs["type"])
                if "hierarchy" in node_attrs:
                    hierarchy_values.add(node_attrs["hierarchy"])
                if "value" in node_attrs:
                    all_values.add(node_attrs["value"])

                # Merge other attributes
                for key, value in node_attrs.items():
                    if key not in ["type", "hierarchy", "value"]:
                        if key in merged_attrs:
                            if isinstance(merged_attrs[key], list):
                                if isinstance(value, list):
                                    merged_attrs[key].extend(value)
                                else:
                                    merged_attrs[key].append(value)
                            else:
                                merged_attrs[key] = [merged_attrs[key], value]
                        else:
                            merged_attrs[key] = value

        # Create the new representative node
        self.add_node(
            node_id=new_node_id,
            value=list(all_values) if len(all_values) > 1 else next(iter(all_values), None),
            type="|".join(type_values) if len(type_values) > 1 else next(iter(type_values), "subsumed"),
            hierarchy="|".join(hierarchy_values) if len(hierarchy_values) > 1 else next(iter(hierarchy_values),
                                                                                        "subsumed"),
            attributes=merged_attrs
        )

        # Create new edges that preserve external connections
        for (source, _), attrs in external_incoming_edges:
            self.add_edge(source, new_node_id, attrs)

        for (_, target), attrs in external_outgoing_edges:
            self.add_edge(new_node_id, target, attrs)

        # Remove all the original nodes and their edges
        for node_id in subgraph_nodes:
            self.remove_node(node_id)

        # Mark caches as inconsistent
        self._mark_inconsistent()

        return new_node_id

    def render_pyvis(self, path: str = "graph.html", height="800px", width="100%", physics=True,
                     spacing_factor=1.0, max_nodes=1000, node_filter=None, sample_strategy="by_type",
                     auto_sample=True):
        """
        Render the graph using PyVis interactive HTML visualization.

        Args:
            path: Path to save the HTML output file
            height: Height of the visualization container
            width: Width of the visualization container
            physics: Whether to enable physics simulation
            spacing_factor: Controls spacing between nodes (higher values = more space)
            max_nodes: Maximum nodes to show before sampling
            node_filter: Dict to filter nodes by type/hierarchy {'type': ['file', 'folder']}
            sample_strategy: "by_type" or "random"
            auto_sample: Whether to automatically sample large graphs
        """
        try:
            from pyvis.network import Network
        except ImportError:
            raise ImportError("PyVis not installed. Install with: pip install pyvis")

        # Simple sampling if needed
        total_nodes = len(self.node_data)
        if auto_sample and total_nodes > max_nodes:
            print(f"Sampling {max_nodes} from {total_nodes} nodes...")
            viz_nodes, viz_edges = self._simple_sample(max_nodes, node_filter)
            is_sample = True
        else:
            viz_nodes = self.node_data
            viz_edges = self.edge_data
            is_sample = False

        # Create a network with appropriate settings
        net = Network(height=height, width=width, notebook=False, directed=True)

        # Configure physics
        if physics:
            # Barnes-Hut with adjusted spacing parameters
            net.barnes_hut(
                gravity=-spacing_factor * 10000,  # Negative gravity pushes nodes apart
                central_gravity=0.1,  # Lower central gravity
                spring_length=spacing_factor * 150  # Longer springs = more space
            )
        else:
            # Repulsion with adjusted spacing
            net.repulsion(
                node_distance=spacing_factor * 150,  # Distance between nodes
                spring_length=spacing_factor * 150  # Length of the springs
            )

        # Define node groups based on types for coloring
        node_types = set()
        for data in viz_nodes.values():
            node_type = data.get("type", "default")
            node_types.add(node_type)

        # Simple color map for node types (like version 2)
        color_map = {}
        colors = ["#4285F4", "#EA4335", "#FBBC05", "#34A853", "#8E24AA", "#16A085", "#D35400", "#C0392B", "#7F8C8D",
                  "#2C3E50"]
        for i, node_type in enumerate(node_types):
            color_map[node_type] = colors[i % len(colors)]

        # Add nodes with proper formatting
        for node_id, data in viz_nodes.items():
            label = str(data.get("value", node_id))
            node_type = data.get("type", "default")

            # Format the tooltip to show node attributes
            tooltip = ""
            for key, value in data.items():
                tooltip += f"{key}: {value} | "

            # Add the node with styling (using group like version 2)
            net.add_node(
                node_id,
                label=label,
                title=tooltip,
                color=color_map.get(node_type, "#7F8C8D"),
                size=25,
                font={'size': 12, 'color': 'black'},
                group=data.get("type", "default")  # Using group for better visualization
            )

        # Add edges with formatting
        edge_types = set(data.get("edge_type", "default") for data in viz_edges.values())
        edge_colors = {}
        for i, edge_type in enumerate(edge_types):
            edge_colors[edge_type] = colors[i % len(colors)]

        # Use viz_edges (sampled edges) not self.edge_data (all edges)
        for (src, tgt), data in viz_edges.items():
            if src in viz_nodes and tgt in viz_nodes:  # Double-check both nodes exist
                edge_type = data.get("edge_type", "")

                # Format the tooltip to show edge attributes
                tooltip = ""
                for key, value in data.items():
                    tooltip += f"{key}: {value} | "

                # Add the edge with styling
                net.add_edge(
                    src,
                    tgt,
                    title=tooltip,
                    label=edge_type,
                    color=edge_colors.get(edge_type, "#7F8C8D"),
                    font={'size': 10, 'align': 'middle'},
                    arrows='to',
                    arrowStrikethrough=False,
                    smooth={'enabled': True, 'type': 'dynamic'}
                )

        net.set_options("""
        var options = {
          "nodes": {
            "borderWidth": 2,
            "borderWidthSelected": 4,
            "shape": "dot",
            "font": {
              "face": "Arial",
              "size": 12
            },
            "scaling": {
              "label": {
                "enabled": true,
                "min": 12,
                "max": 24
              }
            }
          },
          "edges": {
            "color": {
              "inherit": false
            },
            "smooth": {
              "type": "dynamic",
              "forceDirection": "none",
              "roundness": 0.5
            },
            "font": {
              "face": "Arial",
              "size": 10
            }
          },
          "physics": {
            "stabilization": {
              "iterations": 300
            },
            "barnesHut": {
              "springConstant": 0.008,
              "avoidOverlap": 0.3,
              "damping": 0.2
            },
            "maxVelocity": 25,
            "timestep": 0.35
          },
          "interaction": {
            "hover": true,
            "tooltipDelay": 200,
            "hideEdgesOnDrag": false,
            "navigationButtons": true,
            "multiselect": true,
            "zoomView": true
          },
          "layout": {
            "improvedLayout": true
          }
        }
        """)

        # Save the visualization
        net.save_graph(path)

        # Add tiny sample indicator if needed
        if is_sample:
            self._add_tiny_sample_indicator(path, len(viz_nodes), total_nodes)

        return net

    # Keep the sampling methods from version 1
    def _simple_sample(self, max_nodes, node_filter=None):
        """Smart structural sampling - small representative sample showing all patterns"""
        import random
        from collections import defaultdict, deque

        # Apply filter if specified
        if node_filter:
            filtered_nodes = {
                node_id: data for node_id, data in self.node_data.items()
                if all(data.get(filter_key) in filter_values
                       for filter_key, filter_values in node_filter.items())
            }
        else:
            filtered_nodes = self.node_data

        total_nodes = len(filtered_nodes)

        # Smart sample size based on graph size
        if total_nodes <= 1000:
            viz_nodes = filtered_nodes  # Show everything if small enough
        else:
            # For large graphs, aim for small representative sample
            target_size = min(100, max_nodes // 3)  # Much smaller target
            print(f"Large graph ({total_nodes} nodes) - targeting {target_size} representative nodes")
            viz_nodes = self._structural_sample(filtered_nodes, target_size)

        # Get edges between sampled nodes
        viz_edges = {}
        sampled_ids = set(viz_nodes.keys())
        for (src, tgt), data in self.edge_data.items():
            if src in sampled_ids and tgt in sampled_ids:
                viz_edges[(src, tgt)] = data

        return viz_nodes, viz_edges

    def _structural_sample(self, filtered_nodes, target_size):
        """Focus on structural diversity - all types, all edge patterns"""
        import random
        from collections import defaultdict, Counter

        print("Analyzing graph structure...")

        # Step 1: Analyze node and edge type diversity quickly
        node_types = defaultdict(list)
        edge_types = defaultdict(list)

        for node_id, data in filtered_nodes.items():
            node_type = data.get("type", "default")
            node_types[node_type].append(node_id)

        for (src, tgt), data in self.edge_data.items():
            if src in filtered_nodes and tgt in filtered_nodes:
                edge_type = data.get("edge_type", "default")
                edge_types[edge_type].append((src, tgt))

        print(f"Found {len(node_types)} node types, {len(edge_types)} edge types")

        # Step 2: Quick node degree analysis to find pattern examples
        degree_analysis = {}
        for node_id in list(filtered_nodes.keys())[:1000]:  # Sample first 1000 for speed
            try:
                degree_info = self.get_node_degree(node_id)
                degree_analysis[node_id] = degree_info
            except:
                # Skip if degree calculation fails
                continue

        # Step 3: Smart selection strategy
        selected_nodes = set()

        # 3a: Get 2-3 examples of each node type (prioritize rare types)
        for node_type, type_nodes in node_types.items():
            sample_count = min(3, len(type_nodes), target_size // len(node_types))

            # For rare types, take more examples
            if len(type_nodes) < 50:  # Rare type
                sample_count = min(len(type_nodes), 5)

            samples = random.sample(type_nodes, sample_count)
            selected_nodes.update(samples)
            print(f"Selected {len(samples)} examples of type '{node_type}'")

        print(f"Node type coverage: {len(selected_nodes)} nodes")

        # 3b: Add nodes to demonstrate edge patterns
        edge_pattern_nodes = set()

        for edge_type, type_edges in edge_types.items():
            # Get a few examples of this edge type
            sample_edges = random.sample(type_edges, min(3, len(type_edges)))

            for src, tgt in sample_edges:
                # Add both endpoints
                edge_pattern_nodes.add(src)
                edge_pattern_nodes.add(tgt)

                # If we have degree info, prefer interesting patterns
                if src in degree_analysis and tgt in degree_analysis:
                    src_degree = degree_analysis[src]['total_degree']
                    tgt_degree = degree_analysis[tgt]['total_degree']

                    # Prioritize diverse degree patterns
                    if (src_degree == 1 and tgt_degree > 5) or (src_degree > 5 and tgt_degree == 1):
                        # One-to-many pattern
                        edge_pattern_nodes.add(src)
                        edge_pattern_nodes.add(tgt)

        selected_nodes.update(edge_pattern_nodes)
        print(f"Added {len(edge_pattern_nodes)} nodes for edge patterns")

        # 3c: If we're still under target, add some high-degree nodes for connectivity
        if len(selected_nodes) < target_size * 0.8:
            remaining_budget = target_size - len(selected_nodes)

            # Get high-degree nodes for connectivity
            high_degree_nodes = []
            for node_id, degree_info in degree_analysis.items():
                if node_id not in selected_nodes:
                    high_degree_nodes.append((degree_info['total_degree'], node_id))

            high_degree_nodes.sort(reverse=True)  # Highest degree first

            # Add top high-degree nodes
            for _, node_id in high_degree_nodes[:remaining_budget]:
                selected_nodes.add(node_id)

            print(f"Added {min(len(high_degree_nodes), remaining_budget)} high-degree nodes")

        # Step 4: Build result
        viz_nodes = {node_id: filtered_nodes[node_id] for node_id in selected_nodes}

        print(f"Structural sample complete: {len(viz_nodes)} nodes covering {len(node_types)} types")

        return viz_nodes


    def _add_tiny_sample_indicator(self, html_path, sample_size, total_size):
        """Add a very small, unobtrusive sample indicator"""
        try:
            with open(html_path, 'r', encoding='utf-8') as f:
                html_content = f.read()

            # Tiny indicator in bottom right
            sample_indicator = f"""
            <style>
            .tiny-sample {{
                position: fixed;
                bottom: 10px;
                right: 10px;
                background-color: rgba(0,0,0,0.6);
                color: white;
                padding: 2px 6px;
                border-radius: 2px;
                font-family: Arial, sans-serif;
                font-size: 40px;
                z-index: 1000;
            }}
            </style>
            <div class="tiny-sample">Sample: {sample_size}/{total_size}</div>
            """

            html_content = html_content.replace('<body>', f'<body>{sample_indicator}', 1)

            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)

        except Exception as e:
            print(f"Warning: Could not add sample indicator: {e}")


    def render_dash_cytoscape(self, return_component=False):
        """
        Render the graph using Dash Cytoscape for web-based interactive visualization.

        Args:
            return_component: If True, returns the Cytoscape component for embedding in a Dash app.
                             If False, creates and runs a standalone Dash app.

        Returns:
            Dash Cytoscape component or Dash app instance_graph
        """
        try:
            import dash
            from dash import html
            import dash_cytoscape as cyto
        except ImportError:
            raise ImportError("Dash and Dash-Cytoscape not installed. Install with: pip install dash dash-cytoscape")

        # Define the stylesheet
        stylesheet = [
            # Group selectors
            {
                'selector': 'node',
                'style': {
                    'content': 'data(label)',
                    'text-valign': 'center',
                    'text-halign': 'center',
                    'background-color': '#6FB1FC',
                    'width': '30px',
                    'height': '30px',
                    'font-size': '12px',
                    'border-width': '1px',
                    'border-color': '#4080CE'
                }
            },
            {
                'selector': 'edge',
                'style': {
                    'content': 'data(label)',
                    'curve-style': 'bezier',
                    'target-arrow-shape': 'triangle',
                    'line-color': '#9DBAEA',
                    'target-arrow-color': '#9DBAEA',
                    'font-size': '10px',
                    'text-rotation': 'autorotate',
                    'text-margin-y': '-10px'
                }
            }
        ]

        # Add type-specific selectors
        node_types = set()
        for data in self.node_data.values():
            node_type = data.get("type", "default")
            node_types.add(node_type)

        # Create color styles for different node types
        colors = ["#4285F4", "#EA4335", "#FBBC05", "#34A853", "#8E24AA", "#16A085", "#D35400", "#C0392B"]
        for i, node_type in enumerate(node_types):
            stylesheet.append({
                'selector': f'node[type="{node_type}"]',
                'style': {
                    'background-color': colors[i % len(colors)]
                }
            })

        # Prepare nodes and edges for Cytoscape
        nodes = []
        for node_id, data in self.node_data.items():
            nodes.append({
                'data': {
                    'id': node_id,
                    'label': str(data.get("value", node_id)),
                    'type': data.get("type", "default"),
                    **{k: str(v) for k, v in data.items()}  # Include all attributes
                }
            })

        edges = []
        for (src, tgt), data in self.edge_data.items():
            edges.append({
                'data': {
                    'id': f"{src}-{tgt}",
                    'source': src,
                    'target': tgt,
                    'label': data.get("edge_type", ""),
                    **{k: str(v) for k, v in data.items()}  # Include all attributes
                }
            })

        # Create the Cytoscape component
        cytoscape_component = cyto.Cytoscape(
            id='graph',
            layout={'name': 'cose'},  # Force-directed layout
            style={'width': '100%', 'height': '800px'},
            elements=nodes + edges,
            stylesheet=stylesheet,
            minZoom=0.5,
            maxZoom=2
        )

        # If component should be returned for embedding in another app
        if return_component:
            return cytoscape_component

        # Otherwise, create a standalone Dash app
        app = dash.Dash(__name__)
        app.layout = html.Div([
            html.H1("Graph Visualization with Dash Cytoscape"),
            cytoscape_component
        ])

        # Run the app
        app.run_server(debug=True)

        return app

    def render_jaal(self, title_field='value', open_browser=True):
        """
        Render the graph using Jaal for interactive network visualization.

        Args:
            title_field: The node attribute to use as node title/label
            open_browser: Whether to automatically open the visualization in browser

        Returns:
            The Jaal object
        """
        try:
            import pandas as pd
            from jaal import Jaal
        except ImportError:
            raise ImportError("Jaal not installed. Install with: pip install jaal")

        # Convert graph data to Jaal's expected format
        nodes = []
        for node_id, data in self.node_data.items():
            node_data = {"id": node_id}
            # Add all node attributes
            for key, value in data.items():
                node_data[key] = value
            nodes.append(node_data)

        edges = []
        for (src, tgt), data in self.edge_data.items():
            edge_data = {"from": src, "to": tgt}
            # Add all edge attributes
            for key, value in data.items():
                edge_data[key] = value
            edges.append(edge_data)

        # Create DataFrames for nodes and edges
        df_nodes = pd.DataFrame(nodes)
        df_edges = pd.DataFrame(edges)

        # Ensure required columns exist
        if 'id' not in df_nodes.columns:
            raise ValueError("Node DataFrame must have an 'id' column")
        if 'from' not in df_edges.columns or 'to' not in df_edges.columns:
            raise ValueError("Edge DataFrame must have 'from' and 'to' columns")

        # Set title and size for nodes
        if title_field in df_nodes.columns:
            df_nodes['title'] = df_nodes[title_field]
        else:
            df_nodes['title'] = df_nodes['id']

        # Set node size
        df_nodes['size'] = 25

        # Create and display the Jaal visualization
        jaal_obj = Jaal(df_edges, df_nodes)
        jaal_obj.plot(directed=True, vis_opts={'height': '800px'}, open_browser=open_browser)

        return jaal_obj

    def visualize(self, method="pyvis", **kwargs):
        """
        Unified method for visualizing the graph using different backends.

        Args:
            method: The visualization backend to use ("pyvis", "dash_cytoscape", or "jaal")
            **kwargs: Additional arguments to pass to the specific visualization method

        Returns:
            The result of the chosen visualization method
        """
        if method == "pyvis":
            return self.render_pyvis(**kwargs)
        elif method == "dash_cytoscape":
            return self.render_dash_cytoscape(**kwargs)
        elif method == "jaal":
            return self.render_jaal(**kwargs)
        else:
            raise ValueError(
                f"Unsupported visualization method: {method}. Choose from 'pyvis', 'dash_cytoscape', or 'jaal'.")

    '''
    External Integrations (e.g. Neo4j)
    '''

    def sync_to_neo4j(self, uri="bolt://localhost:7687", username="neo4j", password="password", database=None,
                      clear_existing=False):
        """
        Synchronizes the current graph to a Neo4j database.

        Args:
            uri: Neo4j connection URI
            username: Neo4j username
            password: Neo4j password
            database: Specific Neo4j database name (optional)
            clear_existing: If True, clears all existing data in the database before sync

        Returns:
            Dictionary with statistics about the synchronization
        """
        try:
            from neo4j import GraphDatabase
        except ImportError:
            raise ImportError("Neo4j driver not installed. Install with: pip install neo4j")

        # Connect to Neo4j
        driver = GraphDatabase.driver(uri, auth=(username, password))

        # Initialize statistics
        stats = {
            "nodes_created": 0,
            "edges_created": 0,
            "properties_set": 0
        }

        try:
            # Database session context manager
            session_params = {}
            if database:
                session_params["database"] = database

            with driver.session(**session_params) as session:
                # Clear existing data if requested
                if clear_existing:
                    session.run("MATCH (n) DETACH DELETE n")

                # Create nodes with their properties
                for node_id, attributes in self.node_data.items():
                    # Extract node type and prepare properties
                    node_type = attributes.get("type", "Node")

                    # Clean up property names and values for Neo4j
                    properties = {}
                    properties["id"] = node_id  # Always include the original ID

                    for key, value in attributes.items():
                        # Skip None values and sanitize property names
                        if value is not None:
                            # Convert complex values to strings if necessary
                            if isinstance(value, (dict, list, set)):
                                value = str(value)

                            # Sanitize property name
                            neo4j_key = key.replace(" ", "_").replace("-", "_")
                            properties[neo4j_key] = value

                    # Create Cypher parameter string for properties
                    props_string = ", ".join([f"{k}: ${k}" for k in properties.keys()])

                    # Create node with properties
                    result = session.run(
                        f"CREATE (n:{node_type} {{{props_string}}}) RETURN n",
                        **properties
                    )

                    # Update statistics
                    stats["nodes_created"] += result.consume().counters.nodes_created
                    stats["properties_set"] += len(properties)

                # Create edges with their properties
                for (src, tgt), attributes in self.edge_data.items():
                    # Extract edge type
                    edge_type = attributes.get("edge_type", "CONNECTS_TO")

                    # Clean up property names and values for Neo4j
                    properties = {}
                    properties["source_id"] = src
                    properties["target_id"] = tgt

                    for key, value in attributes.items():
                        if key != "edge_type" and value is not None:  # Skip edge_type and None values
                            # Convert complex values to strings if necessary
                            if isinstance(value, (dict, list, set)):
                                value = str(value)

                            # Sanitize property name
                            neo4j_key = key.replace(" ", "_").replace("-", "_")
                            properties[neo4j_key] = value

                    # Create Cypher parameter string for edge properties
                    props_string = ""
                    if len(properties) > 2:  # If there are properties beyond source_id and target_id
                        edge_props = {k: v for k, v in properties.items()
                                      if k not in ["source_id", "target_id"]}
                        props_string = " {" + ", ".join([f"{k}: ${k}" for k in edge_props.keys()]) + "}"

                    # Create edge between nodes
                    result = session.run(
                        f"""
                        MATCH (a), (b)
                        WHERE a.id = $source_id AND b.id = $target_id
                        CREATE (a)-[r:{edge_type}{props_string}]->(b)
                        RETURN r
                        """,
                        **properties
                    )

                    # Update statistics
                    stats["edges_created"] += result.consume().counters.relationships_created
                    if len(properties) > 2:
                        stats["properties_set"] += len(properties) - 2  # Subtract the source_id and target_id

        finally:
            # Close the driver connection
            driver.close()

        return stats

    def query_neo4j(self, cypher_query, parameters=None, uri="bolt://localhost:7687", username="neo4j",
                    password="password", database=None):
        """
        Execute a Cypher query against the Neo4j database.

        Args:
            cypher_query: The Cypher query to execute
            parameters: Dictionary of query parameters
            uri: Neo4j connection URI
            username: Neo4j username
            password: Neo4j password
            database: Specific Neo4j database name (optional)

        Returns:
            List of records returned by the query
        """
        try:
            from neo4j import GraphDatabase
        except ImportError:
            raise ImportError("Neo4j driver not installed. Install with: pip install neo4j")

        # Connect to Neo4j
        driver = GraphDatabase.driver(uri, auth=(username, password))

        try:
            # Database session context manager
            session_params = {}
            if database:
                session_params["database"] = database

            with driver.session(**session_params) as session:
                # Execute the query
                if parameters is None:
                    parameters = {}

                result = session.run(cypher_query, **parameters)

                # Convert result to a list of dictionaries
                records = []
                for record in result:
                    # Convert Neo4j types to Python types
                    record_dict = {}
                    for key, value in record.items():
                        # Handle Neo4j Node objects
                        if hasattr(value, "labels") and hasattr(value, "items"):  # It's a Node
                            node_dict = dict(value.items())
                            node_dict["_labels"] = list(value.labels)
                            record_dict[key] = node_dict
                        # Handle Neo4j Relationship objects
                        elif hasattr(value, "type") and hasattr(value, "start_node") and hasattr(value, "end_node"):
                            rel_dict = dict(value.items())
                            rel_dict["_type"] = value.type
                            rel_dict["_start_node_id"] = value.start_node["id"]
                            rel_dict["_end_node_id"] = value.end_node["id"]
                            record_dict[key] = rel_dict
                        # Handle Neo4j Path objects
                        elif hasattr(value, "nodes") and hasattr(value, "relationships"):
                            # Convert path to simplified representation
                            path_dict = {
                                "nodes": [dict(node.items()) for node in value.nodes],
                                "relationships": [
                                    {
                                        "type": rel.type,
                                        "properties": dict(rel.items())
                                    } for rel in value.relationships
                                ]
                            }
                            record_dict[key] = path_dict
                        else:
                            # Regular value
                            record_dict[key] = value

                    records.append(record_dict)

                return records

        finally:
            # Close the driver connection
            driver.close()

    def load_from_neo4j(self, uri="bolt://localhost:7687", username="neo4j", password="password", database=None,
                        node_query=None, edge_query=None):
        """
        Loads graph data from a Neo4j database into this GraphManager instance_graph.

        Args:
            uri: Neo4j connection URI
            username: Neo4j username
            password: Neo4j password
            database: Specific Neo4j database name (optional)
            node_query: Custom Cypher query to retrieve nodes (optional)
            edge_query: Custom Cypher query to retrieve relationships (optional)

        Returns:
            Dictionary with statistics about the loading process
        """
        try:
            from neo4j import GraphDatabase
        except ImportError:
            raise ImportError("Neo4j driver not installed. Install with: pip install neo4j")

        # Connect to Neo4j
        driver = GraphDatabase.driver(uri, auth=(username, password))

        # Default queries if not provided
        if node_query is None:
            node_query = """
            MATCH (n)
            RETURN n, labels(n) as labels
            """

        if edge_query is None:
            edge_query = """
            MATCH (a)-[r]->(b)
            RETURN a.id as source, b.id as target, type(r) as type, r as properties
            """

        # Initialize statistics
        stats = {
            "nodes_loaded": 0,
            "edges_loaded": 0
        }

        try:
            # Database session context manager
            session_params = {}
            if database:
                session_params["database"] = database

            with driver.session(**session_params) as session:
                # Clear existing graph data
                self.node_data = {}
                self.edge_data = {}

                # Load nodes
                node_result = session.run(node_query)

                for record in node_result:
                    node = record["n"]
                    labels = record["labels"]

                    # Extract node properties
                    properties = dict(node)

                    # Use Neo4j ID as fallback if no id property exists
                    node_id = properties.get("id", f"neo4j_{node.id}")

                    # Create node attributes
                    node_attrs = {}
                    node_type = labels[0] if labels else "Node"  # Use first label as type

                    for key, value in properties.items():
                        if key != "id":  # Skip the id property as it becomes the node_id
                            node_attrs[key] = value

                    # Add standard attributes
                    if "value" not in node_attrs:
                        node_attrs["value"] = properties.get("name", node_id)
                    if "type" not in node_attrs:
                        node_attrs["type"] = node_type
                    if "hierarchy" not in node_attrs:
                        node_attrs["hierarchy"] = "default"

                    # Add node to graph
                    self.node_data[node_id] = node_attrs
                    stats["nodes_loaded"] += 1

                # Load edges
                edge_result = session.run(edge_query)

                for record in edge_result:
                    source = record["source"]
                    target = record["target"]
                    edge_type = record["type"]
                    properties = dict(record["properties"])

                    # Create edge attributes
                    edge_attrs = {
                        "edge_type": edge_type
                    }

                    # Add any additional properties
                    for key, value in properties.items():
                        edge_attrs[key] = value

                    # Add edge to graph
                    edge_key = (source, target)
                    self.edge_data[edge_key] = edge_attrs
                    stats["edges_loaded"] += 1

                # Mark caches as inconsistent
                self._mark_inconsistent()

        finally:
            # Close the driver connection
            driver.close()

        return stats

    '''
    Utilities
    '''

    def _reset(self):
        '''
        Reset key values to their default state.
        '''
        self.node_data = {}
        self.edge_data = {}
        self.graph_json_address = ''

    def _mark_inconsistent(self):
        """Mark the graph state as inconsistent, indicating caches need rebuilding."""
        # Skip if we're in batch mode
        if hasattr(self, '_disable_auto_invalidation') and self._disable_auto_invalidation:
            return
        self.consistent = False

    def _mark_consistent(self):
        self.consistent = True

    def _sync_if_needed(self):
        if not self.consistent:
            self.rebuild_all()
            self._mark_consistent()

    def _is_valid_attribute(attribute: dict) -> bool:
        """
        Check if the attribute is simple (string or integer) and has acceptable cardinality.
        """
        if isinstance(attribute, (str, int)):
            return True
        return False

    def _is_valid_cardinality(values: set, max_cardinality=100) -> bool:
        """
        Check if an attribute's value set has a reasonable cardinality (e.g., no more than 100 unique values).
        """
        return len(values) <= max_cardinality

    def _get_node_index(self, node_id: str):
        """
        Returns the index for a given node. You may need to create a mapping from node_id to indices.
        """
        # Assuming nodes are indexed based on the order in `self.node_data`
        # This will return the index for a node based on its node_id
        # This mapping should match the indices used in Networkit.
        node_list = list(self.node_data.keys())
        return node_list.index(node_id)

    def _convert_timestamps(self, obj: Any):
            if isinstance(obj, dict):
                return {k: self._convert_timestamps(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [self._convert_timestamps(v) for v in obj]
            elif isinstance(obj, str):
                try:
                    return pd.to_datetime(obj, utc=True)
                except Exception:
                    return obj
            else:
                return obj

    def _export_subgraph(self, node_ids: list) -> dict:
        """
        Create a new graph structure containing only the specified nodes and their interconnecting edges.

        Args:
            node_ids: List of node IDs to include in the subgraph

        Returns:
            Dictionary with 'nodes' and 'edges' representing the subgraph
        """
        if not node_ids:
            return {"nodes": {}, "edges": {}}

        # Validate node IDs
        for node_id in node_ids:
            if node_id not in self.node_data:
                raise KeyError(f"Node '{node_id}' does not exist.")

        # Create subgraph structure
        subgraph = {
            "nodes": {},
            "edges": {}
        }

        # Add nodes
        for node_id in node_ids:
            subgraph["nodes"][node_id] = self.node_data[node_id].copy()

        # Add edges (only those connecting nodes within the subgraph)
        for (src, tgt), attrs in self.edge_data.items():
            if src in node_ids and tgt in node_ids:
                subgraph["edges"][(src, tgt)] = attrs.copy()

        return subgraph

    def _get_summary_stats(self) -> dict:
        """
        Generate summary statistics about the graph structure and its properties.

        Returns:
            Dictionary containing various graph statistics
        """
        # Initialize statistics dictionary
        stats = {
            "nodes": {
                "count": len(self.node_data),
                "types": {},
                "hierarchies": {},
                "attribute_counts": {},
                "avg_attributes_per_node": 0
            },
            "edges": {
                "count": len(self.edge_data),
                "types": {},
                "attribute_counts": {},
                "avg_attributes_per_edge": 0
            },
            "structure": {
                "density": 0,
                "is_tree": False,
                "has_cycles": False,
                "connected_components": 0,
                "max_node_degree": 0,
                "avg_node_degree": 0,
                "diameter": None
            }
        }

        # Skip further calculation if graph is empty
        if not self.node_data:
            return stats

        # Node statistics
        total_node_attrs = 0
        node_attr_counts = {}

        for node_id, attrs in self.node_data.items():
            # Count node types
            node_type = attrs.get("type", "unknown")
            stats["nodes"]["types"][node_type] = stats["nodes"]["types"].get(node_type, 0) + 1

            # Count node hierarchies
            hierarchy = attrs.get("hierarchy", "unknown")
            stats["nodes"]["hierarchies"][hierarchy] = stats["nodes"]["hierarchies"].get(hierarchy, 0) + 1

            # Count attribute occurrences
            for attr_name in attrs:
                node_attr_counts[attr_name] = node_attr_counts.get(attr_name, 0) + 1

            # Count total attributes
            total_node_attrs += len(attrs)

        # Calculate average attributes per node
        if stats["nodes"]["count"] > 0:
            stats["nodes"]["avg_attributes_per_node"] = total_node_attrs / stats["nodes"]["count"]

        # Store attribute counts for nodes
        stats["nodes"]["attribute_counts"] = node_attr_counts

        # Edge statistics
        total_edge_attrs = 0
        edge_attr_counts = {}

        for edge_key, attrs in self.edge_data.items():
            # Count edge types
            edge_type = attrs.get("edge_type", "unknown")
            stats["edges"]["types"][edge_type] = stats["edges"]["types"].get(edge_type, 0) + 1

            # Count attribute occurrences
            for attr_name in attrs:
                edge_attr_counts[attr_name] = edge_attr_counts.get(attr_name, 0) + 1

            # Count total attributes
            total_edge_attrs += len(attrs)

        # Calculate average attributes per edge
        if stats["edges"]["count"] > 0:
            stats["edges"]["avg_attributes_per_edge"] = total_edge_attrs / stats["edges"]["count"]

        # Store attribute counts for edges
        stats["edges"]["attribute_counts"] = edge_attr_counts

        # Structure statistics

        # Calculate density
        if stats["nodes"]["count"] > 1:
            max_possible_edges = stats["nodes"]["count"] * (stats["nodes"]["count"] - 1)
            stats["structure"]["density"] = stats["edges"]["count"] / max_possible_edges

        # Check if graph is a tree
        stats["structure"]["is_tree"] = self.is_tree()

        # Check if graph has cycles
        stats["structure"]["has_cycles"] = self.has_cycle()

        # Calculate degree statistics
        degrees = []
        max_degree = 0
        for node_id in self.node_data:
            node_degree = self.get_node_degree(node_id)
            total_degree = node_degree["total_degree"]
            degrees.append(total_degree)
            max_degree = max(max_degree, total_degree)

        stats["structure"]["max_node_degree"] = max_degree
        if degrees:
            stats["structure"]["avg_node_degree"] = sum(degrees) / len(degrees)

        # Calculate connected components
        visited = set()
        connected_components = 0

        for node_id in self.node_data:
            if node_id not in visited:
                # New component found
                connected_components += 1

                # BFS to find all nodes in this component
                queue = [node_id]
                visited.add(node_id)

                while queue:
                    current = queue.pop(0)

                    # Check neighbors
                    neighbors = self.get_node_neighbors(current)["all"]
                    for neighbor in neighbors:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            queue.append(neighbor)

        stats["structure"]["connected_components"] = connected_components

        # Estimate diameter (for small to medium graphs)
        if stats["nodes"]["count"] <= 1000:  # Only calculate for reasonably sized graphs
            try:
                # If we have the igraph backend available, use it for efficiency
                if self.ig_igraph or stats["nodes"]["count"] <= 100:
                    if not self.ig_igraph:
                        self.build_igraph_graph()

                    # Use igraph's diameter calculation
                    if connected_components == 1:  # Only valid for connected graphs
                        stats["structure"]["diameter"] = self.ig_igraph.diameter(directed=True)
                else:
                    # Skip diameter calculation for larger graphs without igraph
                    stats["structure"]["diameter"] = "not calculated (graph too large)"
            except:
                stats["structure"]["diameter"] = "calculation failed"
        else:
            stats["structure"]["diameter"] = "not calculated (graph too large)"

        return stats


def batch_operation(self):
    """
    Returns a context manager for batching operations.
    Delays cache invalidation until all operations in the block are complete.
    """
    return _SimpleContext(self)


class _SimpleContext:
    def __init__(self, graph_manager):
        self.graph_manager = graph_manager
        self.was_consistent = None

    def __enter__(self):
        # Track original consistency state
        self.was_consistent = self.graph_manager.consistent
        # Temporarily disable consistency updates
        self.graph_manager._in_batch = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Re-enable consistency updates
        self.graph_manager._in_batch = False
        # Mark inconsistent once at the end (unless an error occurred)
        if exc_type is None:
            self.graph_manager._mark_inconsistent()