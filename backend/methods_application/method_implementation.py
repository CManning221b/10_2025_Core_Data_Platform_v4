class MethodImplementation:
    """Base class for all method implementations"""

    # Class variables to be overridden by subclasses
    method_id = None
    method_name = None
    description = None

    def __init__(self):
        if not self.method_id:
            raise ValueError("Method implementation must define method_id")

    def execute(self, graph_manager, parameters):
        """
        Execute the method on the graph

        Args:
            graph_manager (GraphManager): The graph to modify
            parameters (dict): Method parameters

        Returns:
            dict: Details of the changes made
        """
        raise NotImplementedError("Method implementations must override execute()")

    def validate_parameters(self, parameters):
        """
        Validate method parameters

        Args:
            parameters (dict): Method parameters

        Returns:
            bool: True if parameters are valid, False otherwise
        """
        return True  # Default implementation accepts any parameters

    def get_nodes_by_type(self, graph_manager, node_type):
        """
        Helper method to get nodes of a specific type

        Args:
            graph_manager (GraphManager): The graph to query
            node_type (str): Type of nodes to find

        Returns:
            dict: Dictionary of node_id: node_data for nodes of the specified type
        """
        if not graph_manager or not hasattr(graph_manager, 'node_data'):
            return {}

        return {
            node_id: node_data
            for node_id, node_data in graph_manager.node_data.items()
            if node_data.get('type') == node_type
        }

    def get_connected_nodes(self, graph_manager, node_id, edge_type=None, direction='outgoing'):
        """
        Helper method to get nodes connected to a specific node

        Args:
            graph_manager (GraphManager): The graph to query
            node_id (str): ID of the node to start from
            edge_type (str, optional): Type of edges to follow
            direction (str): 'outgoing', 'incoming', or 'both'

        Returns:
            dict: Dictionary of node_id: edge_data for connected nodes
        """
        if not graph_manager or not hasattr(graph_manager, 'edge_data'):
            return {}

        connected_nodes = {}

        # Check outgoing edges
        if direction in ['outgoing', 'both']:
            for edge_id, edge_data in graph_manager.edge_data.items():
                if edge_data.get('source') == node_id:
                    # Check edge type if specified
                    if edge_type is None or edge_data.get('edge_type') == edge_type or edge_data.get('attributes',
                                                                                                     {}).get(
                            'edge_type') == edge_type:
                        target_id = edge_data.get('target')
                        connected_nodes[target_id] = edge_data

        # Check incoming edges
        if direction in ['incoming', 'both']:
            for edge_id, edge_data in graph_manager.edge_data.items():
                if edge_data.get('target') == node_id:
                    # Check edge type if specified
                    if edge_type is None or edge_data.get('edge_type') == edge_type or edge_data.get('attributes',
                                                                                                     {}).get(
                            'edge_type') == edge_type:
                        source_id = edge_data.get('source')
                        connected_nodes[source_id] = edge_data

        return connected_nodes

    def generate_unique_id(self, graph_manager, prefix):
        """
        Generate a unique ID for a new node or edge

        Args:
            graph_manager (GraphManager): The graph to query
            prefix (str): Prefix for the ID

        Returns:
            str: A unique ID
        """
        # Get existing IDs
        node_ids = list(graph_manager.node_data.keys()) if hasattr(graph_manager, 'node_data') else []
        edge_ids = list(graph_manager.edge_data.keys()) if hasattr(graph_manager, 'edge_data') else []

        all_ids = node_ids + edge_ids

        # Find the highest number for the given prefix
        max_num = 0
        for id_str in all_ids:
            if id_str.startswith(prefix):
                try:
                    # Extract number part
                    num_part = id_str[len(prefix):]
                    if num_part.isdigit():
                        num = int(num_part)
                        max_num = max(max_num, num)
                except (ValueError, IndexError):
                    pass

        # Generate new ID
        return f"{prefix}{max_num + 1}"