import os
import json
import uuid
import pandas as pd
from ..Utils.file_utils import get_file_extension
from ..core_graph_managers.no1_graphDataIngestor.graphDataIngestion import GraphDataIngestion, NpEncoder


class UploadService:
    def __init__(self):
        self.graph_ingestion = GraphDataIngestion()

    def _is_direct_graph_json(self, json_data):
        """
        Check if JSON is already in the direct graph format with nodes and edges

        Args:
            json_data (dict): JSON data to check

        Returns:
            bool: True if it's a direct graph format, False otherwise
        """
        # Check if it has both 'nodes' and 'edges' keys
        if not isinstance(json_data, dict) or 'nodes' not in json_data or 'edges' not in json_data:
            return False

        nodes = json_data['nodes']
        edges = json_data['edges']

        # Check if nodes is a dict with numeric string keys
        if not isinstance(nodes, dict):
            return False

        # Check if at least some node keys are numeric strings
        node_keys = list(nodes.keys())
        if not node_keys:
            return False

        # Try to convert first few keys to integers to verify they're numeric
        try:
            for key in node_keys[:3]:  # Check first 3 keys
                int(key)
        except (ValueError, TypeError):
            return False

        # Check if edges is a dict with underscore-separated numeric keys
        if not isinstance(edges, dict):
            return False

        edge_keys = list(edges.keys())
        if not edge_keys:
            return True  # Empty edges is still valid

        # Check edge key format (should be like "0_1", "2_3", etc.)
        try:
            for key in edge_keys[:3]:  # Check first 3 keys
                parts = key.split('_')
                if len(parts) != 2:
                    return False
                int(parts[0])
                int(parts[1])
        except (ValueError, TypeError, AttributeError):
            return False

        return True

    def _load_direct_graph_json(self, json_data):
        """
        Load JSON data directly into the graph ingestion system

        Args:
            json_data (dict): Direct graph JSON data
        """
        # Reset the graph ingestion system
        self.graph_ingestion.reset()

        # Load nodes directly
        nodes = json_data['nodes']
        for node_id, node_data in nodes.items():
            # Convert string ID to integer for internal consistency
            int_id = int(node_id)
            self.graph_ingestion.node_data[int_id] = node_data

            # Update node_id_map if needed for consistency
            if hasattr(node_data, 'get'):
                node_type = node_data.get('type', 'unknown')
                node_value = node_data.get('value', node_id)
                key = (node_type, node_value)
                self.graph_ingestion.node_id_map[key] = int_id

        # Load edges directly
        edges = json_data['edges']
        for edge_key, edge_data in edges.items():
            # Parse edge key (e.g., "0_1" -> (0, 1))
            parts = edge_key.split('_')
            src_id = int(parts[0])
            tgt_id = int(parts[1])
            edge_tuple = (src_id, tgt_id)

            self.graph_ingestion.edge_data[edge_tuple] = edge_data

    def process_file(self, file_path):
        """
        Process the uploaded file and return metadata

        Args:
            file_path (str): Path to the uploaded file

        Returns:
            dict: Metadata about the processed graph
        """
        # Generate a unique ID for this graph
        graph_id = str(uuid.uuid4())

        # Determine file type
        file_extension = get_file_extension(file_path)

        # Process based on file type
        if file_extension == '.csv':
            # Process CSV - assuming the GraphDataIngestion has this method
            self.graph_ingestion.ingest_from_dataframe(file_path)
        elif file_extension == '.json':
            # Process JSON file
            with open(file_path, 'r') as f:
                json_data = json.load(f)

            # Check if it's already in direct graph format
            if self._is_direct_graph_json(json_data):
                print("Detected direct graph JSON format - loading directly")
                self._load_direct_graph_json(json_data)
            else:
                print("Processing as hierarchical JSON")
                self.graph_ingestion.ingest_from_json(json_data)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")

        # Save state for later retrieval using the built-in method
        self._save_graph_state(graph_id)

        # Return metadata
        return {
            'graph_id': graph_id,
            'node_count': len(self.graph_ingestion.node_data),
            'edge_count': len(self.graph_ingestion.edge_data),
            'original_file': os.path.basename(file_path),
            'original_source': 'file'
        }

    def process_json(self, json_data):
        """
        Process JSON data directly and return metadata

        Args:
            json_data (dict): JSON data to process

        Returns:
            dict: Metadata about the processed graph
        """
        # Generate a unique ID for this graph
        graph_id = str(uuid.uuid4())

        # Check if it's already in direct graph format
        if self._is_direct_graph_json(json_data):
            print("Detected direct graph JSON format - loading directly")
            self._load_direct_graph_json(json_data)
        else:
            print("Processing as hierarchical JSON")
            self.graph_ingestion.ingest_from_json(json_data)

        # Save state for later retrieval
        self._save_graph_state(graph_id)

        # Return metadata
        return {
            'graph_id': graph_id,
            'node_count': len(self.graph_ingestion.node_data),
            'edge_count': len(self.graph_ingestion.edge_data),
            'original_source': 'json'
        }

    def process_dataframe(self, df, metadata_df):
        """
        Process a dataframe with metadata and return graph information

        Args:
            df (pandas.DataFrame): DataFrame containing the data to visualize
            metadata_df (pandas.DataFrame): DataFrame containing the metadata for graph construction

        Returns:
            dict: Metadata about the processed graph
        """
        # Generate a unique ID for this graph
        graph_id = str(uuid.uuid4())

        # Process dataframe with metadata
        self.graph_ingestion.ingest_from_dataframe(df, metadata_df)

        # Save state for later retrieval
        self._save_graph_state(graph_id)

        # Return metadata
        return {
            'graph_id': graph_id,
            'node_count': len(self.graph_ingestion.node_data),
            'edge_count': len(self.graph_ingestion.edge_data),
            'original_source': 'dataframe'
        }

    def process_paths(self, path_list):
        """
        Process a list of file paths and return metadata

        Args:
            path_list (list): List of file paths to process

        Returns:
            dict: Metadata about the processed graph
        """
        # Generate a unique ID for this graph
        graph_id = str(uuid.uuid4())

        # Process paths
        self.graph_ingestion.ingest_from_paths(path_list)

        # Save state for later retrieval
        self._save_graph_state(graph_id)

        # Return metadata
        return {
            'graph_id': graph_id,
            'node_count': len(self.graph_ingestion.node_data),
            'edge_count': len(self.graph_ingestion.edge_data),
            'path_count': len(path_list),
            'original_source': 'paths'
        }

    def process_directory(self, directory_path, recursive=True):
        """
        Process a directory and return metadata

        Args:
            directory_path (str): Path to the directory to process
            recursive (bool): Whether to process subdirectories recursively

        Returns:
            dict: Metadata about the processed graph
        """
        # Generate a unique ID for this graph
        graph_id = str(uuid.uuid4())

        # Process directory
        self.graph_ingestion.ingest_from_folder(directory_path)

        # Save state for later retrieval
        self._save_graph_state(graph_id)

        # Return metadata
        return {
            'graph_id': graph_id,
            'node_count': len(self.graph_ingestion.node_data),
            'edge_count': len(self.graph_ingestion.edge_data),
            'directory': directory_path,
            'original_source': 'directory'
        }

    def _save_graph_state(self, graph_id):
        """
        Save the current graph state for later retrieval using the built-in method

        Args:
            graph_id (str): Unique identifier for the graph
        """
        # Create a data directory if it doesn't exist
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'graphs')
        os.makedirs(data_dir, exist_ok=True)

        # Use the GraphDataIngestion's built-in method to save the graph
        file_path = os.path.join(data_dir, f"{graph_id}.json")
        self.graph_ingestion.dictionary_to_JSON(return_dict=False, filepath=file_path)

        # If you need to add metadata to the json file, you can do it here
        """
        # Alternative approach if you need more metadata:
        graph_data = self.graph_ingestion.dictionary_to_JSON(return_dict=True)
        graph_data['metadata'] = {
            'graph_id': graph_id,
            'node_count': len(self.graph_ingestion.node_data),
            'edge_count': len(self.graph_ingestion.edge_data)
        }

        # Save to file with the custom encoder
        with open(file_path, 'w') as f:
            json.dump(graph_data, f, indent=2, cls=NpEncoder)
        """