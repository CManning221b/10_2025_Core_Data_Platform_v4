import json
import re
import os
import mimetypes
import pickle

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

from typing import Optional, List
from collections import defaultdict, Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from pyvis.network import Network
from jaal import Jaal
from dash import html


class GraphDataIngestion:
    '''
    Class to ingest data from various sources and transform it into a graph structure.

    Supported Inputs:
     - A Pandas DataFrame with accompanying metadata defining node types, hierarchy levels, edge types, directions, and connection logic.
     - A list of file paths (strings), from which a directory graph can be constructed.
     - A pre-structured JSON tree (e.g., from a hierarchical API or nested file format).
     - A folder containing files and subfolders. Optional enrichment includes:
         - File-level metadata (e.g., MIME type, size, timestamps)
         - File content (e.g., embeddings, textual data)

    Goal:
     - Construct a graph representation of arbitrary structured or semi-structured data.

    Outputs:
     - A dictionary (JSON-serializable) of nodes: {node_id: {node attributes}}
     - A dictionary (JSON-serializable) of edges: {(node_id_1, node_id_2): {edge attributes}}

    These outputs are designed to integrate with external graph frameworks for further analysis or visualsation.

    Visualszation:
     - Supports rendering via scalable tools such as PyVis, Dash Cytoscape, Netgraph, or Jaal.
     - Should enable both macro-level overviews and micro-level exploration of subgraphs or neighborhoods.
    '''

    def __init__(self):
        self.source_data = None                # Raw input (e.g., df, paths, folder tree, etc.)
        self.metadata = None                  # Optional metadata (for df mostly)
        self.graph = None                     # Networkit graph or other internal structure
        self.node_data = {}                   # Dict[node_id] = {attr}
        self.edge_data = {}                   # Dict[(src, tgt)] = {attr}
        self.node_id_map = {}                 # Mapping (type, value) → node_id
        self.visualizer = None                # Optional: track last used visualizer

    def reset(self):
        '''
        Reset key values to their default state.
        '''
        self.source_data = None
        self.metadata = None
        self.graph = None
        self.node_data.clear()
        self.edge_data.clear()
        self.node_id_map.clear()
        self.visualizer = None

    def _get_or_create_node_id(self, type_str, value):
        '''
        Generate a node id from a type and a value, node id is arbitrary to some degree but must be reliable
        '''
        key = (type_str, value)
        if key not in self.node_id_map:
            node_id = len(self.node_id_map)
            self.node_id_map[key] = node_id
            self.node_data[node_id] = {"type": type_str, "value": value}
        return self.node_id_map[key]


    def validate_metadata(self) -> bool:
        ''' Check that metadata contains all required columns. '''
        required_columns = {
            'column', 'node_type', 'edge_type', 'direction', 'hierarchy_level', 'connects_to'
        }
        if not isinstance(self.metadata, pd.DataFrame):
            print("Metadata is not a DataFrame.")
            return False
        metadata_columns = set(self.metadata.columns)
        missing = required_columns - metadata_columns
        if missing:
            print(f"Metadata is missing required columns: {missing}")
            return False
        # Optional: check for NaN in required columns
        for col in required_columns:
            if self.metadata[col].isnull().any():
                print(f"Metadata column '{col}' contains null values.")
                return False
        return True

    def ingest_from_dataframe(self, df: pd.DataFrame, metadata: Optional[pd.DataFrame] = None):
        '''
        Take metadata - metadata describes Node Type, Hierarchy Level, Edge Types, Directions, and Connection Logic.
        Metadata is a dataframe of its own. It lists each column in the original dataframe
        For each column we suggest a node type and hierarchy level, this allows us to go through that column and a make a node
        of each value in the column. We also find an edge that connects that value to the larger graph, we define the edge type (label),
        edge direction (is this column the source or target, node hierarchy helps with this) and what other value in this row (other column)
        it connects to. This way each value in the dataframe is connected to everything else in its row somehow.
        Then connections across rows (same value, in the same column) connect these sub-graphs to the larger graph.
        Check validity of metadata, does it have necessary values, does it have elements for each column in the dataframe.
        For each row in dataframe, assume all values in rows are nodes.
        Add each node to our JSON dictionary, including node id, name, value types, hierarchy as well as value.
        Build edges between nodes in the same row, add them to the dictionary. Edges have types, labels, direction, source, and target.
        '''

        self.metadata = metadata
        if not self.validate_metadata():
            raise ValueError("Invalid metadata. Aborting ingestion.")

        # --- Step 1: Create nodes ---
        for _, row in df.iterrows():
            for _, meta in self.metadata.iterrows():
                col = meta['column']
                node_type = meta['node_type']
                hierarchy_level = meta['hierarchy_level']

                value = row.get(col)
                if pd.isna(value) or value is None:
                    continue

                # Use node_type (not column name) to prevent duplication
                node_id = self._get_or_create_node_id(node_type, value)

                if node_id not in self.node_data:
                    self.node_data[node_id] = {
                        'value': value,
                        'column': col,
                        'node_type': node_type,
                        'hierarchy_level': hierarchy_level
                    }

        # --- Step 2: Create edges ---
        for _, row in df.iterrows():
            for _, meta in self.metadata.iterrows():
                src_col = meta['column']
                tgt_col = meta['connects_to']
                edge_type = meta['edge_type']
                direction = meta['direction']

                src_val = row.get(src_col)
                tgt_val = row.get(tgt_col) if tgt_col != 'central' else None

                if pd.isna(src_val) or src_val is None or (tgt_val is not None and pd.isna(tgt_val)):
                    continue

                src_type = meta['node_type']
                tgt_type = self.metadata[self.metadata['column'] == tgt_col]['node_type'].values[
                    0] if tgt_col != 'central' else None

                src_node_id = self._get_or_create_node_id(src_type, src_val)
                tgt_node_id = self._get_or_create_node_id(tgt_type, tgt_val) if tgt_col != 'central' else None

                if tgt_node_id is not None:
                    edge_key = (src_node_id, tgt_node_id) if direction == 'out' else (tgt_node_id, src_node_id)
                    if edge_key not in self.edge_data:
                        self.edge_data[edge_key] = {
                            'edge_type': edge_type,
                            'direction': direction,
                            'source_column': src_col,
                            'target_column': tgt_col
                        }

        print(f"Ingested {len(self.node_data)} nodes and {len(self.edge_data)} edges.")

    def ingest_from_paths(self, listPaths: List[str]):
        '''
        Take a LIST of file paths (strings) (maybe of different sorts, windows, unix etc.),
        Normalizing path
        Break each path into folders and filenames.
        Take each folder name and filename and tokenize it based on any common delimiters (_, .) found in other names at that same level
        in the tree hierarchy.
        Make a tree like structure:
            - Source folder into subfolders
            - subfolders into subfolders
            - subfolders and folders both have foldernames
            - foldernames have text components
            - subfolders into files
            - files have file addresses and filenames
            - filenames have text components
        The final tree has components connected to the names, and the names in the tree. The components are attached to the place they came from
        Like fruit on the tree.
        This allows us to preserve the original files and folders but to also have a way of connecting common text components of separate files and folders.

        Add node's id, name, values, hierarchy and types to json dictionary. In this case types are (folders - files - filenames - text components - file extension)
        Add edges to json dictionary: Including edge types, labels and directions, source and target.
        '''

        normalized_paths = [os.path.normpath(path) for path in listPaths]
        df_paths = pd.DataFrame(normalized_paths, columns=['full_path'])
        df_paths['dirname'] = df_paths['full_path'].apply(os.path.dirname)
        df_paths['basename'] = df_paths['full_path'].apply(os.path.basename)
        df_paths['filename'], df_paths['ext'] = zip(*df_paths['basename'].apply(lambda x: os.path.splitext(x)))

        # Split into folder components
        def split_path_parts(p):
            return list(Path(p).parts)

        df_paths['folders'] = df_paths['dirname'].apply(split_path_parts)
        max_depth = df_paths['folders'].apply(len).max()

        # Build a dictionary of components at each depth
        level_names = defaultdict(list)
        for folders in df_paths['folders']:
            for i, folder in enumerate(folders):
                level_names[i].append(folder)
        level_names[max_depth] += df_paths['filename'].tolist()

        # Detect common delimiters per level
        def get_delimiters(strs, threshold=0.2):
            delims = ['_', '-', '.', ' ']
            total = len(strs)
            best = []
            for delim in delims:
                count = sum(1 for s in strs if delim in s)
                if count / total >= threshold:
                    best.append(delim)
            return best

        delimiters_by_level = {
            level: get_delimiters(names)
            for level, names in level_names.items()
        }

        # Split string into components based on selected delimiters
        def split_string_components(s, delims):
            if not delims:
                return [s]
            pattern = '|'.join(map(re.escape, delims))
            return [t for t in re.split(pattern, s) if t]

        current_node_id = len(self.node_id_map)

        for _, row in df_paths.iterrows():
            folders = row['folders']
            filename = row['filename']
            ext = row['ext']
            full_path = row['full_path']

            parent_id = None

            # --- Create folder hierarchy ---
            for level, folder in enumerate(folders):
                key = ('folder', '/'.join(folders[:level + 1]))
                if key not in self.node_id_map:
                    self.node_id_map[key] = current_node_id
                    self.node_data[current_node_id] = {
                        'value': folder,
                        'type': 'folder',
                        'hierarchy_level': level
                    }
                    if parent_id is not None:
                        self.edge_data[(parent_id, current_node_id)] = {
                            'edge_type': 'contains',
                            'direction': 'out',
                            'source_type': 'folder',
                            'target_type': 'folder'
                        }
                    parent_id = current_node_id
                    current_node_id += 1
                else:
                    parent_id = self.node_id_map[key]

                # Attach components to folder name
                components = split_string_components(folder, delimiters_by_level.get(level, []))
                for component in components:
                    comp_key = ('component', component)
                    if comp_key not in self.node_id_map:
                        self.node_id_map[comp_key] = current_node_id
                        self.node_data[current_node_id] = {
                            'value': component,
                            'type': 'component',
                            'hierarchy_level': None
                        }
                        current_node_id += 1
                    comp_id = self.node_id_map[comp_key]
                    self.edge_data[(self.node_id_map[key], comp_id)] = {
                        'edge_type': 'has_component',
                        'direction': 'out',
                        'source_type': 'folder',
                        'target_type': 'component'
                    }

            # --- Add file node ---
            file_key = ('file', full_path)
            if file_key not in self.node_id_map:
                self.node_id_map[file_key] = current_node_id
                self.node_data[current_node_id] = {
                    'value': full_path,
                    'type': 'file',
                    'hierarchy_level': max_depth
                }
                self.edge_data[(parent_id, current_node_id)] = {
                    'edge_type': 'contains',
                    'direction': 'out',
                    'source_type': 'folder',
                    'target_type': 'file'
                }
                file_id = current_node_id
                current_node_id += 1
            else:
                file_id = self.node_id_map[file_key]

            # --- Add filename node ---
            name_key = ('filename', filename)
            if name_key not in self.node_id_map:
                self.node_id_map[name_key] = current_node_id
                self.node_data[current_node_id] = {
                    'value': filename,
                    'type': 'filename',
                    'hierarchy_level': max_depth + 1
                }
                self.edge_data[(file_id, current_node_id)] = {
                    'edge_type': 'has_name',
                    'direction': 'out',
                    'source_type': 'file',
                    'target_type': 'filename'
                }
                name_id = current_node_id
                current_node_id += 1
            else:
                name_id = self.node_id_map[name_key]

            # Attach components to filename
            components = split_string_components(filename, delimiters_by_level.get(max_depth, []))
            for component in components:
                comp_key = ('component', component)
                if comp_key not in self.node_id_map:
                    self.node_id_map[comp_key] = current_node_id
                    self.node_data[current_node_id] = {
                        'value': component,
                        'type': 'component',
                        'hierarchy_level': None
                    }
                    current_node_id += 1
                comp_id = self.node_id_map[comp_key]
                self.edge_data[(name_id, comp_id)] = {
                    'edge_type': 'has_component',
                    'direction': 'out',
                    'source_type': 'filename',
                    'target_type': 'component'
                }

            # --- Add file extension node ---
            if ext:
                ext = ext.lower()
                ext_key = ('extension', ext)
                if ext_key not in self.node_id_map:
                    self.node_id_map[ext_key] = current_node_id
                    self.node_data[current_node_id] = {
                        'value': ext,
                        'type': 'extension',
                        'hierarchy_level': None
                    }
                    current_node_id += 1
                ext_id = self.node_id_map[ext_key]
                self.edge_data[(file_id, ext_id)] = {
                    'edge_type': 'has_extension',
                    'direction': 'out',
                    'source_type': 'file',
                    'target_type': 'extension'
                }

        print(f"Ingested {len(self.node_data)} nodes and {len(self.edge_data)} edges.")

    def ingest_from_json(self, json_obj: dict):
        if not hasattr(self, 'node_id_map'):
            self.node_id_map = {}
        if not hasattr(self, 'node_data'):
            self.node_data = {}
        if not hasattr(self, 'edge_data'):
            self.edge_data = {}

        self._node_counter = 0

        def add_node(value, node_type):
            node_id = self._node_counter
            self.node_data[node_id] = {
                'value': value,
                'type': node_type,
            }
            self._node_counter += 1
            return node_id

        def add_edge(source_id, target_id, label, source_type, target_type):
            edge_key = (source_id, target_id)
            if edge_key not in self.edge_data:
                self.edge_data[edge_key] = {
                    'edge_type': label,
                    'direction': 'out',
                    'source_type': source_type,
                    'target_type': target_type,
                }

        def traverse(obj, parent_id=None, parent_type=None, key_from_parent=None):
            if isinstance(obj, dict):
                node_id = add_node(value=None, node_type='dict')
                if parent_id is not None:
                    add_edge(parent_id, node_id, key_from_parent, parent_type, 'dict')
                for k, v in obj.items():
                    traverse(v, parent_id=node_id, parent_type='dict', key_from_parent=k)
            elif isinstance(obj, list):
                node_id = add_node(value=None, node_type='list')
                if parent_id is not None:
                    add_edge(parent_id, node_id, key_from_parent, parent_type, 'list')
                for idx, item in enumerate(obj):
                    traverse(item, parent_id=node_id, parent_type='list', key_from_parent=str(idx))
            else:
                # Primitive value node
                val_type = type(obj).__name__
                node_id = add_node(value=obj, node_type=val_type)
                if parent_id is not None:
                    add_edge(parent_id, node_id, key_from_parent, parent_type, val_type)

        root_id = add_node("root", "root")
        traverse(json_obj, parent_id=root_id, parent_type="root", key_from_parent="root")

        print(f"Ingested {len(self.node_data)} nodes and {len(self.edge_data)} edges from JSON.")

    def ingest_from_folder(self, folder_path: str):
        """
        Mimics ingest_from_paths using actual directory traversal, but adds file size and MIME type.
        The resulting graph will be identical to ingest_from_paths, plus file property nodes.
        """
        import mimetypes

        # Step 1: Walk the directory and collect all file paths
        all_file_paths = []
        for dirpath, _, filenames in os.walk(folder_path):
            for f in filenames:
                full_path = os.path.join(dirpath, f)
                all_file_paths.append(os.path.normpath(full_path))

        # Step 2: Use existing path ingestion method to build the structure
        self.ingest_from_paths(all_file_paths)

        # Step 3: Add file properties (size and type) to each file node
        for path in all_file_paths:
            file_key = ('file', os.path.normpath(path))
            if file_key not in self.node_id_map:
                continue  # Shouldn’t happen unless paths weren't ingested properly

            file_id = self.node_id_map[file_key]

            # --- Add file size node ---
            try:
                file_size = os.path.getsize(path)
                size_key = ('size', file_size)
                if size_key not in self.node_id_map:
                    size_id = len(self.node_id_map)
                    self.node_id_map[size_key] = size_id
                    self.node_data[size_id] = {
                        'value': file_size,
                        'type': 'file_size',
                        'hierarchy_level': None
                    }
                    self.edge_data[(file_id, size_id)] = {
                        'edge_type': 'has_size',
                        'direction': 'out',
                        'source_type': 'file',
                        'target_type': 'file_size'
                    }

            except Exception:
                pass  # Ignore inaccessible files

            # --- Add file type (MIME) node ---
            mime_type, _ = mimetypes.guess_type(path)
            if mime_type is None:
                mime_type = "Unknown"

            type_key = ('file_type', mime_type)
            if type_key not in self.node_id_map:
                type_id = len(self.node_id_map)
                self.node_id_map[type_key] = type_id
                self.node_data[type_id] = {
                    'value': mime_type,
                    'type': 'file_type',
                    'hierarchy_level': None
                }
                self.edge_data[(file_id, type_id)] = {
                    'edge_type': 'has_type',
                    'direction': 'out',
                    'source_type': 'file',
                    'target_type': 'file_type'
                }

        print(f"Ingested {len(self.node_data)} nodes and {len(self.edge_data)} edges (with file properties).")

    def _add_edge(self, src_id, tgt_id, label=None, direction="undirected"):
        '''
        Add edge based on src and tgt & add label and direction as metadata.
        '''
        key = (src_id, tgt_id)
        if key not in self.edge_data:
            self.edge_data[key] = {
                "label": label,
                "direction": direction
            }


    def _type_from_json_depth(self, depth: int) -> str:
        '''
        Take Json depth and convert it into an appropriate type.
        '''
        return f"Group_{depth}"


    def dictionary_to_JSON(self, return_dict: bool = True, filepath: Optional[str] = None):
        data = {
            "nodes": self.node_data,
            "edges": {f"{k[0]}_{k[1]}": v for k, v in self.edge_data.items()}
        }

        if filepath:
            print("Making file as JSON...")
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2, cls=NpEncoder)
        if return_dict:
            return data

    def visualize_pyvis(self, filename: str = "graph.html", height="800px", width="100%", physics=True,
                        spacing_factor=1.0):
        '''
        Generate an enhanced PyVis visualization of the graph

        Args:
            filename: Path to save the HTML output file
            height: Height of the visualization container
            width: Width of the visualization container
            physics: Whether to enable physics simulation
            spacing_factor: Controls spacing between nodes (higher values = more space)

        Returns:
            The PyVis Network object
        '''
        try:
            from pyvis.network import Network
        except ImportError:
            raise ImportError("PyVis not installed. Install with: pip install pyvis")

        # Create network with appropriate settings
        net = Network(height=height, width=width, notebook=False, directed=True)

        # Configure physics
        if physics:
            net.barnes_hut(
                gravity=-spacing_factor * 10000,
                central_gravity=0.1,
                spring_length=spacing_factor * 150
            )
        else:
            net.repulsion(
                node_distance=spacing_factor * 150,
                spring_length=spacing_factor * 150
            )

        # Define color scheme for node types
        node_types = set()
        for data in self.node_data.values():
            node_type = data.get("type", "default")
            node_types.add(node_type)

        color_map = {}
        colors = ["#4285F4", "#EA4335", "#FBBC05", "#34A853", "#8E24AA", "#16A085", "#D35400", "#C0392B", "#7F8C8D",
                  "#2C3E50"]
        for i, node_type in enumerate(node_types):
            color_map[node_type] = colors[i % len(colors)]

        # Define color scheme for edge types
        edge_types = set(data.get("edge_type", "default") for data in self.edge_data.values())
        edge_colors = {}
        for i, edge_type in enumerate(edge_types):
            edge_colors[edge_type] = colors[i % len(colors)]

        # Add nodes
        for node_id, data in self.node_data.items():
            label = str(data.get("value", node_id))
            node_type = data.get("type", "default")

            # Create detailed tooltip
            tooltip = f""
            for key, value in data.items():
                tooltip += f"{key}: {value} | "
            tooltip += " "

            net.add_node(
                node_id,
                label=label,
                title=tooltip,
                group=node_type,
                color=color_map.get(node_type, "#7F8C8D"),
                size=25,
                font={'size': 12, 'color': 'black'}
            )

        # Add edges
        for (src, tgt), data in self.edge_data.items():
            edge_type = data.get("edge_type", "")

            # Create detailed tooltip
            tooltip = f""
            for key, value in data.items():
                tooltip += f"{key}: {value} |"
            tooltip += " "

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

        # Set enhanced visualization options
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
              "iterations": 200
            },
            "barnesHut": {
              "springConstant": 0.01,
              "avoidOverlap": 0.5
            }
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

        # Save visualization
        net.save_graph(filename)

        return net

    def visualize_cytoscape(self):
        '''
        Read JSON files and add nodes, edges and their data to visualisation
        '''
        G = nx.DiGraph()

        # Add nodes and edges
        for node_id, data in self.node_data.items():
            G.add_node(node_id, label=data.get("value", node_id), **data)

        for (src, tgt), data in self.edge_data.items():
            G.add_edge(src, tgt, label=data.get("edge_type", ""), **data)

        pos = nx.spring_layout(G, seed=42)
        node_labels = nx.get_node_attributes(G, 'label')
        edge_labels = nx.get_edge_attributes(G, 'label')

        plt.figure(figsize=(12, 8))
        nx.draw(G, pos, with_labels=True, labels=node_labels, node_size=500, node_color='lightblue', font_size=8)
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)
        plt.title("Cytoscape-style NetworkX Graph")
        plt.show()

    def visualize_jaal(self, title = 'value'):
        '''
        Read JSON files and add nodes, edges and their data to visualisation
        '''
        print('\n'.join(f'node id: {nid}, data: {data}' for nid, data in self.node_data.items()))
        nodes = [{"id": nid, **data} for nid, data in self.node_data.items()]
        edges = [{"from": src, "to": tgt, **edata} for (src, tgt), edata in self.edge_data.items()]

        df_nodes = pd.DataFrame(nodes)
        df_nodes['title'] = df_nodes[title]
        df_nodes['size'] = int(1)
        df_edges = pd.DataFrame(edges)

        pd.set_option('display.max_columns', None)
        print(df_nodes)


        Jaal(df_edges, df_nodes).plot()

    def visualize(self, method: str = "pyvis", **kwargs):
        if method == "pyvis":
            self.visualize_pyvis(**kwargs)
        elif method == "cytoscape":
            self.visualize_cytoscape()
        elif method == "jaal":
            self.visualize_jaal()
        else:
            raise ValueError(f"Unsupported visualization method: {method}")

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, set):
            return list(obj)
        return super().default(obj)
