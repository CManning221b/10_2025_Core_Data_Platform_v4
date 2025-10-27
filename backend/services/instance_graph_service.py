# backend/services/instance_graph_service.py
import os
import json
from datetime import datetime
import glob
from typing import Dict, List, Tuple, Any, Optional
from backend.core_graph_managers.no2_graphManager.graphManager import GraphManager


class InstanceGraphService:
    def __init__(self):
        self.graphs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'graphs')
        self.graph_manager = GraphManager()

    def get_graph_data(self, graph_id):
        """
        Retrieve graph data by its ID

        Args:
            graph_id (str): The unique identifier of the graph

        Returns:
            dict: The graph data or None if not found
        """
        graph_path = os.path.join(self.graphs_dir, f"{graph_id}.json")
        if not os.path.exists(graph_path):
            return None

        try:
            with open(graph_path, 'r') as f:
                graph_data = json.load(f)
            return graph_data
        except Exception as e:
            print(f"Error loading graph {graph_id}: {str(e)}")
            return None

    def get_graph_manager(self, graph_id):
        """
        Load graph data into a GraphManager instance for visualization

        Args:
            graph_id (str): The unique identifier of the graph

        Returns:
            GraphManager: A loaded graph manager instance or None if not found
        """
        graph_path = os.path.join(self.graphs_dir, f"{graph_id}.json")
        if not os.path.exists(graph_path):
            return None

        try:
            graph_manager = GraphManager()
            graph_manager.load_from_json(graph_path)
            return graph_manager
        except Exception as e:
            print(f"Error loading graph {graph_id} into GraphManager: {str(e)}")
            return None

    def generate_pyvis_html(self, graph_id):
        """
        Generate a PyVis HTML visualization for a graph

        Args:
            graph_id (str): The unique identifier of the graph

        Returns:
            str: Path to the generated HTML file or None if failed
        """
        graph_manager = self.get_graph_manager(graph_id)

        if not graph_manager:
            return None

        # Create directory for visualizations if it doesn't exist
        vis_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                               'app', 'static', 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)

        # Generate HTML file path
        html_path = os.path.join(vis_dir, f"{graph_id}.html")

        try:
            # Generate visualization using GraphManager's PyVis method
            graph_manager.render_pyvis(html_path,
                                       height="600px",
                                       width="100%",
                                       physics=True,
                                       spacing_factor=1.2)

            # Return the path relative to the static directory
            return f"visualizations/{graph_id}.html"
        except Exception as e:
            print(f"Error generating PyVis visualization for graph {graph_id}: {str(e)}")
            return None

    def list_graphs(self):
        """
        Alias for get_all_graphs to maintain compatibility with existing code

        Returns:
            list: List of graph metadata dictionaries
        """
        return self.get_all_graphs()

    def get_all_graphs(self):
        """
        Get a list of all available graphs in the system

        Returns:
            list: List of graph metadata dictionaries
        """
        graphs = []

        # Check for graph files in the graphs directory
        if os.path.exists(self.graphs_dir):
            for filename in os.listdir(self.graphs_dir):
                if filename.endswith('.json'):
                    graph_id = filename.split('.')[0]  # Remove file extension

                    # Try to load basic graph info
                    try:
                        graph_path = os.path.join(self.graphs_dir, filename)
                        with open(graph_path, 'r') as f:
                            graph_data = json.load(f)

                        # Extract basic metadata
                        graph_info = {
                            'id': graph_id,
                            'name': graph_data.get('name', f'Graph {graph_id}'),
                            'node_count': len(graph_data.get('nodes', {})),
                            'edge_count': len(graph_data.get('edges', [])),
                            'created': graph_data.get('created', 'Unknown'),
                            'modified': graph_data.get('modified', 'Unknown')
                        }

                        graphs.append(graph_info)
                    except Exception as e:
                        # If there's an error loading the graph, include minimal info
                        graphs.append({
                            'id': graph_id,
                            'name': f'Graph {graph_id}',
                            'node_count': 0,
                            'edge_count': 0,
                            'error': 'Error loading graph data'
                        })

        # Sort graphs by ID
        graphs.sort(key=lambda g: g['id'])

        return graphs

    def generate_search_subgraph_html(self, graph_id: str, subgraph_data: Dict, highlighted_nodes: List[Dict],
                                      highlighted_edges: List[Dict]) -> str:
        """Generate PyVis HTML for search result subgraph with highlighting"""
        try:
            from pyvis.network import Network
            import hashlib

            # Create network with explicit dimensions
            net = Network(
                height="600px",  # Fixed height
                width="100%",  # Full width
                bgcolor="#ffffff",
                font_color="black",
                directed=True
            )

            # Configure physics and layout for proper sizing
            net.set_options("""
            var options = {
              "physics": {
                "enabled": true,
                "stabilization": {"iterations": 150},
                "barnesHut": {
                  "gravitationalConstant": -8000,
                  "centralGravity": 0.3,
                  "springLength": 95,
                  "springConstant": 0.04,
                  "damping": 0.09
                }
              },
              "edges": {
                "arrows": {"to": {"enabled": true, "scaleFactor": 0.8}},
                "smooth": {"enabled": true, "type": "continuous"},
                "font": {"size": 10}
              },
              "nodes": {
                "font": {"size": 12},
                "borderWidth": 2,
                "shadow": true
              },
              "layout": {
                "improvedLayout": true
              },
              "interaction": {
                "dragNodes": true,
                "dragView": true,
                "zoomView": true
              }
            }
            """)

            # Dynamic color generator based on node type
            def get_color_by_type(node_type):
                # Generate a consistent color based on the node type string
                # This ensures same types always get the same color
                hash_object = hashlib.md5(node_type.encode())
                hash_hex = hash_object.hexdigest()

                # Convert first 6 characters to RGB
                r = int(hash_hex[0:2], 16)
                g = int(hash_hex[2:4], 16)
                b = int(hash_hex[4:6], 16)

                # Ensure colors are bright enough to be visible
                r = max(r, 100)
                g = max(g, 100)
                b = max(b, 100)

                return f"#{r:02x}{g:02x}{b:02x}"

            # Get node IDs that were highlighted
            highlighted_node_ids = set([node['id'] for node in highlighted_nodes])
            highlighted_edge_ids = set([edge['id'] for edge in highlighted_edges])

            print(f"DEBUG: Highlighting {len(highlighted_node_ids)} nodes and {len(highlighted_edge_ids)} edges")
            print(
                f"DEBUG: Subgraph has {len(subgraph_data.get('nodes', {}))} nodes and {len(subgraph_data.get('edges', {}))} edges")

            # Add nodes with dynamic type-based coloring
            for node_id, node_data in subgraph_data.get('nodes', {}).items():
                # Get color based on node type
                node_type = node_data.get('type', 'unknown')
                color = get_color_by_type(node_type)

                # Only change size and border for highlighted nodes
                if node_id in highlighted_node_ids:
                    size = 30
                    border_width = 4
                    border_color = "#ff0000"  # Red border for matches
                else:
                    size = 20
                    border_width = 2
                    border_color = "#000000"  # Black border for context

                # Create label
                label = str(node_data.get('value', node_id))
                if len(label) > 20:
                    label = label[:17] + "..."

                # Create comprehensive title for hover
                title = f"ID: {node_id}\nType: {node_data.get('type', 'unknown')}"
                for key, value in node_data.items():
                    if key not in ['type'] and isinstance(value, (str, int, float)):
                        title += f"\n{key}: {value}"

                if node_id in highlighted_node_ids:
                    title = "SEARCH MATCH\n" + title

                net.add_node(
                    node_id,
                    label=label,
                    title=title,
                    color=color,
                    size=size,
                    borderWidth=border_width,
                    borderWidthSelected=border_width + 2
                )

            # Add ALL edges - this is the key fix
            edge_count = 0
            print(f"DEBUG: Processing {len(subgraph_data.get('edges', {}))} edges from subgraph")

            for edge_id, edge_data in subgraph_data.get('edges', {}).items():
                source = edge_data.get('source')
                target = edge_data.get('target')

                # If source/target are None, try to parse from edge ID
                if source is None or target is None:
                    if '_' in edge_id:
                        parts = edge_id.split('_')
                        if len(parts) == 2:
                            source = parts[0]
                            target = parts[1]
                            print(f"DEBUG: Parsed edge {edge_id}: {source} -> {target}")

                print(f"DEBUG: Processing edge {edge_id}: {source} -> {target}")

                if source and target:
                    # Check if both nodes exist in the subgraph
                    if source in subgraph_data.get('nodes', {}) and target in subgraph_data.get('nodes', {}):
                        # Determine styling - only change width for highlighted edges
                        if edge_id in highlighted_edge_ids:
                            width = 4
                            color = "#ff4444"
                        else:
                            width = 2
                            color = "#999999"

                        # Create edge label
                        edge_type = edge_data.get('edge_type', 'connected')
                        label = edge_type if len(edge_type) < 15 else edge_type[:12] + "..."

                        # Create comprehensive title for hover
                        title = f"Edge: {edge_id}\nType: {edge_type}\nFrom: {source}\nTo: {target}"
                        for key, value in edge_data.items():
                            if key not in ['source', 'target', 'type'] and isinstance(value, (str, int, float)):
                                title += f"\n{key}: {value}"

                        if edge_id in highlighted_edge_ids:
                            title = "SEARCH MATCH\n" + title

                        net.add_edge(
                            source,
                            target,
                            label=label,
                            title=title,
                            color=color,
                            width=width
                        )
                        edge_count += 1
                        print(f"DEBUG: Added edge {edge_id}")
                    else:
                        print(f"DEBUG: Skipped edge {edge_id} - missing nodes ({source} or {target})")

            print(f"DEBUG: Successfully added {edge_count} edges to visualization")

            # Generate HTML file
            vis_dir = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                'app', 'static', 'visualizations'
            )
            os.makedirs(vis_dir, exist_ok=True)

            filename = f"{graph_id}_search_subgraph.html"
            filepath = os.path.join(vis_dir, filename)

            net.save_graph(filepath)

            # Simple enhancement without absolute positioning
            self.enhance_search_visualization(filepath)

            return filename

        except Exception as e:
            print(f"Error generating search subgraph visualization: {e}")
            return None

    def enhance_search_visualization(self, filepath: str):
        """Add enhancements to fix compressed display"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()

            # Add CSS to fix the compressed display
            enhancement = """
            <style>
            #mynetworkid {
                width: 100% !important;
                height: 600px !important;
                border: 1px solid #ddd;
            }
            body {
                margin: 0;
                padding: 0;
                font-family: Arial, sans-serif;
            }
            .vis-network {
                width: 100% !important;
                height: 600px !important;
            }
            </style>
            """

            # Insert before closing head tag
            content = content.replace('</head>', enhancement + '</head>')

            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)

        except Exception as e:
            print(f"Error enhancing search visualization: {e}")
