from pyvis.network import Network
from neo4j import GraphDatabase
import json
import random

# Neo4j Connection Details
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "password"

class KnowledgeGraph:
    def __init__(self):
        """Initialize an empty knowledge graph and Neo4j connection."""
        self.nodes = {}  # Dictionary for storing nodes
        self.edges = []  # List for storing edges
        self.node_colors = {}  # Store colors for visualization

        # Neo4j Driver
        # self.driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    def close(self):
        """Close the Neo4j connection."""
        self.driver.close()

    def add_node(self, node_id, node_type, properties=None):
        """Add a node to the knowledge graph."""
        if node_id not in self.nodes:
            self.nodes[node_id] = {"type": node_type, "properties": properties or {}}

    def add_edge(self, source, target, edge_type):
        """Add a directed edge between nodes, preventing duplicate edges of the same type."""
        existing_edges = [(e["source"], e["target"], e["type"]) for e in self.edges]

        if (source, target, edge_type) not in existing_edges:
            self.edges.append({"source": source, "target": target, "type": edge_type})

    def remove_node(self, node_id):
        """Remove a node and all its connected edges from the graph."""
        if node_id in self.nodes:
            del self.nodes[node_id]
        self.edges = [e for e in self.edges if e["source"] != node_id and e["target"] != node_id]

    def _generate_random_color(self):
        """Generate a random HEX color."""
        return "#{:06x}".format(random.randint(0, 0xFFFFFF))

    def get_node_color(self, node_type):
        """Get or assign a color for a node type."""
        if node_type not in self.node_colors:
            self.node_colors[node_type] = self._generate_random_color()
        return self.node_colors[node_type]

    def to_json(self):
        """Return the knowledge graph as a JSON object."""
        return json.dumps({"nodes": self.nodes, "edges": self.edges}, indent=4)

    def visualize(self, output_file="graph.html", highlighted_nodes=None, height="800px", width="100%", physics=True,
                  spacing_factor=1.0):
        """
        Visualize the graph using PyVis interactive HTML visualization.

        Args:
            output_file: Path to save the HTML output file
            highlighted_nodes: List of node values to highlight
            height: Height of the visualization container
            width: Width of the visualization container
            physics: Whether to enable physics simulation
            spacing_factor: Controls spacing between nodes (higher values = more space)

        Returns:
            The PyVis Network object
        """
        try:
            from pyvis.network import Network
        except ImportError:
            raise ImportError("PyVis not installed. Install with: pip install pyvis")

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
        for data in self.nodes.values():
            node_type = data.get("type", "default")
            node_types.add(node_type)

        # Create a color map for node types
        color_map = {}
        colors = ["#4285F4", "#EA4335", "#FBBC05", "#34A853", "#8E24AA", "#16A085", "#D35400", "#C0392B", "#7F8C8D",
                  "#2C3E50"]
        for i, node_type in enumerate(node_types):
            color_map[node_type] = colors[i % len(colors)]

        # Define highlight colors for different types of highlights
        highlight_colors = {
            'matching': "#27AE60",  # Green for exact matches
            'fuzzy': "#3498DB",  # Blue for fuzzy matches
            'method': "#E67E22",  # Orange for methods_application
            'default': "#FF5733"  # Default highlight color (bright orange)
        }

        # Add nodes with proper formatting
        for node_id, data in self.nodes.items():
            # Skip None node IDs
            if node_id is None:
                print("Warning: Skipping node with None ID")
                continue

            node_type = data.get("type", "default")
            properties = data.get("properties", {})

            # Determine label - use property values if available
            if isinstance(properties, dict):
                if "label" in properties:
                    label = str(properties["label"])
                elif "value" in properties:
                    label = str(properties["value"])
                else:
                    label = str(node_id)
            else:
                # Handle non-dict properties
                label = str(node_id)

            # Check if node should be highlighted
            node_color = color_map.get(node_type, "#7F8C8D")
            should_highlight = False

            # Check different ways a node might be highlighted
            if highlighted_nodes and (node_id in highlighted_nodes or label in highlighted_nodes):
                should_highlight = True

            # Get highlight type if specified
            highlight_type = data.get('highlight_type', 'default')

            # Apply appropriate highlighting color
            if should_highlight:
                node_color = highlight_colors.get(highlight_type, highlight_colors['default'])
                # Increase size for highlighted nodes
                node_size = 30
                # Bold font for highlighted nodes
                font = {'size': 14, 'color': 'black', 'face': 'arial', 'bold': True}
            else:
                node_size = 25
                font = {'size': 12, 'color': 'black', 'face': 'arial'}

            # Format the tooltip to show node attributes
            tooltip = f"ID: {node_id}<br>Type: {node_type}<br>"
            if properties and isinstance(properties, dict):
                for key, value in properties.items():
                    tooltip += f"{key}: {value}<br>"

            # Add highlight type to tooltip if highlighted
            if should_highlight:
                tooltip += f"<br>Highlight: {highlight_type.capitalize()}<br>"

            # Add the node with styling
            net.add_node(
                node_id,
                label=label,
                title=tooltip,
                color=node_color,
                size=node_size,
                font=font,
                group=node_type  # Using group for better visualization
            )

        # Add edges with formatting
        edge_types = set()
        for edge in self.edges:
            if "type" in edge:
                edge_types.add(edge.get("type", "default"))

        edge_colors = {}
        for i, edge_type in enumerate(edge_types):
            edge_colors[edge_type] = colors[i % len(colors)]

        # Track which nodes actually exist in the network
        existing_nodes = set(net.get_nodes())

        for edge in self.edges:
            source = edge.get("source")
            target = edge.get("target")
            edge_type = edge.get("type", "default")

            # Skip edges with None source or target, or nodes that don't exist
            if source is None or target is None:
                print(f"Warning: Skipping edge with None source or target: {edge}")
                continue

            # Skip edges where source or target nodes don't exist
            if str(source) not in existing_nodes or str(target) not in existing_nodes:
                print(f"Warning: Skipping edge {source} -> {target} because nodes don't exist")
                continue

            # Format the tooltip
            tooltip = f"Type: {edge_type}<br>Source: {source}<br>Target: {target}"

            # Add the edge with styling
            try:
                net.add_edge(
                    source,
                    target,
                    title=tooltip,
                    label=edge_type,
                    color=edge_colors.get(edge_type, "#7F8C8D"),
                    font={'size': 10, 'align': 'middle'},
                    arrows='to',
                    arrowStrikethrough=False,
                    smooth={'enabled': True, 'type': 'dynamic'}  # Better curve for edges
                )
            except AssertionError as e:
                print(f"Error adding edge {source} -> {target}: {e}")
                continue

        # Add options for a better visualization
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

        # Save the visualization
        net.save_graph(output_file)
        return net

    def render_pyvis(self, output_file, highlighted_nodes=None):
        """
        Render the ontology graph as an interactive PyVis visualization

        Args:
            output_file (str): Path to save the HTML output file
            highlighted_nodes (list): List of node IDs to highlight

        Returns:
            str: Path to the generated HTML file
        """
        return self.visualize(
            output_file=output_file,
            highlighted_nodes=highlighted_nodes,
            height="700px",
            width="100%",
            physics=True,
            spacing_factor=1.2
        )

    @staticmethod
    def _create_node(tx, node_id, node_data):
        """Create a node in Neo4j."""
        query = """
        MERGE (n {id: $node_id})
        SET n.type = $node_type, n += $properties
        RETURN n
        """
        tx.run(query, node_id=node_id, node_type=node_data["type"], properties=node_data["properties"])

    @staticmethod
    def _create_edge(tx, edge):
        """Create an edge in Neo4j."""
        query = """
        MATCH (a {id: $source}), (b {id: $target})
        MERGE (a)-[r:RELATIONSHIP {type: $edge_type}]->(b)
        RETURN r
        """
        tx.run(query, source=edge["source"], target=edge["target"], edge_type=edge["type"])
