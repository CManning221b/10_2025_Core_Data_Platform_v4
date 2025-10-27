# backend/services/inferred_ontology_service.py
import os
import json
import datetime
from backend.core_graph_managers.no2_graphManager.graphManager import GraphManager
from backend.core_graph_managers.no3_OntologyManager.OntologyManager import OntologyManager


class OntologyService:
    def __init__(self):
        self.graphs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data', 'graphs')
        self.ontologies_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'data',
                                           'ontologies')
        # Ensure directories exist
        os.makedirs(self.graphs_dir, exist_ok=True)
        os.makedirs(os.path.join(self.ontologies_dir, 'extracted'), exist_ok=True)
        os.makedirs(os.path.join(self.ontologies_dir, 'visualizations'), exist_ok=True)

    # Add this method to OntologyService
    def get_current_ontology(self, graph_id):
        """
        Get the current ontology object for a graph

        Args:
            graph_id (str): ID of the graph

        Returns:
            OntologyManager: The ontology manager for the graph or None if failed
        """
        # Load the graph using GraphManager
        graph_path = os.path.join(self.graphs_dir, f"{graph_id}.json")
        if not os.path.exists(graph_path):
            return None

        try:
            # Load the graph
            graph_manager = GraphManager()
            graph_manager.load_from_json(graph_path)

            # Initialize ontology manager
            ontology_manager = OntologyManager(graph_manager)
            ontology_manager.extract_ontology()  # Make sure ontology is extracted

            return ontology_manager
        except Exception as e:
            print(f"Error getting ontology for graph {graph_id}: {str(e)}")
            return None

    def extract_ontology_from_graph(self, graph_id):
        """
        Extract ontology from a given graph

        Args:
            graph_id (str): ID of the graph to extract ontology from

        Returns:
            dict: Extracted ontology information or None if failed
        """
        # Load the graph using GraphManager
        graph_path = os.path.join(self.graphs_dir, f"{graph_id}.json")
        if not os.path.exists(graph_path):
            return None

        try:
            # Load the graph
            graph_manager = GraphManager()
            graph_manager.load_from_json(graph_path)

            # Extract ontology
            ontology_manager = OntologyManager(graph_manager)
            ontology_summary = ontology_manager.extract_ontology()

            # Save the ontology
            ontology_path = os.path.join(self.ontologies_dir, 'extracted', f"{graph_id}_ontology.json")
            ontology_manager.export_to_json_schema(ontology_path)

            # Generate ontology visualizations
            vis_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                   'app', 'static', 'visualizations')
            ont_vis_path = os.path.join(vis_dir, f"{graph_id}_ontology.html")

            # Get ontology as graph and visualize
            ont_graph = ontology_manager.export_ontology_schema_graph()
            ont_graph.render_pyvis(ont_vis_path)

            # Return ontology summary with additional metadata
            return {
                'graph_id': graph_id,
                'ontology_summary': ontology_summary,
                'node_types': ontology_manager.get_node_types(),
                'edge_types': ontology_manager.get_edge_types(),
                'relationship_patterns': ontology_manager.extract_relationship_patterns(),
                'visualization_path': f"{graph_id}_ontology.html"
            }
        except Exception as e:
            print(f"Error extracting ontology from graph {graph_id}: {str(e)}")
            return None

    def get_ontology_details(self, graph_id):
        """
        Get detailed ontology information for a graph

        Args:
            graph_id (str): ID of the graph

        Returns:
            dict: Detailed ontology information or None if not found
        """
        # Try to load from cache first
        ontology_path = os.path.join(self.ontologies_dir, 'extracted', f"{graph_id}_ontology.json")

        if os.path.exists(ontology_path):
            try:
                with open(ontology_path, 'r') as f:
                    ontology_data = json.load(f)

                # Check if visualization exists
                vis_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                                        'app', 'static', 'visualizations',
                                        f"{graph_id}_ontology.html")

                has_visualization = os.path.exists(vis_path)

                return {
                    'graph_id': graph_id,
                    'ontology_data': ontology_data,
                    'has_visualization': has_visualization,
                    'visualization_path': f"{graph_id}_ontology.html" if has_visualization else None
                }
            except Exception as e:
                print(f"Error loading ontology for graph {graph_id}: {str(e)}")

        # If not cached, extract it
        return self.extract_ontology_from_graph(graph_id)

    def analyze_node_type(self, graph_id, node_type):
        """
        Analyze a specific node type in the graph

        Args:
            graph_id (str): ID of the graph
            node_type (str): Type of node to analyze

        Returns:
            dict: Analysis results or None if failed
        """
        # Load the graph using GraphManager
        graph_path = os.path.join(self.graphs_dir, f"{graph_id}.json")
        if not os.path.exists(graph_path):
            return None

        try:
            # Load the graph
            graph_manager = GraphManager()
            graph_manager.load_from_json(graph_path)

            # Initialize ontology manager
            ontology_manager = OntologyManager(graph_manager)

            # Analyze node type
            node_analysis = ontology_manager.analyze_node_type(node_type)

            return {
                'graph_id': graph_id,
                'node_type': node_type,
                'analysis': node_analysis
            }
        except Exception as e:
            print(f"Error analyzing node type {node_type} for graph {graph_id}: {str(e)}")
            return None

    def analyze_edge_type(self, graph_id, edge_type):
        """
        Analyze a specific edge type in the graph

        Args:
            graph_id (str): ID of the graph
            edge_type (str): Type of edge to analyze

        Returns:
            dict: Analysis results or None if failed
        """
        # Load the graph using GraphManager
        graph_path = os.path.join(self.graphs_dir, f"{graph_id}.json")
        if not os.path.exists(graph_path):
            return None

        try:
            # Load the graph
            graph_manager = GraphManager()
            graph_manager.load_from_json(graph_path)

            # Initialize ontology manager
            ontology_manager = OntologyManager(graph_manager)

            # Analyze edge type
            edge_analysis = ontology_manager.analyze_edge_type(edge_type)

            return {
                'graph_id': graph_id,
                'edge_type': edge_type,
                'analysis': edge_analysis
            }
        except Exception as e:
            print(f"Error analyzing edge type {edge_type} for graph {graph_id}: {str(e)}")
            return None

    def analyze_relationship_cardinality(self, graph_id, source_type, edge_type, target_type):
        """
        Analyze the cardinality of a specific relationship pattern

        Args:
            graph_id (str): ID of the graph
            source_type (str): Source node type
            edge_type (str): Edge type
            target_type (str): Target node type

        Returns:
            dict: Cardinality analysis or None if failed
        """
        # Load the graph using GraphManager
        graph_path = os.path.join(self.graphs_dir, f"{graph_id}.json")
        if not os.path.exists(graph_path):
            return None

        try:
            # Load the graph
            graph_manager = GraphManager()
            graph_manager.load_from_json(graph_path)

            # Initialize ontology manager
            ontology_manager = OntologyManager(graph_manager)

            # Analyze relationship cardinality
            cardinality_info = ontology_manager.analyze_relationship_cardinality(
                source_type, edge_type, target_type)

            return {
                'graph_id': graph_id,
                'relationship': f"{source_type} → {edge_type} → {target_type}",
                'cardinality': cardinality_info
            }
        except Exception as e:
            print(f"Error analyzing relationship cardinality for graph {graph_id}: {str(e)}")
            return None

    def get_validation_rules(self, graph_id):
        """
        Generate validation rules for a graph

        Args:
            graph_id (str): ID of the graph

        Returns:
            dict: Validation rules or None if failed
        """
        # Load the graph using GraphManager
        graph_path = os.path.join(self.graphs_dir, f"{graph_id}.json")
        if not os.path.exists(graph_path):
            return None

        try:
            # Load the graph
            graph_manager = GraphManager()
            graph_manager.load_from_json(graph_path)

            # Initialize ontology manager
            ontology_manager = OntologyManager(graph_manager)

            # Generate validation rules
            validation_rules = ontology_manager.generate_validation_rules()

            return {
                'graph_id': graph_id,
                'validation_rules': validation_rules
            }
        except Exception as e:
            print(f"Error generating validation rules for graph {graph_id}: {str(e)}")
            return None

    def export_ontology_to_owl(self, graph_id, format='ttl'):
        """
        Export ontology to OWL format with a focus on TTL compatibility

        Args:
            graph_id (str): ID of the graph
            format (str): Output format - 'ttl' (Turtle, default), 'xml' (OWL/XML), or 'rdf' (RDF/XML)

        Returns:
            dict: Information about the exported OWL file or None if failed
        """
        # Make sure we have a directory for OWL ontologies
        owl_dir = os.path.join(self.ontologies_dir, 'owl')
        os.makedirs(owl_dir, exist_ok=True)

        # Load the graph and get ontology manager
        graph_path = os.path.join(self.graphs_dir, f"{graph_id}.json")
        if not os.path.exists(graph_path):
            return None

        try:
            # Load the graph
            graph_manager = GraphManager()
            graph_manager.load_from_json(graph_path)

            # Initialize and extract ontology
            ontology_manager = OntologyManager(graph_manager)
            ontology_manager.extract_ontology()

            # Determine file extension based on format (defaulting to ttl)
            extension = {
                'xml': 'owl',
                'rdf': 'rdf',
                'ttl': 'ttl'
            }.get(format.lower(), 'ttl')

            # Generate filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            owl_filename = f"{graph_id}_ontology_{timestamp}.{extension}"
            owl_path = os.path.join(owl_dir, owl_filename)

            # Export to OWL/TTL format
            exported_path = ontology_manager.export_to_owl(
                filepath=owl_path,
                format=format
            )

            # Get some basic ontology stats for the response
            node_types = ontology_manager.get_node_types()
            edge_types = ontology_manager.get_edge_types()

            return {
                'graph_id': graph_id,
                'ontology_file': owl_filename,
                'ontology_path': exported_path,
                'format': format,
                'timestamp': timestamp,
                'stats': {
                    'node_types': len(node_types),
                    'edge_types': len(edge_types),
                    'relationships': len(ontology_manager.relationships)
                }
            }
        except Exception as e:
            print(f"Error exporting ontology to OWL for graph {graph_id}: {str(e)}")
            return None

