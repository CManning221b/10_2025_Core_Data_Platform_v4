from flask import Blueprint, render_template, request, jsonify
from backend.services.inferred_ontology_service import OntologyService
from backend.services.instance_graph_service import InstanceGraphService
from backend.services.loaded_ontology_service import LoadedOntologyService
from backend.services.reasoning_service import ReasoningService
from backend.services.process_graph_service import ProcessGraphService
from backend.services.text_interface_service import TextInterfaceService
import os

text_interface_bp = Blueprint('text_interface', __name__)

text_interface_service = TextInterfaceService()


def make_json_serializable(obj):
    """
    Recursively convert objects to JSON-serializable format
    """
    if hasattr(obj, '__dict__'):
        # Skip GraphManager and other complex objects
        return str(type(obj).__name__)
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    else:
        return obj

def get_loaded_ontology_info():
    """Get loaded ontology information by actually reading the ontology files"""
    try:
        # Get the base project directory
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        # Construct the path to ontologies directory
        ontology_dir = os.path.join(base_dir, 'data', 'ontologies', 'predefined')

        # Check if directory exists and has files
        if not os.path.exists(ontology_dir):
            return None, "None"

        # Get .ttl, .owl, and .rdf files in the directory
        ontology_files = [os.path.join(ontology_dir, f) for f in os.listdir(ontology_dir)
                          if f.endswith(('.ttl', '.owl', '.rdf'))]

        if not ontology_files:
            return None, "None"

        # Check if rdflib is available
        try:
            from rdflib import Graph, Namespace, URIRef
            from rdflib.namespace import DCTERMS, RDFS, RDF, OWL
            rdflib_available = True
        except ImportError:
            print("rdflib not available - using basic file analysis")
            rdflib_available = False

        # Extract actual ontology names from the files
        loaded_ontologies = []
        ontology_names = []

        for ontology_file in ontology_files:
            try:
                # Extract name from filename (remove path and extension)
                file_name = os.path.basename(ontology_file)
                name_without_ext = os.path.splitext(file_name)[0]
                ontology_name = None
                class_count = 0

                # Try to get proper ontology name and class count from the file itself
                if rdflib_available:
                    try:
                        g = Graph()
                        g.parse(ontology_file)

                        # Count classes in this specific ontology
                        class_count = len(list(g.triples((None, RDF.type, RDFS.Class)))) + \
                                     len(list(g.triples((None, RDF.type, OWL.Class))))

                        # Try to find the ontology title/name from metadata
                        for title_pred in [DCTERMS.title, RDFS.label, URIRef("http://purl.org/dc/elements/1.1/title")]:
                            for _, _, title in g.triples((None, title_pred, None)):
                                if title:
                                    ontology_name = str(title)
                                    break
                            if ontology_name:
                                break

                        # If no title found, try to extract from ontology URI
                        if not ontology_name:
                            for s, p, o in g.triples((None, RDF.type, OWL.Ontology)):
                                ontology_uri = str(s)
                                # Extract name from URI
                                if '#' in ontology_uri:
                                    ontology_name = ontology_uri.split('#')[-1]
                                elif '/' in ontology_uri:
                                    ontology_name = ontology_uri.split('/')[-1]
                                break

                    except Exception as parse_error:
                        print(f"Could not parse {ontology_file} for metadata: {parse_error}")

                # Fall back to filename-based naming if parsing failed or rdflib unavailable
                if not ontology_name:
                    ontology_name = name_without_ext.replace('_', ' ').replace('-', ' ').title()

                loaded_ontologies.append({
                    'name': ontology_name,
                    'file': file_name,
                    'classes': class_count,
                    'format': file_name.split('.')[-1].upper(),
                    'path': ontology_file
                })
                ontology_names.append(ontology_name)

            except Exception as file_error:
                print(f"Error processing ontology file {ontology_file}: {file_error}")
                continue

        if loaded_ontologies:
            # Return list of ontologies and a summary name
            if len(ontology_names) == 1:
                summary_name = ontology_names[0]
            else:
                summary_name = f"{len(ontology_names)} ontologies loaded"

            return loaded_ontologies, summary_name

        return None, "None"

    except Exception as e:
        print(f"Error getting loaded ontology info: {e}")
        return None, "None"


@text_interface_bp.route('/cli/<graph_id>')
def cli_interface(graph_id):
    """Main CLI interface page - gather all available information"""

    # Initialize all services
    ontology_service = OntologyService()
    instance_service = InstanceGraphService()
    reasoning_service = ReasoningService()

    # Collect all available information
    try:
        # 1. Instance Graph Data
        instance_graph = instance_service.get_graph_data(graph_id)
        print(f"Instance graph loaded: {instance_graph is not None}")

        # 2. Inferred Ontology - try multiple methods to get the data
        inferred_ontology = None
        ontology_node_types = {}
        ontology_edge_types = {}

        # First try to get existing ontology details
        inferred_ontology = ontology_service.get_ontology_details(graph_id)

        # If that doesn't work, try to extract fresh
        if not inferred_ontology:
            print("No existing ontology found, extracting fresh...")
            inferred_ontology = ontology_service.extract_ontology_from_graph(graph_id)

        # Try to get the actual ontology manager object for detailed data
        if inferred_ontology:
            # Try to get the current ontology manager
            ontology_manager = ontology_service.get_current_ontology(graph_id)
            if ontology_manager:
                # Get node and edge types from the manager
                if hasattr(ontology_manager, 'get_node_types'):
                    ontology_node_types = ontology_manager.get_node_types()
                elif hasattr(ontology_manager, 'node_types'):
                    ontology_node_types = ontology_manager.node_types

                if hasattr(ontology_manager, 'get_edge_types'):
                    ontology_edge_types = ontology_manager.get_edge_types()
                elif hasattr(ontology_manager, 'edge_types'):
                    ontology_edge_types = ontology_manager.edge_types

                print(
                    f"Ontology Manager - Node types: {len(ontology_node_types)}, Edge types: {len(ontology_edge_types)}")

            # Also check if data is in the inferred_ontology structure
            if 'node_types' in inferred_ontology:
                if not ontology_node_types:
                    ontology_node_types = inferred_ontology['node_types']
            if 'edge_types' in inferred_ontology:
                if not ontology_edge_types:
                    ontology_edge_types = inferred_ontology['edge_types']

        print(f"Final ontology data - Node types: {len(ontology_node_types)}, Edge types: {len(ontology_edge_types)}")

        # 3. Loaded Ontologies - use the working approach
        loaded_ontologies, loaded_ontology_name = get_loaded_ontology_info()
        print(f"Loaded ontology detected: {loaded_ontology_name}")

        # 4. Reasoning Analysis - this gives us the filtered, relevant information
        reasoning_analysis = None
        reasoning_analysis_serializable = None
        try:
            reasoning_analysis = reasoning_service.analyze_instance_with_reasoning(graph_id)
            print(f"Reasoning analysis completed: {reasoning_analysis is not None}")

            # Create a serializable version for JSON
            if reasoning_analysis:
                reasoning_analysis_serializable = {
                    'node_count': reasoning_analysis.get('node_count', 0),
                    'edge_count': reasoning_analysis.get('edge_count', 0),
                    'ontology_comparison': {
                        'matching_classes': reasoning_analysis.get('ontology_comparison', {}).get('matching_classes',
                                                                                                  []),
                        'fuzzy_matching_classes': reasoning_analysis.get('ontology_comparison', {}).get(
                            'fuzzy_matching_classes', []),
                        'only_in_loaded': reasoning_analysis.get('ontology_comparison', {}).get('only_in_loaded', []),
                        'only_in_inferred': reasoning_analysis.get('ontology_comparison', {}).get('only_in_inferred',
                                                                                                  [])
                    },
                    'applicable_methods': [
                        {
                            'id': method.get('id'),
                            'name': method.get('name'),
                            'description': method.get('description'),
                            'appliesTo': method.get('appliesTo', [])
                        }
                        for method in reasoning_analysis.get('applicable_methods', [])
                    ],
                    'matching_processes': [
                        {
                            'id': process.get('id'),
                            'name': process.get('name'),
                            'description': process.get('description'),
                            'step_count': process.get('step_count', 0)
                        }
                        for process in reasoning_analysis.get('matching_processes', [])
                    ]
                }
        except Exception as e:
            print(f"Reasoning analysis failed: {e}")

        # 5. Method executions found in the graph (if any)
        method_executions = []
        if instance_graph and 'nodes' in instance_graph:
            for node_id, node_data in instance_graph['nodes'].items():
                node_type = node_data.get('type', '')
                if 'Method' in node_type or 'ValidationMethod' in node_type:
                    method_executions.append({
                        'node_id': node_id,
                        'node_data': node_data
                    })

        print(f"Found {len(method_executions)} method executions")

        # Pass all information to template
        return render_template('text_interface/cli_interface.html',
                               graph_id=graph_id,
                               # Instance data
                               instance_graph=instance_graph,
                               # Ontology data
                               inferred_ontology=inferred_ontology,
                               ontology_node_types=ontology_node_types,
                               ontology_edge_types=ontology_edge_types,
                               loaded_ontologies=loaded_ontologies or [],
                               # Reasoning data
                               reasoning_analysis=reasoning_analysis,
                               reasoning_analysis_serializable=reasoning_analysis_serializable,
                               # Method execution data
                               method_executions=method_executions,
                               # Summary info for display
                               loaded_ontology_name=loaded_ontology_name)

    except Exception as e:
        print(f"Error in CLI interface: {e}")
        import traceback
        traceback.print_exc()
        return render_template('text_interface/cli_interface.html',
                               graph_id=graph_id,
                               loaded_ontology_name="Error",
                               error=str(e),
                               # Empty data on error
                               instance_graph=None,
                               inferred_ontology=None,
                               ontology_node_types={},
                               ontology_edge_types={},
                               loaded_ontologies=[],
                               reasoning_analysis=None,
                               reasoning_analysis_serializable=None,
                               method_executions=[])


@text_interface_bp.route('/api/cli/command', methods=['POST'])
def process_command():
    """Process CLI commands"""
    command = request.json.get('command', '')
    graph_id = request.json.get('graph_id', '')

    if command.lower() in ['help']:
        output = """SEMANTIC GRAPH EXPLORER COMMANDS

CORE ANALYSIS:
  methods               - Show all executed methods and their results
  warnings              - Show validation failures and alerts
  insights              - Key findings and data quality summary

METHOD DETAILS:
  method [id]           - Deep dive into specific method execution
  validations           - Show all validation results with status
  recommendations       - Suggest additional methods to run

GRAPH EXPLORATION:
  nodes [type]          - List nodes, optionally filter by type
  connections [node_id] - Show what's connected to a node
  timeline              - Show temporal data and measurements

SEARCH & DEBUG:
  search [query]        - Natural language search
  debug                 - System status and embeddings info

Type any command to explore your graph data!"""

    elif command.lower() == 'methods':

        try:

            method_analysis = text_interface_service.get_method_analysis(graph_id)

            if 'error' in method_analysis:

                output = f"Error: {method_analysis['error']}"

            else:

                summary = method_analysis['method_execution_summary']

                executions = method_analysis['method_executions']

                output = f"""METHOD EXECUTIONS
        
        Total Methods Run: {summary['total_methods_run']}
        
        Total Validations: {summary['total_validations']}
        
        Warnings Found: {summary['warnings_found']}
        
        
        EXECUTED METHODS:"""

                for i, method in enumerate(executions):

                    connections = method.get('connections', {})

                    results = connections.get('produced_results', [])

                    output += f"\n  {i + 1}. {method['method_type']}"

                    output += f"\n     Value: {method['value']}"

                    output += f"\n     Results produced: {len(results)}"

                    # Show execution details

                    method_data = method.get('data', {})

                    if 'execution_time' in method_data:
                        output += f"\n     Executed: {method_data['execution_time']}"

                    # Show first few results with their key info

                    for j, result in enumerate(results[:2]):  # Show first 2 results

                        result_data = result.get('data', {})

                        output += f"\n     → Result {j + 1}: {result['value']} ({result['type']})"

                        # Show key result properties based on type

                        if result['type'] == 'ValidationResult':

                            if 'is_valid' in result_data:
                                status = "PASSED" if result_data['is_valid'] else "FAILED"

                                output += f" - {status}"

                            if 'validation_reason' in result_data:
                                reason = result_data['validation_reason'][:60] + "..." if len(
                                    result_data['validation_reason']) > 60 else result_data['validation_reason']

                                output += f"\n       Reason: {reason}"


                        elif result['type'] == 'histogram_data':

                            if 'total_items' in result_data:
                                output += f" - {result_data['total_items']} items"

                            if 'categories' in result_data:
                                output += f", {result_data['categories']} categories"


                        elif result['type'] == 'Timeline':

                            if 'entry_count' in result_data:
                                output += f" - {result_data['entry_count']} events"


                        elif result['type'] == 'channel':

                            if 'channel_id' in result_data:
                                output += f" - ID: {result_data['channel_id']}"

                            if 'confidence' in result_data:
                                output += f", confidence: {result_data['confidence']}"

                    if len(results) > 2:
                        output += f"\n     → ... and {len(results) - 2} more results"

                    output += f"\n     → Use 'method {method['node_id']}' for full details"

                # Show validation summary

                insights = method_analysis['method_insights']

                if insights['validation_summary']:

                    output += f"\n\nVALIDATION RESULTS:"

                    for status, count in insights['validation_summary'].items():
                        output += f"\n  {status}: {count}"

                output += f"\n\nUse 'validations' to see detailed results"

                output += f"\nUse 'warnings' to see issues that need attention"

        except Exception as e:

            output = f"Error analyzing methods: {str(e)}"

    elif command.startswith('method '):
        try:
            node_id = command.split(' ', 1)[1].strip()
            method_analysis = text_interface_service.get_method_analysis(graph_id)

            # Find the specific method
            method_found = None
            for method in method_analysis['method_executions']:
                if method['node_id'] == node_id:
                    method_found = method
                    break

            if method_found:
                connections = method_found['connections']
                results = connections.get('produced_results', [])

                output = f"""METHOD DETAILS: {method_found['method_type']}
    Node ID: {method_found['node_id']}
    Value: {method_found['value']}
    Hierarchy: {method_found['hierarchy']}

    METHOD EXECUTION DATA:"""

                # Show method execution details
                for key, value in method_found['data'].items():
                    if key not in ['type', 'value', 'hierarchy']:
                        output += f"\n  {key}: {value}"

                # Show all results produced
                if results:
                    output += f"\n\nRESULTS PRODUCED ({len(results)}):"
                    for i, result in enumerate(results):
                        result_data = result.get('data', {})
                        output += f"\n  {i + 1}. {result['value']} ({result['type']}) - ID: {result['node_id']}"

                        # Show key validation status first if it's a ValidationResult
                        if result['type'] == 'ValidationResult' and 'is_valid' in result_data:
                            status = "PASSED" if result_data['is_valid'] else "FAILED"
                            output += f" - STATUS: {status}"

                        # Show detailed result data
                        key_fields_shown = 0
                        for key, value in result_data.items():
                            if key not in ['type', 'value', 'hierarchy'] and value and key_fields_shown < 10:
                                # Truncate long values
                                display_value = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                                output += f"\n     {key}: {display_value}"
                                key_fields_shown += 1

                        if len(result_data) > key_fields_shown + 3:  # +3 for type, value, hierarchy
                            remaining = len(result_data) - key_fields_shown - 3
                            output += f"\n     ... and {remaining} more properties"
                else:
                    output += f"\n\nNo results produced by this method."
            else:
                output = f"Method with ID '{node_id}' not found. Use 'methods' to see available methods."
        except Exception as e:
            output = f"Error getting method details: {str(e)}"

    elif command.lower() == 'validations':
        try:
            method_analysis = text_interface_service.get_method_analysis(graph_id)
            validations = method_analysis['validation_results']

            if not validations:
                output = "No validation results found in the graph."
            else:
                output = f"VALIDATION RESULTS\nTotal: {len(validations)}\n"

                # Group by status
                by_status = {}
                for val in validations:
                    status = val['status']
                    if status not in by_status:
                        by_status[status] = []
                    by_status[status].append(val)

                for status, vals in by_status.items():
                    output += f"\n{status} ({len(vals)}):"

                    for val in vals[:3]:  # Show first 3 of each status
                        output += f"\n  • {val['value']} (Node: {val['node_id']})"

                    if len(vals) > 3:
                        output += f"\n  ... and {len(vals) - 3} more"

                output += f"\n\nUse 'method [node_id]' to see validation details"
        except Exception as e:
            output = f"Error getting validations: {str(e)}"

    elif command.lower() == 'warnings':
        try:
            warnings_analysis = text_interface_service.get_method_warnings(graph_id)

            if warnings_analysis['critical_issues'] == 0:
                output = "No critical issues found! All validations are passing."
            else:
                output = f"""WARNINGS & ISSUES
Critical Issues: {warnings_analysis['critical_issues']}
Total Warnings: {warnings_analysis['total_warnings']}

FAILED VALIDATIONS:"""

                for val in warnings_analysis['failed_validations']:
                    output += f"\n  • {val['value']}"
                    output += f"\n    Node: {val['node_id']}, Type: {val['type']}"

                if warnings_analysis['warning_validations']:
                    output += f"\n\nWARNING VALIDATIONS:"
                    for val in warnings_analysis['warning_validations']:
                        output += f"\n  • {val['value']}"
                        output += f"\n    Node: {val['node_id']}"

                high_warnings = warnings_analysis['warnings_by_severity']['HIGH']
                if high_warnings:
                    output += f"\n\nHIGH SEVERITY ALERTS:"
                    for warning in high_warnings:
                        output += f"\n  • {warning['content']} (Node: {warning['node_id']})"
        except Exception as e:
            output = f"Error getting warnings: {str(e)}"

    elif command.lower() == 'insights':
        try:
            method_analysis = text_interface_service.get_method_analysis(graph_id)
            insights = method_analysis['method_insights']

            output = "KEY INSIGHTS\n\n"

            if insights['key_findings']:
                output += "FINDINGS:\n"
                for finding in insights['key_findings']:
                    # Remove emojis from findings
                    clean_finding = finding.replace('⚠️', '').replace('⚡', '').replace('✅', '').strip()
                    output += f"  • {clean_finding}\n"
            else:
                output += "No specific findings to report.\n"

            # Recommendations
            recommendations = method_analysis['method_recommendations']
            if recommendations:
                output += f"\nRECOMMENDATIONS ({len(recommendations)}):\n"
                for rec in recommendations[:5]:
                    output += f"  • {rec['method_name']} ({rec['priority']} Priority)\n"
                    output += f"    {rec['reason']}\n"
            else:
                output += f"\nNo additional method recommendations.\n"

        except Exception as e:
            output = f"Error getting insights: {str(e)}"

    elif command.lower() == 'recommendations':
        try:
            method_analysis = text_interface_service.get_method_analysis(graph_id)
            recommendations = method_analysis['method_recommendations']
            applicable = method_analysis['applicable_methods']

            output = f"METHOD RECOMMENDATIONS\nApplicable Methods: {len(applicable)}\n\n"

            if recommendations:
                output += f"RECOMMENDED TO RUN ({len(recommendations)}):\n"
                for rec in recommendations:
                    output += f"  • {rec['method_name']} - {rec['priority']} Priority\n"
                    applies_to_text = rec['reason'].split('applies to')[1] if 'applies to' in rec[
                        'reason'] else 'Various data types'
                    output += f"    Applies to: {applies_to_text}\n"
                    if rec['description'] != 'No description available':
                        output += f"    Description: {rec['description']}\n"
                    output += "\n"
            else:
                output += "All applicable methods have been executed!\n"

            output += f"ALL APPLICABLE METHODS:\n"
            for method in applicable[:5]:
                output += f"  • {method['name']}\n"
                output += f"    Applies to: {', '.join(method.get('appliesTo', ['Unknown']))}\n"

        except Exception as e:
            output = f"Error getting recommendations: {str(e)}"

    elif command.startswith('nodes'):
        try:
            parts = command.split(' ')
            filter_type = parts[1] if len(parts) > 1 else None

            instance_service = InstanceGraphService()
            graph_data = instance_service.get_graph_data(graph_id)

            if not graph_data:
                output = "No graph data found."
            else:
                nodes = graph_data.get('nodes', {})

                if filter_type:
                    # Filter by type
                    filtered_nodes = {nid: ndata for nid, ndata in nodes.items()
                                      if filter_type.lower() in ndata.get('type', '').lower()}
                    output = f"NODES OF TYPE '{filter_type.upper()}'\nFound: {len(filtered_nodes)}\n"
                    display_nodes = filtered_nodes
                else:
                    # Show all with type breakdown
                    node_types = {}
                    for ndata in nodes.values():
                        ntype = ndata.get('type', 'unknown')
                        node_types[ntype] = node_types.get(ntype, 0) + 1

                    output = f"ALL NODES\nTotal: {len(nodes)}\n\nBy Type:\n"
                    for ntype, count in sorted(node_types.items()):
                        output += f"  {ntype}: {count}\n"

                    output += f"\nUse 'nodes [type]' to filter by type"
                    display_nodes = dict(list(nodes.items())[:5])  # Show first 5

                # Show actual nodes
                if display_nodes:
                    output += f"\n\nNODES:\n"
                    for nid, ndata in list(display_nodes.items())[:10]:
                        output += f"  {nid}: {ndata.get('value', 'No value')} ({ndata.get('type', 'unknown')})\n"

                    if len(display_nodes) > 10:
                        output += f"  ... and {len(display_nodes) - 10} more"

        except Exception as e:
            output = f"Error getting nodes: {str(e)}"

    elif command.lower() == 'debug':
        try:
            embeddings_info = text_interface_service.get_embeddings_info(graph_id)

            output = f"""SYSTEM DEBUG
Graph ID: {graph_id}

EMBEDDINGS STATUS:
Cached: {embeddings_info.get('cached', False)}
Node Embeddings: {embeddings_info.get('node_count', 0)}
Edge Embeddings: {embeddings_info.get('edge_count', 0)}

SERVICES STATUS:
TextInterfaceService: Loaded
Sentence transformer model: {'Loaded' if text_interface_service.model else 'Not loaded'}

Use 'search [query]' to test semantic search"""
        except Exception as e:
            output = f"Error getting debug info: {str(e)}"

    else:
        output = f"""Unknown command: '{command}'

Try these commands:
  help                  - Show all available commands
  methods               - See executed methods (great starting point!)
  warnings              - Check for any issues
  insights              - Get key findings and recommendations

Type 'help' for the complete command list."""

    return jsonify({
        'output': output,
        'status': 'success'
    })


@text_interface_bp.route('/api/cli/search', methods=['POST'])
def natural_language_search():
    """Process natural language queries with structured response and subgraph"""
    query = request.json.get('query', '')
    graph_id = request.json.get('graph_id', '')

    if not query.strip():
        return jsonify({
            'results': 'Please enter a search query.',
            'status': 'error'
        })

    try:
        # Perform the enhanced search
        search_results = text_interface_service.search_graph(graph_id, query, top_k=10)

        # Return the structured response
        return jsonify({
            'results': search_results['structured_response'],
            'search_data': search_results,
            'visualization_path': search_results.get('visualization_path'),
            'status': 'success'
        })

    except Exception as e:
        return jsonify({
            'results': f'Search error: {str(e)}',
            'status': 'error'
        })


@text_interface_bp.route('/api/cli/graph_data/<graph_id>')
def get_graph_data(graph_id):
    """Get graph data for visualization"""
    instance_service = InstanceGraphService()

    try:
        graph_data = instance_service.get_graph_data(graph_id)
        return jsonify(graph_data if graph_data else {'nodes': {}, 'edges': {}})
    except Exception as e:
        return jsonify({
            'nodes': {},
            'edges': {},
            'error': str(e),
            'status': 'error'
        })


@text_interface_bp.route('/api/embeddings/info/<graph_id>')
def get_embeddings_info(graph_id):
    """Get information about embeddings for a graph"""
    try:
        info = text_interface_service.get_embeddings_info(graph_id)
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)})


@text_interface_bp.route('/api/embeddings/generate/<graph_id>', methods=['POST'])
def generate_embeddings(graph_id):
    """Generate embeddings for a graph"""
    try:
        force_refresh = request.json.get('force_refresh', False)
        embeddings = text_interface_service.generate_embeddings(graph_id, force_refresh)
        return jsonify({
            'success': True,
            'metadata': embeddings['metadata']
        })
    except Exception as e:
        return jsonify({'error': str(e)})


@text_interface_bp.route('/api/embeddings/clear', methods=['POST'])
def clear_embeddings():
    """Clear embeddings cache"""
    try:
        graph_id = request.json.get('graph_id')
        result = text_interface_service.clear_embeddings_cache(graph_id)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)})


@text_interface_bp.route('/api/generate_pyvis/<graph_id>', methods=['POST'])
def generate_pyvis_visualization(graph_id):
    """Generate PyVis visualization for the CLI interface"""
    try:
        instance_service = InstanceGraphService()

        # Use the instance service's generate_pyvis_html method
        visualization_path = instance_service.generate_pyvis_html(graph_id)

        if visualization_path:
            return jsonify({
                'success': True,
                'path': visualization_path
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Could not generate visualization'
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@text_interface_bp.route('/api/method-analysis', methods=['POST'])
def get_method_analysis_api():
    """Get method analysis for modal display"""
    graph_id = request.json.get('graph_id', '')

    try:
        analysis = text_interface_service.get_method_analysis(graph_id)
        return jsonify(analysis)
    except Exception as e:
        return jsonify({'error': str(e)}), 500