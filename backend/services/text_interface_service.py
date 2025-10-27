import os
import json
import re
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from sentence_transformers import SentenceTransformer
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from backend.services.instance_graph_service import InstanceGraphService
from backend.services.reasoning_service import ReasoningService
from backend.services.inferred_ontology_service import OntologyService
from backend.services.loaded_ontology_service import LoadedOntologyService
from backend.services.process_graph_service import ProcessGraphService


class TextInterfaceService:
    def __init__(self):
        self.instance_service = InstanceGraphService()
        self.reasoning_service = ReasoningService()
        self.ontology_service = OntologyService()
        self.loaded_ontology_service = LoadedOntologyService()
        self.process_service = ProcessGraphService()
        self.model = None
        self.embeddings_cache = {}
        self.load_model()

    def load_model(self):
        """Load the sentence transformer model"""
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            print("Sentence transformer model loaded successfully")
        except Exception as e:
            print(f"Error loading sentence transformer model: {e}")
            self.model = None

    def get_cache_path(self, graph_id: str) -> str:
        """Get the path for the embeddings cache file"""
        cache_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'data', 'cache', 'embeddings'
        )
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, f"{graph_id}_comprehensive_embeddings.pkl")

    def create_comprehensive_edge_text(self, edge_id: str, edge_data: Dict, context_data: Dict) -> str:
        """Create comprehensive edge text with full context"""
        parts = []

        # Basic edge info - include EVERYTHING
        for key, value in edge_data.items():
            if isinstance(value, (str, int, float, bool)):
                parts.append(f"{key}: {value}")
            elif isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    if isinstance(nested_value, (str, int, float, bool)):
                        parts.append(f"{key}_{nested_key}: {nested_value}")

        # Add source and target node context
        source_id = edge_data.get('source')
        target_id = edge_data.get('target')

        if source_id and 'graph_data' in context_data:
            source_node = context_data['graph_data']['nodes'].get(source_id, {})
            if 'value' in source_node:
                parts.append(f"source_value: {source_node['value']}")
            if 'type' in source_node:
                parts.append(f"source_type: {source_node['type']}")

        if target_id and 'graph_data' in context_data:
            target_node = context_data['graph_data']['nodes'].get(target_id, {})
            if 'value' in target_node:
                parts.append(f"target_value: {target_node['value']}")
            if 'type' in target_node:
                parts.append(f"target_type: {target_node['type']}")

        edge_text = " | ".join(parts)
        return edge_text

    def create_comprehensive_node_text(self, node_id: str, node_data: Dict, context_data: Dict) -> str:
        """Create a comprehensive text representation including ALL available data"""
        parts = []

        # Basic node info - include EVERYTHING
        for key, value in node_data.items():
            if isinstance(value, (str, int, float, bool)):
                parts.append(f"{key}: {value}")
            elif isinstance(value, dict):
                # Flatten nested dictionaries
                for nested_key, nested_value in value.items():
                    if isinstance(nested_value, (str, int, float, bool)):
                        parts.append(f"{key}_{nested_key}: {nested_value}")

        # Add ontology context if available
        if 'ontology_types' in context_data:
            node_type = node_data.get('type', 'unknown')
            if node_type in context_data['ontology_types']:
                type_info = context_data['ontology_types'][node_type]
                parts.append(f"ontology_type_info: {type_info}")

        # Add reasoning context
        if 'reasoning_data' in context_data:
            reasoning = context_data['reasoning_data']
            node_type = node_data.get('type', 'unknown')

            # Check if this node is part of any applicable methods
            if 'applicable_methods' in reasoning:
                for method in reasoning['applicable_methods']:
                    if node_type in method.get('appliesTo', []):
                        parts.append(f"applicable_method: {method['name']}")
                        parts.append(f"method_description: {method.get('description', '')}")

            # Check if this node matches any ontology classes
            if 'ontology_comparison' in reasoning:
                comparison = reasoning['ontology_comparison']
                if node_type in comparison.get('matching_classes', []):
                    parts.append(f"ontology_match: exact_match")

                # Check fuzzy matches
                for loaded, inferred in comparison.get('fuzzy_matching_classes', []):
                    if node_type == inferred:
                        parts.append(f"ontology_match: fuzzy_match_with_{loaded}")

        # Add process context
        if 'process_data' in context_data:
            for process in context_data['process_data']:
                if 'uses_methods' in process:
                    # If this node is a method execution, link to processes
                    if node_data.get('type') in ['TemporalValidationMethod',
                                                 'ValidationMethod'] or 'Method' in node_data.get('type', ''):
                        parts.append(f"related_process: {process['name']}")

        # Add connected nodes context
        if 'connections' in context_data and node_id in context_data['connections']:
            connections = context_data['connections'][node_id]
            for conn_type, connected_nodes in connections.items():
                parts.append(f"{conn_type}_connected_to: {', '.join(connected_nodes)}")

        return " | ".join(parts)

    def get_comprehensive_context(self, graph_id: str) -> Dict:
        """Gather ALL available context data for the graph"""
        context = {}

        try:
            # Get graph data
            graph_data = self.instance_service.get_graph_data(graph_id)
            context['graph_data'] = graph_data

            # Get reasoning analysis
            reasoning_analysis = self.reasoning_service.analyze_instance_with_reasoning(graph_id)
            context['reasoning_data'] = reasoning_analysis

            # Get ontology types
            inferred_ontology = self.ontology_service.get_current_ontology(graph_id)
            if inferred_ontology:
                if hasattr(inferred_ontology, 'get_node_types'):
                    context['ontology_types'] = inferred_ontology.get_node_types()
                elif hasattr(inferred_ontology, 'node_types'):
                    context['ontology_types'] = inferred_ontology.node_types

            # Get process data
            if reasoning_analysis and 'matching_processes' in reasoning_analysis:
                context['process_data'] = reasoning_analysis['matching_processes']

            # Build connection maps
            connections = {}
            if graph_data and 'edges' in graph_data:
                for edge_id, edge_data in graph_data['edges'].items():
                    source = edge_data.get('source')
                    target = edge_data.get('target')
                    edge_type = edge_data.get('type', 'connected')

                    if source:
                        if source not in connections:
                            connections[source] = {}
                        if f"outgoing_{edge_type}" not in connections[source]:
                            connections[source][f"outgoing_{edge_type}"] = []
                        connections[source][f"outgoing_{edge_type}"].append(target)

                    if target:
                        if target not in connections:
                            connections[target] = {}
                        if f"incoming_{edge_type}" not in connections[target]:
                            connections[target][f"incoming_{edge_type}"] = []
                        connections[target][f"incoming_{edge_type}"].append(source)

            context['connections'] = connections

            # Get loaded ontology data
            try:
                self.reasoning_service.ensure_ontology_loaded()
                if hasattr(self.loaded_ontology_service,
                           'ontology_bridge') and self.loaded_ontology_service.ontology_bridge:
                    ontology_data = self.loaded_ontology_service.get_ontology_data()
                    context['loaded_ontology'] = ontology_data
            except Exception as e:
                print(f"Could not get loaded ontology context: {e}")

        except Exception as e:
            print(f"Error gathering comprehensive context: {e}")

        return context

    def generate_embeddings(self, graph_id: str, force_refresh: bool = False) -> Dict:
        """Generate comprehensive embeddings using ALL available data"""
        if not self.model:
            raise Exception("Sentence transformer model not loaded")

        cache_path = self.get_cache_path(graph_id)

        # Check cache first
        if not force_refresh and os.path.exists(cache_path):
            try:
                with open(cache_path, 'rb') as f:
                    cached_embeddings = pickle.load(f)
                print(f"Loaded cached comprehensive embeddings for graph {graph_id}")
                return cached_embeddings
            except Exception as e:
                print(f"Error loading cached embeddings: {e}")

        # Get ALL context data
        print(f"Gathering comprehensive context for graph {graph_id}...")
        context_data = self.get_comprehensive_context(graph_id)

        graph_data = context_data.get('graph_data')
        if not graph_data:
            raise Exception(f"Graph {graph_id} not found")

        embeddings = {
            'nodes': {},
            'edges': {},
            'node_texts': {},
            'edge_texts': {},
            'context_data': context_data,  # Store the context for later use
            'metadata': {
                'graph_id': graph_id,
                'model_name': 'all-MiniLM-L6-v2',
                'total_nodes': len(graph_data.get('nodes', {})),
                'total_edges': len(graph_data.get('edges', {})),
                'has_reasoning': 'reasoning_data' in context_data,
                'has_ontology': 'ontology_types' in context_data,
                'has_processes': 'process_data' in context_data
            }
        }

        # Process nodes with full context
        node_texts = []
        node_ids = []

        print("Creating comprehensive node representations...")
        for node_id, node_data in graph_data.get('nodes', {}).items():
            text_repr = self.create_comprehensive_node_text(node_id, node_data, context_data)
            node_texts.append(text_repr)
            node_ids.append(node_id)
            embeddings['node_texts'][node_id] = text_repr

        if node_texts:
            print(f"Encoding {len(node_texts)} node texts...")
            node_embeddings = self.model.encode(node_texts)
            for i, node_id in enumerate(node_ids):
                embeddings['nodes'][node_id] = node_embeddings[i].tolist()

        # Process edges with full context
        edge_texts = []
        edge_ids = []

        print("Creating comprehensive edge representations...")
        for edge_id, edge_data in graph_data.get('edges', {}).items():
            text_repr = self.create_comprehensive_edge_text(edge_id, edge_data, context_data)
            edge_texts.append(text_repr)
            edge_ids.append(edge_id)
            embeddings['edge_texts'][edge_id] = text_repr

        if edge_texts:
            print(f"Encoding {len(edge_texts)} edge texts...")
            edge_embeddings = self.model.encode(edge_texts)
            for i, edge_id in enumerate(edge_ids):
                embeddings['edges'][edge_id] = edge_embeddings[i].tolist()

        # Cache the embeddings
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(embeddings, f)
            print(f"Cached comprehensive embeddings for graph {graph_id}")
        except Exception as e:
            print(f"Error caching embeddings: {e}")

        return embeddings

    def extract_subgraph(self, graph_id: str, matching_nodes: List[Dict], matching_edges: List[Dict]) -> Dict:
        """Extract a focused subgraph prioritizing exact matches"""
        graph_data = self.instance_service.get_graph_data(graph_id)
        if not graph_data:
            return {'nodes': {}, 'edges': {}}

        # Start with matching node IDs
        subgraph_node_ids = set([node['id'] for node in matching_nodes])
        subgraph_edge_ids = set([edge['id'] for edge in matching_edges])

        print(f"DEBUG: Starting with {len(subgraph_node_ids)} matching nodes")

        # Helper function to parse edge source and target from edge ID
        def parse_edge_id(edge_id):
            if '_' in edge_id:
                parts = edge_id.split('_')
                if len(parts) == 2:
                    return parts[0], parts[1]
            return None, None

        # For exact matches (high similarity), add more neighbors
        # For semantic matches, add fewer neighbors
        neighbors_added = 0

        for node in matching_nodes:
            node_id = node['id']
            is_exact_match = node['similarity'] >= 0.9
            max_neighbors = 5 if is_exact_match else 2  # More neighbors for exact matches

            node_neighbors = 0
            for edge_id, edge_data in graph_data.get('edges', {}).items():
                if node_neighbors >= max_neighbors:
                    break

                source = edge_data.get('source')
                target = edge_data.get('target')

                # Parse from edge ID if needed
                if source is None or target is None:
                    parsed_source, parsed_target = parse_edge_id(edge_id)
                    if parsed_source and parsed_target:
                        source = parsed_source
                        target = parsed_target

                # Add neighbors
                if source == node_id and target and target not in subgraph_node_ids:
                    subgraph_node_ids.add(target)
                    subgraph_edge_ids.add(edge_id)
                    neighbors_added += 1
                    node_neighbors += 1
                elif target == node_id and source and source not in subgraph_node_ids:
                    subgraph_node_ids.add(source)
                    subgraph_edge_ids.add(edge_id)
                    neighbors_added += 1
                    node_neighbors += 1

        print(f"DEBUG: Added {neighbors_added} neighbors (prioritizing exact matches)")

        # Add connecting edges
        connecting_edges_added = 0
        for edge_id, edge_data in graph_data.get('edges', {}).items():
            source = edge_data.get('source')
            target = edge_data.get('target')

            if source is None or target is None:
                parsed_source, parsed_target = parse_edge_id(edge_id)
                if parsed_source and parsed_target:
                    source = parsed_source
                    target = parsed_target

            if source in subgraph_node_ids and target in subgraph_node_ids:
                if edge_id not in subgraph_edge_ids:
                    subgraph_edge_ids.add(edge_id)
                    connecting_edges_added += 1

        # Build subgraph
        subgraph = {
            'nodes': {node_id: graph_data['nodes'][node_id]
                      for node_id in subgraph_node_ids
                      if node_id in graph_data['nodes']},
            'edges': {edge_id: graph_data['edges'][edge_id]
                      for edge_id in subgraph_edge_ids
                      if edge_id in graph_data['edges']}
        }

        print(f"DEBUG: Final focused subgraph: {len(subgraph['nodes'])} nodes, {len(subgraph['edges'])} edges")
        return subgraph

    def get_embeddings_info(self, graph_id: str) -> Dict:
        """Get information about cached embeddings"""
        cache_path = self.get_cache_path(graph_id)

        if not os.path.exists(cache_path):
            return {'cached': False, 'graph_id': graph_id}

        try:
            with open(cache_path, 'rb') as f:
                embeddings = pickle.load(f)

            return {
                'cached': True,
                'graph_id': graph_id,
                'cache_file': cache_path,
                'metadata': embeddings.get('metadata', {}),
                'node_count': len(embeddings.get('nodes', {})),
                'edge_count': len(embeddings.get('edges', {}))
            }
        except Exception as e:
            return {
                'cached': False,
                'error': str(e),
                'graph_id': graph_id
            }

    def clear_embeddings_cache(self, graph_id: str = None) -> Dict:
        """Clear embeddings cache for a specific graph or all graphs"""
        cache_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            'data', 'cache', 'embeddings'
        )

        if graph_id:
            # Clear specific graph
            cache_path = self.get_cache_path(graph_id)
            if os.path.exists(cache_path):
                os.remove(cache_path)
                return {'cleared': True, 'graph_id': graph_id}
            else:
                return {'cleared': False, 'graph_id': graph_id, 'reason': 'Cache file not found'}
        else:
            # Clear all caches
            cleared_files = []
            if os.path.exists(cache_dir):
                for file in os.listdir(cache_dir):
                    if file.endswith('_embeddings.pkl'):
                        os.remove(os.path.join(cache_dir, file))
                        cleared_files.append(file)

            return {'cleared': True, 'files_cleared': cleared_files}

    def structure_response_with_ai(self, query: str, search_results: Dict) -> str:
        """Create a focused response prioritizing exact matches"""
        if search_results['summary']['total_matches'] == 0:
            return f"No results found for '{query}' in the graph."

        summary = search_results['summary']
        exact_count = summary.get('exact_matches', 0)
        shown_results = len(search_results.get('nodes', [])) + len(search_results.get('edges', []))

        # More accurate summary
        if exact_count > 0:
            output = f"**Found exact matches for '{query}'** ({exact_count} exact, showing {shown_results - exact_count} most relevant related)\n\n"
        else:
            output = f"**Found {shown_results} most relevant matches** (no exact matches)\n\n"

        nodes = search_results.get('nodes', [])[:5]
        edges = search_results.get('edges', [])[:3]

        if nodes:
            output += "**Entities:**\n"

            # Separate exact from semantic matches
            exact_nodes = [n for n in nodes if n['similarity'] >= 0.9]
            semantic_nodes = [n for n in nodes if n['similarity'] < 0.9]

            # Show exact matches first
            if exact_nodes:
                for node in exact_nodes:
                    details = self.extract_key_details_from_text(node['text'])
                    node_type = self.extract_node_type_from_text(node['text'])
                    output += f"• **{details['name']}** ({node_type}) - EXACT MATCH\n"
                    if details['description']:
                        clean_desc = details['description'].replace('hierarchy:', '').strip()
                        output += f"  {clean_desc}\n"

            # Then show semantic matches
            if semantic_nodes:
                output += "\n**Related entities:**\n"
                for node in semantic_nodes[:3]:
                    details = self.extract_key_details_from_text(node['text'])
                    node_type = self.extract_node_type_from_text(node['text'])
                    output += f"• {details['name']} ({node_type}) - similarity: {node['similarity']:.2f}\n"

        # Show meaningful relationships
        if edges:
            output += f"\n**Connections:**\n"
            meaningful_edges = []
            for edge in edges[:3]:
                relationship = self.extract_meaningful_relationship(edge['text'])
                if relationship != "Connection":  # Only show meaningful relationships
                    if edge['similarity'] >= 0.9:
                        meaningful_edges.append(f"• {relationship} - EXACT MATCH")
                    else:
                        meaningful_edges.append(f"• {relationship}")

            if meaningful_edges:
                output += "\n".join(meaningful_edges) + "\n"

        # Subgraph info
        if search_results.get('subgraph'):
            subgraph = search_results['subgraph']
            output += f"\n*Visualization shows {len(subgraph['nodes'])} entities with {len(subgraph['edges'])} connections.*"

        return output

    def extract_meaningful_relationship(self, text: str) -> str:
        """Extract a meaningful relationship description"""
        parts = text.split('|')

        # Look for source and target values
        source_value = None
        target_value = None
        edge_type = None

        for part in parts:
            part = part.strip()
            if part.startswith('source_value:'):
                source_value = part.split(':', 1)[1].strip()
            elif part.startswith('target_value:'):
                target_value = part.split(':', 1)[1].strip()
            elif part.startswith('type:'):
                edge_type = part.split(':', 1)[1].strip()
            elif part.startswith('source_type:'):
                if not source_value:  # fallback to type if no value
                    source_type = part.split(':', 1)[1].strip()
                    source_value = f"({source_type})"
            elif part.startswith('target_type:'):
                if not target_value:  # fallback to type if no value
                    target_type = part.split(':', 1)[1].strip()
                    target_value = f"({target_type})"

        # Build meaningful description
        if source_value and target_value:
            if edge_type and edge_type not in ['connected', 'Connection']:
                return f"{source_value} --[{edge_type}]--> {target_value}"
            else:
                return f"{source_value} --> {target_value}"
        elif edge_type and edge_type not in ['connected', 'Connection']:
            return f"Relationship: {edge_type}"
        else:
            return "Connection"

    def extract_node_type_from_text(self, text: str) -> str:
        """Extract node type from text representation"""
        parts = text.split('|')
        for part in parts:
            if 'type:' in part.lower():
                return part.split(':')[1].strip()
        return 'Unknown'

    def extract_key_details_from_text(self, text: str) -> Dict[str, str]:
        """Extract key details like name and description from text"""
        details = {'name': 'Unknown', 'description': ''}

        parts = text.split('|')
        for part in parts:
            part = part.strip()
            if 'value:' in part.lower():
                details['name'] = part.split(':', 1)[1].strip()
            elif 'id:' in part.lower() and details['name'] == 'Unknown':
                details['name'] = f"ID {part.split(':', 1)[1].strip()}"

        # Create description from other parts
        desc_parts = []
        for part in parts[:3]:  # Take first 3 parts for description
            if not any(key in part.lower() for key in ['value:', 'id:', 'type:']):
                desc_parts.append(part.strip())

        details['description'] = ' | '.join(desc_parts) if desc_parts else ''
        return details

    def extract_temporal_info_from_text(self, text: str) -> Dict:
        """Extract temporal information from text"""
        temporal = {}
        parts = text.split('|')

        for part in parts:
            part = part.strip().lower()
            if 'created:' in part:
                try:
                    temporal['created'] = part.split(':')[1].strip()
                except:
                    pass
            elif 'destroyed:' in part:
                try:
                    temporal['destroyed'] = part.split(':')[1].strip()
                except:
                    pass
            elif 'timestamp:' in part:
                try:
                    temporal['timestamp'] = float(part.split(':')[1].strip())
                except:
                    pass

        return temporal

    def extract_location_from_text(self, text: str) -> str:
        """Extract location from text"""
        parts = text.split('|')
        for part in parts:
            if 'location:' in part.lower():
                return part.split(':', 1)[1].strip()
        return None

    def extract_relationship_from_text(self, text: str) -> str:
        """Extract relationship description from edge text"""
        parts = text.split('|')
        relationship_parts = []

        for part in parts:
            part = part.strip()
            if any(key in part.lower() for key in ['from:', 'to:', 'relationship:']):
                relationship_parts.append(part)

        return ' → '.join(relationship_parts) if relationship_parts else 'Unknown relationship'

    def generate_subgraph_visualization(self, graph_id: str, search_results: Dict) -> str:
        """Generate a PyVis visualization of the search results subgraph"""
        if not search_results.get('subgraph'):
            return None

        try:
            # Create a modified version of generate_pyvis_html for subgraph
            subgraph_path = self.instance_service.generate_search_subgraph_html(
                graph_id,
                search_results['subgraph'],
                search_results.get('nodes', []),
                search_results.get('edges', [])
            )
            return subgraph_path
        except Exception as e:
            print(f"Error generating subgraph visualization: {e}")
            return None

    def search_graph(self, graph_id: str, query: str, top_k: int = 5, threshold: float = 0.15) -> Dict:
        """Enhanced search with intelligent context expansion for exact matches"""
        if not self.model:
            raise Exception("Sentence transformer model not loaded")

        embeddings = self.generate_embeddings(graph_id)

        results = {
            'query': query,
            'nodes': [],
            'edges': [],
            'subgraph': None,
            'summary': {},
            'context_used': embeddings.get('metadata', {}),
            'structured_response': '',
            'visualization_path': None
        }

        try:
            # STEP 1: Find exact matches first
            exact_matches = self.find_exact_matches(query, embeddings)
            print(f"DEBUG: Found {len(exact_matches['nodes'])} exact node matches")

            # STEP 2: Get semantic matches regardless (we'll combine them)
            query_embedding = self.model.encode([query])[0]
            semantic_matches = self.find_semantic_matches(query_embedding, embeddings, threshold,
                                                          exclude_exact=exact_matches)
            print(
                f"DEBUG: Found {len(semantic_matches['nodes'])} semantic node matches, {len(semantic_matches['edges'])} semantic edge matches")

            # STEP 3: If we have exact matches, get their meaningful context
            if exact_matches['nodes']:
                context_results = self.get_meaningful_context_for_exact_matches(
                    exact_matches['nodes'], embeddings, graph_id
                )
                print(
                    f"DEBUG: Found {len(context_results['nodes'])} contextual nodes, {len(context_results['edges'])} contextual edges")

                # Combine exact matches + context + semantic matches
                all_node_results = exact_matches['nodes'] + context_results['nodes'] + semantic_matches['nodes']
                all_edge_results = exact_matches['edges'] + context_results['edges'] + semantic_matches['edges']
            else:
                # No exact matches, use semantic search results
                print("DEBUG: No exact matches found, using semantic search only")
                all_node_results = semantic_matches['nodes']
                all_edge_results = semantic_matches['edges']

            # Take top results
            results['nodes'] = all_node_results[:top_k]
            results['edges'] = all_edge_results[:top_k]

            print(f"DEBUG: Final results - {len(results['nodes'])} nodes, {len(results['edges'])} edges")

            # Generate subgraph
            if results['nodes'] or results['edges']:
                results['subgraph'] = self.extract_subgraph(graph_id, results['nodes'], results['edges'])
                results['visualization_path'] = self.generate_subgraph_visualization(graph_id, results)

            # Create summary
            results['summary'] = {
                'total_matches': len(all_node_results) + len(all_edge_results),
                'node_matches': len(results['nodes']),
                'edge_matches': len(results['edges']),
                'exact_matches': len(exact_matches['nodes']) + len(exact_matches['edges']),
                'avg_similarity': np.mean([r['similarity'] for r in results['nodes'] + results['edges']]) if (
                            results['nodes'] or results['edges']) else 0.0,
                'max_similarity': max([r['similarity'] for r in results['nodes'] + results['edges']]) if (
                            results['nodes'] or results['edges']) else 0.0,
                'threshold_used': threshold
            }

            # Choose appropriate response based on what we found
            if exact_matches['nodes']:
                results['structured_response'] = self.create_context_aware_response(query, results)
            else:
                results['structured_response'] = self.create_semantic_response(query, results)

            return results

        except Exception as e:
            print(f"ERROR in search_graph: {e}")
            import traceback
            traceback.print_exc()
            # Return empty results on error
            results['structured_response'] = f"Search error: {str(e)}"
            return results

    def create_semantic_response(self, query: str, search_results: Dict) -> str:
        """Create a response for semantic-only matches"""
        if search_results['summary']['total_matches'] == 0:
            return f"No results found for '{query}' in the graph."

        summary = search_results['summary']
        output = f"**Found {summary['total_matches']} related matches for '{query}'**\n\n"

        nodes = search_results.get('nodes', [])
        edges = search_results.get('edges', [])

        if nodes:
            output += "**Related entities:**\n"

            # Group nodes by type for better organization
            node_groups = {}
            for node in nodes:
                node_type = self.extract_node_type_from_text(node['text'])
                if node_type not in node_groups:
                    node_groups[node_type] = []
                node_groups[node_type].append(node)

            for node_type, type_nodes in node_groups.items():
                if len(type_nodes) == 1:
                    node = type_nodes[0]
                    details = self.extract_key_details_from_text(node['text'])
                    output += f"• **{details['name']}** ({node_type}) - similarity: {node['similarity']:.2f}\n"
                else:
                    output += f"• **{node_type.title()}s:** "
                    names = []
                    for node in type_nodes[:3]:  # Show top 3
                        details = self.extract_key_details_from_text(node['text'])
                        names.append(details['name'])
                    if len(type_nodes) > 3:
                        names.append(f"and {len(type_nodes) - 3} more")
                    output += ", ".join(names) + "\n"

        # Show meaningful relationships
        if edges:
            output += f"\n**Relationships:**\n"
            for edge in edges[:3]:  # Show top 3 relationships
                relationship = self.extract_meaningful_relationship(edge['text'])
                if relationship != "Connection":
                    output += f"• {relationship} - similarity: {edge['similarity']:.2f}\n"

        # Subgraph info
        if search_results.get('subgraph'):
            subgraph = search_results['subgraph']
            output += f"\n*Showing {len(subgraph['nodes'])} related entities in visualization.*"

        return output

    def get_meaningful_context_for_exact_matches(self, exact_nodes: List[Dict], embeddings: Dict,
                                                 graph_id: str) -> Dict:
        """Get meaningful context (files, folders, etc.) connected to exact matches"""
        graph_data = self.instance_service.get_graph_data(graph_id)
        if not graph_data:
            return {'nodes': [], 'edges': []}

        # Helper function to parse edge connections
        def parse_edge_id(edge_id):
            if '_' in edge_id:
                parts = edge_id.split('_')
                if len(parts) == 2:
                    return parts[0], parts[1]
            return None, None

        context_nodes = []
        context_edges = []
        exact_node_ids = [node['id'] for node in exact_nodes]

        # Find all nodes connected to our exact matches
        connected_node_ids = set()
        relevant_edge_ids = set()

        for edge_id, edge_data in graph_data.get('edges', {}).items():
            source = edge_data.get('source')
            target = edge_data.get('target')

            # Parse from edge ID if needed
            if source is None or target is None:
                parsed_source, parsed_target = parse_edge_id(edge_id)
                if parsed_source and parsed_target:
                    source = parsed_source
                    target = parsed_target

            # If this edge connects to our exact match, it's relevant
            if source in exact_node_ids and target:
                connected_node_ids.add(target)
                relevant_edge_ids.add(edge_id)
            elif target in exact_node_ids and source:
                connected_node_ids.add(source)
                relevant_edge_ids.add(edge_id)

        print(f"DEBUG: Found {len(connected_node_ids)} nodes connected to exact matches")

        # Prioritize certain types of connected nodes (files, folders, locations, etc.)
        priority_types = ['file', 'folder', 'filename', 'location', 'measurement', 'validation']

        # Add connected nodes, prioritizing meaningful types
        for node_id in connected_node_ids:
            if node_id in graph_data['nodes']:
                node_data = graph_data['nodes'][node_id]
                node_type = node_data.get('type', '').lower()
                node_text = embeddings['node_texts'].get(node_id, '')

                # Give higher similarity to meaningful context
                if any(ptype in node_type for ptype in priority_types):
                    similarity = 0.85  # High context relevance
                else:
                    similarity = 0.70  # Standard context relevance

                context_nodes.append({
                    'id': node_id,
                    'similarity': similarity,
                    'text': node_text,
                    'type': 'node'
                })

        # Add relevant edges
        for edge_id in relevant_edge_ids:
            if edge_id in embeddings['edge_texts']:
                context_edges.append({
                    'id': edge_id,
                    'similarity': 0.80,  # Context edge relevance
                    'text': embeddings['edge_texts'][edge_id],
                    'type': 'edge'
                })

        # Sort by relevance
        context_nodes.sort(key=lambda x: x['similarity'], reverse=True)
        context_edges.sort(key=lambda x: x['similarity'], reverse=True)

        return {
            'nodes': context_nodes[:8],  # Top 8 contextual nodes
            'edges': context_edges[:5]  # Top 5 contextual edges
        }

    def create_context_aware_response(self, query: str, search_results: Dict) -> str:
        """Create a response that highlights the context around exact matches"""
        if search_results['summary']['total_matches'] == 0:
            return f"No results found for '{query}' in the graph."

        summary = search_results['summary']
        exact_count = summary.get('exact_matches', 0)

        if exact_count > 0:
            output = f"**Found exact match for '{query}'** with related context\n\n"
        else:
            output = f"**Found {summary['total_matches']} related matches** (no exact matches)\n\n"

        nodes = search_results.get('nodes', [])
        edges = search_results.get('edges', [])

        # Separate exact matches from context
        exact_nodes = [n for n in nodes if n['similarity'] >= 0.95]
        context_nodes = [n for n in nodes if n['similarity'] < 0.95]

        # Show the exact match first
        if exact_nodes:
            for node in exact_nodes:
                details = self.extract_key_details_from_text(node['text'])
                node_type = self.extract_node_type_from_text(node['text'])
                output += f"**{details['name']}** ({node_type})\n"
                if details['description']:
                    clean_desc = details['description'].replace('hierarchy:', '').strip()
                    output += f"  {clean_desc}\n"

        # Show meaningful context
        if context_nodes:
            output += f"\n**Related context:**\n"

            # Group context by type for better organization
            context_by_type = {}
            for node in context_nodes:
                node_type = self.extract_node_type_from_text(node['text'])
                if node_type not in context_by_type:
                    context_by_type[node_type] = []
                context_by_type[node_type].append(node)

            # Show different types of context
            for context_type, type_nodes in context_by_type.items():
                if context_type.lower() in ['file', 'filename', 'folder']:
                    output += f"**Files & Folders:**\n"
                    for node in type_nodes[:3]:  # Top 3
                        details = self.extract_key_details_from_text(node['text'])
                        output += f"  • {details['name']}\n"
                elif context_type.lower() in ['location']:
                    output += f"**Locations:**\n"
                    for node in type_nodes[:2]:  # Top 2
                        details = self.extract_key_details_from_text(node['text'])
                        output += f"  • {details['name']}\n"
                elif context_type.lower() in ['measurement']:
                    output += f"**Measurements:**\n"
                    for node in type_nodes[:3]:  # Top 3
                        details = self.extract_key_details_from_text(node['text'])
                        output += f"  • {details['name']}\n"

        # Show key connections
        if edges:
            meaningful_connections = []
            for edge in edges[:3]:
                relationship = self.extract_meaningful_relationship(edge['text'])
                if relationship not in ["Connection", "(ChannelDetectionMethod) --> (channel)"]:
                    meaningful_connections.append(relationship)

            if meaningful_connections:
                output += f"\n**Key connections:**\n"
                for connection in meaningful_connections:
                    output += f"  • {connection}\n"

        # Subgraph info
        if search_results.get('subgraph'):
            subgraph = search_results['subgraph']
            output += f"\n*Showing {len(subgraph['nodes'])} related entities in visualization.*"

        return output

    def find_semantic_matches(self, query_embedding, embeddings: Dict, threshold: float, exclude_exact: Dict) -> Dict:
        """Find semantic matches excluding exact matches"""
        semantic_nodes = []
        semantic_edges = []

        # Use a lower threshold for semantic matches when no exact matches
        if not exclude_exact['nodes']:
            semantic_threshold = max(threshold, 0.10)  # Lower threshold when no exact matches
            print(f"DEBUG: No exact matches, using lower semantic threshold: {semantic_threshold}")
        else:
            semantic_threshold = max(threshold, 0.25)  # Higher threshold when we have exact matches
            print(f"DEBUG: Have exact matches, using higher semantic threshold: {semantic_threshold}")

        # Get IDs of exact matches to exclude
        exact_node_ids = set([node['id'] for node in exclude_exact['nodes']])
        exact_edge_ids = set([edge['id'] for edge in exclude_exact['edges']])

        # Search nodes semantically
        node_similarities = []
        for node_id, node_embedding in embeddings['nodes'].items():
            if node_id in exact_node_ids:
                continue

            similarity = cosine_similarity([query_embedding], [node_embedding])[0][0]
            node_similarities.append((node_id, similarity))

            if similarity >= semantic_threshold:
                semantic_nodes.append({
                    'id': node_id,
                    'similarity': float(similarity),
                    'text': embeddings['node_texts'][node_id],
                    'type': 'node'
                })

        # Debug: show top similarities even if they don't meet threshold
        node_similarities.sort(key=lambda x: x[1], reverse=True)
        print(f"DEBUG: Top 5 node similarities: {[(nid, f'{sim:.3f}') for nid, sim in node_similarities[:5]]}")

        # Search edges semantically
        edge_similarities = []
        for edge_id, edge_embedding in embeddings['edges'].items():
            if edge_id in exact_edge_ids:
                continue

            similarity = cosine_similarity([query_embedding], [edge_embedding])[0][0]
            edge_similarities.append((edge_id, similarity))

            if similarity >= semantic_threshold:
                semantic_edges.append({
                    'id': edge_id,
                    'similarity': float(similarity),
                    'text': embeddings['edge_texts'][edge_id],
                    'type': 'edge'
                })

        # Debug: show top similarities
        edge_similarities.sort(key=lambda x: x[1], reverse=True)
        print(f"DEBUG: Top 5 edge similarities: {[(eid, f'{sim:.3f}') for eid, sim in edge_similarities[:5]]}")

        # Sort and limit results
        semantic_nodes.sort(key=lambda x: x['similarity'], reverse=True)
        semantic_edges.sort(key=lambda x: x['similarity'], reverse=True)

        # Limit semantic matches
        semantic_nodes = semantic_nodes[:8]  # More when no exact matches
        semantic_edges = semantic_edges[:5]

        print(
            f"DEBUG: Semantic search found {len(semantic_nodes)} nodes, {len(semantic_edges)} edges above threshold {semantic_threshold}")

        return {'nodes': semantic_nodes, 'edges': semantic_edges}

    def find_exact_matches(self, query: str, embeddings: Dict) -> Dict:
        """Find exact string matches in node and edge data"""
        exact_nodes = []
        exact_edges = []

        # Clean the query - remove punctuation and extra spaces
        import re
        query_clean = re.sub(r'[^\w\s]', ' ', query).strip()
        query_lower = query_clean.lower()

        print(f"DEBUG: Searching for exact matches of '{query_clean}' (cleaned from '{query}')")

        # Search in node texts for exact matches
        for node_id, node_text in embeddings.get('node_texts', {}).items():
            node_text_lower = node_text.lower()

            # Check for exact substring match
            if query_lower in node_text_lower:
                # Calculate how exact the match is
                if f"value: {query_lower}" in node_text_lower:
                    similarity = 0.99  # Very high for exact value match
                elif query_lower in node_text_lower:
                    similarity = 0.95  # High for substring match
                else:
                    continue

                exact_nodes.append({
                    'id': node_id,
                    'similarity': similarity,
                    'text': node_text,
                    'type': 'node'
                })
                print(f"DEBUG: Exact node match - {node_id}: {similarity}")

        # Search in edge texts for exact matches
        for edge_id, edge_text in embeddings.get('edge_texts', {}).items():
            edge_text_lower = edge_text.lower()

            if query_lower in edge_text_lower:
                similarity = 0.90  # High but slightly lower than nodes
                exact_edges.append({
                    'id': edge_id,
                    'similarity': similarity,
                    'text': edge_text,
                    'type': 'edge'
                })
                print(f"DEBUG: Exact edge match - {edge_id}: {similarity}")

        # Sort by similarity (exact matches first)
        exact_nodes.sort(key=lambda x: x['similarity'], reverse=True)
        exact_edges.sort(key=lambda x: x['similarity'], reverse=True)

        return {'nodes': exact_nodes, 'edges': exact_edges}

    # Validation and warning analysis methods with semantic similarity
    # Keyword fallback methods
    def _determine_validation_status(self, node_data: Dict) -> str:
        """Determine validation status using config-driven logic"""
        config = self._load_status_config()

        # Check if this node has any status indicator fields
        found_status_fields = []
        for field_config in config['status_fields']:
            field_name = field_config['field']
            if field_name in node_data:
                found_status_fields.append({
                    'config': field_config,
                    'value': node_data[field_name]
                })

        # If no status fields found, check node type for classification
        if not found_status_fields:
            node_type = node_data.get('type', '').lower()
            if any(info_type in node_type for info_type in ['histogram', 'timeline', 'channel', 'timestamp']):
                return 'INFORMATIONAL'
            return config['default_status']

        # Process status fields according to config
        for status_field in found_status_fields:
            field_config = status_field['config']
            field_value = status_field['value']

            if field_config['type'] == 'boolean':
                if field_value is True and field_config.get('true_status'):
                    return field_config['true_status']
                elif field_value is False and field_config.get('false_status'):
                    return field_config['false_status']

            elif field_config['type'] == 'string':
                str_value = str(field_value).upper()
                mappings = field_config.get('mappings', {})
                if str_value in mappings:
                    return mappings[str_value]

        # Check text fields for failure keywords as fallback
        text_fields = ['validation_reason', 'error_message', 'description']
        for field in text_fields:
            if field in node_data:
                text_value = str(node_data[field]).lower()
                for keyword in config['failure_keywords']:
                    if keyword in text_value:
                        return 'FAILED'

        # If we found status fields but couldn't determine status, assume it's important
        return 'WARNING'

    def _contains_warning_indicators(self, node_data: Dict) -> bool:
        """Check if node has status indicators suggesting problems"""
        config = self._load_status_config()

        # Check if node has any status fields
        has_status_fields = any(
            field_config['field'] in node_data
            for field_config in config['status_fields']
        )

        if not has_status_fields:
            return False

        # Determine status and see if it indicates problems
        status = self._determine_validation_status(node_data)
        return status in ['FAILED', 'WARNING']

    def _load_status_config(self) -> Dict:
        """Load status configuration from JSON file"""
        if not hasattr(self, '_status_config'):
            config_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                'config', 'status_config.json'
            )
            try:
                with open(config_path, 'r') as f:
                    self._status_config = json.load(f)
            except FileNotFoundError:
                # Fallback config
                self._status_config = {
                    'status_fields': [
                        {
                            'field': 'is_valid',
                            'type': 'boolean',
                            'true_status': 'PASSED',
                            'false_status': 'FAILED'
                        }
                    ],
                    'failure_keywords': ['anachronism', 'error'],
                    'default_status': 'INFORMATIONAL'
                }

            print(f"Loaded status config with {len(self._status_config['status_fields'])} field rules")

        return self._status_config

    def _assess_warning_severity(self, node_data: Dict) -> str:
        """Assess warning severity using semantic similarity on all node data"""
        if not self.model:
            return self._assess_warning_severity_keywords(node_data)

        # Combine all text data from the node
        text_parts = []
        for key, value in node_data.items():
            if isinstance(value, (str, bool, int, float)) and value is not None:
                text_parts.append(f"{key}: {value}")

        text_to_analyze = " | ".join(text_parts)

        if not text_to_analyze.strip():
            return 'UNKNOWN'

        severity_categories = {
            'HIGH': [
                "critical failure with immediate attention required",
                "severe error causing system malfunction",
                "validation failed completely with data corruption",
                "anachronism detected with temporal violation",
                "measurement exceeds safety limits dangerously",
                "process terminated due to unrecoverable errors"
            ],
            'MEDIUM': [
                "warning condition requiring review and consideration",
                "potential issue that needs investigation",
                "validation passed but flagged inconsistencies",
                "measurement borderline requiring attention",
                "process completed with minor errors",
                "data quality concerns but not critical"
            ],
            'LOW': [
                "informational notice for reference only",
                "routine status update with normal operation",
                "data processed successfully without issues",
                "analysis completed providing useful insights",
                "measurement recorded within expected range",
                "process generated standard informational output"
            ]
        }

        text_embedding = self.model.encode([text_to_analyze])[0]

        best_match = 'UNKNOWN'
        highest_similarity = 0.0

        for severity, examples in severity_categories.items():
            example_embeddings = self.model.encode(examples)
            similarities = cosine_similarity([text_embedding], example_embeddings)[0]
            max_similarity = max(similarities)

            if max_similarity > highest_similarity:
                highest_similarity = max_similarity
                best_match = severity

        confidence_threshold = 0.2
        if highest_similarity >= confidence_threshold:
            return best_match
        else:
            return 'UNKNOWN'


    def get_method_analysis(self, graph_id: str) -> Dict:
        """OPTIMIZED: Pre-filter nodes and analyze semantically"""
        try:
            context_data = self.get_comprehensive_context(graph_id)
            graph_data = context_data.get('graph_data')
            reasoning_data = context_data.get('reasoning_data', {})

            if not graph_data:
                return {'error': 'Graph not found'}

            nodes = graph_data.get('nodes', {})
            edges = graph_data.get('edges', {})

            # PERFORMANCE: Pre-filter to only relevant nodes
            method_nodes = {
                node_id: node_data for node_id, node_data in nodes.items()
                if 'Method' in node_data.get('type', '')
            }

            result_nodes = {
                node_id: node_data for node_id, node_data in nodes.items()
                if any(result_type in node_data.get('type', '') for result_type in
                       ['ValidationResult', 'Result', 'histogram_data', 'Timeline', 'Timestamp', 'channel'])
            }

            print(
                f"Performance: Analyzing {len(method_nodes)} methods and {len(result_nodes)} results from {len(nodes)} total nodes")

            method_executions = []
            validation_results = []
            warnings = []

            # Analyze method executions
            for node_id, node_data in method_nodes.items():
                method_exec = {
                    'node_id': node_id,
                    'method_type': node_data.get('type', ''),
                    'value': node_data.get('value', ''),
                    'hierarchy': node_data.get('hierarchy', ''),
                    'data': node_data,
                    'connections': self._analyze_method_connections(node_id, edges, nodes)
                }
                method_executions.append(method_exec)

            # Analyze result nodes (including ValidationResult and all other result types)
            for node_id, node_data in result_nodes.items():
                # Determine status semantically for ALL result types
                status = self._determine_validation_status(node_data)

                validation = {
                    'node_id': node_id,
                    'type': node_data.get('type', ''),
                    'value': node_data.get('value', ''),
                    'hierarchy': node_data.get('hierarchy', ''),
                    'status': status,
                    'data': node_data  # Include full data for analysis
                }
                validation_results.append(validation)

                # Check for warnings in result nodes
                if self._contains_warning_indicators(node_data):
                    warnings.append({
                        'node_id': node_id,
                        'type': node_data.get('type', ''),
                        'content': node_data.get('value', ''),
                        'severity': self._assess_warning_severity(node_data),
                        'data': node_data
                    })

            # Also check method execution nodes for warnings
            for node_id, node_data in method_nodes.items():
                if self._contains_warning_indicators(node_data):
                    warnings.append({
                        'node_id': node_id,
                        'type': node_data.get('type', ''),
                        'content': node_data.get('value', ''),
                        'severity': self._assess_warning_severity(node_data),
                        'data': node_data
                    })

            # Analyze method outcomes
            method_insights = self._analyze_method_insights(method_executions, validation_results, reasoning_data)

            return {
                'method_execution_summary': {
                    'total_methods_run': len(method_executions),
                    'total_validations': len(validation_results),
                    'warnings_found': len(warnings),
                    'method_types': list(set([m['method_type'] for m in method_executions]))
                },
                'method_executions': method_executions,
                'validation_results': validation_results,
                'warnings_and_alerts': warnings,
                'method_insights': method_insights,
                'applicable_methods': reasoning_data.get('applicable_methods', []),
                'method_recommendations': self._generate_method_recommendations(method_executions, reasoning_data)
            }
        except Exception as e:
            return {'error': str(e)}

    def get_method_warnings(self, graph_id: str) -> Dict:
        """Get all warnings and alerts from method executions"""
        method_analysis = self.get_method_analysis(graph_id)

        if 'error' in method_analysis:
            return method_analysis

        warnings = method_analysis['warnings_and_alerts']
        validations = method_analysis['validation_results']

        # Group warnings by severity
        warnings_by_severity = {'HIGH': [], 'MEDIUM': [], 'LOW': []}
        for warning in warnings:
            severity = warning.get('severity', 'UNKNOWN')
            if severity in warnings_by_severity:
                warnings_by_severity[severity].append(warning)

        # Failed validations are high priority
        failed_validations = [v for v in validations if v['status'] == 'FAILED']
        warning_validations = [v for v in validations if v['status'] == 'WARNING']

        return {
            'total_warnings': len(warnings),
            'warnings_by_severity': warnings_by_severity,
            'failed_validations': failed_validations,
            'warning_validations': warning_validations,
            'critical_issues': len(warnings_by_severity['HIGH']) + len(failed_validations),
            'summary': f"{len(failed_validations)} failed validations, {len(warnings)} warnings detected"
        }

    def _analyze_method_connections(self, node_id: str, edges: Dict, nodes: Dict) -> Dict:
        """Find the results produced by this method"""

        def parse_edge_id(edge_id):
            if '_' in edge_id:
                parts = edge_id.split('_')
                if len(parts) == 2:
                    return parts[0], parts[1]
            return None, None

        produced_results = []
        total_connections = 0

        for edge_id, edge_data in edges.items():
            source = edge_data.get('source')
            target = edge_data.get('target')
            edge_type = edge_data.get('edge_type', edge_data.get('type', 'unknown'))

            # Parse from edge ID if needed
            if source is None or target is None:
                parsed_source, parsed_target = parse_edge_id(edge_id)
                if parsed_source and parsed_target:
                    source = parsed_source
                    target = parsed_target

            # Look for 'produces' relationships from this method
            if source == node_id and edge_type == 'produces' and target in nodes:
                result_node = nodes[target]
                produced_results.append({
                    'node_id': target,
                    'type': result_node.get('type', 'unknown'),
                    'value': result_node.get('value', 'Unknown'),
                    'data': result_node  # Include full node data
                })
                total_connections += 1

        return {
            'produced_results': produced_results,
            'total_connections': total_connections,
            'result_count': len(produced_results),
            # Keep legacy for compatibility
            'inputs': [],
            'outputs': produced_results,  # Map to outputs for backward compatibility
            'total_inputs': 0,
            'total_outputs': len(produced_results)
        }

    def _analyze_method_insights(self, method_executions: List[Dict], validation_results: List[Dict],
                                 reasoning_data: Dict) -> Dict:
        """Generate insights about what methods are telling us"""
        insights = {
            'validation_summary': {},
            'key_findings': [],
            'method_effectiveness': {},
            'data_quality_indicators': []
        }

        # Analyze validation patterns
        validation_statuses = {}
        for validation in validation_results:
            status = validation['status']
            validation_statuses[status] = validation_statuses.get(status, 0) + 1

        insights['validation_summary'] = validation_statuses

        # Generate key findings (cleaned up - no emojis in service layer)
        if validation_statuses.get('FAILED', 0) > 0:
            insights['key_findings'].append(
                f"{validation_statuses['FAILED']} validation(s) failed - data quality issues detected")

        if validation_statuses.get('WARNING', 0) > 0:
            insights['key_findings'].append(
                f"{validation_statuses['WARNING']} warning(s) issued - review recommended")

        if validation_statuses.get('PASSED', 0) > 0:
            insights['key_findings'].append(
                f"{validation_statuses['PASSED']} validation(s) passed - data quality good")

        return insights

    def _generate_method_recommendations(self, method_executions: List[Dict], reasoning_data: Dict) -> List[Dict]:
        """Generate recommendations for additional methods to run"""
        recommendations = []
        executed_types = set([m['method_type'] for m in method_executions])
        applicable_methods = reasoning_data.get('applicable_methods', [])

        for method in applicable_methods:
            method_name = method.get('name', '')
            if not any(method_name in exec_type for exec_type in executed_types):
                recommendations.append({
                    'method_name': method_name,
                    'reason': f"Available but not executed - applies to {', '.join(method.get('appliesTo', []))}",
                    'priority': 'HIGH' if 'Validation' in method_name else 'MEDIUM',
                    'description': method.get('description', 'No description available')
                })

        return recommendations[:5]