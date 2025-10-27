# backend/routes/graph_propagation_routes.py

from flask import Blueprint, render_template, request, jsonify, abort, Response
from backend.services.graph_propagation_service import GraphPropagationService
import json

propagation_bp = Blueprint('propagation', __name__, url_prefix='/propagation')
propagation_service = GraphPropagationService()


@propagation_bp.route('/<graph_id>')
def propagation_interface(graph_id):
    """Main propagation interface page"""
    # Get graph metadata for dropdowns
    metadata = propagation_service.get_graph_metadata(graph_id)

    if not metadata:
        abort(404, description=f"Graph {graph_id} not found")

    return render_template('propagation/propagation_interface.html',
                           graph_id=graph_id,
                           metadata=metadata)


@propagation_bp.route('/api/metadata/<graph_id>')
def get_metadata(graph_id):
    """API endpoint to get graph metadata for dropdowns"""
    metadata = propagation_service.get_graph_metadata(graph_id)

    if not metadata:
        return jsonify({'error': 'Graph not found'}), 404

    return jsonify(metadata)


@propagation_bp.route('/api/propagate', methods=['POST'])
def api_propagate():
    """Main API endpoint for property propagation"""
    try:
        config = request.json
        graph_id = config.get('graph_id')

        if not graph_id:
            return jsonify({'error': 'graph_id is required'}), 400

        # Validate required fields
        required_fields = ['source_node_types', 'source_property', 'method']
        for field in required_fields:
            if field not in config:
                return jsonify({'error': f'{field} is required'}), 400

        # Run propagation
        results = propagation_service.propagate_property(graph_id, config)

        return jsonify({
            'success': True,
            'results': results
        })

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Propagation failed: {str(e)}'}), 500


@propagation_bp.route('/api/filter', methods=['POST'])
def api_filter_range():
    """API endpoint for range filtering"""
    try:
        data = request.json
        propagation_results = data.get('propagation_results')
        min_val = data.get('min_value')
        max_val = data.get('max_value')

        if not all([propagation_results, min_val is not None, max_val is not None]):
            return jsonify({'error': 'propagation_results, min_value, and max_value are required'}), 400

        filtered_results = propagation_service.filter_by_range(
            propagation_results, min_val, max_val
        )

        return jsonify({
            'success': True,
            'filtered_results': filtered_results
        })

    except Exception as e:
        return jsonify({'error': f'Filtering failed: {str(e)}'}), 500


@propagation_bp.route('/api/layout', methods=['POST'])
def api_calculate_layout():
    """API endpoint for similarity-based layout calculation"""
    try:
        data = request.json
        propagation_results = data.get('propagation_results')
        layout_config = data.get('layout_config', {})

        if not propagation_results:
            return jsonify({'error': 'propagation_results is required'}), 400

        layout_results = propagation_service.calculate_similarity_layout(
            propagation_results, layout_config
        )

        return jsonify({
            'success': True,
            'layout': layout_results
        })

    except Exception as e:
        return jsonify({'error': f'Layout calculation failed: {str(e)}'}), 500


@propagation_bp.route('/api/generate-visualization/<graph_id>', methods=['POST'])
def api_generate_visualization(graph_id):
    """Generate interactive propagation visualization"""
    try:
        print("DEBUG: Visualization route called")
        data = request.json
        print(f"DEBUG: Received data keys: {data.keys() if data else 'None'}")

        propagation_results = data.get('propagation_results')
        layout_positions = data.get('layout_positions')

        if not propagation_results:
            return jsonify({'error': 'No propagation results provided'}), 400

        print("DEBUG: Calling generate_interactive_propagation_viz")
        # Generate interactive HTML visualization
        viz_path = propagation_service.generate_interactive_propagation_viz(
            graph_id, propagation_results, layout_positions
        )

        print(f"DEBUG: Generated viz path: {viz_path}")

        if viz_path:
            return jsonify({
                'success': True,
                'visualization_path': viz_path
            })
        else:
            return jsonify({'error': 'Failed to generate visualization'}), 500

    except Exception as e:
        print(f"DEBUG: Visualization error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@propagation_bp.route('/api/full-workflow', methods=['POST'])
def api_full_workflow():
    try:
        print("DEBUG: Starting full workflow")
        config = request.json
        graph_id = config.get('graph_id')

        # Check if this is a request for a specific chunk
        chunk_size = config.get('chunk_size', 1000)
        chunk_index = config.get('chunk_index')
        metadata_only = config.get('metadata_only', False)

        if not graph_id:
            print("DEBUG: Missing graph_id")
            return jsonify({'success': False, 'error': 'graph_id is required'}), 400

        print("DEBUG: Calling propagation service")
        propagation_results = propagation_service.propagate_property(graph_id, config)
        print("DEBUG: Propagation completed successfully")

        # If requesting metadata only
        if metadata_only:
            total_nodes = len(propagation_results['propagated_values'])
            total_chunks = (total_nodes + chunk_size - 1) // chunk_size
            print(f"DEBUG: Metadata request - total_nodes: {total_nodes}, total_chunks: {total_chunks}")

            return jsonify({
                'success': True,
                'metadata': {
                    'total_nodes': total_nodes,
                    'total_chunks': total_chunks,
                    'chunk_size': chunk_size,
                    'value_range': propagation_results['value_range'],
                    'source_nodes': propagation_results['source_nodes'],
                    'method': propagation_results['method']
                }
            })

        # If chunk_index is specified, return specific chunk
        if chunk_index is not None:
            total_nodes = len(propagation_results['propagated_values'])
            total_chunks = (total_nodes + chunk_size - 1) // chunk_size

            print(f"DEBUG: Chunk request - chunk_index: {chunk_index}, total_chunks: {total_chunks}")

            if chunk_index >= total_chunks:
                print(f"DEBUG: Invalid chunk index {chunk_index}, max is {total_chunks - 1}")
                return jsonify({
                    'success': False,
                    'error': f'Invalid chunk index {chunk_index}, max is {total_chunks - 1}'
                }), 400

            # Calculate chunk boundaries
            start_idx = chunk_index * chunk_size
            end_idx = min(start_idx + chunk_size, total_nodes)

            print(f"DEBUG: Chunk {chunk_index} - nodes {start_idx} to {end_idx - 1}")

            # Get chunk of nodes
            all_nodes = list(propagation_results['propagated_values'].keys())
            chunk_nodes = all_nodes[start_idx:end_idx]

            print(f"DEBUG: Chunk {chunk_index} contains {len(chunk_nodes)} nodes")

            # Extract chunk data
            chunk_propagated_values = {
                node: propagation_results['propagated_values'][node]
                for node in chunk_nodes
            }

            chunk_gmm_data = {
                node: propagation_results['gmm_data'][node]
                for node in chunk_nodes
                if node in propagation_results['gmm_data']
            }

            response = {
                'success': True,
                'chunk_index': chunk_index,
                'total_chunks': total_chunks,
                'chunk_data': {
                    'propagated_values': chunk_propagated_values,
                    'gmm_data': chunk_gmm_data
                }
            }

            # Add additional results only on the last chunk
            if chunk_index == total_chunks - 1:
                print(f"DEBUG: Adding additional results to final chunk {chunk_index}")

                if 'min_value' in config and 'max_value' in config:
                    print("DEBUG: Applying filter by range")
                    filtered_results = propagation_service.filter_by_range(
                        propagation_results, config['min_value'], config['max_value']
                    )
                    response['filtered_results'] = filtered_results

                if config.get('calculate_layout', False):
                    print("DEBUG: Calculating layout")
                    layout_config = config.get('layout_config', {})
                    layout_results = propagation_service.calculate_similarity_layout(
                        propagation_results, layout_config
                    )
                    response['layout'] = layout_results

            print(f"DEBUG: Returning chunk {chunk_index} with {len(chunk_propagated_values)} propagated values")
            return jsonify(response)

        # Original non-chunked response (fallback)
        print("DEBUG: Non-chunked request")
        response = {
            'success': True,
            'propagation_results': propagation_results
        }

        if 'min_value' in config and 'max_value' in config:
            print("DEBUG: Applying filter by range")
            filtered_results = propagation_service.filter_by_range(
                propagation_results, config['min_value'], config['max_value']
            )
            response['filtered_results'] = filtered_results

        if config.get('calculate_layout', False):
            print("DEBUG: Calculating layout")
            layout_config = config.get('layout_config', {})
            layout_results = propagation_service.calculate_similarity_layout(
                propagation_results, layout_config
            )
            response['layout'] = layout_results

        print("DEBUG: Returning full response")
        return jsonify(response)

    except Exception as e:
        print(f"DEBUG: Exception occurred: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Workflow failed: {str(e)}'}), 500

# Helper route for debugging
@propagation_bp.route('/api/debug/<graph_id>')
def api_debug_graph(graph_id):
    """Debug endpoint to see graph structure"""
    try:
        metadata = propagation_service.get_graph_metadata(graph_id)
        return jsonify({
            'graph_id': graph_id,
            'metadata': metadata,
            'debug': True
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@propagation_bp.route('/api/extract-subgraph/<graph_id>', methods=['POST'])
def api_extract_subgraph(graph_id):
    """API endpoint for subgraph extraction"""
    try:
        data = request.json
        node_ids = data.get('node_ids', [])
        include_neighbors = data.get('include_neighbors', True)

        if not node_ids:
            return jsonify({'error': 'node_ids is required'}), 400

        subgraph = propagation_service.extract_subgraph(
            graph_id, node_ids, include_neighbors
        )

        return jsonify({
            'success': True,
            'subgraph': subgraph
        })

    except Exception as e:
        return jsonify({'error': f'Subgraph extraction failed: {str(e)}'}), 500