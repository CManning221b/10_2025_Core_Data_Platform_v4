from flask import Blueprint, render_template, request, abort, jsonify, redirect, url_for, flash
from backend.services.instance_graph_service import InstanceGraphService
from backend.services.visualization.time_series_service import TimeSeriesService
from backend.core_graph_managers.no2_graphManager.graphManager import GraphManager
from backend.services.inferred_ontology_service import OntologyService
from backend.services.visualization.core_plot_service import ChannelDataService

# Create blueprint
visualisation_bp = Blueprint('visualisation', __name__, url_prefix='/visualisation')
instance_service = InstanceGraphService()
time_series_service = TimeSeriesService()
channel_data_service = ChannelDataService()

@visualisation_bp.route('/timeline/<graph_id>', methods=['GET'])
def timeline(graph_id):
    """Display timeline visualization for a specific graph"""
    graph_data = instance_service.get_graph_data(graph_id)
    if not graph_data:
        abort(404)

    # Load graph into GraphManager
    graph_manager = GraphManager()
    graph_manager.node_data = graph_data.get('nodes', {})

    # Convert edges back to tuple format
    edges = graph_data.get('edges', {})
    graph_manager.edge_data = {}
    for edge_key, edge_data in edges.items():
        if isinstance(edge_key, str) and '_' in edge_key:
            source, target = edge_key.split('_', 1)
            graph_manager.edge_data[(source, target)] = edge_data

    # Get ontology for this graph
    ontology_service = OntologyService()
    ontology_manager = ontology_service.get_current_ontology(graph_id)

    # Generate timeline data and plot (default to showing events only)
    timeline_data = time_series_service.generate_timeline_data(
        graph_manager,
        ontology_manager,
        category_filter='events'  # Default to events only
    )

    return render_template('visualisation/time_series.html',
                           graph_id=graph_id,
                           timeline_data=timeline_data,
                           title=f"Timeline Analysis - {graph_id}")

@visualisation_bp.route('/timeline/<graph_id>/update', methods=['POST'])
def update_timeline(graph_id):
    """Update timeline with new sorting/filtering options"""
    try:
        graph_data = instance_service.get_graph_data(graph_id)
        if not graph_data:
            abort(404)

        # Get parameters from request
        sort_criteria = request.json.get('sort_criteria', {})
        color_strategy = request.json.get('color_strategy', {})
        filter_options = request.json.get('filter_options', {})
        visualization_mode = request.json.get('visualization_mode', 'timeline')
        category_filter = request.json.get('category_filter', 'all')

        print(f"Update request: mode={visualization_mode}, category={category_filter}, sort={sort_criteria}")

        # Load graph
        graph_manager = GraphManager()
        graph_manager.node_data = graph_data.get('nodes', {})
        edges = graph_data.get('edges', {})
        graph_manager.edge_data = {}
        for edge_key, edge_data in edges.items():
            if isinstance(edge_key, str) and '_' in edge_key:
                source, target = edge_key.split('_', 1)
                graph_manager.edge_data[(source, target)] = edge_data

        # Get ontology
        ontology_service = OntologyService()
        ontology_manager = ontology_service.get_current_ontology(graph_id)

        # Generate timeline with custom options
        timeline_data = time_series_service.generate_timeline_data(
            graph_manager,
            ontology_manager,
            sort_criteria=sort_criteria,
            color_strategy=color_strategy,
            filter_options=filter_options,
            visualization_mode=visualization_mode,
            category_filter=category_filter
        )

        print(f"Generated timeline: has_data={timeline_data.get('has_data')}, error={timeline_data.get('error')}")

        return jsonify(timeline_data)

    except Exception as e:
        import traceback
        print(f"ERROR in update_timeline: {e}")
        traceback.print_exc()
        return jsonify({
            'plot_html': None,
            'has_data': False,
            'error': f"Server error: {str(e)}"
        }), 500

@visualisation_bp.route('/timeline/<graph_id>/options', methods=['GET'])
def get_timeline_options(graph_id):
    """Get available sorting and filtering options for the timeline"""
    graph_data = instance_service.get_graph_data(graph_id)
    if not graph_data:
        abort(404)

    # Load graph
    graph_manager = GraphManager()
    graph_manager.node_data = graph_data.get('nodes', {})
    edges = graph_data.get('edges', {})
    graph_manager.edge_data = {}
    for edge_key, edge_data in edges.items():
        if isinstance(edge_key, str) and '_' in edge_key:
            source, target = edge_key.split('_', 1)
            graph_manager.edge_data[(source, target)] = edge_data

    # Get ontology
    ontology_service = OntologyService()
    ontology_manager = ontology_service.get_current_ontology(graph_id)

    # Analyze available options
    options = time_series_service.get_available_options(graph_manager, ontology_manager)

    return jsonify(options)


# Keep all your other existing routes (histogram, similarity, core_plot, etc.)
@visualisation_bp.route('/histogram/<graph_id>', methods=['GET'])
def histogram(graph_id):
    """Display histogram visualization for a specific graph"""
    # Check if graph exists
    graph_data = instance_service.get_graph_data(graph_id)
    if not graph_data:
        abort(404)

    return render_template('visualisation/histogram.html',
                           graph_id=graph_id,
                           title=f"Histogram Analysis - {graph_id}")


@visualisation_bp.route('/similarity/<graph_id>', methods=['GET'])
def similarity(graph_id):
    """Display similarity visualization for a specific graph"""
    # Check if graph exists
    graph_data = instance_service.get_graph_data(graph_id)
    if not graph_data:
        abort(404)

    return render_template('visualisation/similarity.html',
                           graph_id=graph_id,
                           title=f"Similarity Analysis - {graph_id}")


@visualisation_bp.route('/core_plot/<graph_id>', methods=['GET'])
def core_plot(graph_id):
    """Display ontology-driven core plot visualization for a specific graph"""
    # Check if graph exists
    graph_data = instance_service.get_graph_data(graph_id)
    if not graph_data:
        abort(404)

    # Load graph into GraphManager
    graph_manager = GraphManager()
    graph_manager.node_data = graph_data.get('nodes', {})

    # Convert edges back to tuple format
    edges = graph_data.get('edges', {})
    graph_manager.edge_data = {}
    for edge_key, edge_data in edges.items():
        if isinstance(edge_key, str) and '_' in edge_key:
            source, target = edge_key.split('_', 1)
            graph_manager.edge_data[(source, target)] = edge_data

    # Get ontology for this graph
    ontology_service = OntologyService()
    ontology_manager = ontology_service.get_current_ontology(graph_id)

    # Create channel data service with ontology
    enhanced_channel_service = ChannelDataService(ontology_manager)

    # Extract channel data and reactor configuration
    extraction_result = enhanced_channel_service.extract_channel_data_from_graph(graph_manager)

    # Handle both old and new return formats
    if isinstance(extraction_result, dict) and 'channels' in extraction_result:
        # New format with reactor info
        channel_data = extraction_result['channels']
        reactor_type = extraction_result.get('reactor_type', 'Unknown')
        reactor_config = extraction_result.get('reactor_config', {})
        detection_confidence = extraction_result.get('detection_confidence', 0.0)
        channel_display_mapping = extraction_result.get('channel_display_mapping', {})  # ADD THIS
    else:
        # Old format - just channel data
        channel_data = extraction_result
        reactor_type = 'Unknown'
        reactor_config = enhanced_channel_service._get_reactor_config('CANDU')  # Default
        detection_confidence = 0.0
        channel_display_mapping = {}  # ADD THIS

    # Debug info
    print(f"DEBUG: Extracted {len(channel_data)} channels for core plot")
    print(f"DEBUG: Detected reactor type: {reactor_type} (confidence: {detection_confidence:.2f})")
    print(f"DEBUG: Reactor config: {reactor_config}")

    for channel_name, data in list(channel_data.items())[:5]:  # Show first 5
        print(f"  {channel_name}: {data['status']}, {data['measurement_count']} measurements")

    # Prepare reactor info for template
    reactor_info = {
        'type': reactor_type,
        'type_display': _get_reactor_display_name(reactor_type),
        'confidence': detection_confidence,
        'grid_width': reactor_config.get('grid_width', 24),
        'grid_height': reactor_config.get('grid_height', 25),
        'total_positions': reactor_config.get('grid_width', 24) * reactor_config.get('grid_height', 25),
        'excluded_channels': reactor_config.get('excluded_channels', []),
        'channel_rule': reactor_config.get('channel_rule', 'default'),
        'regex_pattern': reactor_config.get('regex_pattern', r'^[A-Z]\d{2}$')
    }

    # Generate title with reactor information
    if reactor_type and reactor_type != 'Unknown':
        confidence_text = f" (Confidence: {detection_confidence:.1%})" if detection_confidence > 0 else ""
        title = f"Core Plot - {reactor_info['type_display']} Reactor{confidence_text} - {graph_id}"
    else:
        title = f"Core Plot - {graph_id}"

    return render_template('visualisation/core_plot.html',
                           graph_id=graph_id,
                           channel_data=channel_data,
                           reactor_info=reactor_info,
                           reactor_config=reactor_config,
                           channel_display_mapping=channel_display_mapping,  # ADD THIS
                           title=title)


def _get_reactor_display_name(reactor_type: str) -> str:
    """Convert reactor type code to display name"""
    reactor_names = {
        'CANDU': 'CANDU',
        'AGR': 'Advanced Gas-cooled Reactor (AGR)',
        'PWR': 'Pressurized Water Reactor (PWR)',
        'BWR': 'Boiling Water Reactor (BWR)'
    }
    return reactor_names.get(reactor_type, reactor_type or 'Unknown')

def extract_channel_data_from_graph(graph_manager):
    """Extract channel information from graph data"""
    channel_data = {}

    # Find all channel nodes
    for node_id, node_data in graph_manager.node_data.items():
        if node_data.get('type') == 'channel':
            channel_name = node_data.get('value', node_id)

            # Get associated measurement count
            measurement_count = 0
            temporal_data = []
            parent_folder = None

            # Find measurements linked to this channel
            for (source, target), edge_attrs in graph_manager.edge_data.items():
                if source == node_id:  # This channel links to something
                    target_node = graph_manager.node_data.get(target)
                    if target_node and target_node.get('type') == 'Timestamp':
                        measurement_count += 1
                        # Extract temporal info
                        temporal_data.append({
                            'timestamp_id': target,
                            'datetime': target_node.get('value', ''),
                            'components': {
                                'year': target_node.get('year'),
                                'month': target_node.get('month'),
                                'day': target_node.get('day'),
                                'hour': target_node.get('hour', 0),
                                'minute': target_node.get('minute', 0),
                                'second': target_node.get('second', 0)
                            }
                        })

                # Find parent folder
                if target == node_id:  # Something links to this channel
                    source_node = graph_manager.node_data.get(source)
                    if source_node and source_node.get('type') == 'folder':
                        parent_folder = source_node.get('value', '')

            channel_data[channel_name] = {
                'node_id': node_id,
                'node_type': 'channel',
                'measurement_count': measurement_count,
                'temporal_data': temporal_data,
                'parent_folder': parent_folder,
                'last_measurement': temporal_data[-1]['datetime'] if temporal_data else None,
                'status': 'active' if measurement_count > 0 else 'inactive'
            }

    return channel_data