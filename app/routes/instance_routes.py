# app/routes/instance_routes.py
from flask import Blueprint, render_template, request, abort, jsonify, redirect, url_for
from backend.services.instance_graph_service import InstanceGraphService

# Create blueprint
instance_bp = Blueprint('instance', __name__, url_prefix='/instance')
instance_service = InstanceGraphService()


@instance_bp.route('/', methods=['GET'])
def list_graphs():
    """Display a list of all available graphs"""
    print(instance_service.list_graphs())
    graphs = instance_service.get_all_graphs()
    return render_template('instance_graph/instance_graph_list.html', graphs=graphs)


@instance_bp.route('/view/<graph_id>', methods=['GET'])
def view_graph(graph_id):
    """View a specific graph by ID"""
    print("Inside instance routes. view graph {}".format(graph_id))
    graph_data = instance_service.get_graph_data(graph_id)
    if not graph_data:
        abort(404)

    # Generate PyVis visualization
    vis_path = instance_service.generate_pyvis_html(graph_id)

    return render_template('instance_graph/instance_graph_view.html',
                           graph_id=graph_id,
                           graph_data=graph_data,
                           vis_path=vis_path)


@instance_bp.route('/visualize/<graph_id>', methods=['GET'])
def visualize_graph(graph_id):
    """Generate and view visualization for a specific graph"""
    # Generate PyVis visualization
    vis_path = instance_service.generate_pyvis_html(graph_id)

    if not vis_path:
        abort(404)

    return redirect(url_for('static', filename=vis_path))


@instance_bp.route('/details/<graph_id>', methods=['GET'])
def graph_details(graph_id):
    """View detailed information about a graph"""
    graph_data = instance_service.get_graph_data(graph_id)

    if not graph_data:
        abort(404)

    # Extract statistics
    nodes = graph_data.get('nodes', {})
    edges = graph_data.get('edges', {})

    # Count node types
    node_types = {}
    for node_id, node in nodes.items():
        node_type = node.get('type', 'unknown')
        node_types[node_type] = node_types.get(node_type, 0) + 1

    # Count edge types
    edge_types = {}
    for edge_id, edge in edges.items():
        edge_type = edge.get('edge_type', 'unknown')
        edge_types[edge_type] = edge_types.get(edge_type, 0) + 1

    stats = {
        'node_count': len(nodes),
        'edge_count': len(edges),
        'node_types': node_types,
        'edge_types': edge_types
    }

    return render_template('instance_graph/instance_graph_details.html',
                           graph_id=graph_id,
                           graph_data=graph_data,
                           stats=stats)


@instance_bp.route('/data/<graph_id>', methods=['GET'])
def get_graph_data(graph_id):
    """API endpoint to get graph data as JSON"""
    graph_data = instance_service.get_graph_data(graph_id)

    if not graph_data:
        abort(404)

    return jsonify(graph_data)