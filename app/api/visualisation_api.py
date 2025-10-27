from flask import Blueprint, jsonify, abort
from backend.services.instance_graph_service import InstanceGraphService
from backend.services.visualization.time_series_service import TimeSeriesService
from backend.core_graph_managers.no2_graphManager.graphManager import GraphManager

# Create API blueprint
visualisation_api = Blueprint('visualisation_api', __name__, url_prefix='/api/visualization')
instance_service = InstanceGraphService()


@visualisation_api.route('/graphs', methods=['GET'])
def get_available_graphs():
    """API endpoint to get list of available graphs for visualization"""
    try:
        graphs = instance_service.get_all_graphs()
        return jsonify({
            "success": True,
            "graphs": graphs
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Error retrieving graphs: {str(e)}"
        }), 500