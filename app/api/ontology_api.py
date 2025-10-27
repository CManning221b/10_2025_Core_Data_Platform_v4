# app/api/ontology_api.py
from flask import Blueprint, jsonify, request
from backend.services.inferred_ontology_service import OntologyService

ontology_api = Blueprint('ontology_api', __name__, url_prefix='/api/ontology')
ontology_service = OntologyService()


@ontology_api.route('/check/<graph_id>', methods=['GET'])
def check_ontology(graph_id):
    """Check if ontology visualization exists for a graph"""
    ontology_data = ontology_service.get_ontology_details(graph_id)

    if ontology_data and 'visualization_path' in ontology_data:
        return jsonify({
            'exists': True,
            'visualization_path': ontology_data['visualization_path']
        })

    return jsonify({
        'exists': False
    })


@ontology_api.route('/generate/<graph_id>', methods=['POST'])
def generate_ontology(graph_id):
    """Generate ontology visualization for a graph"""
    try:
        ontology_data = ontology_service.extract_ontology_from_graph(graph_id)

        if ontology_data and 'visualization_path' in ontology_data:
            return jsonify({
                'success': True,
                'visualization_path': ontology_data['visualization_path']
            })

        return jsonify({
            'success': False,
            'error': 'Failed to generate ontology visualization'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })