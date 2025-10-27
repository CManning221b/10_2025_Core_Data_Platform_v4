# app/routes/ontology_routes.py
from flask import Blueprint, render_template, request, abort, jsonify, redirect, url_for
from backend.services.inferred_ontology_service import OntologyService
from backend.services.instance_graph_service import InstanceGraphService
from flask import render_template, request, redirect, url_for, flash, current_app
from backend.services.loaded_ontology_service import LoadedOntologyService
from backend.services.inferred_ontology_service import OntologyService
from backend.core_graph_managers.no2_graphManager.graphManager import GraphManager
from backend.core_graph_managers.no3_OntologyManager.OntologyManager import OntologyManager
import os

# Create blueprint
ontology_bp = Blueprint('ontology', __name__, url_prefix='/ontology')
ontology_service = OntologyService()
instance_service = InstanceGraphService()  # For getting graph data

@ontology_bp.route('/extract/<graph_id>', methods=['GET', 'POST'])
def extract_ontology(graph_id):
    """Extract ontology from a graph"""
    # Check if graph exists
    graph_data = instance_service.get_graph_data(graph_id)
    if not graph_data:
        abort(404)

    # If POST, force re-extraction
    if request.method == 'POST':
        ontology_data = ontology_service.extract_ontology_from_graph(graph_id)
    else:
        # Try to get existing ontology first
        ontology_data = ontology_service.get_ontology_details(graph_id)

        # If not found, extract it
        if not ontology_data:
            ontology_data = ontology_service.extract_ontology_from_graph(graph_id)

    if not ontology_data:
        abort(500)

    ontology_manager = ontology_service.get_current_ontology(graph_id)
    ontology_node_types = {}
    ontology_edge_types = {}

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

        print(f"Ontology Manager - Node types: {len(ontology_node_types)}, Edge types: {len(ontology_edge_types)}")

    if ontology_node_types:
        ontology_data['node_types'] = ontology_node_types
    if ontology_edge_types:
        ontology_data['edge_types'] = ontology_edge_types

    # Debug print to see the structure
    import pprint
    print("Ontology Data Structure:")
    pprint.pprint(ontology_data)

    return render_template('ontology/inferred_ontology.html',
                           graph_id=graph_id,
                           ontology_data=ontology_data)

@ontology_bp.route('/analyze/<graph_id>', methods=['GET'])
def analyze_ontology(graph_id):
    """Show detailed ontology analysis page"""
    # Get ontology data

    ontology_data = ontology_service.extract_ontology_from_graph(graph_id)

    if not ontology_data:
        abort(404)

    return render_template('ontology/ontology_analysis.html',
                           graph_id=graph_id,
                           ontology_data=ontology_data)

@ontology_bp.route('/node_type/<graph_id>/<node_type>', methods=['GET'])
def node_type_details(graph_id, node_type):
    """Show details for a specific node type"""
    # Get analysis for the node type
    analysis = ontology_service.analyze_node_type(graph_id, node_type)

    if not analysis:
        abort(404)

    return render_template('ontology/node_type_details.html',
                           graph_id=graph_id,
                           node_type=node_type,
                           analysis=analysis)

@ontology_bp.route('/edge_type/<graph_id>/<edge_type>', methods=['GET'])
def edge_type_details(graph_id, edge_type):
    """Show details for a specific edge type"""
    # Get analysis for the edge type
    analysis = ontology_service.analyze_edge_type(graph_id, edge_type)

    if not analysis:
        abort(404)

    return render_template('ontology/edge_type_details.html',
                           graph_id=graph_id,
                           edge_type=edge_type,
                           analysis=analysis)

@ontology_bp.route('/relationship/<graph_id>', methods=['GET'])
def relationship_analysis(graph_id):
    """Analyze a relationship pattern"""
    source_type = request.args.get('source_type')
    edge_type = request.args.get('edge_type')
    target_type = request.args.get('target_type')

    if not all([source_type, edge_type, target_type]):
        # Get ontology data to display available types
        ontology_data = ontology_service.get_ontology_details(graph_id)
        if not ontology_data:
            abort(404)

        return render_template('ontology/relationship_form.html',
                               graph_id=graph_id,
                               ontology_data=ontology_data)

    # Analyze the relationship
    analysis = ontology_service.analyze_relationship_cardinality(
        graph_id, source_type, edge_type, target_type)

    if not analysis:
        abort(404)

    return render_template('ontology/relationship_details.html',
                           graph_id=graph_id,
                           analysis=analysis)

@ontology_bp.route('/validation/<graph_id>', methods=['GET'])
def validation_rules(graph_id):
    """Show validation rules for a graph"""
    # Get validation rules
    rules = ontology_service.get_validation_rules(graph_id)

    if not rules:
        abort(404)

    return render_template('ontology/validation_rules.html',
                           graph_id=graph_id,
                           rules=rules)

@ontology_bp.route('/visualize/<graph_id>', methods=['GET'])
def visualize_ontology(graph_id):
    """View the ontology visualization"""
    # Get ontology data to check if visualization exists
    ontology_data = ontology_service.get_ontology_details(graph_id)

    if not ontology_data or not ontology_data.get('has_visualization'):
        # Try to extract ontology if visualization doesn't exist
        ontology_data = ontology_service.extract_ontology_from_graph(graph_id)

        if not ontology_data or not ontology_data.get('visualization_path'):
            abort(404)

    # Redirect to the static visualization file
    return redirect(url_for('static', filename=f"visualizations/{ontology_data['visualization_path']}"))

@ontology_bp.route('/loaded', methods=['GET'])
def loaded_ontology():
    """Display the loaded OWL ontology with management options"""
    service = LoadedOntologyService()

    # Get the base project directory (two levels up from the app directory)
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Construct the path to ontologies directory
    ontology_dir = os.path.join(base_dir, 'data', 'ontologies', 'predefined')

    # Make sure the directory exists
    os.makedirs(ontology_dir, exist_ok=True)

    # Get .ttl, .owl, and .rdf files in the directory
    ontology_files = []
    if os.path.exists(ontology_dir):
        ontology_files = [os.path.join(ontology_dir, f) for f in os.listdir(ontology_dir)
                          if f.endswith('.ttl') or f.endswith('.owl') or f.endswith('.rdf')]

    # Get file sizes
    file_sizes = {}
    for file_path in ontology_files:
        try:
            size_bytes = os.path.getsize(file_path)
            # Convert to human-readable format
            if size_bytes < 1024:
                file_sizes[file_path] = f"{size_bytes} B"
            elif size_bytes < 1024 * 1024:
                file_sizes[file_path] = f"{size_bytes / 1024:.1f} KB"
            else:
                file_sizes[file_path] = f"{size_bytes / (1024 * 1024):.1f} MB"
        except:
            file_sizes[file_path] = "Unknown"

    # If no files found, show a message
    if not ontology_files:
        flash("No ontology files found in the predefined directory.", "warning")
        # Create a minimal ontology_data structure to avoid template errors
        empty_ontology_data = {
            "ontology_summary": {
                "class_count": 0,
                "property_count": 0
            },
            "classes": {}
        }
        return render_template('ontology/loaded_ontology.html',
                               title="Loaded OWL Ontology",
                               ontology_files=[],
                               file_sizes={},
                               ontology_data=empty_ontology_data,
                               visualization_path=None)

    # Load the ontologies
    knowledge_graph = service.load_ontologies(ontology_files)

    # Get structured ontology data
    ontology_data = service.get_ontology_data() or {
        "ontology_summary": {
            "class_count": 0,
            "property_count": 0
        },
        "classes": {}
    }

    # Generate visualization
    vis_dir = os.path.join(base_dir, 'app', 'static', 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    vis_path = os.path.join(vis_dir, 'loaded_ontology.html')
    service.visualize(vis_path)

    return render_template('ontology/loaded_ontology.html',
                           title="Loaded OWL Ontology",
                           ontology_files=ontology_files,
                           file_sizes=file_sizes,
                           ontology_data=ontology_data,
                           visualization_path='loaded_ontology.html')

@ontology_bp.route('/compare', methods=['GET'])
def compare_ontologies():
    """Compare loaded and inferred ontologies"""
    # Get graph_id from query parameter
    graph_id = request.args.get('graph_id')

    # If no graph_id, redirect to the selection page
    if not graph_id:
        flash("Please select a graph to compare with the OWL ontology", "warning")
        return redirect(url_for('ontology.compare_select'))

    # Load inferred ontology service
    inferred_service = OntologyService()

    # Use the new method to get the ontology directly
    inferred_ontology = inferred_service.get_current_ontology(graph_id)
    if not inferred_ontology:
        flash("Could not get ontology for the selected graph", "error")
        return redirect(url_for('instance.view_graph', graph_id=graph_id))

    # Get the ontology data for visualization path
    inferred_ontology_data = inferred_service.get_ontology_details(graph_id)
    if not inferred_ontology_data:
        inferred_ontology_data = inferred_service.extract_ontology_from_graph(graph_id)

    # Load OWL ontology service
    loaded_service = LoadedOntologyService()

    # Get the base project directory (two levels up from the app directory)
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Construct the path to ontologies directory
    ontology_dir = os.path.join(base_dir, 'data', 'ontologies', 'predefined')

    # Make sure the directory exists
    os.makedirs(ontology_dir, exist_ok=True)

    # Get .ttl files in the directory
    ontology_files = []
    if os.path.exists(ontology_dir):
        ontology_files = [os.path.join(ontology_dir, f) for f in os.listdir(ontology_dir)
                          if f.endswith('.ttl')]

    # If no files found, show a message
    if not ontology_files:
        flash("No ontology files found in the predefined directory.", "warning")
        return redirect(url_for('ontology.loaded_ontology'))

    loaded_service.load_ontologies(ontology_files)

    # Compare ontologies
    comparison = loaded_service.compare_with_inferred(inferred_ontology)

    # Generate visualizations with highlights
    vis_dir = os.path.join(base_dir, 'app', 'static', 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)

    # Generate visualization for loaded ontology with all highlighting types
    loaded_vis_path = os.path.join(vis_dir, f'loaded_ontology_compared_{graph_id}.html')
    try:
        loaded_service.visualize(
            loaded_vis_path,
            highlighted_nodes=comparison['matching_classes'],
            fuzzy_matches=comparison['fuzzy_matching_classes'],
            methods=comparison['methods_application']
        )
        loaded_vis_filename = f'loaded_ontology_compared_{graph_id}.html'
    except Exception as e:
        print(f"Error visualizing loaded ontology: {str(e)}")
        # Fallback to using the standard visualization
        loaded_vis_filename = 'loaded_ontology.html'

    # Use the existing inferred ontology visualization as fallback
    inferred_vis_filename = f'{graph_id}_ontology.html'

    # Try to create a highlighted version of inferred ontology
    try:
        # Generate visualization for inferred ontology with the same highlighting
        inferred_vis_path = os.path.join(vis_dir, f'inferred_ontology_compared_{graph_id}.html')

        # Here we need to adapt the highlighting for the inferred ontology
        inferred_matching = comparison['matching_classes']
        inferred_methods = []  # No methods_application in inferred ontology

        # For fuzzy matches, we need to use the inferred side of the tuple
        inferred_fuzzy_matches = []
        if comparison['fuzzy_matching_classes']:
            for loaded, inferred in comparison['fuzzy_matching_classes']:
                inferred_fuzzy_matches.append((inferred, inferred))  # Use same value for both sides

        # If visualize_ontology is available, use it
        if hasattr(inferred_service, 'visualize_ontology'):
            result = inferred_service.visualize_ontology(
                graph_id,
                inferred_vis_path,
                highlighted_nodes=inferred_matching,
                fuzzy_matches=inferred_fuzzy_matches,
                methods=inferred_methods
            )
            if result:
                inferred_vis_filename = result
    except Exception as e:
        print(f"Error visualizing inferred ontology: {str(e)}")
        # Use existing visualization or default back to standard one
        if inferred_ontology_data and 'visualization_path' in inferred_ontology_data:
            inferred_vis_filename = inferred_ontology_data['visualization_path']

    return render_template('ontology/compare_ontologies.html',
                           title="Ontology Comparison",
                           comparison=comparison,
                           graph_id=graph_id,
                           loaded_vis_path=loaded_vis_filename,
                           inferred_vis_path=inferred_vis_filename)

@ontology_bp.route('/compare_select', methods=['GET'])
def compare_select():
    """Select a graph to compare with OWL ontology"""
    # Get available graphs from the instance service
    # Use get_all_graphs instead of list_graphs
    graphs = instance_service.get_all_graphs()

    # If no graphs, show a message
    if not graphs:
        flash("No graphs available for comparison. Please create some graphs first.", "warning")
        return redirect(url_for('instance.list_graphs'))

    return render_template('ontology/compare_selected.html',
                           title="Select Graph for Comparison",
                           graphs=graphs)

@ontology_bp.route('/upload_ontology', methods=['POST'])
def upload_ontology():
    """Handle uploading a new ontology file"""
    # Check if the post request has the file part
    if 'ontology_file' not in request.files:
        flash('No ontology file provided', 'error')
        return redirect(url_for('ontology.loaded_ontology'))

    file = request.files['ontology_file']

    # If the user does not select a file
    if file.filename == '':
        flash('No ontology file selected', 'error')
        return redirect(url_for('ontology.loaded_ontology'))

    # Check file type
    if not file.filename.endswith(('.ttl', '.owl', '.rdf')):
        flash('Invalid file format. Please upload .ttl, .owl, or .rdf files only', 'error')
        return redirect(url_for('ontology.loaded_ontology'))

    # Get the ontologies directory
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ontology_dir = os.path.join(base_dir, 'data', 'ontologies', 'predefined')
    os.makedirs(ontology_dir, exist_ok=True)

    # Save the file
    try:
        filepath = os.path.join(ontology_dir, file.filename)
        file.save(filepath)
        flash(f'Successfully uploaded {file.filename}', 'success')

        # Reload the ontology service
        service = LoadedOntologyService()
        ontology_files = [os.path.join(ontology_dir, f) for f in os.listdir(ontology_dir)
                          if f.endswith('.ttl') or f.endswith('.owl') or f.endswith('.rdf')]
        service.load_ontologies(ontology_files)


        # Regenerate visualization
        vis_dir = os.path.join(base_dir, 'app', 'static', 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        vis_path = os.path.join(vis_dir, 'loaded_ontology.html')
        service.visualize(vis_path)

    except Exception as e:
        flash(f'Error uploading ontology: {str(e)}', 'error')

    return redirect(url_for('ontology.loaded_ontology'))

@ontology_bp.route('/delete_ontology', methods=['POST'])
def delete_ontology():
    """Delete a specific ontology file"""
    filename = request.form.get('filename')
    if not filename:
        flash('No filename provided', 'error')
        return redirect(url_for('ontology.loaded_ontology'))

    # Get the ontologies directory
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ontology_dir = os.path.join(base_dir, 'data', 'ontologies', 'predefined')

    # Check if file exists
    filepath = os.path.join(ontology_dir, filename)
    if not os.path.exists(filepath):
        flash(f'File {filename} not found', 'error')
        return redirect(url_for('ontology.loaded_ontology'))

    # Delete the file
    try:
        os.remove(filepath)
        flash(f'Successfully deleted {filename}', 'success')

        # Reload the ontology service if there are remaining files
        remaining_files = [os.path.join(ontology_dir, f) for f in os.listdir(ontology_dir)
                           if f.endswith('.ttl') or f.endswith('.owl') or f.endswith('.rdf')]

        if remaining_files:
            service = LoadedOntologyService()
            service.load_ontologies(remaining_files)

            # Regenerate visualization
            vis_dir = os.path.join(base_dir, 'app', 'static', 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)
            vis_path = os.path.join(vis_dir, 'loaded_ontology.html')
            service.visualize(vis_path)

    except Exception as e:
        flash(f'Error deleting ontology: {str(e)}', 'error')

    return redirect(url_for('ontology.loaded_ontology'))

@ontology_bp.route('/clear_ontologies', methods=['POST'])
def clear_ontologies():
    """Delete all ontology files"""
    # Get the ontologies directory
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ontology_dir = os.path.join(base_dir, 'data', 'ontologies', 'predefined')

    # Check if directory exists
    if not os.path.exists(ontology_dir):
        flash('Ontology directory not found', 'error')
        return redirect(url_for('ontology.loaded_ontology'))

    # Delete all ontology files
    try:
        file_count = 0
        for filename in os.listdir(ontology_dir):
            if filename.endswith(('.ttl', '.owl', '.rdf')):
                os.remove(os.path.join(ontology_dir, filename))
                file_count += 1

        if file_count > 0:
            flash(f'Successfully deleted {file_count} ontology files', 'success')
        else:
            flash('No ontology files found to delete', 'info')

    except Exception as e:
        flash(f'Error clearing ontologies: {str(e)}', 'error')

    return redirect(url_for('ontology.loaded_ontology'))

@ontology_bp.route('/reload_ontologies', methods=['GET'])
def reload_ontologies():
    """Reload all ontology files and regenerate the visualization"""
    # Get the ontologies directory
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ontology_dir = os.path.join(base_dir, 'data', 'ontologies', 'predefined')

    # Ensure directory exists
    os.makedirs(ontology_dir, exist_ok=True)

    # Get ontology files
    ontology_files = [os.path.join(ontology_dir, f) for f in os.listdir(ontology_dir)
                      if f.endswith('.ttl') or f.endswith('.owl') or f.endswith('.rdf')]

    if not ontology_files:
        flash('No ontology files found to reload', 'info')
        return redirect(url_for('ontology.loaded_ontology'))

    # Reload ontologies
    try:
        service = LoadedOntologyService()
        service.load_ontologies(ontology_files)

        # Regenerate visualization
        vis_dir = os.path.join(base_dir, 'app', 'static', 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        vis_path = os.path.join(vis_dir, 'loaded_ontology.html')
        service.visualize(vis_path)

        flash(f'Successfully reloaded {len(ontology_files)} ontology files', 'success')

    except Exception as e:
        flash(f'Error reloading ontologies: {str(e)}', 'error')

    return redirect(url_for('ontology.loaded_ontology'))

@ontology_bp.route('/view_ontology_file/<filename>', methods=['GET'])
def view_ontology_file(filename):
    """View the content of an ontology file"""
    # Get the ontologies directory
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ontology_dir = os.path.join(base_dir, 'data', 'ontologies', 'predefined')

    # Check if file exists
    filepath = os.path.join(ontology_dir, filename)
    if not os.path.exists(filepath):
        flash(f'File {filename} not found', 'error')
        return redirect(url_for('ontology.loaded_ontology'))

    # Read the file content
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Determine the content type for proper formatting
        if filename.endswith('.ttl'):
            language = 'turtle'
        elif filename.endswith('.owl') or filename.endswith('.rdf'):
            language = 'xml'
        else:
            language = 'text'

        return render_template('ontology/view_ontology_file.html',
                               title=f"Viewing {filename}",
                               filename=filename,
                               content=content,
                               language=language)

    except Exception as e:
        flash(f'Error reading file: {str(e)}', 'error')
        return redirect(url_for('ontology.loaded_ontology'))


@ontology_bp.route('/export_owl/<graph_id>', methods=['GET', 'POST'])
def export_owl(graph_id):
    """Export ontology to OWL format (TTL)"""
    # Check if graph exists
    graph_data = instance_service.get_graph_data(graph_id)
    if not graph_data:
        abort(404)

    # Call the service to export as TTL
    result = ontology_service.export_ontology_to_owl(
        graph_id=graph_id,
        format='ttl'  # Always use TTL format for now
    )

    if not result:
        flash("Failed to export ontology to TTL format", "error")
        return redirect(url_for('ontology.extract_ontology', graph_id=graph_id))

    # Generate a download URL that's accessible from the browser
    download_url = url_for('static',
                           filename=f"ontologies/owl/{result['ontology_file']}")

    # Redirect directly to the file for immediate download
    flash(f"Successfully exported ontology to TTL format", "success")
    return redirect(download_url)