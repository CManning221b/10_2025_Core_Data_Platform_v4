# app/routes/process_routes.py
from flask import Blueprint, render_template, request, abort, jsonify, redirect, url_for, flash, current_app
from backend.services.instance_graph_service import InstanceGraphService
from backend.services.loaded_ontology_service import LoadedOntologyService
from backend.services.process_graph_service import ProcessGraphService
from backend.core_graph_managers.no2_graphManager.graphManager import GraphManager
from backend.core_graph_managers.no3_OntologyManager.OntologyManager import OntologyManager
import os
import json

# Create blueprint
process_bp = Blueprint('process', __name__, url_prefix='/process')
instance_service = InstanceGraphService()  # For getting graph data
process_service = ProcessGraphService()  # Process graph service to create
ontology_service = LoadedOntologyService()  # For getting available methods_application


@process_bp.route('/', methods=['GET'])
@process_bp.route('/graphs', methods=['GET'])
def process_graphs():
    """Display all loaded process graphs and tools for creating new ones"""
    # Get the base project directory
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Construct the path to process graphs directory
    process_dir = os.path.join(base_dir, 'data', 'processes', 'definitions')

    # Make sure the directory exists
    os.makedirs(process_dir, exist_ok=True)

    # Get process graph files in the directory
    process_files = []
    if os.path.exists(process_dir):
        for filename in os.listdir(process_dir):
            if filename.endswith(('.ttl', '.owl', '.rdf', '.json')):
                process_path = os.path.join(process_dir, filename)

                # Extract process metadata (in a real implementation, this would parse the file)
                process_name = filename.split('.')[0].replace('_', ' ').title()

                try:
                    # Try to load the process file to get details
                    process_details = process_service.get_process_details(process_path)
                    process_files.append({
                        'id': filename,
                        'name': process_details.get('name', process_name),
                        'step_count': process_details.get('step_count', 0),
                        'description': process_details.get('description', '')
                    })
                except Exception as e:
                    # If loading fails, add with minimal details
                    print(f"Error loading process {filename}: {e}")
                    process_files.append({
                        'id': filename,
                        'name': process_name,
                        'step_count': 0,
                        'description': 'Could not load process details'
                    })

    # Get execution history (empty for now)
    execution_history = []

    return render_template('process/process_graphs.html',
                           title="Process Graphs",
                           process_files=process_files,
                           execution_history=execution_history)


@process_bp.route('/view/<process_id>', methods=['GET'])
def view_process(process_id):
    """View details and visualization of a specific process graph"""
    # Get the base directory
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Construct the path to the process file
    process_path = os.path.join(base_dir, 'data', 'processes', 'definitions', process_id)

    # Check if file exists
    if not os.path.exists(process_path):
        flash(f"Process graph {process_id} not found", "error")
        return redirect(url_for('process.process_graphs'))

    try:
        # Get process details
        process_details = process_service.get_process_details(process_path)

        # Generate visualization if it doesn't exist
        vis_dir = os.path.join(base_dir, 'app', 'static', 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)

        vis_filename = f'process_{process_id.replace(".", "_")}.html'
        vis_path = os.path.join(vis_dir, vis_filename)

        if not os.path.exists(vis_path) or process_service.is_newer(process_path, vis_path):
            # Generate new visualization
            process_service.visualize_process(process_path, vis_path)

        # Generate a preview of the process file for the source tab
        process_preview = ""
        try:
            with open(process_path, 'r', encoding='utf-8') as f:
                content = f.read()

                # Limit preview to first 500 characters or 20 lines
                lines = content.split('\n')
                if len(lines) > 20:
                    process_preview = '\n'.join(lines[:20]) + "\n\n... (file continues) ..."
                elif len(content) > 2000:
                    process_preview = content[:2000] + "\n\n... (file continues) ..."
                else:
                    process_preview = content

                # Handle special characters for HTML display
                process_preview = process_preview.replace('<', '&lt;').replace('>', '&gt;')
        except Exception as e:
            process_preview = f"Error loading file preview: {str(e)}"

        return render_template('process/view_process.html',
                               title=f"Process: {process_details.get('name', process_id)}",
                               process=process_details,
                               visualization_path=vis_filename,
                               process_preview=process_preview)

    except Exception as e:
        flash(f"Error loading process: {str(e)}", "error")
        return redirect(url_for('process.process_graphs'))


@process_bp.route('/upload_process', methods=['POST'])
def upload_process():
    """Handle uploading a new process graph file"""
    # Check if the post request has the file part
    if 'process_file' not in request.files:
        flash('No process file provided', 'error')
        return redirect(url_for('process.process_graphs'))

    file = request.files['process_file']

    # If the user does not select a file
    if file.filename == '':
        flash('No process file selected', 'error')
        return redirect(url_for('process.process_graphs'))

    # Check file type
    if not file.filename.endswith(('.ttl', '.owl', '.rdf', '.json')):
        flash('Invalid file format. Please upload .ttl, .owl, .rdf, or .json files only', 'error')
        return redirect(url_for('process.process_graphs'))

    # Get the processes directory
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    process_dir = os.path.join(base_dir, 'data', 'processes', 'definitions')
    os.makedirs(process_dir, exist_ok=True)

    # Save the file
    try:
        filepath = os.path.join(process_dir, file.filename)
        file.save(filepath)
        flash(f'Successfully uploaded {file.filename}', 'success')

        # Validate and process the file
        process_service.validate_process_file(filepath)

        # Generate visualization
        vis_dir = os.path.join(base_dir, 'app', 'static', 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)

        vis_filename = f'process_{file.filename.replace(".", "_")}.html'
        vis_path = os.path.join(vis_dir, vis_filename)

        process_service.visualize_process(filepath, vis_path)

    except Exception as e:
        flash(f'Error processing uploaded file: {str(e)}', 'error')

    return redirect(url_for('process.process_graphs'))


@process_bp.route('/delete_process', methods=['POST'])
def delete_process():
    """Delete a specific process graph file"""
    process_id = request.form.get('process_id')
    if not process_id:
        flash('No process ID provided', 'error')
        return redirect(url_for('process.process_graphs'))

    # Get the processes directory
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    process_dir = os.path.join(base_dir, 'data', 'processes', 'definitions')

    # Check if file exists
    filepath = os.path.join(process_dir, process_id)
    if not os.path.exists(filepath):
        flash(f'Process {process_id} not found', 'error')
        return redirect(url_for('process.process_graphs'))

    # Delete the file
    try:
        os.remove(filepath)
        flash(f'Successfully deleted {process_id}', 'success')

        # Also delete visualization if it exists
        vis_dir = os.path.join(base_dir, 'app', 'static', 'visualizations')
        vis_path = os.path.join(vis_dir, f'process_{process_id.replace(".", "_")}.html')
        if os.path.exists(vis_path):
            os.remove(vis_path)

    except Exception as e:
        flash(f'Error deleting process: {str(e)}', 'error')

    return redirect(url_for('process.process_graphs'))


@process_bp.route('/clear_processes', methods=['POST'])
def clear_processes():
    """Delete all process graph files"""
    # Get the processes directory
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    process_dir = os.path.join(base_dir, 'data', 'processes', 'definitions')

    # Check if directory exists
    if not os.path.exists(process_dir):
        flash('Process directory not found', 'error')
        return redirect(url_for('process.process_graphs'))

    # Delete all process files
    try:
        file_count = 0
        for filename in os.listdir(process_dir):
            if filename.endswith(('.ttl', '.owl', '.rdf', '.json')):
                os.remove(os.path.join(process_dir, filename))
                file_count += 1

                # Also delete visualization if it exists
                vis_dir = os.path.join(base_dir, 'app', 'static', 'visualizations')
                vis_path = os.path.join(vis_dir, f'process_{filename.replace(".", "_")}.html')
                if os.path.exists(vis_path):
                    os.remove(vis_path)

        if file_count > 0:
            flash(f'Successfully deleted {file_count} process files', 'success')
        else:
            flash('No process files found to delete', 'info')

    except Exception as e:
        flash(f'Error clearing processes: {str(e)}', 'error')

    return redirect(url_for('process.process_graphs'))


@process_bp.route('/reload_processes', methods=['GET'])
def reload_processes():
    """Reload all process graphs and regenerate visualizations"""
    # Get the processes directory
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    process_dir = os.path.join(base_dir, 'data', 'processes', 'definitions')

    # Ensure directory exists
    os.makedirs(process_dir, exist_ok=True)

    # Get process files
    process_files = [f for f in os.listdir(process_dir)
                     if f.endswith(('.ttl', '.owl', '.rdf', '.json'))]

    if not process_files:
        flash('No process files found to reload', 'info')
        return redirect(url_for('process.process_graphs'))

    # Reload processes and regenerate visualizations
    try:
        processed_count = 0
        for filename in process_files:
            filepath = os.path.join(process_dir, filename)

            try:
                # Validate the process file
                process_service.validate_process_file(filepath)

                # Regenerate visualization
                vis_dir = os.path.join(base_dir, 'app', 'static', 'visualizations')
                os.makedirs(vis_dir, exist_ok=True)

                vis_filename = f'process_{filename.replace(".", "_")}.html'
                vis_path = os.path.join(vis_dir, vis_filename)

                process_service.visualize_process(filepath, vis_path)

                processed_count += 1
            except Exception as e:
                flash(f'Error processing {filename}: {str(e)}', 'error')

        flash(f'Successfully reloaded {processed_count} process files', 'success')

    except Exception as e:
        flash(f'Error reloading processes: {str(e)}', 'error')

    return redirect(url_for('process.process_graphs'))


@process_bp.route('/view_process_file/<filename>', methods=['GET'])
def view_process_file(filename):
    """View the content of a process graph file"""
    # Get the processes directory
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    process_dir = os.path.join(base_dir, 'data', 'processes', 'definitions')

    # Check if file exists
    filepath = os.path.join(process_dir, filename)
    if not os.path.exists(filepath):
        flash(f'File {filename} not found', 'error')
        return redirect(url_for('process.process_graphs'))

    # Read the file content
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        # Determine the content type for proper formatting
        if filename.endswith('.ttl'):
            language = 'turtle'
        elif filename.endswith('.owl') or filename.endswith('.rdf'):
            language = 'xml'
        elif filename.endswith('.json'):
            language = 'json'
        else:
            language = 'text'

        return render_template('process/view_process_file.html',
                               title=f"Viewing {filename}",
                               filename=filename,
                               content=content,
                               language=language)

    except Exception as e:
        flash(f'Error reading file: {str(e)}', 'error')
        return redirect(url_for('process.process_graphs'))