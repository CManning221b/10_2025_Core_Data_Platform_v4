from flask import Blueprint, render_template, request, flash, redirect, url_for, current_app, jsonify
import os
import json
import pandas as pd
from werkzeug.utils import secure_filename
from backend.services.upload_service import UploadService
from backend.Utils.file_utils import allowed_file

# Create blueprint
upload_bp = Blueprint('upload', __name__, url_prefix='/upload')
upload_service = UploadService()


@upload_bp.route('/', methods=['GET'])
def upload_page():
    return render_template('upload/upload.html')


@upload_bp.route('/process_file_upload', methods=['POST'])
def process_file_upload():
    # Check if file is present in request
    if 'file' not in request.files:
        flash('No file part', 'error')
        return redirect(url_for('upload.upload_page'))

    file = request.files['file']

    # Check if file is selected
    if file.filename == '':
        flash('No file selected', 'error')
        return redirect(url_for('upload.upload_page'))

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)

        # Ensure upload directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Save file
        file.save(file_path)

        # Process file
        try:
            result = upload_service.process_file(file_path)

            return render_template('upload/upload_status.html',
                                   success=True,
                                   status_title="File Processed Successfully",
                                   status_message=f"File {filename} was uploaded and processed successfully.",
                                   graph_stats=result)

            # Change all similar redirects to use 'instance.view_graph' instead of 'visualization.view'

        except Exception as e:
            return render_template('upload/upload_status.html',
                                   success=False,
                                   status_title="Processing Error",
                                   status_message=f"Error processing file: {str(e)}",
                                   errors=[str(e)])
    else:
        return render_template('upload/upload_status.html',
                               success=False,
                               status_title="Invalid File Type",
                               status_message="The selected file type is not supported.",
                               errors=[f"Supported file types are: {', '.join(allowed_file(''))}"])


@upload_bp.route('/process_dataframe_upload', methods=['POST'])
def process_dataframe_upload():
    # Check if both files are present in request
    if 'dataframe_file' not in request.files or 'metadata_file' not in request.files:
        return render_template('upload/upload_status.html',
                               success=False,
                               status_title="Missing Files",
                               status_message="Both dataframe and metadata files are required.",
                               errors=["Please upload both a dataframe CSV and metadata CSV file."])

    dataframe_file = request.files['dataframe_file']
    metadata_file = request.files['metadata_file']

    # Check if files are selected
    if dataframe_file.filename == '' or metadata_file.filename == '':
        return render_template('upload/upload_status.html',
                               success=False,
                               status_title="Missing Files",
                               status_message="Both dataframe and metadata files are required.",
                               errors=["Please upload both a dataframe CSV and metadata CSV file."])

    # Check file types
    if not (dataframe_file.filename.endswith('.csv') and metadata_file.filename.endswith('.csv')):
        return render_template('upload/upload_status.html',
                               success=False,
                               status_title="Invalid File Type",
                               status_message="Both files must be CSV format.",
                               errors=["Both dataframe and metadata files must be in CSV format."])

    try:
        # Save the files temporarily
        dataframe_filename = secure_filename(dataframe_file.filename)
        metadata_filename = secure_filename(metadata_file.filename)

        dataframe_path = os.path.join(current_app.config['UPLOAD_FOLDER'], dataframe_filename)
        metadata_path = os.path.join(current_app.config['UPLOAD_FOLDER'], metadata_filename)

        # Ensure upload directory exists
        os.makedirs(os.path.dirname(dataframe_path), exist_ok=True)

        # Save files
        dataframe_file.save(dataframe_path)
        metadata_file.save(metadata_path)

        # Read the files
        df = pd.read_csv(dataframe_path)
        metadata_df = pd.read_csv(metadata_path)

        # Validate metadata format
        required_columns = {'column', 'node_type', 'edge_type', 'direction', 'hierarchy_level', 'connects_to'}
        if not all(col in metadata_df.columns for col in required_columns):
            return render_template('upload/upload_status.html',
                                   success=False,
                                   status_title="Invalid Metadata Format",
                                   status_message="Metadata CSV is missing required columns.",
                                   errors=[f"Metadata CSV must contain these columns: {', '.join(required_columns)}"])

        # Process dataframe with metadata
        result = upload_service.process_dataframe(df, metadata_df)

        return render_template('upload/upload_status.html',
                               success=True,
                               status_title="DataFrame Processed Successfully",
                               status_message=f"DataFrame was uploaded and processed successfully.",
                               graph_stats=result,
                               details={
                                   "DataFrame Size": f"{df.shape[0]} rows × {df.shape[1]} columns",
                                   "Metadata Rules": f"{len(metadata_df)} column mappings"
                               })

    except ValueError as e:
        return render_template('upload/upload_status.html',
                               success=False,
                               status_title="Validation Error",
                               status_message="Error in dataframe or metadata format.",
                               errors=[str(e)])
    except Exception as e:
        return render_template('upload/upload_status.html',
                               success=False,
                               status_title="Processing Error",
                               status_message=f"Error processing dataframe: {str(e)}",
                               errors=[str(e)])


@upload_bp.route('/process_paths_upload', methods=['POST'])
def process_paths_upload():
    # Handle path list from textarea
    path_list = []
    if 'paths' in request.form and request.form['paths'].strip():
        path_list = [path.strip() for path in request.form['paths'].split('\n') if path.strip()]

    # Handle paths from uploaded file
    if 'paths_file' in request.files and request.files['paths_file'].filename != '':
        paths_file = request.files['paths_file']
        try:
            # Read paths from file
            path_content = paths_file.read().decode('utf-8')
            file_paths = [path.strip() for path in path_content.split('\n') if path.strip()]
            path_list.extend(file_paths)
        except Exception as e:
            return render_template('upload/upload_status.html',
                                   success=False,
                                   status_title="Path File Error",
                                   status_message=f"Error reading paths file: {str(e)}",
                                   errors=[str(e)])

    # Check if we have any paths
    if not path_list:
        return render_template('upload/upload_status.html',
                               success=False,
                               status_title="No Paths Provided",
                               status_message="No valid paths were provided for processing.",
                               errors=["Please enter at least one valid file path."])

    # Process the paths
    try:
        result = upload_service.process_paths(path_list)
        return render_template('upload/upload_status.html',
                               success=True,
                               status_title="Paths Processed Successfully",
                               status_message=f"Successfully processed {len(path_list)} paths.",
                               graph_stats=result,
                               details={"Paths Processed": len(path_list)})
    except Exception as e:
        return render_template('upload/upload_status.html',
                               success=False,
                               status_title="Processing Error",
                               status_message=f"Error processing paths: {str(e)}",
                               errors=[str(e)])


@upload_bp.route('/process_directory', methods=['POST'])
def process_directory():
    # Get directory path from form
    directory_path = request.form.get('directory_path', '').strip()

    # Check if directory path is provided
    if not directory_path:
        return render_template('upload/upload_status.html',
                               success=False,
                               status_title="No Directory Specified",
                               status_message="No directory path was provided for processing.",
                               errors=["Please enter a valid directory path."])

    # Check if directory exists
    if not os.path.isdir(directory_path):
        return render_template('upload/upload_status.html',
                               success=False,
                               status_title="Invalid Directory",
                               status_message=f"The directory '{directory_path}' does not exist or is not accessible.",
                               errors=[f"Directory not found: {directory_path}"])

    # Process the directory
    try:
        result = upload_service.process_directory(directory_path)
        return render_template('upload/upload_status.html',
                               success=True,
                               status_title="Directory Processed Successfully",
                               status_message=f"Successfully processed directory: {directory_path}",
                               graph_stats=result,
                               details={"Directory": directory_path})
    except Exception as e:
        return render_template('upload/upload_status.html',
                               success=False,
                               status_title="Processing Error",
                               status_message=f"Error processing directory: {str(e)}",
                               errors=[str(e)])


@upload_bp.route('/process_json_input', methods=['POST'])
def process_json_input():
    # Get JSON data from form
    json_data = request.form.get('json_data', '').strip()

    # Check if JSON data is provided
    if not json_data:
        return render_template('upload/upload_status.html',
                               success=False,
                               status_title="No JSON Data",
                               status_message="No JSON data was provided for processing.",
                               errors=["Please enter valid JSON data."])

    # Parse JSON data
    try:
        parsed_json = json.loads(json_data)
    except json.JSONDecodeError as e:
        return render_template('upload/upload_status.html',
                               success=False,
                               status_title="Invalid JSON",
                               status_message="The provided JSON data is not valid.",
                               errors=[f"JSON parsing error: {str(e)}"])

    # Process JSON data
    try:
        result = upload_service.process_json(parsed_json)
        return render_template('upload/upload_status.html',
                               success=True,
                               status_title="JSON Processed Successfully",
                               status_message="Successfully processed JSON data.",
                               graph_stats=result)
    except Exception as e:
        return render_template('upload/upload_status.html',
                               success=False,
                               status_title="Processing Error",
                               status_message=f"Error processing JSON data: {str(e)}",
                               errors=[str(e)])