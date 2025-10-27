from flask import Blueprint, render_template, request, jsonify, abort, redirect, url_for, flash, json
from backend.services.reasoning_service import ReasoningService
from backend.services.instance_graph_service import InstanceGraphService
from backend.services.loaded_ontology_service import LoadedOntologyService
from backend.services.process_graph_service import ProcessGraphService
from backend.services.inferred_ontology_service import OntologyService

import os

reasoning_bp = Blueprint('reasoning', __name__, url_prefix='/reasoning')

# Initialize services
reasoning_service = ReasoningService()
instance_service = InstanceGraphService()
ontology_service = LoadedOntologyService()
process_service = ProcessGraphService()
inferred_ontology_service = OntologyService()


@reasoning_bp.route('/', methods=['GET'])
def reasoning_dashboard():
    """Main reasoning dashboard"""
    # Get available instance graphs
    instance_graphs = instance_service.get_all_graphs()

    # Get available processes
    processes = process_service.get_all_processes()

    # Use the enhanced method detection from ReasoningService
    methods = reasoning_service.get_all_methods_with_applied_classes()

    return render_template('reasoning/reasoning_dashboard.html',
                           instance_graphs=instance_graphs,
                           processes=processes,
                           methods=methods)


@reasoning_bp.route('/analyze/<graph_id>', methods=['GET'])
def analyze_instance(graph_id):
    """Analyze an instance graph with ontology comparison and reasoning"""
    # Get the instance graph
    instance_graph = instance_service.get_graph_data(graph_id)

    if not instance_graph:
        flash("Graph not found", "error")
        return redirect(url_for('reasoning.reasoning_dashboard'))

    # Perform comprehensive analysis with reasoning
    analysis = reasoning_service.analyze_instance_with_reasoning(graph_id)

    # Generate visualizations
    instance_vis_path = instance_service.generate_pyvis_html(graph_id)

    # Generate ontology visualization if not already done
    ontology_data = inferred_ontology_service.get_ontology_details(graph_id)
    if not ontology_data or not ontology_data.get('visualization_path'):
        ontology_data = inferred_ontology_service.extract_ontology_from_graph(graph_id)

    ontology_vis_path = ontology_data.get('visualization_path') if ontology_data else None

    # Organize methods_application by class for better display
    # Start with your existing mapping
    methods_by_class = {}
    for method in analysis.get('applicable_methods', []):
        for applies_to in method.get('appliesTo', []):
            if applies_to not in methods_by_class:
                methods_by_class[applies_to] = []
            methods_by_class[applies_to].append(method)

    # Merge in fuzzy matches
    for loaded_class, inferred_list in analysis.get('fuzzy_matching_classes', []):
        for inferred_class, score, reason in inferred_list:
            # Copy methods from inferred class to loaded class if not already present
            if inferred_class in methods_by_class:
                methods_by_class.setdefault(loaded_class, []).extend(methods_by_class[inferred_class])

    print("methods_by_class:", methods_by_class)
    print(f'analysis {analysis}')

    return render_template('reasoning/analyze_results.html',
                           graph_id=graph_id,
                           instance_graph=instance_graph,
                           analysis=analysis,
                           methods_by_class=methods_by_class,
                           instance_vis_path=instance_vis_path,
                           ontology_vis_path=ontology_vis_path)


@reasoning_bp.route('/methods/<graph_id>', methods=['GET'])
def applicable_methods(graph_id):
    """Get methods_application applicable to a graph based on ontology comparison"""
    # Get the instance graph
    instance_graph = instance_service.get_graph_data(graph_id)

    if not instance_graph:
        flash("Graph not found", "error")
        return redirect(url_for('reasoning.reasoning_dashboard'))

    # Get ontology comparison
    comparison = reasoning_service.get_ontology_comparison(graph_id)

    # Extract all relevant classes (matching + fuzzy matching)
    relevant_classes = list(comparison.get('matching_classes', []))
    for loaded, inferred in comparison.get('fuzzy_matching_classes', []):
        if loaded not in relevant_classes:
            relevant_classes.append(loaded)

    # Get methods_application for these classes
    applicable_methods = reasoning_service.get_methods_for_classes(relevant_classes)

    # Organize methods_application by class for better display
    methods_by_class = {}
    for method in applicable_methods:
        for applies_to in method.get('appliesTo', []):
            if applies_to not in methods_by_class:
                methods_by_class[applies_to] = []
            methods_by_class[applies_to].append(method)

    return render_template('reasoning/applicable_methods.html',
                           graph_id=graph_id,
                           comparison=comparison,
                           methods=applicable_methods,
                           methods_by_class=methods_by_class,
                           relevant_classes=relevant_classes)


@reasoning_bp.route('/prepare_method/<graph_id>/<method_id>', methods=['GET'])
def prepare_method(graph_id, method_id):
    """Prepare to execute a method on a graph"""
    # Get the instance graph
    instance_graph = instance_service.get_graph_data(graph_id)

    if not instance_graph:
        flash("Graph not found", "error")
        return redirect(url_for('reasoning.reasoning_dashboard'))

    # Get all methods_application
    all_methods = reasoning_service.get_all_methods_with_applied_classes()
    method = next((m for m in all_methods if m.get('id') == method_id), None)

    if not method:
        flash(f"Method {method_id} not found", "error")
        return redirect(url_for('reasoning.analyze_instance', graph_id=graph_id))

    # Set default parameters based on method type
    default_parameters = "{}"

    if method_id == "reason:GenerateHistogramMethod":
        default_parameters = json.dumps({"property": "extension", "bin_count": 5}, indent=2)

    return render_template('reasoning/execute_method.html',
                           graph_id=graph_id,
                           method=method,
                           default_parameters=default_parameters,
                           execution_result=None)


@reasoning_bp.route('/execute_method', methods=['POST'])
def execute_method():
    """Execute a specific method on an instance graph"""
    graph_id = request.form.get('graph_id')
    method_id = request.form.get('method_id')

    # Get parameters from form
    parameters = {}
    parameters_str = request.form.get('parameters', '{}')
    if parameters_str and parameters_str.strip():
        try:
            parameters = json.loads(parameters_str)
        except json.JSONDecodeError:
            flash("Invalid JSON in parameters field", "error")
            return redirect(url_for('reasoning.prepare_method', graph_id=graph_id, method_id=method_id))

    # Execute the method
    result = reasoning_service.execute_method(graph_id, method_id, parameters)

    # Auto re-extract ontology if method was successful and modified the graph
    auto_refresh_ontology = request.form.get('auto_refresh_ontology', 'true').lower() == 'true'

    if result.get('success', False) and auto_refresh_ontology:
        try:
            flash("Reasoning applied successfully. Re-extracting ontology...", "info")
            updated_ontology = inferred_ontology_service.extract_ontology_from_graph(graph_id)
            flash("Ontology updated to reflect graph changes.", "success")
        except Exception as e:
            flash(f"Warning: Could not update ontology: {e}", "warning")

    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        # AJAX request, return JSON
        return jsonify(result)

    # Get method details for the result page
    all_methods = reasoning_service.get_all_methods_with_applied_classes()
    method = next((m for m in all_methods if m.get('id') == method_id), None)

    if not method:
        flash(f"Method {method_id} not found", "error")
        return redirect(url_for('reasoning.analyze_instance', graph_id=graph_id))

    # Regular form submission - show the execution result page
    return render_template('reasoning/execute_method.html',
                           graph_id=graph_id,
                           method=method,
                           default_parameters=parameters_str,
                           execution_result=result)


@reasoning_bp.route('/prepare_process/<graph_id>/<process_id>', methods=['GET'])
def prepare_process(graph_id, process_id):
    """Prepare to execute a process on a graph"""
    # Get the instance graph
    instance_graph = instance_service.get_graph_data(graph_id)

    if not instance_graph:
        flash("Graph not found", "error")
        return redirect(url_for('reasoning.reasoning_dashboard'))

    # Get process details
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    process_path = os.path.join(base_dir, 'data', 'processes', 'definitions', process_id)

    process = process_service.get_process_details(process_path)
    if not process:
        flash(f"Process {process_id} not found", "error")
        return redirect(url_for('reasoning.analyze_instance', graph_id=graph_id))

    return render_template('reasoning/execute_process.html',
                           graph_id=graph_id,
                           process=process,
                           execution_result=None)


@reasoning_bp.route('/execute_process', methods=['POST'])
def execute_process():
    """Execute a complete process on an instance graph"""
    graph_id = request.form.get('graph_id')
    process_id = request.form.get('process_id')

    # Execute the process
    result = reasoning_service.execute_process(graph_id, process_id)

    if request.headers.get('X-Requested-With') == 'XMLHttpRequest':
        # AJAX request, return JSON
        return jsonify(result)

    # Get process details for the result page
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    process_path = os.path.join(base_dir, 'data', 'processes', 'definitions', process_id)

    process = process_service.get_process_details(process_path)
    if not process:
        flash(f"Process {process_id} not found", "error")
        return redirect(url_for('reasoning.analyze_instance', graph_id=graph_id))

    # Regular form submission - show the execution result page
    return render_template('reasoning/execute_process.html',
                           graph_id=graph_id,
                           process=process,
                           execution_result=result)


@reasoning_bp.route('/processes/<graph_id>', methods=['GET'])
def matching_processes(graph_id):
    """Get processes applicable to a graph based on methods_application"""
    # Get the instance graph
    instance_graph = instance_service.get_graph_data(graph_id)

    if not instance_graph:
        flash("Graph not found", "error")
        return redirect(url_for('reasoning.reasoning_dashboard'))

    # Get ontology comparison
    comparison = reasoning_service.get_ontology_comparison(graph_id)

    # Extract all relevant classes (matching + fuzzy matching)
    relevant_classes = list(comparison.get('matching_classes', []))
    for loaded, inferred in comparison.get('fuzzy_matching_classes', []):
        if loaded not in relevant_classes:
            relevant_classes.append(loaded)

    # Get methods_application for these classes
    applicable_methods = reasoning_service.get_methods_for_classes(relevant_classes)

    # Find processes that use these methods_application
    matching_processes = reasoning_service.find_processes_with_methods(applicable_methods)

    return render_template('reasoning/matching_processes.html',
                           graph_id=graph_id,
                           processes=matching_processes,
                           comparison=comparison,
                           methods=applicable_methods)


@reasoning_bp.route('/ontology_comparison/<graph_id>', methods=['GET'])
def ontology_comparison(graph_id):
    """Show ontology comparison details for a graph"""
    # Get the instance graph
    instance_graph = instance_service.get_graph_data(graph_id)

    if not instance_graph:
        flash("Graph not found", "error")
        return redirect(url_for('reasoning.reasoning_dashboard'))

    # Get ontology comparison
    comparison = reasoning_service.get_ontology_comparison(graph_id)

    # Generate ontology visualization if not already done
    ontology_data = inferred_ontology_service.get_ontology_details(graph_id)
    if not ontology_data or not ontology_data.get('visualization_path'):
        ontology_data = inferred_ontology_service.extract_ontology_from_graph(graph_id)

    ontology_vis_path = ontology_data.get('visualization_path') if ontology_data else None

    return render_template('reasoning/ontology_comparison.html',
                           graph_id=graph_id,
                           comparison=comparison,
                           ontology_vis_path=ontology_vis_path)


@reasoning_bp.route('/api/methods/<graph_id>', methods=['GET'])
def api_applicable_methods(graph_id):
    """API endpoint to get methods_application applicable to a graph"""
    # Get ontology comparison
    comparison = reasoning_service.get_ontology_comparison(graph_id)

    # Extract all relevant classes (matching + fuzzy matching)
    relevant_classes = list(comparison.get('matching_classes', []))
    for loaded, inferred in comparison.get('fuzzy_matching_classes', []):
        if loaded not in relevant_classes:
            relevant_classes.append(loaded)

    # Get methods_application for these classes
    applicable_methods = reasoning_service.get_methods_for_classes(relevant_classes)

    return jsonify({
        'methods_application': applicable_methods,
        'relevantClasses': relevant_classes
    })


@reasoning_bp.route('/api/processes/<graph_id>', methods=['GET'])
def api_matching_processes(graph_id):
    """API endpoint to get processes applicable to a graph"""
    # Get ontology comparison
    comparison = reasoning_service.get_ontology_comparison(graph_id)

    # Extract all relevant classes (matching + fuzzy matching)
    relevant_classes = list(comparison.get('matching_classes', []))
    for loaded, inferred in comparison.get('fuzzy_matching_classes', []):
        if loaded not in relevant_classes:
            relevant_classes.append(loaded)

    # Get methods_application for these classes
    applicable_methods = reasoning_service.get_methods_for_classes(relevant_classes)

    # Find processes that use these methods_application
    matching_processes = reasoning_service.find_processes_with_methods(applicable_methods)

    return jsonify({
        'processes': matching_processes
    })


@reasoning_bp.route('/debug/ontology_triples', methods=['GET'])
def debug_ontology_triples():
    """Debug endpoint to show all loaded ontology triples"""
    triples = reasoning_service.debug_ontology_triples()
    return render_template('reasoning/debug_triples.html',
                           triples=triples)