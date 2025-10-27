import os
import importlib.util
import inspect
from backend.core_graph_managers.no2_graphManager.graphManager import GraphManager
from backend.services.instance_graph_service import InstanceGraphService


class MethodExecutor:
    """Framework for executing methods_application on instance graphs"""

    def __init__(self, instance_service=None):
        self.instance_service = instance_service or InstanceGraphService()
        self.method_implementations = {}
        self._load_method_implementations()

    def _load_method_implementations(self):
        """Load all method implementations from the methods_application directory, including subdirectories."""
        methods_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                   'methods_application/methods')

        if not os.path.exists(methods_dir):
            print(f"Methods directory not found: {methods_dir}")
            return

        # Walk through the directory and its subdirectories
        for root, _, files in os.walk(methods_dir):
            for filename in files:
                if filename.endswith('.py') and not filename.startswith('__'):
                    try:
                        module_path = os.path.join(root, filename)
                        relative_path = os.path.relpath(module_path, methods_dir)
                        module_name = os.path.splitext(relative_path.replace(os.sep, "."))[0]

                        spec = importlib.util.spec_from_file_location(module_name, module_path)
                        module = importlib.util.module_from_spec(spec)
                        spec.loader.exec_module(module)

                        for name, obj in inspect.getmembers(module):
                            if (inspect.isclass(obj) and
                                    hasattr(obj, 'method_id') and
                                    hasattr(obj, 'execute') and
                                    obj.method_id is not None and
                                    obj.__name__ != 'MethodImplementation'):
                                self.method_implementations[obj.method_id] = obj
                                print(f"Loaded method implementation: {obj.method_id}")

                    except Exception as e:
                        print(f"Error loading method module {filename}: {e}")

    def execute_method(self, graph_id, method_id, parameters=None):
        """
        Execute a method on an instance graph

        Args:
            graph_id (str): ID of the graph to modify
            method_id (str): ID of the method to execute
            parameters (dict): Optional parameters for method execution

        Returns:
            dict: Result of the execution
        """
        # Get the graph path directly
        graphs_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'data',
                                  'graphs')
        graph_path = os.path.join(graphs_dir, f"{graph_id}.json")

        if not os.path.exists(graph_path):
            return {
                "success": False,
                "error": f"Graph {graph_id} not found"
            }

        # ✅ FIXED: Strip namespace prefix from method_id
        short_method_id = method_id.split(':')[-1] if ':' in method_id else method_id
        print(f"DEBUG: Looking for method implementation: '{method_id}' -> '{short_method_id}'")

        # Check if we have an implementation for this method (using short name)
        if short_method_id not in self.method_implementations:
            print(f"DEBUG: Available implementations: {list(self.method_implementations.keys())}")
            return {
                "success": False,
                "error": f"No implementation found for method {short_method_id} (from {method_id})"
            }

        # Get the method implementation
        method_class = self.method_implementations[short_method_id]
        method_instance = method_class()

        try:
            # Create a GraphManager instance and load the graph
            graph_manager = GraphManager()
            graph_manager.load_from_json(graph_path)

            print(f"DEBUG: Executing method {short_method_id} on graph {graph_id}")

            # Execute the method
            result = method_instance.execute(graph_manager, parameters or {})

            # Save the updated graph using GraphManager's native method
            graph_manager.save_to_json(return_dict=False, filepath=graph_path)

            print(f"DEBUG: Method {short_method_id} executed successfully")

            # Return success
            return {
                "success": True,
                "method_id": method_id,
                "graph_id": graph_id,
                "changes": result,
                "message": f"Method {short_method_id} executed successfully"
            }

        except Exception as e:
            print(f"DEBUG: Error executing method {short_method_id}: {str(e)}")
            return {
                "success": False,
                "error": f"Error executing method {short_method_id}: {str(e)}"
            }

    def execute_process(self, graph_id, process, parameters=None):
        """Execute a complete process on an instance graph"""
        print(f"DEBUG: Starting process execution for {process.get('id')}")

        if not process or 'nodes' not in process:
            return {"success": False, "error": "Invalid process definition"}

        # ✅ NEW: If process ID contains #, extract the specific process to run
        target_process_id = None
        if '#' in process.get('id', ''):
            _, target_process_id = process.get('id').split('#', 1)
            print(f"DEBUG: Target process ID: {target_process_id}")

        # ✅ IMPROVED: Find the correct start node for the specific process
        start_node = None

        if target_process_id:
            # Find the specific process node and its start step
            process_node = process.get('nodes', {}).get(target_process_id)
            if process_node and process_node.get('type') == 'Process':
                print(f"DEBUG: Found target process node: {target_process_id}")

                # Find the start step connected to this specific process
                for edge_id, edge in process.get('edges', {}).items():
                    edge_source = edge.get('source') or edge.get('from')
                    edge_target = edge.get('target') or edge.get('to')

                    if edge_source == target_process_id:
                        target_node = process['nodes'].get(edge_target)
                        if target_node and target_node.get('type') in ['StartStep', 'Start']:
                            start_node = edge_target
                            print(f"DEBUG: Found specific start node: {start_node}")
                            break

        # Fallback: find any start node if process-specific search failed
        if not start_node:
            for node_id, node in process.get('nodes', {}).items():
                if node.get('type') in ['StartStep', 'Start']:
                    start_node = node_id
                    print(f"DEBUG: Using fallback start node: {start_node}")
                    break

        if not start_node:
            return {"success": False, "error": "Process has no start node"}

        # ✅ FILTER: Only include nodes that are part of the target process
        if target_process_id:
            relevant_nodes = self._get_process_related_nodes(process, target_process_id)
            print(f"DEBUG: Process {target_process_id} has {len(relevant_nodes)} related nodes: {relevant_nodes}")
        else:
            relevant_nodes = set(process.get('nodes', {}).keys())

        # Execute steps in sequence
        current_node = start_node
        step_results = []
        max_steps = 20
        step_count = 0

        while current_node and step_count < max_steps:
            step_count += 1
            print(f"DEBUG: Step {step_count} - Processing node {current_node}")

            node = process['nodes'].get(current_node)
            if not node:
                return {"success": False, "error": f"Node {current_node} not found"}

            # ✅ CHECK: Only process nodes that belong to our target process
            if target_process_id and current_node not in relevant_nodes:
                print(f"DEBUG: Skipping node {current_node} - not part of target process")
                break

            if node.get('type') == 'MethodStep':
                method_id = node.get('methodId')
                if not method_id and 'attributes' in node and 'implementsMethod' in node['attributes']:
                    method_id = node['attributes']['implementsMethod']

                if method_id:
                    print(f"DEBUG: Executing method {method_id}")
                    result = self.execute_method(graph_id, method_id, parameters)
                    step_results.append({
                        "node": current_node,
                        "method_id": method_id,
                        "success": result.get('success', False),
                        "result": result
                    })

                    # Update ontology after each successful method execution
                    if result.get('success', False):
                        try:
                            from backend.services.inferred_ontology_service import OntologyService
                            ontology_service = OntologyService()
                            ontology_service.extract_ontology_from_graph(graph_id)
                            print(f"DEBUG: Ontology updated after {method_id}")
                        except Exception as e:
                            print(f"DEBUG: Warning - Could not update ontology: {e}")

            elif node.get('type') == 'SubprocessStep':
                print(f"DEBUG: SubprocessStep found but not implemented yet: {current_node}")
                step_results.append({
                    "node": current_node,
                    "method_id": "SubprocessStep",
                    "success": True,
                    "result": {"message": "Subprocess step skipped"}
                })

            # Find next node
            next_node = None
            for edge_id, edge in process.get('edges', {}).items():
                edge_source = edge.get('source') or edge.get('from')
                edge_target = edge.get('target') or edge.get('to')

                if edge_source == current_node:
                    # ✅ CHECK: Only follow edges to nodes in our target process
                    if not target_process_id or edge_target in relevant_nodes:
                        next_node = edge_target
                        print(f"DEBUG: Next node: {next_node}")
                        break

            if not next_node or node.get('type') in ['EndStep', 'End']:
                print(f"DEBUG: Reached end - Next: {next_node}, Type: {node.get('type')}")
                break

            current_node = next_node

        return {
            "success": True,
            "graph_id": graph_id,
            "process_id": process.get('id'),
            "steps_executed": len(step_results),
            "step_results": step_results
        }

    def _get_process_related_nodes(self, process, target_process_id):
        """Get all nodes that are part of a specific process"""
        related_nodes = set()

        # Add the process node itself
        related_nodes.add(target_process_id)

        # Find all steps listed in the process's hasStep
        process_node = process.get('nodes', {}).get(target_process_id)
        if process_node:
            # Get steps from edges starting from this process
            for edge_id, edge in process.get('edges', {}).items():
                edge_source = edge.get('source') or edge.get('from')
                edge_target = edge.get('target') or edge.get('to')

                if edge_source == target_process_id:
                    related_nodes.add(edge_target)
                    # Follow the chain of nextStep relationships
                    self._follow_step_chain(process, edge_target, related_nodes)

        return related_nodes

    def _follow_step_chain(self, process, start_node, related_nodes):
        """Follow nextStep relationships to find all connected steps"""
        current = start_node
        visited = set()

        while current and current not in visited:
            visited.add(current)
            related_nodes.add(current)

            # Find next step
            next_node = None
            for edge_id, edge in process.get('edges', {}).items():
                edge_source = edge.get('source') or edge.get('from')
                edge_target = edge.get('target') or edge.get('to')

                if edge_source == current and edge.get('label') == 'nextStep':
                    next_node = edge_target
                    break

            current = next_node