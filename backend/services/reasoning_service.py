import os
import json
from backend.core_graph_managers.no2_graphManager.graphManager import GraphManager
from backend.core_graph_managers.no3_OntologyManager.OntologyManager import OntologyManager
from backend.services.instance_graph_service import InstanceGraphService
from backend.services.loaded_ontology_service import LoadedOntologyService
from backend.services.process_graph_service import ProcessGraphService
from backend.services.inferred_ontology_service import OntologyService
from backend.methods_application.method_executor import MethodExecutor


class ReasoningService:
    def __init__(self):
        self.instance_service = InstanceGraphService()
        self.loaded_ontology_service = LoadedOntologyService()
        self.process_service = ProcessGraphService()
        self.inferred_ontology_service = OntologyService()
        self.method_executor = MethodExecutor(self.instance_service)
        # Make sure ontology is loaded when service is initialized
        self.ensure_ontology_loaded()

    def get_inferred_types(self, graph_id):
        """Get inferred types for nodes in the instance graph"""
        # Get the instance graph
        instance_graph = self.instance_service.get_graph_data(graph_id)

        if not instance_graph:
            return {}  # Return empty dict if graph not found

        # Extract ontology from graph
        ontology_data = self.inferred_ontology_service.extract_ontology_from_graph(graph_id)

        if not ontology_data:
            return {}  # Return empty dict if ontology extraction fails

        # Get inferred types from the ontology
        inferred_types = {}

        # Map nodes to their types
        nodes = instance_graph.get('nodes', {})
        for node_id, node_data in nodes.items():
            # Get node type from instance graph
            node_type = node_data.get('type', 'Unknown')
            inferred_types[node_id] = {
                'type': node_type,
                'confidence': 1.0  # Set confidence to 1.0 for directly assigned types
            }

        return inferred_types

    def ensure_ontology_loaded(self):
        """
        Ensure an OWL ontology is loaded before performing comparison

        Returns:
            bool: True if ontology is loaded, False otherwise
        """
        # Check if any ontology is already loaded
        if hasattr(self.loaded_ontology_service, 'ontology_manager') and self.loaded_ontology_service.ontology_manager:
            return True

        # Try to load default ontologies
        try:
            # Get the base project directory
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

            # Construct the path to ontologies directory
            ontology_dir = os.path.join(base_dir, 'data', 'ontologies', 'predefined')

            # Check if directory exists
            if not os.path.exists(ontology_dir):
                return False

            # Look for ontology files
            ontology_files = [os.path.join(ontology_dir, f) for f in os.listdir(ontology_dir)
                              if f.endswith(('.ttl', '.owl', '.rdf'))]

            if not ontology_files:
                return False

            # Load ontologies
            self.loaded_ontology_service.load_ontologies(ontology_files)
            return True

        except Exception as e:
            print(f"Error loading ontologies: {str(e)}")
            return False

    def debug_ontology_triples(self):
        """Debug helper to show all loaded ontology triples"""
        if not self.ensure_ontology_loaded():
            return "No ontology loaded"

        if not hasattr(self.loaded_ontology_service,
                       'ontology_bridge') or self.loaded_ontology_service.ontology_bridge is None:
            return "No ontology bridge available"

        bridge = self.loaded_ontology_service.ontology_bridge
        all_properties = bridge.get_all_properties()

        # Format as nice output
        output = []
        method_related = []

        for src, pred, obj in all_properties:
            triple = f"{src} {pred} {obj}"
            output.append(triple)

            # Also create a list of method-related triples for easier review
            if ("Method" in src or "Method" in obj or
                    "method" in pred or "Method" in pred or
                    "appliesTo" in pred):
                method_related.append(triple)

        return {
            "all_triples": output,
            "method_related": method_related,
            "total_triples": len(output),
            "method_related_count": len(method_related)
        }

    def get_ontology_comparison(self, graph_id):
        """
        Compare inferred ontology with loaded OWL ontology

        Args:
            graph_id (str): ID of the instance graph

        Returns:
            dict: Comparison results with matching classes, methods_application, etc.
        """
        # Get inferred ontology from the graph
        inferred_ontology = self.inferred_ontology_service.get_current_ontology(graph_id)

        if not inferred_ontology:
            return {
                'matching_classes': [],
                'fuzzy_matching_classes': [],
                'only_in_loaded': [],
                'only_in_inferred': [],
                'methods_application': [],
                'error': "Could not extract inferred ontology from graph"
            }

        # Ensure an OWL ontology is loaded
        if not self.ensure_ontology_loaded():
            return {
                'matching_classes': [],
                'fuzzy_matching_classes': [],
                'only_in_loaded': [],
                'only_in_inferred': [],
                'methods_application': [],
                'error': "No OWL ontology loaded. Please upload an ontology file first."
            }

        # Compare with loaded OWL ontology
        try:
            comparison = self.loaded_ontology_service.compare_with_inferred(inferred_ontology)

            # Enhance the comparison with methods_application - use our custom method fetching
            methods = self.get_all_methods_with_applied_classes()
            if methods:
                # Store method information in comparison
                comparison['methods_application'] = methods

            return comparison
        except ValueError as e:
            # Handle the specific error when no ontology is loaded
            return {
                'matching_classes': [],
                'fuzzy_matching_classes': [],
                'only_in_loaded': [],
                'only_in_inferred': [],
                'methods_application': [],
                'error': str(e)
            }
        except Exception as e:
            # Handle any other exceptions
            return {
                'matching_classes': [],
                'fuzzy_matching_classes': [],
                'only_in_loaded': [],
                'only_in_inferred': [],
                'methods_application': [],
                'error': f"Error comparing ontologies: {str(e)}"
            }

    def get_all_methods_with_applied_classes(self):
        """
        Enhanced method to extract all method definitions and their applicable classes from the ontology

        Returns:
            list: A list of method dictionaries with id, name, description, and appliesTo fields
        """
        # Ensure we have an ontology loaded
        if not self.ensure_ontology_loaded():
            return []

        methods = []

        # Use Bridge to get triples directly
        if not hasattr(self.loaded_ontology_service,
                       'ontology_bridge') or self.loaded_ontology_service.ontology_bridge is None:
            print("No ontology bridge available")
            return []

        bridge = self.loaded_ontology_service.ontology_bridge

        # 1. Find all classes that might be methods
        all_classes = bridge.get_all_classes()
        method_classes = []

        print(f"Total classes in ontology: {len(all_classes)}")

        # Filter for potential method classes
        for cls in all_classes:
            # Check if the class name contains "Method"
            if "Method" in cls:
                method_classes.append(cls)
                print(f"Found method class by name: {cls}")

            # Check annotations for method indicators
            annotations = bridge.get_annotations_for(cls)
            if "isMethod" in annotations or "appliesTo" in annotations:
                if cls not in method_classes:
                    method_classes.append(cls)
                    print(f"Found method class by annotation: {cls}")

        print(f"Total method classes found: {len(method_classes)}")

        # 2. Get all properties and find appliesTo-related predicates dynamically
        all_properties = bridge.get_all_properties()

        # Find all predicates that are subproperties of appliesTo or contain appliesTo
        applies_to_predicates = set()

        for src, pred, obj in all_properties:
            # Look for rdfs:subPropertyOf core:appliesTo relationships
            if "subPropertyOf" in pred and ("appliesTo" in obj or "core:appliesTo" in obj):
                applies_to_predicates.add(src)
                print(f"Found appliesTo subproperty: {src}")

            # Also include any predicate that contains "appliesTo" in its name
            if "appliesTo" in pred:
                applies_to_predicates.add(pred)
                print(f"Found appliesTo-related predicate: {pred}")

        # Add base appliesTo predicates
        base_applies_to_predicates = ["appliesTo", "core:appliesTo", "applies_to"]
        applies_to_predicates.update(base_applies_to_predicates)

        print(f"Total appliesTo-related predicates found: {applies_to_predicates}")

        # Also include other method-related predicates for completeness
        method_predicates = applies_to_predicates.union({
            "implementsMethod", "isMethod", "isFileExtensionMethod",
            "isHistogramMethod", "isFolderStatsMethod"
        })

        # Define meta-classes to filter out
        meta_classes = ["core:Method", "Method", "owl:Class", "rdfs:Class"]

        # Dictionary to store methods and what they apply to
        method_applications = {}

        # 3. For each potential method class, find what it applies to
        for method_class in method_classes:
            # Initialize if not exists
            if method_class not in method_applications:
                method_applications[method_class] = []

            # Check direct triples in both directions
            for src, pred, obj in all_properties:
                # Check if this triple relates the method to classes using any appliesTo-related predicate
                if any(method_pred in pred for method_pred in method_predicates):
                    # Case 1: Method is subject, Class is object
                    if (src == method_class and
                            obj not in method_applications[method_class] and
                            obj not in meta_classes and  # Filter out meta-classes
                            not obj.startswith('core:') and  # Filter out core namespace classes
                            not obj.startswith('xsd:')):  # Filter out XSD types
                        method_applications[method_class].append(obj)
                        print(f"Method {method_class} applies to {obj} via {pred} (method is subject)")

                    # Case 2: Class is subject, Method is object
                    elif (obj == method_class and
                          src not in method_applications[method_class] and
                          src not in meta_classes and  # Filter out meta-classes
                          not src.startswith('core:') and  # Filter out core namespace classes
                          not src.startswith('xsd:') and  # Filter out XSD types
                          not any(meta in src for meta in ['Method', 'Class'])):  # Filter out method/class names
                        method_applications[method_class].append(src)
                        print(f"Method {method_class} applies to {src} via {pred} (method is object)")

        # 4. Check for core:Method relationships
        for src, pred, obj in all_properties:
            # Look for core:Method isXMethod relationships (e.g., isFileExtensionMethod)
            if "core:Method" in src and "is" in pred and "Method" in pred:
                method_class = obj
                if method_class not in method_applications:
                    method_applications[method_class] = []
                print(f"Found method via core:Method relationship: {method_class}")

                # Now find what this method applies to using any appliesTo-related predicate
                for src2, pred2, obj2 in all_properties:
                    if (src2 == method_class and
                            any(applies_pred in pred2 for applies_pred in applies_to_predicates) and
                            obj2 not in method_applications[method_class] and
                            obj2 not in meta_classes and  # Filter out meta-classes
                            not obj2.startswith('core:') and  # Filter out core namespace
                            not obj2.startswith('xsd:')):  # Filter out XSD types
                        method_applications[method_class].append(obj2)
                        print(f"Method {method_class} applies to {obj2} via {pred2}")

        # 5. Build method objects with all the information
        for method_class, applies_to_classes in method_applications.items():
            # Get annotations for the method
            annotations = bridge.get_annotations_for(method_class)

            # Create method object
            method = {
                'id': method_class,
                'name': annotations.get('label', method_class.split(':')[-1]),
                'description': annotations.get('comment', ''),
                'appliesTo': applies_to_classes,
                'implementable': method_class in self.method_executor.method_implementations
            }

            methods.append(method)

        # Also check the process ontology for method definitions
        self._enhance_methods_from_process_ontology(methods)

        print(f"Total methods found with relationships: {len(methods)}")
        for method in methods:
            print(
                f"Method: {method['name']} - Applies to: {method['appliesTo']} - Implementable: {method['implementable']}")

        return methods

    def _enhance_methods_from_process_ontology(self, methods):
        """
        Enhance methods_application with information from the process ontology

        Args:
            methods (list): Existing list of methods_application to be enhanced
        """
        if not hasattr(self.loaded_ontology_service,
                       'ontology_bridge') or self.loaded_ontology_service.ontology_bridge is None:
            return

        bridge = self.loaded_ontology_service.ontology_bridge
        all_properties = bridge.get_all_properties()

        # Find all implementsMethod relationships from process ontology
        method_implementations = {}

        for src, pred, obj in all_properties:
            if "implementsMethod" in pred:
                if obj not in method_implementations:
                    method_implementations[obj] = []
                # Store the method step that implements this method
                method_implementations[obj].append(src)
                print(f"Process step {src} implements method {obj}")

        # Add implementation info to existing methods_application
        for method in methods:
            method_id = method['id']
            if method_id in method_implementations:
                method['implementedBy'] = method_implementations[method_id]

        # Look for method steps that aren't already in our methods_application list
        for method_id, step_ids in method_implementations.items():
            # Check if this method is already in our list
            existing_method = next((m for m in methods if m['id'] == method_id), None)

            if existing_method is None:
                # Create a new method entry
                annotations = bridge.get_annotations_for(method_id)

                new_method = {
                    'id': method_id,
                    'name': annotations.get('label', method_id.split(':')[-1]),
                    'description': annotations.get('comment', ''),
                    'appliesTo': [],  # We don't know what it applies to from process ontology
                    'implementedBy': step_ids,
                    'implementable': method_id in self.method_executor.method_implementations
                }

                # Try to infer what this method applies to from the process steps
                for step_id in step_ids:
                    step_annotations = bridge.get_annotations_for(step_id)
                    step_label = step_annotations.get('label', '').lower()
                    step_comment = step_annotations.get('comment', '').lower()

                    # Look for hints in label and comment
                    keywords = {
                        "file:File": ["file", "files"],
                        "file:Folder": ["folder", "directory", "directories"],
                        "file:FileExtension": ["extension", "file extension"],
                        "file:Filename": ["filename", "file name"],
                        "file:FileMetadata": ["metadata", "file metadata"]
                    }

                    for class_name, class_keywords in keywords.items():
                        if any(keyword in step_label or keyword in step_comment for keyword in class_keywords):
                            if class_name not in new_method['appliesTo']:
                                new_method['appliesTo'].append(class_name)
                                print(f"Inferred that method {method_id} applies to {class_name} from step description")

                methods.append(new_method)
                print(f"Added method from process ontology: {new_method['name']}")

    def get_methods_for_classes(self, class_names):
        """
        Get methods_application that can be applied to the given class names

        Args:
            class_names (list): List of class names to find methods_application for

        Returns:
            list: Methods applicable to the given classes
        """
        # Get all methods_application including their appliesTo relationships
        all_methods = self.get_all_methods_with_applied_classes()

        if not all_methods or not class_names:
            return []  # Return empty list if no methods_application or classes

        # Filter methods_application based on appliesTo field
        applicable_methods = []
        for method in all_methods:
            # Check if method applies to any of the given classes
            applies_to = method.get('appliesTo', [])
            if any(cls in applies_to for cls in class_names):
                applicable_methods.append(method)
                print(
                    f"Found applicable method {method['name']} for class {[cls for cls in class_names if cls in applies_to]}")

        return applicable_methods

    def find_processes_with_methods(self, methods):
        """
        Find processes that utilize the given methods_application
        """
        # Get all processes
        processes = self.process_service.get_all_processes()
        print(f"DEBUG: Found {len(processes)} total processes")

        # Create both full and short method IDs for flexible matching
        method_ids = []
        method_short_names = []

        for method in methods:
            full_id = method.get('id')
            if full_id:
                method_ids.append(full_id)
                if ':' in full_id:
                    short_id = full_id.split(':')[-1]
                    method_short_names.append(short_id)
                else:
                    method_short_names.append(full_id)

        print(f"DEBUG: Looking for method IDs: {method_ids}")
        print(f"DEBUG: Looking for short method names: {method_short_names}")

        # Find processes that use these methods
        matching_processes = []
        for process in processes:
            process_id = process.get('id')
            print(f"DEBUG: Checking process {process_id}")
            if not process_id:
                continue

            # Get full process details
            try:
                # ✅ FIXED: Handle process ID format with specific process identifier
                if '#' in process_id:
                    filename, specific_process_id = process_id.split('#', 1)
                    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    process_path = os.path.join(base_dir, 'data', 'processes', 'definitions', filename)

                    # Get the full TTL content and filter for this specific process
                    process_details = self.process_service.get_process_details(process_path)

                    # For now, just use the full process details but check if it contains our specific process
                    if not self._process_contains_specific_id(process_details, specific_process_id):
                        print(f"DEBUG: Process {process_id} doesn't contain {specific_process_id}")
                        continue

                else:
                    # Handle existing single-process files
                    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                    process_path = os.path.join(base_dir, 'data', 'processes', 'definitions', process_id)
                    process_details = self.process_service.get_process_details(process_path)

                print(f"DEBUG: Process {process_id} has {len(process_details.get('nodes', []))} nodes")

                # Check if process uses any of the methods
                uses_methods = []

                # Look at each node in the process
                for node in process_details.get('nodes', {}).values():
                    print(f"DEBUG: Checking node {node.get('id')} of type {node.get('type')}")

                    if node.get('type') == 'MethodStep':
                        method_id = node.get('methodId', node.get('id'))

                        if 'attributes' in node and 'implementsMethod' in node['attributes']:
                            method_id = node['attributes']['implementsMethod']

                        print(f"DEBUG: MethodStep found with method_id: '{method_id}'")

                        # Check both full method ID and short name
                        if method_id:
                            # Direct match
                            if method_id in method_ids:
                                uses_methods.append(method_id)
                                print(f"DEBUG: ✅ DIRECT MATCH! {method_id}")

                            # Short name match (extract part after colon)
                            elif ':' in method_id:
                                short_method = method_id.split(':')[-1]
                                if short_method in method_short_names:
                                    uses_methods.append(method_id)
                                    print(f"DEBUG: ✅ SHORT MATCH! {short_method} from {method_id}")

                            # Reverse check - method_id is short, looking for long
                            elif method_id in method_short_names:
                                uses_methods.append(method_id)
                                print(f"DEBUG: ✅ REVERSE MATCH! {method_id}")

                            else:
                                print(f"DEBUG: ❌ No match for '{method_id}'")

                # Also inspect relationships for implementsMethod relations
                for edge in process_details.get('edges', {}).values():
                    if 'attributes' in edge and 'edge_type' in edge['attributes']:
                        if edge['attributes']['edge_type'] == 'implementsMethod':
                            method_id = edge.get('target')
                            print(f"DEBUG: Found implementsMethod edge targeting: '{method_id}'")

                            if method_id and method_id not in uses_methods:
                                if (method_id in method_ids or
                                        ((':' in method_id and method_id.split(':')[-1] in method_short_names)) or
                                        method_id in method_short_names):
                                    uses_methods.append(method_id)
                                    print(f"DEBUG: ✅ EDGE MATCH! {method_id}")

                if uses_methods:
                    # Add process with the methods it uses
                    process_details['uses_methods'] = uses_methods
                    matching_processes.append(process_details)
                    print(f"DEBUG: ✅ Added matching process: {process_details['name']} using methods: {uses_methods}")
                else:
                    print(f"DEBUG: ❌ Process {process_id} uses no matching methods")

            except Exception as e:
                print(f"DEBUG: Error processing {process_id}: {e}")
                continue

        print(f"DEBUG: Total matching processes found: {len(matching_processes)}")
        return matching_processes

    def _process_contains_specific_id(self, process_details, specific_process_id):
        """Check if process details contain a specific process ID"""
        # Simple check - look for the specific process ID in the nodes
        if process_details and 'nodes' in process_details:
            for node_id, node in process_details.get('nodes', {}).items():
                if node_id == specific_process_id:
                    return True
        return True  # For now, assume all processes are valid

    def analyze_instance_with_reasoning(self, graph_id):
        """
        Analyze an instance graph with ontology comparison and reasoning

        Args:
            graph_id (str): ID of the instance graph

        Returns:
            dict: Comprehensive analysis results
        """
        # Get the instance graph
        instance_graph = self.instance_service.get_graph_data(graph_id)

        if not instance_graph:
            return {"error": "Graph not found"}

        # Get ontology comparison
        comparison = self.get_ontology_comparison(graph_id)

        # Check if there was an error in comparison
        if 'error' in comparison:
            return {
                "graph_id": graph_id,
                "node_count": len(instance_graph.get('nodes', {})),
                "edge_count": len(instance_graph.get('edges', {})),
                "ontology_comparison": comparison,
                "relevant_classes": [],
                "applicable_methods": [],
                "matching_processes": [],
                "error": comparison.get('error')
            }

        # Extract all relevant classes (exact + fuzzy matches)
        relevant_classes = list(comparison.get('matching_classes', []))

        fuzzy_matches_dict = comparison.get('fuzzy_matching_classes', {})
        for loaded_class, match_list in fuzzy_matches_dict.items():
            if loaded_class not in relevant_classes:
                relevant_classes.append(loaded_class)

        # Get methods_application for these classes
        applicable_methods = self.get_methods_for_classes(relevant_classes)

        # Find processes that use these methods_application
        matching_processes = self.find_processes_with_methods(applicable_methods)

        # Return comprehensive analysis
        return {
            "graph_id": graph_id,
            "node_count": len(instance_graph.get('nodes', {})),
            "edge_count": len(instance_graph.get('edges', {})),
            "ontology_comparison": comparison,
            "relevant_classes": relevant_classes,
            "applicable_methods": applicable_methods,
            "matching_processes": matching_processes
        }

    def execute_method(self, graph_id, method_id, parameters=None):
        """Execute a specific method on an instance graph"""
        # Ensure an OWL ontology is loaded
        if not self.ensure_ontology_loaded():
            return {"success": False, "error": "No OWL ontology loaded. Please upload an ontology file first."}

        # Execute the method using the MethodExecutor
        return self.method_executor.execute_method(graph_id, method_id, parameters or {})

    def execute_process(self, graph_id, process_id):
        """Execute a complete process on an instance graph"""
        graph = self.instance_service.get_graph_data(graph_id)
        if not graph:
            return {"success": False, "error": "Graph not found"}

        # ✅ Handle process ID format with specific process identifier
        if '#' in process_id:
            filename, specific_process_id = process_id.split('#', 1)
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            process_path = os.path.join(base_dir, 'data', 'processes', 'definitions', filename)

            # For now, get all processes and filter manually
            process_details = self.process_service.get_process_details(process_path)

            # Filter the process to only include nodes related to the specific process
            filtered_process = self._filter_process_for_specific_id(process_details, specific_process_id)

            if not filtered_process:
                return {"success": False, "error": f"Specific process {specific_process_id} not found"}

            process = filtered_process
        else:
            # Handle existing single-process files
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            process_path = os.path.join(base_dir, 'data', 'processes', 'definitions', process_id)
            process = self.process_service.get_process_details(process_path)

        if not process:
            return {"success": False, "error": "Process not found"}

        return self.method_executor.execute_process(graph_id, process, None)

    def _filter_process_for_specific_id(self, process_details, specific_process_id):
        """Filter process details to only include nodes for a specific process"""
        # This is a simplified version - you could enhance this to properly filter
        # For now, just update the process name to help with start node detection
        if process_details:
            process_details['name'] = specific_process_id.split(':')[-1]
            print(f"DEBUG: Filtered process for {specific_process_id}")
        return process_details