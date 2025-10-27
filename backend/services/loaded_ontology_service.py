from backend.core_graph_managers.no3_OntologyManager.OntologyManager import OntologyManager
from backend.core_graph_managers.no3_OntologyManager.ontologyBridge import OntologyBridge
from backend.core_graph_managers.no2_graphManager.graphManager import GraphManager
from backend.services.comparators.ontology_matcher import FuzzyOntologyMatcher

import os
from datetime import datetime

class LoadedOntologyService:
    def __init__(self):
        self.ontology_bridge = None
        self.knowledge_graph = None
        self.ontology_manager = None  # New field to store the OntologyManager

    def load_ontologies(self, ontology_paths):
        """Load multiple OWL ontologies from paths and generate inferred processes"""
        self.ontology_bridge = OntologyBridge(ontology_paths)
        self.knowledge_graph = self.ontology_bridge.get_ontology_graph()

        # Convert to OntologyManager format
        self.ontology_manager = self.ontology_bridge.to_ontology_manager()

        # Generate process definitions from inferred methods
        try:
            self.generate_inferred_processes_ttl()
            print("Successfully generated inferred processes from ontology")
        except Exception as e:
            print(f"Warning: Could not generate inferred processes: {e}")

        return self.knowledge_graph

    def visualize(self, output_path, highlighted_nodes=None, fuzzy_matches=None, methods=None):
        """
        Generate visualization with direct relationships between nodes,
        distinguishing between classes, methods_application, and properties,
        excluding unattached primitives and method/class base nodes
        """
        if not self.ontology_bridge:
            return None

        # Create a fresh graph manager for our schema
        schema_graph = GraphManager(preload=False)

        # Initialize methods_application list if None
        if methods is None:
            methods = []

        # Skip these specific node IDs (abstract base classes or concepts we don't want to visualize)
        skip_nodes = {"Method", "Class", "None"}

        # Add relevant classes as nodes with appropriate types
        classes = self.ontology_bridge.get_all_classes()
        for cls in classes:
            # Skip nodes we don't want to visualize
            if cls in skip_nodes:
                continue

            # Determine if this is a method class by checking the name
            is_method = "Method" in cls
            # Get any annotations to check for method indicators
            annotations = self.ontology_bridge.get_annotations_for(cls)
            if "appliesTo" in annotations or "isMethod" in annotations:
                is_method = True

            node_type = "Method" if is_method else "Class"

            schema_graph.add_node(
                node_id=cls,
                value=cls,
                type=node_type,
                hierarchy="OntologyClass",
                attributes=annotations  # Include all annotations
            )

            # If it's a method, add it to the methods_application list for highlighting
            if is_method and cls not in methods:
                methods.append(cls)

        # Keep track of nodes that are actually used in relationships
        connected_nodes = set()

        # Add direct edges for relationships and track connected nodes
        properties = self.ontology_bridge.get_all_properties()
        for src, pred, obj in properties:
            # Skip if either source or target should be skipped
            if src in skip_nodes or obj in skip_nodes:
                continue

            # Add source and target to connected nodes set
            connected_nodes.add(src)
            connected_nodes.add(obj)

            # Make sure both nodes exist (add them if not)
            if src not in schema_graph.node_data:
                # Check if this is a method
                is_method = "Method" in src
                node_type = "Method" if is_method else "Class"

                schema_graph.add_node(
                    node_id=src,
                    value=src,
                    type=node_type,
                    hierarchy="OntologyClass",
                    attributes={}
                )

                # If it's a method, add it to the methods_application list
                if is_method and src not in methods:
                    methods.append(src)

            if obj not in schema_graph.node_data:
                # Determine if this is a primitive type
                is_primitive = obj.startswith("xsd:")
                node_type = "Primitive" if is_primitive else "Class"

                schema_graph.add_node(
                    node_id=obj,
                    value=obj,
                    type=node_type,
                    hierarchy="XSDType" if is_primitive else "OntologyClass",
                    attributes={}
                )

            # Add the direct edge with additional type information
            try:
                # Determine if this is a property edge (connects to a primitive)
                edge_type = "Property" if obj.startswith("xsd:") else "Relationship"

                schema_graph.add_edge(
                    source=src,
                    target=obj,
                    attributes={
                        "edge_type": pred,
                        "relationship_type": edge_type
                    }
                )
            except ValueError:
                # Edge already exists, might need to handle duplicates
                pass

        # Only add primitive types that are actually used in relationships
        primitives = ["string", "integer", "float", "boolean", "dateTime"]
        for ptype in primitives:
            ptype_id = f"xsd:{ptype}"
            if ptype_id in connected_nodes and ptype_id not in schema_graph.node_data:
                schema_graph.add_node(
                    node_id=ptype_id,
                    value=ptype_id,
                    type="Primitive",
                    hierarchy="XSDType",
                    attributes={"comment": f"XSD Primitive Type: {ptype}"}
                )

        # Remove any nodes that aren't connected (no incoming or outgoing edges)
        # Make a copy of node_data keys to avoid modifying during iteration
        for node_id in list(schema_graph.node_data.keys()):
            if node_id not in connected_nodes:
                schema_graph.node_data.pop(node_id, None)

        # Apply custom styling based on node and edge types
        for node_id, node_data in schema_graph.node_data.items():
            if node_data.get("type") == "Method":
                node_data["color"] = "#33ff57"  # Green for methods_application
                node_data["shape"] = "diamond"
            elif node_data.get("type") == "Primitive":
                node_data["color"] = "#ffcc00"  # Yellow for primitives
                node_data["shape"] = "dot"
                node_data["size"] = 15  # Smaller size for primitives
            else:
                node_data["color"] = "#3498db"  # Blue for regular classes
                node_data["shape"] = "dot"

        # Style edges based on their type
        for edge_key, edge_data in schema_graph.edge_data.items():
            if edge_data.get("relationship_type") == "Property":
                edge_data["color"] = "#ff9900"  # Orange for property edges
                edge_data["dashes"] = True  # Dashed lines for properties
            else:
                edge_data["color"] = "#666666"  # Gray for regular relationships

        # Apply highlighting if needed (this will override the default styling)
        if highlighted_nodes or fuzzy_matches or methods:
            self._apply_highlighting(schema_graph, highlighted_nodes, fuzzy_matches, methods)

        # Render the visualization
        try:
            return schema_graph.render_pyvis(
                path=output_path,
                height="700px",
                width="100%",
                physics=True,
                spacing_factor=1.2
            )
        except Exception as e:
            print(f"Error rendering visualization: {e}")
            try:
                return schema_graph.render_pyvis(path=output_path)
            except Exception as e2:
                print(f"Simplified rendering also failed: {e2}")
                return None

    def _apply_highlighting(self, schema_graph, highlighted_nodes, fuzzy_matches, methods):
        """Apply highlighting to the schema graph"""
        if not schema_graph or not hasattr(schema_graph, 'node_data'):
            return

        # Apply highlighting to nodes in GraphManager format
        if highlighted_nodes:
            for node_id in highlighted_nodes:
                if node_id in schema_graph.node_data:
                    # Add color attribute for PyVis rendering
                    schema_graph.node_data[node_id]['color'] = '#ff5733'  # Bright orange
                    schema_graph.node_data[node_id]['borderWidth'] = 3
                    schema_graph.node_data[node_id]['borderColor'] = '#000000'

                    # Also store in attributes
                    if 'attributes' not in schema_graph.node_data[node_id]:
                        schema_graph.node_data[node_id]['attributes'] = {}
                    schema_graph.node_data[node_id]['attributes']['highlight_type'] = 'matching'

        # Apply highlighting for fuzzy matches
        if fuzzy_matches:
            for loaded_class, _ in fuzzy_matches:
                if loaded_class in schema_graph.node_data:
                    # Add color attribute for PyVis rendering
                    schema_graph.node_data[loaded_class]['color'] = '#33a1ff'  # Light blue
                    schema_graph.node_data[loaded_class]['borderWidth'] = 3
                    schema_graph.node_data[loaded_class]['borderColor'] = '#000000'

                    # Also store in attributes
                    if 'attributes' not in schema_graph.node_data[loaded_class]:
                        schema_graph.node_data[loaded_class]['attributes'] = {}
                    schema_graph.node_data[loaded_class]['attributes']['highlight_type'] = 'fuzzy'

        # Apply highlighting for methods_application
        if methods:
            for node_id in methods:
                if node_id in schema_graph.node_data:
                    # Add color attribute for PyVis rendering
                    schema_graph.node_data[node_id]['color'] = '#33ff57'  # Light green
                    schema_graph.node_data[node_id]['borderWidth'] = 3
                    schema_graph.node_data[node_id]['borderColor'] = '#000000'

                    # Also store in attributes
                    if 'attributes' not in schema_graph.node_data[node_id]:
                        schema_graph.node_data[node_id]['attributes'] = {}
                    schema_graph.node_data[node_id]['attributes']['highlight_type'] = 'method'

    # Add methods_application that leverage OntologyManager's capabilities
    def generate_validation_rules(self):
        """Generate validation rules using OntologyManager"""
        if self.ontology_manager:
            return self.ontology_manager.generate_validation_rules()
        return None

    def discover_property_patterns(self, class_name, property_name):
        """Discover patterns in property values using OntologyManager"""
        if self.ontology_manager:
            return self.ontology_manager.discover_property_patterns(class_name, property_name)
        return None

    def compare_with_inferred(self, inferred_ontology):
        """Compare loaded ontology with inferred ontology.

        Args:
            inferred_ontology: An OntologyManager instance containing the inferred ontology.

        Returns:
            dict: Comparison results
        """
        if not self.ontology_manager:
            raise ValueError("No ontology loaded or ontology_manager not initialized")

        # --------------------------
        # 1. Extract class lists
        # --------------------------
        loaded_classes = list(self.ontology_bridge.get_all_classes())

        if hasattr(inferred_ontology, 'node_types'):
            inferred_classes = [k for k in inferred_ontology.node_types.keys()]
        elif hasattr(inferred_ontology, 'get_node_types'):
            inferred_classes = list(inferred_ontology.get_node_types().keys())
        else:
            raise ValueError("Inferred ontology does not provide node types for comparison")

        # --------------------------
        # 2. Exact matches (case-sensitive)
        # --------------------------
        exact_matching_classes = set(loaded_classes).intersection(set(inferred_classes))

        # --------------------------
        # 3. Fuzzy matches
        # --------------------------
        matcher = FuzzyOntologyMatcher(
            loaded_classes=loaded_classes,
            inferred_classes=inferred_classes,
            loaded_node_types=getattr(self.ontology_manager, 'node_types', None),
            inferred_node_types=getattr(inferred_ontology, 'node_types', None),
            loaded_graph_manager=getattr(self.ontology_manager, 'graph_manager', None),
            inferred_graph_manager=getattr(inferred_ontology, 'graph_manager', None),
            loaded_relationships=getattr(self.ontology_manager, 'relationships', []),
            inferred_relationships=getattr(inferred_ontology, 'relationships', [])
        )
        fuzzy_matches_raw = matcher.find_fuzzy_matches(threshold=0.6)

        # Exclude anything that is already an exact match
        fuzzy_matches = {lc: matches for lc, matches in fuzzy_matches_raw.items() if lc not in exact_matching_classes}

        # --------------------------
        # 4. Combine matches
        # --------------------------
        all_matching_loaded = set(exact_matching_classes) | set(fuzzy_matches.keys())

        only_in_loaded = [cls for cls in loaded_classes if cls not in all_matching_loaded]
        only_in_inferred = [cls for cls in inferred_classes if cls not in
                            {ic for matches in fuzzy_matches.values() for ic, *_ in matches} | exact_matching_classes]

        # --------------------------
        # 5. Methods / application
        # --------------------------
        methods = []
        for cls in loaded_classes:
            if 'Method' in cls:
                methods.append(cls)
            annotations = self.ontology_bridge.get_annotations_for(cls)
            if 'appliesTo' in annotations or 'isMethod' in annotations:
                if cls not in methods:
                    methods.append(cls)

        # --------------------------
        # 6. Return clean result
        # --------------------------
        return {
            'matching_classes': list(exact_matching_classes),
            'fuzzy_matching_classes': fuzzy_matches,  # dict of loaded -> [(inferred, score, reason)]
            'only_in_loaded': only_in_loaded,
            'only_in_inferred': only_in_inferred,
            'methods_application': methods
        }

    def get_ontology_data(self):
        """Get structured data about the loaded ontology"""
        if not self.ontology_bridge:
            return None

        # If we have the OntologyManager, use its capabilities
        if self.ontology_manager:
            # Get node types (classes)
            if hasattr(self.ontology_manager, 'get_node_types'):
                node_types = self.ontology_manager.get_node_types()
            else:
                node_types = self.ontology_manager.node_types if hasattr(self.ontology_manager, 'node_types') else {}

            # Format for display - similar to original format
            class_data = {}
            for cls, cls_info in node_types.items():
                # Get annotations from ontology bridge for compatibility
                annotations = self.ontology_bridge.get_annotations_for(cls)

                # Get properties for this class
                properties = []

                # Find properties from edge_types - edges where this class is the source
                if hasattr(self.ontology_manager, 'get_edge_types'):
                    edge_types = self.ontology_manager.get_edge_types()
                else:
                    edge_types = self.ontology_manager.edge_types if hasattr(self.ontology_manager,
                                                                             'edge_types') else {}

                # Check relationships for properties
                if hasattr(self.ontology_manager, 'relationships'):
                    for (src, pred, obj), count in self.ontology_manager.relationships.items():
                        if src == cls:
                            properties.append({
                                "predicate": pred,
                                "object": obj,
                                "count": count
                            })

                # Add any properties found in class_info
                if cls_info and 'properties' in cls_info:
                    for prop in cls_info['properties']:
                        if not any(p['predicate'] == prop for p in properties):
                            properties.append({
                                "predicate": prop,
                                "object": "Unknown",
                                "count": cls_info.get('count', 0)
                            })

                class_data[cls] = {
                    "annotations": annotations,
                    "properties": properties,
                    "count": cls_info.get('count', 0) if cls_info else 0,
                    "sample_nodes": cls_info.get('sample_nodes', []) if cls_info else []
                }

            # Also check the original ontology bridge for any missing classes
            bridge_classes = self.ontology_bridge.get_all_classes()
            bridge_properties = self.ontology_bridge.get_all_properties()

            # Add any classes from the bridge that aren't already in class_data
            for cls in bridge_classes:
                if cls not in class_data:
                    annotations = self.ontology_bridge.get_annotations_for(cls)
                    class_data[cls] = {
                        "annotations": annotations,
                        "properties": []
                    }

            # Associate bridge properties with classes
            for src, pred, obj in bridge_properties:
                if src in class_data:
                    # Check if this property already exists
                    if not any(p['predicate'] == pred and p['object'] == obj for p in class_data[src]["properties"]):
                        class_data[src]["properties"].append({
                            "predicate": pred,
                            "object": obj
                        })

            # Prepare ontology summary
            ontology_summary = {
                "class_count": len(class_data),
                "property_count": len(bridge_properties),
                "node_type_count": len(node_types),
                "edge_type_count": len(edge_types) if isinstance(edge_types, dict) else 0,
                "relationship_count": len(self.ontology_manager.relationships)
                if hasattr(self.ontology_manager, 'relationships') else 0
            }

            return {
                "classes": class_data,
                "ontology_summary": ontology_summary
            }

        # Fall back to original approach if ontology_manager isn't available
        classes = self.ontology_bridge.get_all_classes()
        properties = self.ontology_bridge.get_all_properties()

        # Format for display
        class_data = {}
        for cls in classes:
            annotations = self.ontology_bridge.get_annotations_for(cls)
            class_data[cls] = {
                "annotations": annotations,
                "properties": []
            }

        # Associate properties with classes
        for src, pred, obj in properties:
            if src in class_data:
                class_data[src]["properties"].append({
                    "predicate": pred,
                    "object": obj
                })

        # Prepare ontology summary
        ontology_summary = {
            "class_count": len(classes),
            "property_count": len(properties)
        }

        return {
            "classes": class_data,
            "ontology_summary": ontology_summary
        }

    def get_methods(self):
        """Get all methods_application from the loaded ontology"""
        methods = []

        if not self.ontology_bridge:
            return methods

        # Get all classes from the ontology
        classes = self.ontology_bridge.get_all_classes()

        # Filter for method classes
        for cls in classes:
            is_method = False

            # Check if this is a method class by name
            if "Method" in cls:
                is_method = True

            # Check if it's a method through annotations
            annotations = self.ontology_bridge.get_annotations_for(cls)
            if "appliesTo" in annotations or "isMethod" in annotations:
                is_method = True

            if is_method:
                # Get applies to information
                applies_to = []
                for src, pred, obj in self.ontology_bridge.get_all_properties():
                    if src == cls and pred == "appliesTo":
                        applies_to.append(obj)

                # Get description
                description = annotations.get("comment", "")

                methods.append({
                    'id': cls,
                    'name': cls,
                    'description': description,
                    'appliesTo': applies_to
                })

        return methods

    ## ------ Process Graph Creation

    def generate_inferred_processes_ttl(self, output_dir=None):
        """
        Generate all possible process paths from ontology and write to TTL file.
        Called after loading ontologies.
        """
        if not self.ontology_bridge:
            return None

        if output_dir is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            output_dir = os.path.join(base_dir, 'data', 'processes', 'definitions')

        os.makedirs(output_dir, exist_ok=True)

        # Clear old process files
        self._clear_old_processes(output_dir)

        # Build method dependency graph
        method_graph = self._build_method_dependency_graph()

        if not method_graph:
            print("No methods found in ontology to generate processes")
            return None

        # Find all valid transformation paths
        all_paths = self._find_all_transformation_paths(method_graph, max_depth=6)

        # Generate TTL content
        ttl_content = self._generate_process_ontology_ttl(all_paths)

        # Write to file
        output_file = os.path.join(output_dir, 'inferred_processes.ttl')
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(ttl_content)

        print(f"Generated {len(all_paths)} process paths in {output_file}")
        return output_file

    def _clear_old_processes(self, process_dir):
        """Remove old generated process files"""
        try:
            for filename in os.listdir(process_dir):
                if filename.endswith('.ttl') or filename.endswith('.owl'):
                    filepath = os.path.join(process_dir, filename)
                    os.remove(filepath)
                    print(f"Cleared old process file: {filename}")
        except Exception as e:
            print(f"Error clearing old processes: {e}")

    def _build_method_dependency_graph(self):
        """Build {input_class: [(method_id, output_class), ...]}"""
        graph = {}

        all_classes = self.ontology_bridge.get_all_classes()
        all_properties = self.ontology_bridge.get_all_properties()
        method_classes = [c for c in all_classes if "Method" in c]

        print(f"\n=== METHOD DEPENDENCY GRAPH DEBUG ===")
        print(f"Total classes: {len(all_classes)}")
        print(f"Method classes found: {method_classes}")
        print(f"Total properties: {len(all_properties)}")

        for method in method_classes:
            applies_to_classes = []
            produces_classes = []

            print(f"\n--- Processing {method} ---")

            for src, pred, obj in all_properties:
                if src == method:
                    print(f"  Triple: {src} {pred} {obj}")

                    if "appliesTo" in pred:
                        applies_to_classes.append(obj)
                        print(f"    ✓ Found appliesTo: {obj}")

                    if "produces" in pred:
                        produces_classes.append(obj)
                        print(f"    ✓ Found produces: {obj}")

            print(f"  Summary: applies_to={applies_to_classes}, produces={produces_classes}")

            # Register method in graph
            for input_class in applies_to_classes:
                if input_class not in graph:
                    graph[input_class] = []

                for output_class in produces_classes:
                    graph[input_class].append({
                        'method': method,
                        'output': output_class
                    })
                    print(f"    → Added path: {input_class} --{method}--> {output_class}")

        print(f"\n=== FINAL GRAPH ===")
        for input_cls, transitions in graph.items():
            print(f"{input_cls}:")
            for t in transitions:
                print(f"  → {t['output']} via {t['method']}")

        return graph

    def _find_all_transformation_paths(self, method_graph, max_depth=6):
        """Find all valid transformation paths through the method graph"""
        all_paths = []

        # For each starting data type
        for start_class in method_graph.keys():
            # Find all reachable endpoints
            paths_from_start = self._dfs_find_paths(
                current_class=start_class,
                graph=method_graph,
                path=[],
                visited=set(),
                max_depth=max_depth
            )

            for end_class, path in paths_from_start:
                all_paths.append({
                    'start': start_class,
                    'end': end_class,
                    'steps': path
                })

        return all_paths

    def _dfs_find_paths(self, current_class, graph, path, visited, max_depth):
        """DFS to find all transformation paths"""
        paths_found = []

        if len(path) >= max_depth:
            return paths_found

        if current_class in visited:
            return paths_found

        # Record this as a valid endpoint if we have steps
        if path:
            paths_found.append((current_class, path.copy()))

        visited = visited | {current_class}

        if current_class in graph:
            for transition in graph[current_class]:
                new_path = path + [transition]

                paths_found.extend(self._dfs_find_paths(
                    current_class=transition['output'],
                    graph=graph,
                    path=new_path,
                    visited=visited,
                    max_depth=max_depth
                ))

        return paths_found

    def _generate_process_ontology_ttl(self, paths):
        """Generate TTL content for all inferred processes"""
        timestamp = datetime.now().isoformat()

        ttl = f"""@prefix rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
    @prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
    @prefix owl:  <http://www.w3.org/2002/07/owl#> .
    @prefix xsd:  <http://www.w3.org/2001/XMLSchema#> .
    @prefix process: <http://example.com/process_ontology#> .

    <http://example.com/inferred_processes> rdf:type owl:Ontology ;
        rdfs:label "Inferred Process Ontology" ;
        rdfs:comment "Auto-generated process definitions from method dependencies ({timestamp})" .

    ### === Process Core Classes ===

    process:Process rdf:type owl:Class ;
        rdfs:label "Process" ;
        rdfs:comment "A defined workflow process" .

    process:Step rdf:type owl:Class ;
        rdfs:label "Process Step" ;
        rdfs:comment "A step within a process" .

    process:Start rdf:type owl:Class ;
        rdfs:subClassOf process:Step ;
        rdfs:label "Start Step" .

    process:End rdf:type owl:Class ;
        rdfs:subClassOf process:Step ;
        rdfs:label "End Step" .

    process:MethodStep rdf:type owl:Class ;
        rdfs:subClassOf process:Step ;
        rdfs:label "Method Step" .

    ### === Process Properties ===

    process:hasStep rdf:type owl:ObjectProperty ;
        rdfs:domain process:Process ;
        rdfs:range process:Step .

    process:nextStep rdf:type owl:ObjectProperty ;
        rdfs:domain process:Step ;
        rdfs:range process:Step .

    process:implementsMethod rdf:type owl:ObjectProperty ;
        rdfs:domain process:MethodStep ;
        rdfs:comment "Links a method step to its implementation" .

    """

        process_counter = 0
        for path_info in paths:
            start_class = path_info['start'].split(':')[-1]
            end_class = path_info['end'].split(':')[-1]
            steps_count = len(path_info['steps'])

            process_id = f"process:P{process_counter}"
            start_label = start_class.replace('_', ' ').title()
            end_label = end_class.replace('_', ' ').title()

            # Get first method's ID to connect Start to it
            first_method = path_info['steps'][0]['method'].split(':')[-1]
            first_method_label = first_method.replace('Method', '').strip()
            first_method_node_id = first_method_label.replace(' ', '')

            ttl += f"""### === Process {process_counter + 1}: {start_label} → {end_label} ===

    {process_id} rdf:type process:Process ;
        rdfs:label "Transform {start_label} to {end_label}" ;
        rdfs:comment "Inferred workflow: {steps_count} step(s)" .

    process:P{process_counter}_Start rdf:type process:Start ;
        rdfs:label "Start" ;
        process:nextStep process:P{process_counter}_{first_method_node_id} .

    {process_id} process:hasStep process:P{process_counter}_Start .

    """

            # Generate method steps
            for step_idx, step in enumerate(path_info['steps']):
                method_full_id = step['method']
                method_short = method_full_id.split(':')[-1]
                method_label = method_short.replace('Method', '').strip()
                method_node_id = method_label.replace(' ', '')

                # Next step is either the next method or End
                if step_idx < len(path_info['steps']) - 1:
                    next_method = path_info['steps'][step_idx + 1]['method'].split(':')[-1]
                    next_method_label = next_method.replace('Method', '').strip()
                    next_node_id = f"process:P{process_counter}_{next_method_label.replace(' ', '')}"
                else:
                    next_node_id = f"process:P{process_counter}_End"

                output_class = step['output'].split(':')[-1].replace('_', ' ').title()

                ttl += f"""process:P{process_counter}_{method_node_id} rdf:type process:MethodStep ;
        rdfs:label "{method_label} Execution" ;
        rdfs:comment "Execute {method_short} → {output_class}" ;
        process:implementsMethod {method_full_id} ;
        process:nextStep {next_node_id} .

    {process_id} process:hasStep process:P{process_counter}_{method_node_id} .

    """

            # Generate end node
            ttl += f"""process:P{process_counter}_End rdf:type process:End ;
        rdfs:label "End" ;
        rdfs:comment "Complete, produces {end_label}" .

    {process_id} process:hasStep process:P{process_counter}_End .

    """

            process_counter += 1

        return ttl