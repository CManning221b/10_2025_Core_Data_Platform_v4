from flask import current_app
import os
import json
import re
import html
from backend.core_graph_managers.no2_graphManager.graphManager import GraphManager


class ProcessGraphService:
    """Service for managing process graph definitions and visualizations"""

    def get_all_processes(self):
        """
        Get a list of all available process graphs, detecting multiple processes per file

        Returns:
            list: List of process metadata dictionaries
        """
        # Get the processes directory
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        process_dir = os.path.join(base_dir, 'data', 'processes', 'definitions')

        # Make sure the directory exists
        os.makedirs(process_dir, exist_ok=True)

        # Get process files
        process_files = []
        if os.path.exists(process_dir):
            for filename in os.listdir(process_dir):
                if filename.endswith(('.ttl', '.owl', '.rdf', '.json')):
                    process_path = os.path.join(process_dir, filename)

                    try:
                        if filename.endswith(('.ttl', '.owl', '.rdf')):
                            # ✅ NEW: Extract individual processes from TTL files
                            individual_processes = self._extract_individual_processes(process_path)
                            process_files.extend(individual_processes)
                        else:
                            # Handle JSON files as before
                            process_details = self.get_process_details(process_path)
                            process_files.append({
                                'id': filename,
                                'name': process_details.get('name', filename),
                                'description': process_details.get('description', ''),
                                'step_count': process_details.get('step_count', 0),
                                'file_source': filename
                            })
                    except Exception as e:
                        # If loading fails, add with minimal details
                        print(f"Error loading process {filename}: {e}")
                        process_files.append({
                            'id': filename,
                            'name': filename.split('.')[0].replace('_', ' ').title(),
                            'description': 'Could not load process details',
                            'step_count': 0,
                            'file_source': filename
                        })

        return process_files

    def _extract_individual_processes(self, process_path):
        """Extract individual processes from a TTL file that contains multiple processes"""
        filename = os.path.basename(process_path)

        try:
            # Read the file content
            with open(process_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Find all processes
            processes = re.findall(r'(process:\w+)\s+rdf:type\s+process:Process', content)

            individual_processes = []

            for process_id in processes:
                # Get process label
                label_match = re.search(rf'{re.escape(process_id)}\s+rdfs:label\s+"([^"]+)"', content)
                process_name = label_match.group(1) if label_match else process_id.split(':')[-1]

                # Get process description
                desc_match = re.search(rf'{re.escape(process_id)}\s+rdfs:comment\s+"([^"]+)"', content)
                process_description = desc_match.group(1) if desc_match else ""

                # Find hasStep relationships to count steps
                hasStep_pattern = rf'{re.escape(process_id)}.*?process:hasStep\s+([\w:,\s]+)'
                hasStep_match = re.search(hasStep_pattern, content, re.DOTALL)

                step_count = 0
                if hasStep_match:
                    steps_text = hasStep_match.group(1)
                    # Count comma-separated steps
                    steps = [s.strip() for s in re.split(r',\s*', steps_text) if s.strip()]
                    step_count = len(steps)

                individual_processes.append({
                    'id': f"{filename}#{process_id}",  # Keep the # syntax
                    'name': process_name,  # ✅ Use the actual process label
                    'description': process_description,
                    'step_count': step_count,
                    'file_source': filename,
                    'process_id': process_id
                })

                print(f"DEBUG: Found individual process: {process_name} with {step_count} steps")

            return individual_processes

        except Exception as e:
            print(f"Error extracting individual processes from {filename}: {e}")
            return []

    def get_process_details(self, process_path):
        """Extract details from a process file"""
        if process_path.endswith('.json'):
            # Handle JSON files (unchanged)
            with open(process_path, 'r', encoding='utf-8') as f:
                process_data = json.load(f)

            id_value = os.path.basename(process_path)
            return {
                'id': id_value,
                'name': process_data.get('name', 'Unnamed Process'),
                'description': process_data.get('description', ''),
                'step_count': len(process_data.get('nodes', [])) if 'nodes' in process_data else len(
                    process_data.get('steps', [])),
                'nodes': process_data.get('nodes', []),
                'edges': process_data.get('edges', []),
                'steps': process_data.get('steps', [])
            }

        elif process_path.endswith(('.ttl', '.owl', '.rdf')):
            # Handle RDF/OWL files
            filename = os.path.basename(process_path)

            # Load the file content WITH dependencies
            content = self._load_process_content_with_dependencies(process_path)

            # Extract process name from filename
            process_name = filename.split('.')[0].replace('_', ' ').title()

            # Find all processes
            processes = re.findall(r'(process:\w+)\s+rdf:type\s+process:Process', content)

            # Find all steps
            steps = []
            for step_type in ['process:Start', 'process:End', 'process:MethodStep', 'process:SubprocessStep',
                              'process:Step']:
                steps.extend(re.findall(rf'(process:\w+)\s+rdf:type\s+{step_type}', content))

            # Find labels
            labels = {}
            for entity in processes + steps:
                label_match = re.search(rf'{re.escape(entity)}\s+rdfs:label\s+"([^"]+)"', content)
                if label_match:
                    labels[entity] = label_match.group(1)
                else:
                    labels[entity] = entity.split(':')[-1]

            # Extract implementsMethod relationships with step-specific regex
            implements_methods = {}
            print(f"DEBUG: Starting implementsMethod extraction...")
            print(f"DEBUG: Found steps: {steps}")

            # For each step, find its implementsMethod in its definition block
            for step in steps:
                # Find the block for this specific step (from declaration to next declaration or end)
                step_pattern = rf'{re.escape(step)}\s+rdf:type.*?(?=process:\w+\s+rdf:type|$)'
                step_block_match = re.search(step_pattern, content, re.DOTALL)

                if step_block_match:
                    step_block = step_block_match.group(0)

                    # Look for implementsMethod in this specific block
                    method_match = re.search(r'process:implementsMethod\s+([\w:]+)', step_block)
                    if method_match:
                        method_id = method_match.group(1)
                        implements_methods[step] = method_id
                        print(f"DEBUG: Found implementsMethod: {step} implements {method_id}")

            print(f"DEBUG: Found {len(implements_methods)} implementsMethod relationships")
            for step_id, method_id in implements_methods.items():
                print(f"DEBUG: {step_id} -> {method_id}")

            # Create nodes
            nodes = []

            # Add processes as nodes
            for proc in processes:
                nodes.append({
                    'id': proc,
                    'label': labels.get(proc, proc.split(':')[-1]),
                    'type': 'Process'
                })

            # Add steps as nodes WITH implementsMethod attributes
            for step in steps:
                # Determine step type
                step_type = 'Step'
                if re.search(rf'{re.escape(step)}\s+rdf:type\s+process:Start', content):
                    step_type = 'StartStep'
                elif re.search(rf'{re.escape(step)}\s+rdf:type\s+process:End', content):
                    step_type = 'EndStep'
                elif re.search(rf'{re.escape(step)}\s+rdf:type\s+process:MethodStep', content):
                    step_type = 'MethodStep'
                elif re.search(rf'{re.escape(step)}\s+rdf:type\s+process:SubprocessStep', content):
                    step_type = 'SubprocessStep'

                # Create node with implementsMethod if available
                node = {
                    'id': step,
                    'label': labels.get(step, step.split(':')[-1]),
                    'type': step_type
                }

                # Add implementsMethod to attributes if this step implements a method
                if step in implements_methods:
                    node['attributes'] = {
                        'implementsMethod': implements_methods[step]
                    }
                    print(f"DEBUG: Added implementsMethod {implements_methods[step]} to node {step}")

                nodes.append(node)

            # Find nextStep relationships
            next_steps = []

            # Look at each step's definition block in the file
            for step in steps:
                # Find the entire block for this step (from its declaration to the next period)
                step_block_match = re.search(rf'{re.escape(step)}\s+rdf:type.*?\.', content, re.DOTALL)
                if step_block_match:
                    step_block = step_block_match.group(0)

                    # Search for nextStep in this block
                    next_step_match = re.search(r'process:nextStep\s+(process:\w+)', step_block)
                    if next_step_match:
                        target_step = next_step_match.group(1)
                        next_steps.append((step, target_step))

            # Find hasStep relationships to identify start steps for each process
            process_starts = []

            for proc in processes:
                # Look for ALL hasStep relationships for this process (they're separate statements)
                has_step_matches = re.findall(rf'{re.escape(proc)}\s+process:hasStep\s+([^\s.]+_Start)', content)

                for start_step in has_step_matches:
                    process_starts.append((proc, start_step))
                    print(f"DEBUG: Found start: {proc} -> {start_step}")

            # Create edges for nextStep relationships
            edges = []

            # Add nextStep edges
            for i, (src, tgt) in enumerate(next_steps):
                edges.append({
                    'id': f'edge_{i}',
                    'from': src,
                    'to': tgt,
                    'label': 'nextStep'
                })

            # Add process-to-start edges
            for i, (proc, start) in enumerate(process_starts):
                edges.append({
                    'id': f'start_edge_{i}',
                    'from': proc,
                    'to': start,
                    'label': 'startsAt',
                    'dashes': True
                })

            # Debug info
            step_links = [{'from': src, 'to': tgt} for src, tgt in next_steps]
            process_start_links = [{'from': src, 'to': tgt} for src, tgt in process_starts]
            step_labels = [node['label'] for node in nodes if node['type'] != 'Process']

            print(f"Found {len(processes)} processes, {len(steps)} steps")
            print(f"Found {len(next_steps)} nextStep relationships")
            print(f"Found {len(process_starts)} process-to-start relationships")

            debug_info = {
                'process_nodes': [node for node in nodes if node['type'] == 'Process'],
                'step_nodes': [node for node in nodes if node['type'] != 'Process'],
                'step_links': step_links,
                'process_start_links': process_start_links,
                'has_step_links': []
            }

            # Create a GraphManager representation of this process
            graph_manager = self._create_graph_manager(nodes, edges)

            nodes_dict = {node['id']: node for node in nodes}
            edges_dict = {edge['id']: edge for edge in edges}

            return {
                'id': filename,
                'name': process_name,
                'description': "Process extracted from OWL/TTL file",
                'step_count': len(steps),
                'nodes': nodes_dict,
                'edges': edges_dict,
                'steps': step_labels,
                'debug_info': debug_info,
                'graph_manager': graph_manager
            }
        else:
            raise ValueError(f"Unsupported file format: {process_path}")

    def _load_process_content_with_dependencies(self, process_path):
        """Load a process file and its dependencies (like process_ontology.ttl)"""

        # Get the directory containing the process file
        process_dir = os.path.dirname(process_path)

        # Look for process_ontology.ttl in the same directory
        ontology_path = os.path.join(process_dir, 'process_ontology.ttl')

        combined_content = ""

        # Load the process ontology first if it exists
        if os.path.exists(ontology_path):
            with open(ontology_path, 'r', encoding='utf-8') as f:
                ontology_content = f.read()
            combined_content += ontology_content + "\n\n"
            print(f"DEBUG: Loaded process ontology from {ontology_path}")
        else:
            print(f"WARNING: Process ontology not found at {ontology_path}")
            print(f"DEBUG: Looking for ontology at: {ontology_path}")
            print(f"DEBUG: Process file is at: {process_path}")

        # Load the individual process file
        with open(process_path, 'r', encoding='utf-8') as f:
            process_content = f.read()
        combined_content += process_content

        print(f"DEBUG: Combined content is {len(combined_content)} characters")
        return combined_content

    def _create_graph_manager(self, nodes, edges):
        """Create a GraphManager representation of the process"""
        graph_manager = GraphManager(preload=False)

        # Add nodes to the GraphManager
        for node in nodes:
            node_id = node.get('id')
            node_label = node.get('label')
            node_type = node.get('type')

            # Add node with styling based on type
            color = "#97C2FC"  # Default blue
            shape = "dot"
            size = 20

            if node_type == 'Process':
                color = "#FFD700"  # Gold
                shape = "diamond"
                size = 30
            elif node_type == 'StartStep':
                color = "#3498db"  # Blue
                shape = "triangle"
                size = 25
            elif node_type == 'EndStep':
                color = "#e74c3c"  # Red
                shape = "square"
                size = 25
            elif node_type == 'MethodStep':
                color = "#2ecc71"  # Green
                shape = "dot"
            elif node_type == 'SubprocessStep':
                color = "#9b59b6"  # Purple
                shape = "hexagon"
                size = 25

            try:
                graph_manager.add_node(
                    node_id=node_id,
                    value=node_label,
                    type=node_type,
                    hierarchy="ProcessStep",
                    attributes={
                        "color": color,
                        "shape": shape,
                        "size": size
                    }
                )
            except Exception as e:
                print(f"Warning: Could not add node {node_id} to GraphManager: {e}")

        # Add edges to the GraphManager
        for edge in edges:
            source = edge.get('from')
            target = edge.get('to')
            label = edge.get('label', 'nextStep')

            edge_color = "#666666"  # Gray for nextStep
            dashes = False

            if label == 'startsAt':
                edge_color = "#FFD700"  # Gold for startsAt
                dashes = True

            try:
                graph_manager.add_edge(
                    source=source,
                    target=target,
                    attributes={
                        "edge_type": label,
                        "color": edge_color,
                        "dashes": dashes,
                        "arrows": {"to": True},
                        "smooth": {"type": "cubicBezier", "forceDirection": "horizontal"}
                    }
                )
            except Exception as e:
                print(f"Warning: Could not add edge {source} -> {target} to GraphManager: {e}")

        return graph_manager

    def validate_process_file(self, filepath):
        """Basic validation - just make sure the file exists and can be opened"""
        if not os.path.exists(filepath):
            raise ValueError(f"File not found: {filepath}")

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                if not content.strip():
                    raise ValueError("Empty file")
        except Exception as e:
            raise ValueError(f"Error reading file: {str(e)}")

        return True

    def visualize_process(self, process_path, vis_path):
        """Create visualization HTML for a process graph with detailed debugging"""
        # Get process details
        process_details = self.get_process_details(process_path)

        # Try to use GraphManager for visualization if available
        if 'graph_manager' in process_details and process_details['graph_manager']:
            try:
                # Get the GraphManager
                graph_manager = process_details['graph_manager']

                # Define visualization options
                options = {
                    "layout": {
                        "hierarchical": {
                            "direction": "LR",
                            "sortMethod": "directed",
                            "nodeSpacing": 150,
                            "levelSeparation": 200
                        }
                    },
                    "physics": {
                        "enabled": False
                    },
                    "edges": {
                        "smooth": {
                            "type": "cubicBezier",
                            "forceDirection": "horizontal"
                        }
                    }
                }

                # Render using GraphManager
                graph_manager.render_pyvis(
                    path=vis_path,
                    height="700px",
                    width="100%",
                    physics=False,
                    options=options
                )

                # Enhance the HTML with custom controls and legend
                self._enhance_visualization_html(vis_path, process_details.get('name', 'Process Visualization'))

                return True
            except Exception as e:
                print(f"GraphManager visualization failed: {e}, falling back to direct HTML generation")

        # Fall back to direct HTML generation if GraphManager fails
        try:
            # Create direct HTML visualization without using GraphManager
            with open(vis_path, 'w', encoding='utf-8') as f:
                html_content = self._generate_visualization(process_details)
                f.write(html_content)
            return True
        except Exception as e:
            print(f"Direct HTML visualization failed: {e}")
            return False

    def _enhance_visualization_html(self, vis_path, title):
        """Add custom elements to the visualization HTML created by GraphManager"""
        try:
            with open(vis_path, 'r', encoding='utf-8') as f:
                html = f.read()

            # Add legend and controls
            legend_html = """
            <div style="margin-bottom: 20px;">
                <h1>{title}</h1>

                <div style="margin-bottom: 15px;">
                    <button onclick="network.fit()" style="padding: 6px 12px; background: #3498db; color: white; border: none; border-radius: 4px; cursor: pointer; margin-right: 10px;">Reset View</button>
                    <button onclick="togglePhysics()" style="padding: 6px 12px; background: #3498db; color: white; border: none; border-radius: 4px; cursor: pointer; margin-right: 10px;">Toggle Physics</button>
                </div>

                <div style="display: flex; flex-wrap: wrap; margin-bottom: 15px;">
                    <div style="display: flex; align-items: center; margin-right: 20px; margin-bottom: 10px;">
                        <div style="width: 20px; height: 20px; background: #FFD700; transform: rotate(45deg); margin-right: 5px; border: 1px solid #ccc;"></div>
                        <span>Process</span>
                    </div>
                    <div style="display: flex; align-items: center; margin-right: 20px; margin-bottom: 10px;">
                        <div style="width: 0; height: 0; border-left: 10px solid transparent; border-right: 10px solid transparent; border-bottom: 20px solid #3498db; margin-right: 5px;"></div>
                        <span>Start Step</span>
                    </div>
                    <div style="display: flex; align-items: center; margin-right: 20px; margin-bottom: 10px;">
                        <div style="width: 20px; height: 20px; background: #e74c3c; margin-right: 5px; border: 1px solid #ccc;"></div>
                        <span>End Step</span>
                    </div>
                    <div style="display: flex; align-items: center; margin-right: 20px; margin-bottom: 10px;">
                        <div style="width: 20px; height: 20px; background: #2ecc71; border-radius: 50%; margin-right: 5px; border: 1px solid #ccc;"></div>
                        <span>Method Step</span>
                    </div>
                    <div style="display: flex; align-items: center; margin-right: 20px; margin-bottom: 10px;">
                        <div style="width: 20px; height: 20px; background: #9b59b6; clip-path: polygon(25% 0%, 75% 0%, 100% 50%, 75% 100%, 25% 100%, 0% 50%); margin-right: 5px; border: 1px solid #ccc;"></div>
                        <span>Subprocess Step</span>
                    </div>
                </div>
            </div>
            """.format(title=title)

            # Add physics toggle script
            physics_script = """
            <script>
            function togglePhysics() {
                var physics = network.physics.options.enabled;
                network.setOptions({ physics: { enabled: !physics } });
            }
            </script>
            """

            # Insert legend after the body tag
            html = html.replace("<body>", "<body>\n" + legend_html)

            # Insert physics script before the </body> tag
            html = html.replace("</body>", physics_script + "</body>")

            # Write the enhanced HTML back to the file
            with open(vis_path, 'w', encoding='utf-8') as f:
                f.write(html)

        except Exception as e:
            print(f"Error enhancing visualization HTML: {e}")
            # Continue even if enhancement fails

    def _generate_visualization(self, process_details):
        """Generate a simple HTML visualization (fallback method)"""
        nodes_dict = process_details.get('nodes', {})  # Now expecting a dict
        edges_dict = process_details.get('edges', {})  # Now expecting a dict

        # Convert dict values to lists for iteration
        nodes = list(nodes_dict.values()) if isinstance(nodes_dict, dict) else nodes_dict
        edges = list(edges_dict.values()) if isinstance(edges_dict, dict) else edges_dict

        # Create node data for vis.js
        nodes_data = []
        for i, node in enumerate(nodes):  # Now node is the actual node object
            node_type = node.get('type', 'Unknown')

            # Set color and shape based on type
            color = "#97C2FC"  # Default blue
            shape = "dot"

            if node_type == 'Process':
                color = "#FFD700"  # Gold
                shape = "diamond"
            elif node_type == 'StartStep':
                color = "#3498db"  # Blue
                shape = "triangle"
            elif node_type == 'EndStep':
                color = "#e74c3c"  # Red
                shape = "square"
            elif node_type == 'MethodStep':
                color = "#2ecc71"  # Green
                shape = "dot"
            elif node_type == 'SubprocessStep':
                color = "#9b59b6"  # Purple
                shape = "hexagon"

            nodes_data.append({
                "id": node.get('id'),
                "label": node.get('label', f'Node {i}'),
                "color": color,
                "shape": shape,
                "font": {"size": 14, "face": "Arial"},
                "size": 25 if node_type == 'Process' else 20
            })

        # Create edge data for vis.js
        edges_data = []
        for edge in edges:  # Now edge is the actual edge object
            edge_data = {
                "from": edge.get('from'),
                "to": edge.get('to'),
                "arrows": "to",
                "label": edge.get('label', ''),
                "color": "#666666"  # Gray
            }

            # Style process-to-start connections differently
            if edge.get('label') == 'startsAt':
                edge_data["dashes"] = True
                edge_data["color"] = "#FFD700"  # Match the process color
                edge_data["width"] = 2
            else:
                edge_data["width"] = 2
                edge_data["smooth"] = {"type": "curvedCW", "roundness": 0.2}

            edges_data.append(edge_data)

        # Generate HTML with vis.js visualization
        html_code = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>{process_details.get('name', 'Process Visualization')}</title>
            <script type="text/javascript" src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 0;
                    padding: 20px;
                }}

                h1, h2 {{
                    color: #333;
                }}

                #visualization {{
                    width: 100%;
                    height: 700px;
                    border: 1px solid #ddd;
                    border-radius: 4px;
                }}

                .info-panel {{
                    background: #f8f8f8;
                    padding: 15px;
                    border-radius: 4px;
                    margin-bottom: 20px;
                }}

                .legend {{
                    display: flex;
                    flex-wrap: wrap;
                    margin-bottom: 15px;
                }}

                .legend-item {{
                    display: flex;
                    align-items: center;
                    margin-right: 20px;
                    margin-bottom: 10px;
                }}

                .legend-color {{
                    width: 20px;
                    height: 20px;
                    margin-right: 5px;
                    border: 1px solid #ccc;
                }}

                .dashed-line {{
                    width: 30px;
                    height: 0;
                    border-top: 2px dashed #FFD700;
                    margin-right: 5px;
                }}

                .controls {{
                    margin-bottom: 15px;
                }}

                button {{
                    padding: 6px 12px;
                    background: #3498db;
                    color: white;
                    border: none;
                    border-radius: 4px;
                    cursor: pointer;
                    margin-right: 10px;
                }}

                button:hover {{
                    background: #2980b9;
                }}
            </style>
        </head>
        <body>
            <h1>{process_details.get('name', 'Process Visualization')}</h1>

            <div class="info-panel">
                <p><strong>File:</strong> {process_details.get('id')}</p>
                <p><strong>Description:</strong> {process_details.get('description')}</p>
                <p><strong>Steps:</strong> {process_details.get('step_count', 0)}</p>
            </div>

            <div class="controls">
                <button onclick="resetView()">Reset View</button>
                <button onclick="togglePhysics()">Toggle Physics</button>
            </div>

            <div class="legend">
                <div class="legend-item">
                    <div class="legend-color" style="background:#FFD700; transform: rotate(45deg);"></div>
                    <span>Process</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background:#3498db; clip-path: polygon(50% 0%, 0% 100%, 100% 100%);"></div>
                    <span>Start Step</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background:#e74c3c; border-radius: 0;"></div>
                    <span>End Step</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background:#2ecc71;"></div>
                    <span>Method Step</span>
                </div>
                <div class="legend-item">
                    <div class="legend-color" style="background:#9b59b6; clip-path: polygon(25% 0%, 75% 0%, 100% 50%, 75% 100%, 25% 100%, 0% 50%);"></div>
                    <span>Subprocess Step</span>
                </div>
                <div class="legend-item">
                    <div class="dashed-line"></div>
                    <span>Process Start</span>
                </div>
            </div>

            <div id="visualization"></div>

            <script type="text/javascript">
                // Initialize vis.js network
                var container = document.getElementById('visualization');

                // Create nodes and edges datasets
                var nodes = new vis.DataSet({json.dumps(nodes_data)});
                var edges = new vis.DataSet({json.dumps(edges_data)});

                // Create the network
                var data = {{
                    nodes: nodes,
                    edges: edges
                }};

                var options = {{
                    layout: {{
                        hierarchical: {{
                            direction: "LR",
                            sortMethod: "directed",
                            nodeSpacing: 150,
                            levelSeparation: 200
                        }}
                    }},
                    physics: {{
                        enabled: false
                    }},
                    edges: {{
                        smooth: {{
                            type: "cubicBezier",
                            forceDirection: "horizontal"
                        }}
                    }},
                    interaction: {{
                        hover: true,
                        tooltipDelay: 200
                    }}
                }};

                var network = new vis.Network(container, data, options);

                // Fit the view once the network is loaded
                network.once('afterDrawing', function() {{
                    network.fit();
                }});

                // Add tooltips
                network.on("hoverNode", function(params) {{
                    var nodeId = params.node;
                    var node = nodes.get(nodeId);
                    node.title = 'ID: ' + nodeId;
                    nodes.update(node);
                }});

                // Reset view function
                function resetView() {{
                    network.fit({{
                        animation: {{
                            duration: 1000,
                            easingFunction: 'easeInOutQuad'
                        }}
                    }});
                }}

                // Toggle physics function
                var physicsEnabled = false;
                function togglePhysics() {{
                    physicsEnabled = !physicsEnabled;
                    network.setOptions({{ physics: {{ enabled: physicsEnabled }} }});
                }}
            </script>
        </body>
        </html>
        """

        return html_code

    def is_newer(self, file1, file2):
        """Check if file1 is newer than file2"""
        if not os.path.exists(file2):
            return True

        return os.path.getmtime(file1) > os.path.getmtime(file2)

    def get_process_graph_manager(self, process_path):
        """Get the GraphManager representation of a process - for integration with other systems"""
        process_details = self.get_process_details(process_path)

        if 'graph_manager' in process_details and process_details['graph_manager']:
            return process_details['graph_manager']

        # If not already created, create it now
        return self._create_graph_manager(
            process_details.get('nodes', []),
            process_details.get('edges', [])
        )

