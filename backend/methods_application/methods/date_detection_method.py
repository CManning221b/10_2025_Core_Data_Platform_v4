# backend/methods_application/methods/date_detection_method.py
from backend.methods_application.method_implementation import MethodImplementation
import re
from datetime import datetime
from dateutil import parser


class DateDetectionMethod(MethodImplementation):
    method_id = "DateDetectionMethod"
    method_name = "Date Detection in Components"
    description = "Searches component text for date patterns and creates Date nodes"

    def execute(self, graph_manager, parameters):
        changes = {
            "nodes_added": 0,
            "edges_added": 0,
            "method_instances_created": 0,
            "components_processed": 0,
            "dates_found": 0,
            "date_patterns": {}
        }

        # Create method instance node for traceability
        method_instance_id = self._create_method_instance(graph_manager, changes)

        # Find all component nodes
        component_nodes = []
        for node_id, node_data in graph_manager.node_data.items():
            if node_data.get('type') == 'component':
                component_nodes.append((node_id, node_data))

        print(f" Found {len(component_nodes)} component nodes to process")

        for comp_id, comp_data in component_nodes:
            changes["components_processed"] += 1

            # Extract text from component
            text = comp_data.get('value', '').strip()
            if not text:
                continue

            # Try to detect date patterns
            date_info = self._detect_date_patterns(text)

            if date_info:
                # Create Date node using GraphManager's add_node method
                date_id = self._get_next_numeric_id(graph_manager)

                try:
                    graph_manager.add_node(
                        node_id=date_id,
                        value=date_info['formatted_date'],
                        type='Date',
                        hierarchy='temporal',
                        attributes={
                            'original_text': text,
                            'pattern': date_info['pattern'],
                            'confidence': date_info['confidence'],
                            'parsed_datetime': date_info['datetime_str']
                        }
                    )
                    changes["nodes_added"] += 1
                    print(f" Created Date node {date_id} for text '{text}' -> {date_info['formatted_date']}")

                    # Connect method instance to the date it created
                    self._connect_method_to_output(graph_manager, method_instance_id, date_id, changes)

                    # Connect method instance to the component it analyzed
                    self._connect_method_to_input(graph_manager, method_instance_id, comp_id, changes)

                    # Find what entity this component belongs to and link those entities to the date
                    parent_entities = self._find_parent_entities(graph_manager, comp_id)
                    print(f" Found parent entities for component {comp_id}: {parent_entities}")

                    for parent_id, parent_type in parent_entities:
                        self._connect_entity_to_date(graph_manager, parent_id, parent_type, date_id, changes)

                    changes["dates_found"] += 1
                    pattern = date_info['pattern']
                    if pattern in changes["date_patterns"]:
                        changes["date_patterns"][pattern] += 1
                    else:
                        changes["date_patterns"][pattern] = 1

                except KeyError as e:
                    print(f" Error creating Date node {date_id}: {e}")

        print(
            f" DateDetectionMethod processed {changes['components_processed']} components, found {changes['dates_found']} dates")
        return changes

    def _create_method_instance(self, graph_manager, changes):
        """Create a method instance node for execution traceability"""
        method_instance_id = self._get_next_numeric_id(graph_manager)

        try:
            graph_manager.add_node(
                node_id=method_instance_id,
                value=f"DateDetectionMethod execution",
                type='method_instance',
                hierarchy='analysis',
                attributes={
                    'method_type': 'DateDetectionMethod',
                    'execution_time': datetime.now().isoformat(),
                    'method_id': self.method_id,
                    'method_name': self.method_name
                }
            )
            changes["nodes_added"] += 1
            changes["method_instances_created"] += 1
            print(f" Created method_instance node {method_instance_id}")

        except KeyError as e:
            print(f" Method instance node {method_instance_id} already exists")

        return method_instance_id

    def _connect_method_to_output(self, graph_manager, method_instance_id, date_id, changes):
        """Connect method instance to its output Date node"""
        print(f" Connecting method {method_instance_id} -> Date {date_id}")

        try:
            graph_manager.add_edge(
                source=method_instance_id,
                target=date_id,
                attributes={
                    'edge_type': 'produces',
                    'direction': 'out',
                    'provenance_type': 'generated_by',
                    'created': datetime.now().isoformat()
                }
            )
            changes["edges_added"] += 1
            print(f" Created output edge {method_instance_id} -> {date_id}")

        except (KeyError, ValueError) as e:
            print(f" Error creating output edge: {e}")

    def _connect_method_to_input(self, graph_manager, method_instance_id, comp_id, changes):
        """Connect method instance to the component it analyzed"""
        print(f" Connecting method {method_instance_id} -> component {comp_id}")

        try:
            graph_manager.add_edge(
                source=method_instance_id,
                target=comp_id,
                attributes={
                    'edge_type': 'uses',
                    'direction': 'out',
                    'provenance_type': 'derived_from',
                    'created': datetime.now().isoformat()
                }
            )
            changes["edges_added"] += 1
            print(f" Created input edge {method_instance_id} -> {comp_id}")

        except (KeyError, ValueError) as e:
            print(f" Error creating input edge: {e}")

    def _connect_entity_to_date(self, graph_manager, parent_id, parent_type, date_id, changes):
        """Connect parent entity to the date"""
        print(f" Connecting {parent_type} {parent_id} -> Date {date_id}")

        try:
            graph_manager.add_edge(
                source=parent_id,
                target=date_id,
                attributes={
                    'edge_type': 'has_date',
                    'direction': 'out',
                    'provenance_type': 'contains_temporal_info',
                    'created': datetime.now().isoformat()
                }
            )
            changes["edges_added"] += 1
            print(f" Created date relationship edge {parent_id} -> {date_id}")

        except (KeyError, ValueError) as e:
            print(f" Error creating date relationship edge: {e}")

    def _get_next_numeric_id(self, graph_manager):
        """Generate next available numeric ID"""
        existing_ids = set()
        for node_id in graph_manager.node_data.keys():
            try:
                existing_ids.add(int(node_id))
            except (ValueError, TypeError):
                pass

        next_id = 0
        while next_id in existing_ids:
            next_id += 1

        result = str(next_id)
        print(f" Generated next ID: {result}")
        return result

    def _detect_date_patterns(self, text):
        """Detect various date patterns in text"""
        patterns = [
            # YYMMDD format (like 241005)
            {
                'regex': r'^(\d{2})(\d{2})(\d{2})$',
                'pattern': 'YYMMDD',
                'parser': lambda m: f"20{m.group(1)}-{m.group(2)}-{m.group(3)}"
            },
            # YYYYMMDD format
            {
                'regex': r'^(\d{4})(\d{2})(\d{2})$',
                'pattern': 'YYYYMMDD',
                'parser': lambda m: f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
            },
            # HHMMSS format (time)
            {
                'regex': r'^(\d{2})(\d{2})(\d{2})$',
                'pattern': 'HHMMSS',
                'parser': lambda m: f"{m.group(1)}:{m.group(2)}:{m.group(3)}"
            },
            # Year only
            {
                'regex': r'^(20\d{2})$',
                'pattern': 'YYYY',
                'parser': lambda m: f"{m.group(1)}-01-01"
            }
        ]

        for pattern_info in patterns:
            match = re.match(pattern_info['regex'], text)
            if match:
                try:
                    formatted = pattern_info['parser'](match)

                    # Try to parse as datetime
                    if pattern_info['pattern'] == 'HHMMSS':
                        # For time, create a datetime with today's date
                        dt = datetime.strptime(formatted, '%H:%M:%S')
                        confidence = 0.7  # Lower confidence for time-only
                    else:
                        dt = datetime.strptime(formatted, '%Y-%m-%d')
                        confidence = 0.9

                    return {
                        'datetime_str': dt.isoformat(),
                        'formatted_date': formatted,
                        'pattern': pattern_info['pattern'],
                        'confidence': confidence
                    }
                except ValueError:
                    continue

        return None

    def _find_parent_entities(self, graph_manager, comp_id):
        """Find what entities (files, folders) this component belongs to"""
        parents = []

        # Look for incoming edges to this component using proper tuple iteration
        for (source, target), edge_attrs in graph_manager.edge_data.items():
            if target == comp_id:
                source_node = graph_manager.node_data.get(source)
                if source_node:
                    source_type = source_node.get('type')
                    if source_type in ['file', 'folder', 'filename']:
                        parents.append((source, source_type))
                        print(f" Found parent {source_type} {source} for component {comp_id}")

                        # If it's a filename, also find the file that has this filename
                        if source_type == 'filename':
                            for (file_src, file_tgt), file_edge_attrs in graph_manager.edge_data.items():
                                if (file_tgt == source and
                                        file_edge_attrs.get('edge_type') == 'has_name'):
                                    file_node = graph_manager.node_data.get(file_src)
                                    if file_node and file_node.get('type') == 'file':
                                        parents.append((file_src, 'file'))
                                        print(f" Found parent file {file_src} via filename {source}")

        return parents