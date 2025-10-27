# backend/methods_application/methods/reactor/channel_detection_method.py
from backend.methods_application.method_implementation import MethodImplementation
import re
from datetime import datetime


class ChannelDetectionMethod(MethodImplementation):
    method_id = "ChannelDetectionMethod"
    method_name = "Ontology-Driven Channel ID Detection"
    description = "Searches component text for reactor-specific channel ID patterns and creates Channel nodes"

    def __init__(self, ontology_manager=None):
        super().__init__()
        self.ontology_manager = ontology_manager
        self.reactor_patterns = self._load_reactor_patterns()

    def execute(self, graph_manager, parameters):
        changes = {
            "nodes_added": 0,
            "edges_added": 0,
            "method_instances_created": 0,
            "components_processed": 0,
            "channels_found": 0,
            "channel_patterns": {},
            "reactor_type_detected": None,
            "detection_confidence": 0.0
        }

        # Load reactor patterns from ontology if available
        if self.ontology_manager:
            self.reactor_patterns = self._load_reactor_patterns_from_ontology()

        # Detect reactor type first
        reactor_type, confidence = self._detect_reactor_type(graph_manager)
        changes["reactor_type_detected"] = reactor_type
        changes["detection_confidence"] = confidence

        print(f" Detected reactor type: {reactor_type} (confidence: {confidence:.2f})")

        # Find all component nodes that might contain channel IDs
        component_nodes = []
        for node_id, node_data in graph_manager.node_data.items():
            if node_data.get('type') == 'component':
                text = node_data.get('value', '').strip()
                if text:
                    component_nodes.append((node_id, node_data))

        print(f" Found {len(component_nodes)} component nodes with text to analyze")

        if not component_nodes:
            print(" No components with text found - skipping channel detection")
            return changes

        # Create method instance
        method_instance_id = self._create_method_instance(graph_manager, changes, reactor_type)

        for comp_id, comp_data in component_nodes:
            changes["components_processed"] += 1
            text = comp_data.get('value', '').strip()

            # Try to detect channel ID patterns using reactor-specific patterns
            channel_info = self._detect_channel_patterns(text, reactor_type)

            if channel_info:
                # Find existing channel node or create new one
                existing_channel = self._find_existing_channel(graph_manager, channel_info['channel_id'])

                if existing_channel:
                    channel_id = existing_channel
                    print(f" Found existing channel {channel_id} for '{channel_info['channel_id']}'")
                else:
                    # Create new Channel node
                    channel_id = self._get_next_numeric_id(graph_manager)

                    try:
                        graph_manager.add_node(
                            node_id=channel_id,
                            value=f"Channel {channel_info['channel_id']}",
                            type='channel',
                            hierarchy='reactor',
                            attributes={
                                'channel_id': channel_info['channel_id'],
                                'reactor_type': reactor_type,
                                'original_text': text,
                                'pattern': channel_info['pattern'],
                                'confidence': channel_info['confidence'],
                                'created': datetime.now().isoformat()
                            }
                        )
                        changes["nodes_added"] += 1
                        print(f" Created Channel node {channel_id} for '{channel_info['channel_id']}'")

                        # Connect method to the channel it created
                        self._connect_method_to_output(graph_manager, method_instance_id, channel_id, changes)

                    except KeyError as e:
                        print(f" Error creating Channel node: {e}")
                        continue

                # Connect method instance to the component it analyzed
                self._connect_method_to_input(graph_manager, method_instance_id, comp_id, changes)

                # Find parent entities and connect them to the channel
                parent_entities = self._find_parent_entities(graph_manager, comp_id)
                for parent_id, parent_type in parent_entities:
                    self._connect_entity_to_channel(graph_manager, parent_id, parent_type, channel_id, changes)

                changes["channels_found"] += 1
                pattern = channel_info['pattern']
                changes["channel_patterns"][pattern] = changes["channel_patterns"].get(pattern, 0) + 1

        print(f" ChannelDetectionMethod processed {changes['components_processed']} components, "
              f"found {changes['channels_found']} channels for {reactor_type} reactor")
        return changes

    def _load_reactor_patterns(self):
        """Load default reactor patterns"""
        return {
            'CANDU': {
                'regex': r'^[A-Y](?:0[1-9]|1[0-9]|2[0-5])$',
                'format': 'Letter (A-Y) + 2 digits (01-25)',
                'examples': ['A01', 'B12', 'Y25']
            },
            'AGR': {
                'regex': r'^(?:0[2-9]|[1-4][0-9]|5[2-9]|[6-9][0-9])(?:[0-9][0-9])$',
                'format': '4 digits: even numbers 02-48, 52-98',
                'examples': ['0878', '1270', '5264']
            }
        }

    def _load_reactor_patterns_from_ontology(self):
        """Load reactor patterns from ontology"""
        patterns = self._load_reactor_patterns()  # Start with defaults

        try:
            if hasattr(self.ontology_manager, 'node_types'):
                for class_name, class_data in self.ontology_manager.node_types.items():
                    if 'ChannelDetection' in class_name:
                        reactor_type = self._infer_reactor_type_from_method(class_name)
                        if reactor_type:
                            annotations = class_data.get('attributes', {})
                            if 'hasRegexPattern' in annotations:
                                patterns[reactor_type] = {
                                    'regex': annotations['hasRegexPattern'],
                                    'format': f'Ontology pattern for {reactor_type}',
                                    'examples': []
                                }
                                print(
                                    f" Loaded {reactor_type} pattern from ontology: {annotations['hasRegexPattern']}")
        except Exception as e:
            print(f" Error loading patterns from ontology: {e}")

        return patterns

    def _infer_reactor_type_from_method(self, method_name):
        """Infer reactor type from method name"""
        method_upper = method_name.upper()
        if 'CANDU' in method_upper:
            return 'CANDU'
        elif 'AGR' in method_upper:
            return 'AGR'
        return None

    def _detect_reactor_type(self, graph_manager):
        """Detect reactor type based on existing channel patterns"""
        channel_samples = []

        # Look for existing channel nodes
        for node_data in graph_manager.node_data.values():
            if node_data.get('type') == 'channel':
                channel_id = node_data.get('channel_id') or node_data.get('value', '')
                if channel_id:
                    channel_samples.append(channel_id)

        # Look for channel patterns in component text
        for node_data in graph_manager.node_data.values():
            if node_data.get('type') == 'component':
                text = node_data.get('value', '')
                for reactor_type, config in self.reactor_patterns.items():
                    matches = re.findall(config['regex'], text, re.IGNORECASE)
                    channel_samples.extend(matches)

        if not channel_samples:
            return None, 0.0

        # Test against each reactor pattern
        best_match = None
        best_confidence = 0.0

        for reactor_type, config in self.reactor_patterns.items():
            pattern = config['regex']
            matches = sum(1 for ch in channel_samples if re.match(pattern, ch, re.IGNORECASE))
            confidence = matches / len(channel_samples) if channel_samples else 0.0

            if confidence > best_confidence:
                best_confidence = confidence
                best_match = reactor_type

        return best_match, best_confidence

    def _detect_channel_patterns(self, text, reactor_type=None):
        """Detect channel ID patterns in text using reactor-specific patterns"""
        patterns_to_try = []

        # If reactor type is known, prioritize its pattern
        if reactor_type and reactor_type in self.reactor_patterns:
            config = self.reactor_patterns[reactor_type]
            patterns_to_try.append({
                'regex': config['regex'],
                'pattern': f'{reactor_type}_PATTERN',
                'confidence': 0.95,  # High confidence for reactor-specific match
                'parser': lambda m: m.group(0).upper()
            })

        # Add fallback patterns
        fallback_patterns = [
            {
                'regex': r'^([A-Z])(\d{2,3})$',
                'pattern': 'LETTER_DIGITS',
                'confidence': 0.8,
                'parser': lambda m: f"{m.group(1)}{m.group(2).zfill(2)}"
            },
            {
                'regex': r'^([A-Z])-(\d{2,3})$',
                'pattern': 'LETTER_DASH_DIGITS',
                'confidence': 0.8,
                'parser': lambda m: f"{m.group(1)}{m.group(2).zfill(2)}"
            },
            {
                'regex': r'^(\d{4})$',
                'pattern': 'FOUR_DIGITS',
                'confidence': 0.7,
                'parser': lambda m: m.group(0)
            }
        ]

        patterns_to_try.extend(fallback_patterns)

        for pattern_info in patterns_to_try:
            match = re.match(pattern_info['regex'], text)
            if match:
                try:
                    channel_id = pattern_info['parser'](match)
                    return {
                        'channel_id': channel_id,
                        'pattern': pattern_info['pattern'],
                        'confidence': pattern_info['confidence']
                    }
                except ValueError:
                    continue

        return None

    def _find_existing_channel(self, graph_manager, channel_id):
        """Find existing channel node with the same channel_id"""
        for node_id, node_data in graph_manager.node_data.items():
            if (node_data.get('type') == 'channel' and
                    node_data.get('channel_id') == channel_id):
                return node_id
        return None

    def _create_method_instance(self, graph_manager, changes, reactor_type):
        """Create a method instance node for execution traceability"""
        method_instance_id = self._get_next_numeric_id(graph_manager)

        try:
            graph_manager.add_node(
                node_id=method_instance_id,
                value=f"ChannelDetectionMethod execution ({reactor_type or 'Unknown'} reactor)",
                type='ChannelDetectionMethod',
                hierarchy='analysis',
                attributes={
                    'method_type': 'ChannelDetectionMethod',
                    'reactor_type': reactor_type,
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

    # Keep all the existing helper methods unchanged
    def _connect_method_to_output(self, graph_manager, method_instance_id, channel_id, changes):
        """Connect method instance to the channel it created"""
        try:
            graph_manager.add_edge(
                source=method_instance_id,
                target=channel_id,
                attributes={
                    'edge_type': 'produces',
                    'direction': 'out',
                    'provenance_type': 'generated_by',
                    'created': datetime.now().isoformat()
                }
            )
            changes["edges_added"] += 1

        except (KeyError, ValueError) as e:
            print(f" Error creating output edge: {e}")

    def _connect_method_to_input(self, graph_manager, method_instance_id, comp_id, changes):
        """Connect method instance to the component it analyzed"""
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

        except (KeyError, ValueError) as e:
            print(f" Error creating input edge: {e}")

    def _connect_entity_to_channel(self, graph_manager, parent_id, parent_type, channel_id, changes):
        """Connect parent entity to the channel"""
        try:
            graph_manager.add_edge(
                source=parent_id,
                target=channel_id,
                attributes={
                    'edge_type': 'has_channel',
                    'direction': 'out',
                    'provenance_type': 'contains_channel_reference',
                    'created': datetime.now().isoformat()
                }
            )
            changes["edges_added"] += 1

        except (KeyError, ValueError) as e:
            print(f"Error creating channel relationship edge: {e}")

    def _find_parent_entities(self, graph_manager, comp_id):
        """Find what entities (files, folders) this component belongs to"""
        parents = []

        for (source, target), edge_attrs in graph_manager.edge_data.items():
            if target == comp_id:
                source_node = graph_manager.node_data.get(source)
                if source_node:
                    source_type = source_node.get('type')
                    if source_type in ['file', 'folder', 'filename']:
                        parents.append((source, source_type))

                        if source_type == 'filename':
                            for (file_src, file_tgt), file_edge_attrs in graph_manager.edge_data.items():
                                if (file_tgt == source and
                                        file_edge_attrs.get('edge_type') == 'has_name'):
                                    file_node = graph_manager.node_data.get(file_src)
                                    if file_node and file_node.get('type') == 'file':
                                        parents.append((file_src, 'file'))

        return parents

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

        return str(next_id)