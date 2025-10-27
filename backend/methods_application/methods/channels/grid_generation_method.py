# backend/methods_application/methods/reactor/grid_generation_method.py
from backend.methods_application.method_implementation import MethodImplementation
import re
from datetime import datetime
import json


class GridGenerationMethod(MethodImplementation):
    method_id = "GridGenerationMethod"
    method_name = "Reactor Grid Layout Generation"
    description = "Generates complete grid layout with valid channel positions for reactor types"

    def __init__(self, ontology_manager=None):
        super().__init__()
        self.ontology_manager = ontology_manager
        self.grid_configurations = self._load_grid_configurations()

    def execute(self, graph_manager, parameters):
        changes = {
            "nodes_added": 0,
            "edges_added": 0,
            "method_instances_created": 0,
            "grid_layouts_created": 0,
            "valid_positions_generated": 0,
            "excluded_positions": 0,
            "reactor_type": None,
            "grid_dimensions": None
        }

        # Load grid configurations from ontology if available
        if self.ontology_manager:
            self.grid_configurations = self._load_grid_configurations_from_ontology()

        # Detect reactor type from existing graph data
        reactor_type = self._detect_reactor_type_from_graph(graph_manager)
        changes["reactor_type"] = reactor_type

        if not reactor_type:
            print(" No reactor type detected - cannot generate grid")
            return changes

        print(f" Generating grid layout for {reactor_type} reactor")

        # Get grid configuration for this reactor type
        grid_config = self.grid_configurations.get(reactor_type)
        if not grid_config:
            print(f" No grid configuration found for {reactor_type}")
            return changes

        # Create method instance
        method_instance_id = self._create_method_instance(graph_manager, changes, reactor_type, grid_config)

        # Generate grid layout
        grid_layout_id = self._create_grid_layout_node(graph_manager, reactor_type, grid_config, changes)

        # Generate all valid channel positions
        valid_positions = self._generate_valid_positions(reactor_type, grid_config)
        changes["valid_positions_generated"] = len(valid_positions)
        changes["excluded_positions"] = len(grid_config.get('excluded_channels', []))
        changes["grid_dimensions"] = f"{grid_config['grid_width']}x{grid_config['grid_height']}"

        # Create position nodes for each valid position
        position_nodes = self._create_position_nodes(graph_manager, valid_positions, reactor_type, grid_config, changes)

        # Connect grid layout to position nodes
        self._connect_grid_to_positions(graph_manager, grid_layout_id, position_nodes, changes)

        # Connect method to grid layout
        self._connect_method_to_output(graph_manager, method_instance_id, grid_layout_id, changes)

        # Link existing channels to their grid positions
        self._link_channels_to_grid_positions(graph_manager, position_nodes, reactor_type, changes)

        # Find reactor type node and connect grid to it
        reactor_type_node = self._find_reactor_type_node(graph_manager, reactor_type)
        if reactor_type_node:
            self._connect_reactor_type_to_grid(graph_manager, reactor_type_node, grid_layout_id, changes)

        print(
            f" Generated {grid_config['grid_width']}x{grid_config['grid_height']} grid with {len(valid_positions)} valid positions")
        return changes

    def _load_grid_configurations(self):
        """Load default grid configurations for different reactor types"""
        return {
            'CANDU': {
                'grid_width': 24,
                'grid_height': 25,
                'channel_pattern': r'^[A-Y](?:0[1-9]|1[0-9]|2[0-5])$',
                'excluded_channels': [
                    'A01', 'A02', 'A23', 'A24', 'Y01', 'Y02', 'Y23', 'Y24'
                ],
                'channel_rule': 'exclude_corners',
                'coordinate_system': 'letter_number',
                'row_letters': 'ABCDEFGHJKLMNPQRSTUVWXY',  # Excluding I and O
                'col_range': (1, 25),
                'description': 'CANDU reactor 24x25 grid with corner exclusions'
            },
            'AGR': {
                'grid_width': 49,
                'grid_height': 49,
                'channel_pattern': r'^(?:0[2-9]|[1-4][0-9]|5[2-9]|[6-9][0-9])(?:[0-9][0-9])$',
                'excluded_channels': ['50', '51', '99'],
                'channel_rule': 'even_numbers_only',
                'coordinate_system': 'four_digit',
                'row_range': (2, 98, 2),  # Start, end, step (even numbers only)
                'col_range': (2, 98, 2),
                'description': 'AGR reactor 49x49 grid with even-number constraints'
            }
        }

    def _load_grid_configurations_from_ontology(self):
        """Load grid configurations from ontology"""
        configs = self._load_grid_configurations()  # Start with defaults

        try:
            if hasattr(self.ontology_manager, 'node_types'):
                for class_name, class_data in self.ontology_manager.node_types.items():
                    if 'GridGeneration' in class_name:
                        reactor_type = self._infer_reactor_type_from_method(class_name)
                        if reactor_type and reactor_type in configs:
                            annotations = class_data.get('attributes', {})
                            config = configs[reactor_type]

                            # Update grid dimensions
                            if 'hasGridDimensions' in annotations:
                                dims = annotations['hasGridDimensions']
                                if ',' in str(dims):
                                    width, height = str(dims).split(',')
                                    config['grid_width'] = int(width.strip())
                                    config['grid_height'] = int(height.strip())

                            # Update individual dimensions
                            if 'hasGridWidth' in annotations:
                                config['grid_width'] = int(annotations['hasGridWidth'])
                            if 'hasGridHeight' in annotations:
                                config['grid_height'] = int(annotations['hasGridHeight'])

                            # Update excluded positions
                            if 'hasExcludedPositions' in annotations:
                                excluded = str(annotations['hasExcludedPositions'])
                                config['excluded_channels'] = [
                                    ch.strip() for ch in excluded.split(',') if ch.strip()
                                ]

                            # Update channel rule
                            if 'hasChannelRule' in annotations:
                                config['channel_rule'] = annotations['hasChannelRule']

                            # Update pattern
                            if 'hasRegexPattern' in annotations:
                                config['channel_pattern'] = annotations['hasRegexPattern']

                            print(f" Updated {reactor_type} grid config from ontology")

        except Exception as e:
            print(f" Error loading grid configs from ontology: {e}")

        return configs

    def _detect_reactor_type_from_graph(self, graph_manager):
        """Detect reactor type from existing graph nodes"""
        # Look for existing reactor type nodes
        for node_id, node_data in graph_manager.node_data.items():
            if node_data.get('type') == 'reactor_type':
                reactor_type = node_data.get('reactor_type_code') or node_data.get('attributes', {}).get(
                    'reactor_type_code')
                if reactor_type:
                    print(f" Found reactor type node: {reactor_type}")
                    return reactor_type

        # Look for reactor instance nodes
        for node_id, node_data in graph_manager.node_data.items():
            if node_data.get('type') == 'reactor':
                reactor_type = node_data.get('reactor_type') or node_data.get('attributes', {}).get('reactor_type')
                if reactor_type:
                    print(f" Found reactor instance with type: {reactor_type}")
                    return reactor_type

        # Try to infer from channel patterns
        channel_samples = []
        for node_data in graph_manager.node_data.values():
            if node_data.get('type') == 'channel':
                channel_id = node_data.get('channel_id') or node_data.get('value', '')
                if channel_id:
                    channel_samples.append(channel_id.upper())

        if channel_samples:
            return self._infer_reactor_type_from_channels(channel_samples)

        print(" Could not detect reactor type from graph")
        return None

    def _infer_reactor_type_from_channels(self, channel_samples):
        """Infer reactor type from channel samples"""
        for reactor_type, config in self.grid_configurations.items():
            pattern = config['channel_pattern']
            matches = sum(1 for ch in channel_samples if re.match(pattern, ch))
            confidence = matches / len(channel_samples) if channel_samples else 0.0

            if confidence > 0.7:  # High confidence threshold
                print(f" Inferred {reactor_type} from channel patterns (confidence: {confidence:.2f})")
                return reactor_type

        return None

    def _generate_valid_positions(self, reactor_type, grid_config):
        """Generate all valid channel positions for the reactor type"""
        valid_positions = []
        excluded = set(grid_config.get('excluded_channels', []))

        if reactor_type == 'CANDU':
            # CANDU uses letter-number system (A01-Y25)
            row_letters = grid_config.get('row_letters', 'ABCDEFGHJKLMNPQRSTUVWXY')
            col_start, col_end = grid_config.get('col_range', (1, 25))

            for row_idx, letter in enumerate(row_letters):
                for col_num in range(col_start, col_end + 1):
                    position = f"{letter}{col_num:02d}"
                    if position not in excluded:
                        valid_positions.append({
                            'channel_id': position,
                            'row_letter': letter,
                            'row_index': row_idx,
                            'col_number': col_num,
                            'col_index': col_num - 1,
                            'grid_x': col_num - 1,
                            'grid_y': row_idx
                        })

        elif reactor_type == 'AGR':
            # AGR uses four-digit system with even numbers
            width = grid_config['grid_width']
            height = grid_config['grid_height']

            # Generate positions based on even number constraints
            for row in range(2, 99, 2):  # Even rows from 02 to 98
                if row == 50:  # Skip 50 as per exclusions
                    continue
                for col in range(2, 99, 2):  # Even columns from 02 to 98
                    if col == 50:  # Skip 50 as per exclusions
                        continue

                    position = f"{row:02d}{col:02d}"
                    if position not in excluded:
                        # Calculate grid coordinates
                        grid_x = (col - 2) // 2
                        grid_y = (row - 2) // 2

                        # Skip if outside grid bounds
                        if grid_x < width and grid_y < height:
                            valid_positions.append({
                                'channel_id': position,
                                'row_number': row,
                                'col_number': col,
                                'grid_x': grid_x,
                                'grid_y': grid_y
                            })

        print(f" Generated {len(valid_positions)} valid positions for {reactor_type}")
        return valid_positions

    def _create_grid_layout_node(self, graph_manager, reactor_type, grid_config, changes):
        """Create main grid layout node"""
        grid_layout_id = self._get_next_numeric_id(graph_manager)

        graph_manager.add_node(
            node_id=grid_layout_id,
            value=f"{reactor_type} Grid Layout",
            type='grid_layout',
            hierarchy='reactor',
            attributes={
                'reactor_type': reactor_type,
                'grid_width': grid_config['grid_width'],
                'grid_height': grid_config['grid_height'],
                'total_positions': grid_config['grid_width'] * grid_config['grid_height'],
                'excluded_count': len(grid_config.get('excluded_channels', [])),
                'coordinate_system': grid_config.get('coordinate_system', 'unknown'),
                'channel_rule': grid_config.get('channel_rule', 'default'),
                'description': grid_config.get('description', f'{reactor_type} reactor grid'),
                'created': datetime.now().isoformat(),
                'generation_method': 'GridGenerationMethod'
            }
        )
        changes["nodes_added"] += 1
        changes["grid_layouts_created"] += 1
        print(f" Created grid layout node {grid_layout_id}")
        return grid_layout_id

    def _create_position_nodes(self, graph_manager, valid_positions, reactor_type, grid_config, changes):
        """Create nodes for each valid grid position"""
        position_nodes = {}

        for position_info in valid_positions:
            position_id = self._get_next_numeric_id(graph_manager)
            channel_id = position_info['channel_id']

            graph_manager.add_node(
                node_id=position_id,
                value=f"Position {channel_id}",
                type='grid_position',
                hierarchy='reactor',
                attributes={
                    'channel_id': channel_id,
                    'reactor_type': reactor_type,
                    'grid_x': position_info['grid_x'],
                    'grid_y': position_info['grid_y'],
                    'coordinate_system': grid_config.get('coordinate_system'),
                    'is_valid_position': True,
                    'created': datetime.now().isoformat(),
                    **{k: v for k, v in position_info.items() if k not in ['channel_id', 'grid_x', 'grid_y']}
                }
            )

            position_nodes[channel_id] = position_id
            changes["nodes_added"] += 1

        print(f" Created {len(position_nodes)} position nodes")
        return position_nodes

    def _connect_grid_to_positions(self, graph_manager, grid_layout_id, position_nodes, changes):
        """Connect grid layout to all position nodes"""
        for channel_id, position_node_id in position_nodes.items():
            try:
                graph_manager.add_edge(
                    source=grid_layout_id,
                    target=position_node_id,
                    attributes={
                        'edge_type': 'has_position',
                        'direction': 'out',
                        'position_id': channel_id,
                        'created': datetime.now().isoformat()
                    }
                )
                changes["edges_added"] += 1
            except (KeyError, ValueError) as e:
                print(f" Error connecting grid to position {channel_id}: {e}")

    def _link_channels_to_grid_positions(self, graph_manager, position_nodes, reactor_type, changes):
        """Link existing channel nodes to their corresponding grid positions"""
        linked_channels = 0

        for node_id, node_data in graph_manager.node_data.items():
            if node_data.get('type') == 'channel':
                channel_id = (node_data.get('channel_id') or
                              node_data.get('attributes', {}).get('channel_id') or
                              node_data.get('value', '')).upper()

                # Clean channel ID (remove "Channel " prefix if present)
                if channel_id.startswith('CHANNEL '):
                    channel_id = channel_id[8:]

                if channel_id in position_nodes:
                    position_node_id = position_nodes[channel_id]
                    try:
                        graph_manager.add_edge(
                            source=node_id,
                            target=position_node_id,
                            attributes={
                                'edge_type': 'has_grid_position',
                                'direction': 'out',
                                'created': datetime.now().isoformat()
                            }
                        )
                        changes["edges_added"] += 1
                        linked_channels += 1

                        # Update channel node with grid position info
                        if 'attributes' not in node_data:
                            node_data['attributes'] = {}

                        position_data = graph_manager.node_data[position_node_id]['attributes']
                        node_data['attributes'].update({
                            'grid_x': position_data.get('grid_x'),
                            'grid_y': position_data.get('grid_y'),
                            'grid_position_linked': True,
                            'grid_link_time': datetime.now().isoformat()
                        })

                    except (KeyError, ValueError) as e:
                        print(f" Error linking channel {channel_id} to grid position: {e}")

        print(f" Linked {linked_channels} existing channels to grid positions")

    def _find_reactor_type_node(self, graph_manager, reactor_type):
        """Find existing reactor type node"""
        for node_id, node_data in graph_manager.node_data.items():
            if (node_data.get('type') == 'reactor_type' and
                    (node_data.get('reactor_type_code') == reactor_type or
                     node_data.get('attributes', {}).get('reactor_type_code') == reactor_type)):
                return node_id
        return None

    def _connect_reactor_type_to_grid(self, graph_manager, reactor_type_node, grid_layout_id, changes):
        """Connect reactor type to its grid layout"""
        try:
            graph_manager.add_edge(
                source=reactor_type_node,
                target=grid_layout_id,
                attributes={
                    'edge_type': 'has_grid_layout',
                    'direction': 'out',
                    'created': datetime.now().isoformat()
                }
            )
            changes["edges_added"] += 1
            print(f" Connected reactor type to grid layout")
        except (KeyError, ValueError) as e:
            print(f" Error connecting reactor type to grid: {e}")

    def _create_method_instance(self, graph_manager, changes, reactor_type, grid_config):
        """Create method instance node"""
        method_instance_id = self._get_next_numeric_id(graph_manager)

        graph_manager.add_node(
            node_id=method_instance_id,
            value=f"GridGenerationMethod execution ({reactor_type})",
            type='GridGenerationMethod',
            hierarchy='analysis',
            attributes={
                'method_type': 'GridGenerationMethod',
                'reactor_type': reactor_type,
                'grid_dimensions': f"{grid_config['grid_width']}x{grid_config['grid_height']}",
                'execution_time': datetime.now().isoformat(),
                'method_id': self.method_id,
                'method_name': self.method_name,
                'grid_config': json.dumps(grid_config, default=str)
            }
        )
        changes["nodes_added"] += 1
        changes["method_instances_created"] += 1
        return method_instance_id

    def _connect_method_to_output(self, graph_manager, method_instance_id, grid_layout_id, changes):
        """Connect method to its output"""
        try:
            graph_manager.add_edge(
                source=method_instance_id,
                target=grid_layout_id,
                attributes={
                    'edge_type': 'produces',
                    'direction': 'out',
                    'provenance_type': 'generated_by',
                    'created': datetime.now().isoformat()
                }
            )
            changes["edges_added"] += 1
        except (KeyError, ValueError) as e:
            print(f" Error creating method output edge: {e}")

    def _infer_reactor_type_from_method(self, method_name):
        """Infer reactor type from method name"""
        method_upper = method_name.upper()
        if 'CANDU' in method_upper:
            return 'CANDU'
        elif 'AGR' in method_upper:
            return 'AGR'
        elif 'PWR' in method_upper:
            return 'PWR'
        elif 'BWR' in method_upper:
            return 'BWR'
        return None

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