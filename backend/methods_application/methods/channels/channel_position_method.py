# backend/methods_application/methods/reactor/channel_position_method.py
from backend.methods_application.method_implementation import MethodImplementation
import re
import math
from datetime import datetime


class ChannelPositionMethod(MethodImplementation):
    method_id = "ChannelPositionMethod"
    method_name = "Channel Grid Position Calculation"
    description = "Calculates grid position coordinates for individual channels and creates position mapping"

    def __init__(self, ontology_manager=None):
        super().__init__()
        self.ontology_manager = ontology_manager
        self.position_algorithms = self._load_position_algorithms()

    def execute(self, graph_manager, parameters):
        changes = {
            "nodes_added": 0,
            "edges_added": 0,
            "method_instances_created": 0,
            "channels_positioned": 0,
            "position_calculations_created": 0,
            "reactor_type": None,
            "positioning_algorithm": None,
            "coordinate_mappings": {}
        }

        # Load position algorithms from ontology if available
        if self.ontology_manager:
            self.position_algorithms = self._load_position_algorithms_from_ontology()

        # Detect reactor type and positioning algorithm
        reactor_type = self._detect_reactor_type_from_graph(graph_manager)
        changes["reactor_type"] = reactor_type

        if not reactor_type:
            print("DEBUG: No reactor type detected - using generic positioning")
            reactor_type = "generic"

        algorithm = self.position_algorithms.get(reactor_type, self.position_algorithms["generic"])
        changes["positioning_algorithm"] = algorithm["name"]

        print(f"DEBUG: Using {algorithm['name']} positioning algorithm for {reactor_type}")

        # Create method instance
        method_instance_id = self._create_method_instance(graph_manager, changes, reactor_type, algorithm)

        # Find all channel nodes that need positioning
        channel_nodes = self._find_channel_nodes(graph_manager)
        print(f"DEBUG: Found {len(channel_nodes)} channel nodes to position")

        # Find existing grid layout and positions
        grid_layout = self._find_grid_layout(graph_manager, reactor_type)
        existing_positions = self._find_existing_positions(graph_manager, reactor_type)

        for channel_id, channel_data in channel_nodes.items():
            channel_identifier = self._extract_channel_identifier(channel_data)

            if not channel_identifier:
                print(f"DEBUG: Could not extract channel identifier from node {channel_id}")
                continue

            # Calculate position using appropriate algorithm
            position_info = self._calculate_position(channel_identifier, reactor_type, algorithm)

            if position_info:
                # Create or update position calculation node
                position_calc_id = self._create_position_calculation_node(
                    graph_manager, channel_id, channel_identifier, position_info, reactor_type, changes
                )

                # Connect channel to position calculation
                self._connect_channel_to_position_calc(graph_manager, channel_id, position_calc_id, changes)

                # Connect method to position calculation
                self._connect_method_to_output(graph_manager, method_instance_id, position_calc_id, changes)

                # Update channel node with position information
                self._update_channel_with_position(graph_manager, channel_id, position_info, changes)

                # Link to existing grid position if available
                if channel_identifier in existing_positions:
                    self._link_to_existing_grid_position(
                        graph_manager, position_calc_id, existing_positions[channel_identifier], changes
                    )

                # Link to grid layout if available
                if grid_layout:
                    self._link_to_grid_layout(graph_manager, position_calc_id, grid_layout, changes)

                changes["channels_positioned"] += 1
                changes["coordinate_mappings"][channel_identifier] = position_info

        print(f"DEBUG: Successfully positioned {changes['channels_positioned']} channels")
        return changes

    def _load_position_algorithms(self):
        """Load default position calculation algorithms"""
        return {
            'CANDU': {
                'name': 'letter_to_row_number_to_col',
                'coordinate_system': 'letter_number',
                'description': 'Maps CANDU channels (A01-Y25) to grid coordinates',
                'row_mapping': 'ABCDEFGHJKLMNPQRSTUVWXY',  # 24 letters, excluding I and O
                'col_offset': 1,  # Columns start at 1
                'grid_width': 24,
                'grid_height': 25,
                'origin': 'top_left'
            },
            'AGR': {
                'name': 'four_digit_even_mapping',
                'coordinate_system': 'four_digit',
                'description': 'Maps AGR channels (0202-9898) to grid coordinates with even constraints',
                'row_start': 2,
                'row_step': 2,
                'col_start': 2,
                'col_step': 2,
                'grid_width': 49,
                'grid_height': 49,
                'origin': 'top_left'
            },
            'generic': {
                'name': 'pattern_based_mapping',
                'coordinate_system': 'auto_detect',
                'description': 'Generic pattern-based position mapping',
                'fallback': True
            }
        }

    def _load_position_algorithms_from_ontology(self):
        """Load position algorithms from ontology"""
        algorithms = self._load_position_algorithms()  # Start with defaults

        try:
            if hasattr(self.ontology_manager, 'node_types'):
                for class_name, class_data in self.ontology_manager.node_types.items():
                    if 'ChannelPosition' in class_name:
                        reactor_type = self._infer_reactor_type_from_method(class_name)
                        if reactor_type and reactor_type in algorithms:
                            annotations = class_data.get('attributes', {})

                            # Update position mapping algorithm
                            if 'hasPositionMapping' in annotations:
                                algorithms[reactor_type]['name'] = annotations['hasPositionMapping']

                            # Update coordinate system
                            if 'coordinate_system' in annotations:
                                algorithms[reactor_type]['coordinate_system'] = annotations['coordinate_system']

                            print(f"DEBUG: Updated {reactor_type} position algorithm from ontology")

        except Exception as e:
            print(f"DEBUG: Error loading position algorithms from ontology: {e}")

        return algorithms

    def _detect_reactor_type_from_graph(self, graph_manager):
        """Detect reactor type from existing graph nodes"""
        # Look for reactor type nodes
        for node_data in graph_manager.node_data.values():
            if node_data.get('type') == 'reactor_type':
                reactor_type = (node_data.get('reactor_type_code') or
                                node_data.get('attributes', {}).get('reactor_type_code'))
                if reactor_type:
                    return reactor_type

        # Look for grid layout nodes
        for node_data in graph_manager.node_data.values():
            if node_data.get('type') == 'grid_layout':
                reactor_type = node_data.get('attributes', {}).get('reactor_type')
                if reactor_type:
                    return reactor_type

        # Try to infer from channel patterns
        channel_samples = []
        for node_data in graph_manager.node_data.values():
            if node_data.get('type') == 'channel':
                channel_id = self._extract_channel_identifier(node_data)
                if channel_id:
                    channel_samples.append(channel_id)

        return self._infer_reactor_type_from_channels(channel_samples)

    def _infer_reactor_type_from_channels(self, channel_samples):
        """Infer reactor type from channel patterns"""
        if not channel_samples:
            return None

        # Test CANDU pattern (letter + 2-3 digits)
        candu_pattern = r'^[A-Y]\d{2,3}$'
        candu_matches = sum(1 for ch in channel_samples if re.match(candu_pattern, ch))

        # Test AGR pattern (4 digits)
        agr_pattern = r'^\d{4}$'
        agr_matches = sum(1 for ch in channel_samples if re.match(agr_pattern, ch))

        total = len(channel_samples)
        candu_confidence = candu_matches / total if total > 0 else 0
        agr_confidence = agr_matches / total if total > 0 else 0

        if candu_confidence > 0.7:
            return 'CANDU'
        elif agr_confidence > 0.7:
            return 'AGR'

        return None

    def _find_channel_nodes(self, graph_manager):
        """Find all channel nodes in the graph"""
        channel_nodes = {}
        for node_id, node_data in graph_manager.node_data.items():
            if node_data.get('type') == 'channel':
                channel_nodes[node_id] = node_data
        return channel_nodes

    def _find_grid_layout(self, graph_manager, reactor_type):
        """Find grid layout node for the reactor type"""
        for node_id, node_data in graph_manager.node_data.items():
            if (node_data.get('type') == 'grid_layout' and
                    node_data.get('attributes', {}).get('reactor_type') == reactor_type):
                return node_id
        return None

    def _find_existing_positions(self, graph_manager, reactor_type):
        """Find existing grid position nodes"""
        positions = {}
        for node_id, node_data in graph_manager.node_data.items():
            if (node_data.get('type') == 'grid_position' and
                    node_data.get('attributes', {}).get('reactor_type') == reactor_type):
                channel_id = node_data.get('attributes', {}).get('channel_id')
                if channel_id:
                    positions[channel_id] = node_id
        return positions

    def _extract_channel_identifier(self, channel_data):
        """Extract channel identifier from channel node data"""
        # Try various fields that might contain the channel ID
        candidates = [
            channel_data.get('channel_id'),
            channel_data.get('attributes', {}).get('channel_id'),
            channel_data.get('value', ''),
            channel_data.get('name', '')
        ]

        for candidate in candidates:
            if candidate:
                # Clean the identifier
                clean_id = str(candidate).upper().strip()

                # Remove "CHANNEL " prefix if present
                if clean_id.startswith('CHANNEL '):
                    clean_id = clean_id[8:]

                # Validate the identifier
                if self._is_valid_channel_identifier(clean_id):
                    return clean_id

        return None

    def _is_valid_channel_identifier(self, identifier):
        """Check if identifier looks like a valid channel ID"""
        if not identifier:
            return False

        # CANDU pattern (A01, B12, etc.)
        if re.match(r'^[A-Y]\d{2,3}$', identifier):
            return True

        # AGR pattern (0202, 1234, etc.)
        if re.match(r'^\d{4}$', identifier):
            return True

        # Generic patterns
        if re.match(r'^[A-Z]-?\d{1,4}$', identifier):
            return True

        return False

    def _calculate_position(self, channel_identifier, reactor_type, algorithm):
        """Calculate grid position for a channel using the appropriate algorithm"""
        try:
            if reactor_type == 'CANDU':
                return self._calculate_candu_position(channel_identifier, algorithm)
            elif reactor_type == 'AGR':
                return self._calculate_agr_position(channel_identifier, algorithm)
            else:
                return self._calculate_generic_position(channel_identifier, algorithm)
        except Exception as e:
            print(f"DEBUG: Error calculating position for {channel_identifier}: {e}")
            return None

    def _calculate_candu_position(self, channel_id, algorithm):
        """Calculate position for CANDU channels (A01-Y25)"""
        if not re.match(r'^[A-Y]\d{2,3}$', channel_id):
            return None

        letter = channel_id[0]
        number_str = channel_id[1:]

        try:
            number = int(number_str)
        except ValueError:
            return None

        # Get row mapping
        row_letters = algorithm.get('row_mapping', 'ABCDEFGHJKLMNPQRSTUVWXY')

        if letter not in row_letters:
            return None

        # Calculate grid coordinates
        row_index = row_letters.index(letter)
        col_index = number - algorithm.get('col_offset', 1)

        # Validate bounds
        grid_width = algorithm.get('grid_width', 24)
        grid_height = algorithm.get('grid_height', 25)

        if not (0 <= row_index < grid_height and 0 <= col_index < grid_width):
            return None

        # Calculate additional position information
        center_x = grid_width / 2
        center_y = grid_height / 2
        distance_from_center = math.sqrt((col_index - center_x) ** 2 + (row_index - center_y) ** 2)

        # Determine sector (core regions)
        sector = self._determine_candu_sector(row_index, col_index, grid_width, grid_height)

        return {
            'channel_id': channel_id,
            'grid_x': col_index,
            'grid_y': row_index,
            'row_letter': letter,
            'row_index': row_index,
            'col_number': number,
            'col_index': col_index,
            'distance_from_center': round(distance_from_center, 2),
            'sector': sector,
            'coordinate_system': 'letter_number',
            'algorithm_used': algorithm['name'],
            'is_valid': True
        }

    def _calculate_agr_position(self, channel_id, algorithm):
        """Calculate position for AGR channels (four-digit format)"""
        if not re.match(r'^\d{4}$', channel_id):
            return None

        try:
            row_part = int(channel_id[:2])
            col_part = int(channel_id[2:])
        except ValueError:
            return None

        # Calculate grid coordinates based on even number mapping
        row_start = algorithm.get('row_start', 2)
        row_step = algorithm.get('row_step', 2)
        col_start = algorithm.get('col_start', 2)
        col_step = algorithm.get('col_step', 2)

        # Map to grid coordinates
        if row_part < row_start or (row_part - row_start) % row_step != 0:
            return None
        if col_part < col_start or (col_part - col_start) % col_step != 0:
            return None

        grid_x = (col_part - col_start) // col_step
        grid_y = (row_part - row_start) // row_step

        # Validate bounds
        grid_width = algorithm.get('grid_width', 49)
        grid_height = algorithm.get('grid_height', 49)

        if not (0 <= grid_x < grid_width and 0 <= grid_y < grid_height):
            return None

        # Calculate additional position information
        center_x = grid_width / 2
        center_y = grid_height / 2
        distance_from_center = math.sqrt((grid_x - center_x) ** 2 + (grid_y - center_y) ** 2)

        # Determine assembly and sub-channel
        assembly_x = grid_x // 7  # Assuming 7x7 sub-grid per assembly
        assembly_y = grid_y // 7
        sub_channel_x = grid_x % 7
        sub_channel_y = grid_y % 7

        return {
            'channel_id': channel_id,
            'grid_x': grid_x,
            'grid_y': grid_y,
            'row_number': row_part,
            'col_number': col_part,
            'assembly_x': assembly_x,
            'assembly_y': assembly_y,
            'sub_channel_x': sub_channel_x,
            'sub_channel_y': sub_channel_y,
            'distance_from_center': round(distance_from_center, 2),
            'coordinate_system': 'four_digit',
            'algorithm_used': algorithm['name'],
            'is_valid': True
        }

    def _calculate_generic_position(self, channel_id, algorithm):
        """Generic position calculation for unknown reactor types"""
        # Try to extract numeric parts
        numbers = re.findall(r'\d+', channel_id)
        letters = re.findall(r'[A-Z]+', channel_id)

        if not numbers:
            return None

        try:
            if letters and numbers:
                # Letter-number format
                letter = letters[0][0] if letters[0] else 'A'
                number = int(numbers[0])

                # Simple mapping
                row_index = ord(letter) - ord('A')
                col_index = number - 1

                return {
                    'channel_id': channel_id,
                    'grid_x': col_index,
                    'grid_y': row_index,
                    'row_letter': letter,
                    'col_number': number,
                    'coordinate_system': 'generic',
                    'algorithm_used': algorithm['name'],
                    'is_valid': True
                }
            else:
                # Pure numeric format
                number = int(numbers[0])

                # Assume square grid
                grid_size = int(math.sqrt(number * 2))  # Rough estimate
                grid_x = number % grid_size
                grid_y = number // grid_size

                return {
                    'channel_id': channel_id,
                    'grid_x': grid_x,
                    'grid_y': grid_y,
                    'position_number': number,
                    'coordinate_system': 'numeric',
                    'algorithm_used': algorithm['name'],
                    'is_valid': True
                }
        except ValueError:
            return None

    def _determine_candu_sector(self, row_index, col_index, grid_width, grid_height):
        """Determine reactor sector for CANDU reactors"""
        center_x = grid_width / 2
        center_y = grid_height / 2

        # Simple quadrant system
        if col_index < center_x and row_index < center_y:
            return 'NW'
        elif col_index >= center_x and row_index < center_y:
            return 'NE'
        elif col_index < center_x and row_index >= center_y:
            return 'SW'
        else:
            return 'SE'

    def _create_position_calculation_node(self, graph_manager, channel_node_id, channel_id, position_info, reactor_type,
                                          changes):
        """Create a position calculation node"""
        position_calc_id = self._get_next_numeric_id(graph_manager)

        graph_manager.add_node(
            node_id=position_calc_id,
            value=f"Position Calculation for {channel_id}",
            type='position_calculation',
            hierarchy='reactor',
            attributes={
                'channel_id': channel_id,
                'reactor_type': reactor_type,
                'original_channel_node': channel_node_id,
                'calculation_method': 'ChannelPositionMethod',
                'created': datetime.now().isoformat(),
                **position_info
            }
        )
        changes["nodes_added"] += 1
        changes["position_calculations_created"] += 1
        return position_calc_id

    def _update_channel_with_position(self, graph_manager, channel_id, position_info, changes):
        """Update channel node with position information"""
        channel_data = graph_manager.node_data[channel_id]

        if 'attributes' not in channel_data:
            channel_data['attributes'] = {}

        # Add position information to channel attributes
        channel_data['attributes'].update({
            'grid_x': position_info.get('grid_x'),
            'grid_y': position_info.get('grid_y'),
            'distance_from_center': position_info.get('distance_from_center'),
            'coordinate_system': position_info.get('coordinate_system'),
            'position_calculated': True,
            'position_calculation_time': datetime.now().isoformat()
        })

        # Add reactor-specific position info
        if 'sector' in position_info:
            channel_data['attributes']['sector'] = position_info['sector']
        if 'assembly_x' in position_info:
            channel_data['attributes']['assembly_x'] = position_info['assembly_x']
            channel_data['attributes']['assembly_y'] = position_info['assembly_y']

    def _connect_channel_to_position_calc(self, graph_manager, channel_id, position_calc_id, changes):
        """Connect channel to its position calculation"""
        try:
            graph_manager.add_edge(
                source=channel_id,
                target=position_calc_id,
                attributes={
                    'edge_type': 'has_position_calculation',
                    'direction': 'out',
                    'created': datetime.now().isoformat()
                }
            )
            changes["edges_added"] += 1
        except (KeyError, ValueError) as e:
            print(f"DEBUG: Error connecting channel to position calculation: {e}")

    def _link_to_existing_grid_position(self, graph_manager, position_calc_id, grid_position_id, changes):
        """Link position calculation to existing grid position"""
        try:
            graph_manager.add_edge(
                source=position_calc_id,
                target=grid_position_id,
                attributes={
                    'edge_type': 'corresponds_to_grid_position',
                    'direction': 'out',
                    'created': datetime.now().isoformat()
                }
            )
            changes["edges_added"] += 1
        except (KeyError, ValueError) as e:
            print(f"DEBUG: Error linking to grid position: {e}")

    def _link_to_grid_layout(self, graph_manager, position_calc_id, grid_layout_id, changes):
        """Link position calculation to grid layout"""
        try:
            graph_manager.add_edge(
                source=position_calc_id,
                target=grid_layout_id,
                attributes={
                    'edge_type': 'part_of_grid_layout',
                    'direction': 'out',
                    'created': datetime.now().isoformat()
                }
            )
            changes["edges_added"] += 1
        except (KeyError, ValueError) as e:
            print(f"DEBUG: Error linking to grid layout: {e}")

    def _create_method_instance(self, graph_manager, changes, reactor_type, algorithm):
        """Create method instance node"""
        method_instance_id = self._get_next_numeric_id(graph_manager)

        graph_manager.add_node(
            node_id=method_instance_id,
            value=f"ChannelPositionMethod execution ({reactor_type})",
            type='ChannelPositionMethod',
            hierarchy='analysis',
            attributes={
                'method_type': 'ChannelPositionMethod',
                'reactor_type': reactor_type,
                'positioning_algorithm': algorithm['name'],
                'coordinate_system': algorithm.get('coordinate_system', 'unknown'),
                'execution_time': datetime.now().isoformat(),
                'method_id': self.method_id,
                'method_name': self.method_name
            }
        )
        changes["nodes_added"] += 1
        changes["method_instances_created"] += 1
        return method_instance_id

    def _connect_method_to_output(self, graph_manager, method_instance_id, position_calc_id, changes):
        """Connect method to its output"""
        try:
            graph_manager.add_edge(
                source=method_instance_id,
                target=position_calc_id,
                attributes={
                    'edge_type': 'produces',
                    'direction': 'out',
                    'provenance_type': 'generated_by',
                    'created': datetime.now().isoformat()
                }
            )
            changes["edges_added"] += 1
        except (KeyError, ValueError) as e:
            print(f"DEBUG: Error creating method output edge: {e}")

    def _infer_reactor_type_from_method(self, method_name):
        """Infer reactor type from method name"""
        method_upper = method_name.upper()
        if 'CANDU' in method_upper:
            return 'CANDU'
        elif 'AGR' in method_upper:
            return 'AGR'
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