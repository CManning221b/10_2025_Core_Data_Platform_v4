# backend/services/channel_data_service.py
from typing import Dict, List, Optional, Any
from datetime import datetime
import re

ANONYMIZATION_CONFIG = {
    'enabled': True,  # Set to True to enable anonymization
    'channel_mapping': {},
    'reactor_type_mapping': {
        'CANDU': 'REACTOR_A',
        'AGR': 'REACTOR_B',
        'PWR': 'REACTOR_C',
        'BWR': 'REACTOR_D',
    },
    'grid_dimensions_mapping': {
        (24, 25): (28, 28),
        (49, 49): (45, 50),
    },
}


class ChannelDataService:
    """Service for extracting and analyzing channel data from graph structures using ontology-driven patterns"""

    def __init__(self, ontology_manager=None):
        self.ontology_manager = ontology_manager
        self.reactor_config = self._load_reactor_config()
        self.detected_reactor_type = None

    def extract_channel_data_from_graph(self, graph_manager) -> Dict[str, Any]:
        """
        Extract comprehensive channel data from the graph for CorePlot visualization
        Uses ontology to determine reactor type and apply appropriate patterns

        Returns:
            Dict mapping channel names to their data, plus reactor configuration
        """
        print("DEBUG: Starting ontology-driven channel data extraction from graph")

        # Detect reactor type first
        reactor_type, confidence = self.detect_reactor_type(graph_manager)
        self.detected_reactor_type = reactor_type

        print(f"DEBUG: Detected reactor type: {reactor_type} (confidence: {confidence:.2f})")

        # Find all channel-related nodes
        channel_nodes = self._find_channel_nodes(graph_manager)
        timestamp_nodes = self._find_timestamp_nodes(graph_manager)
        folder_nodes = self._find_folder_nodes(graph_manager)

        print(
            f"DEBUG: Found {len(channel_nodes)} channel nodes, {len(timestamp_nodes)} timestamp nodes, {len(folder_nodes)} folder nodes")

        # Build comprehensive channel data
        channel_data = {}

        # Process explicit channel nodes
        for node_id, node_data in channel_nodes.items():
            channel_name = self._extract_channel_name(node_data, reactor_type)
            if channel_name and self._is_valid_channel_name(channel_name, reactor_type):
                channel_info = self._build_channel_info(
                    node_id, node_data, graph_manager,
                    timestamp_nodes, folder_nodes
                )
                channel_data[channel_name] = channel_info
                print(
                    f"DEBUG: Added explicit channel {channel_name} with {channel_info['measurement_count']} measurements")

        # Look for channel references in other node types
        implicit_channels = self._find_implicit_channels(graph_manager, timestamp_nodes, folder_nodes, reactor_type)
        for channel_name, channel_info in implicit_channels.items():
            if channel_name not in channel_data:
                channel_data[channel_name] = channel_info
                print(
                    f"DEBUG: Added implicit channel {channel_name} with {channel_info['measurement_count']} measurements")

        print(f"DEBUG: Total channels found: {len(channel_data)}")

        # Apply anonymization if enabled
        if ANONYMIZATION_CONFIG['enabled']:
            channel_data = self._anonymize_channel_keys(channel_data)
            print("DEBUG: Channel names anonymized (A01 → 0101, etc.)")

        # Build result
        result = {
            'channels': channel_data,
            'reactor_type': reactor_type,
            'reactor_config': self._get_reactor_config(reactor_type),
            'detection_confidence': confidence
        }

        return result

    def _anonymize_channel_keys(self, channel_data: Dict) -> Dict:
        """
        Rename channel dict keys from A01 → 0101, B02 → 0202, etc.
        Everything else stays the same—exclusions, data, etc.
        """
        anonymized = {}
        for real_name, info in channel_data.items():
            anon_name = self._anonymize_channel_name(real_name)
            anonymized[anon_name] = info
            print(f"DEBUG: Renamed channel: {real_name} → {anon_name}")
        return anonymized

    def _anonymize_channel_name(self, channel_name: str) -> str:
        """A01 → 0101, B02 → 0202, Y25 → 2525, etc."""
        if channel_name in ANONYMIZATION_CONFIG['channel_mapping']:
            return ANONYMIZATION_CONFIG['channel_mapping'][channel_name]

        if len(channel_name) >= 3 and channel_name[0].isalpha():
            letter_pos = ord(channel_name[0].upper()) - ord('A') + 1
            numbers = channel_name[1:]
            return f"{letter_pos:02d}{numbers}"

        return channel_name

    def _load_reactor_config(self) -> Dict[str, Dict]:
        """Load reactor configuration from ontology or use defaults"""
        if not self.ontology_manager:
            return self._get_default_configs()

        reactor_configs = {}

        # Query ontology for reactor layout methods and channel patterns
        try:
            # Look for method classes that contain grid generation info
            if hasattr(self.ontology_manager, 'node_types'):
                for class_name, class_data in self.ontology_manager.node_types.items():
                    if 'GridGenerationMethod' in class_name or 'GridGeneration' in class_name:
                        config = self._extract_reactor_config_from_ontology(class_name, class_data)
                        if config:
                            reactor_type = self._infer_reactor_type_from_method(class_name)
                            if reactor_type:
                                reactor_configs[reactor_type] = config
                                print(f"DEBUG: Loaded {reactor_type} config from ontology: {config}")
        except Exception as e:
            print(f"DEBUG: Error loading ontology config: {e}")

        # Fall back to defaults if no ontology configs found
        if not reactor_configs:
            reactor_configs = self._get_default_configs()
            print("DEBUG: Using default reactor configurations")

        return reactor_configs

    def _extract_reactor_config_from_ontology(self, method_name: str, method_data: Dict) -> Optional[Dict]:
        """Extract reactor configuration from ontology method"""
        config = {}

        # Get method properties from ontology annotations
        annotations = method_data.get('attributes', {})

        # Try to get regex pattern from the ontology
        regex_sources = ['hasRegexPattern', 'core:hasRegexPattern', 'regex_pattern']
        for source in regex_sources:
            if source in annotations:
                config['regex_pattern'] = annotations[source]
                break

        # Try to get grid dimensions
        if 'hasGridDimensions' in annotations:
            dimensions = annotations['hasGridDimensions']
            if ',' in str(dimensions):
                width, height = str(dimensions).split(',')
                config['grid_width'] = int(width.strip())
                config['grid_height'] = int(height.strip())

        # Individual dimension properties
        if 'hasGridWidth' in annotations:
            config['grid_width'] = int(annotations['hasGridWidth'])
        if 'hasGridHeight' in annotations:
            config['grid_height'] = int(annotations['hasGridHeight'])

        # Excluded positions/channels
        excluded_sources = ['hasExcludedPositions', 'hasExcludedChannels', 'excluded_positions']
        for source in excluded_sources:
            if source in annotations:
                excluded = str(annotations[source])
                config['excluded_channels'] = [ch.strip() for ch in excluded.split(',') if ch.strip()]
                break

        # Channel rules
        if 'hasChannelRule' in annotations:
            config['channel_rule'] = annotations['hasChannelRule']

        return config if config else None

    def _infer_reactor_type_from_method(self, method_name: str) -> Optional[str]:
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

    def _get_default_configs(self) -> Dict[str, Dict]:
        """Fallback reactor configurations"""
        return {
            'CANDU': {
                'regex_pattern': r'^[A-Y](?:0[1-9]|1[0-9]|2[0-5])$',
                'grid_width': 24,
                'grid_height': 25,
                'excluded_channels': ['A01', 'A02', 'A23', 'A24', 'Y01', 'Y02', 'Y23', 'Y24'],
                'channel_rule': 'exclude_corners'
            },
            'AGR': {
                'regex_pattern': r'^(?:0[2-9]|[1-4][0-9]|5[2-9]|[6-9][0-9])(?:[0-9][0-9])$',
                'grid_width': 49,
                'grid_height': 49,
                'excluded_channels': ['50', '51', '99'],
                'channel_rule': 'even_numbers_only'
            }
        }

    def _get_reactor_config(self, reactor_type: str) -> Dict:
        """Get configuration for a specific reactor type"""
        if reactor_type and reactor_type in self.reactor_config:
            return self.reactor_config[reactor_type]

        # Return CANDU as default
        return self.reactor_config.get('CANDU', {
            'regex_pattern': r'^[A-Z]\d{2}$',
            'grid_width': 24,
            'grid_height': 25,
            'excluded_channels': [],
            'channel_rule': 'default'
        })

    def detect_reactor_type(self, graph_manager) -> tuple[Optional[str], float]:
        """Detect reactor type based on channel patterns in the graph"""
        channel_samples = []

        # Collect channel name samples from explicit channel nodes
        for node_id, node_data in graph_manager.node_data.items():
            if node_data.get('type', '').lower() == 'channel':
                channel_name = self._extract_channel_name(node_data)
                if channel_name:
                    channel_samples.append(channel_name)

        # Also look for channel patterns in text values
        for node_data in graph_manager.node_data.values():
            for value in node_data.values():
                if isinstance(value, str):
                    # Try each reactor pattern to find channels
                    for reactor_type, config in self.reactor_config.items():
                        pattern = config.get('regex_pattern', r'^[A-Z]\d{2}$')
                        matches = re.findall(pattern, value.upper())
                        channel_samples.extend(matches)

        if not channel_samples:
            print("DEBUG: No channel samples found for reactor type detection")
            return None, 0.0

        print(f"DEBUG: Testing {len(channel_samples)} channel samples: {channel_samples[:10]}...")

        # Test against each reactor type's pattern
        reactor_scores = {}
        for reactor_type, config in self.reactor_config.items():
            pattern = config.get('regex_pattern', r'^[A-Z]\d{2}$')
            matches = sum(1 for ch in channel_samples if re.match(pattern, ch, re.IGNORECASE))
            if matches > 0:
                confidence = matches / len(channel_samples)
                reactor_scores[reactor_type] = confidence
                print(
                    f"DEBUG: {reactor_type} pattern matched {matches}/{len(channel_samples)} samples (confidence: {confidence:.2f})")

        # Return the reactor type with the highest match rate
        if reactor_scores:
            best_match = max(reactor_scores.items(), key=lambda x: x[1])
            return best_match[0], best_match[1]

        print("DEBUG: No reactor type patterns matched the samples")
        return None, 0.0

    def _extract_channel_name(self, node_data: Dict, reactor_type: str = None) -> Optional[str]:
        """Extract channel name from node data"""
        candidates = ['value', 'name', 'channel_name', 'channel_id', 'label']

        for candidate in candidates:
            value = node_data.get(candidate, '')
            if isinstance(value, str) and self._is_valid_channel_name(value, reactor_type):
                return value.upper()

        return None

    def _is_valid_channel_name(self, name: str, reactor_type: str = None) -> bool:
        """Check if a string is a valid channel name for the given reactor type"""
        if not isinstance(name, str):
            return False

        name = name.upper()

        # If reactor type is specified, use its specific pattern
        if reactor_type and reactor_type in self.reactor_config:
            config = self.reactor_config[reactor_type]
            pattern = config.get('regex_pattern', r'^[A-Z]\d{2}$')
            return bool(re.match(pattern, name))

        # Try all known reactor patterns
        for config in self.reactor_config.values():
            pattern = config.get('regex_pattern', r'^[A-Z]\d{2}$')
            if re.match(pattern, name):
                return True

        # Fallback to generic pattern
        return bool(re.match(r'^[A-Z]\d{2,4}$', name))

    def _find_implicit_channels(self, graph_manager, timestamp_nodes: Dict, folder_nodes: Dict,
                                reactor_type: str = None) -> Dict[str, Dict]:
        """Find channels referenced in folder names, file paths, etc."""
        implicit_channels = {}

        # Look for channel patterns in folder names
        for folder_id, folder_data in folder_nodes.items():
            folder_path = folder_data.get('value', '')
            channels_in_folder = self._extract_channels_from_text(folder_path, reactor_type)

            for channel_name in channels_in_folder:
                if channel_name not in implicit_channels:
                    implicit_channels[channel_name] = {
                        'node_id': f"implicit_{channel_name}",
                        'node_type': 'implicit_channel',
                        'measurement_count': 0,
                        'temporal_data': [],
                        'parent_folder': self._extract_folder_name(folder_data),
                        'last_measurement': None,
                        'status': 'referenced',
                        'raw_node_data': {'inferred_from': folder_path}
                    }
                    print(f"DEBUG: Found implicit channel {channel_name} in folder: {folder_path}")

        # Look for channel patterns in other node values
        for node_id, node_data in graph_manager.node_data.items():
            if node_data.get('type') not in ['channel', 'Timestamp', 'folder']:
                text_value = str(node_data.get('value', ''))
                channels_in_text = self._extract_channels_from_text(text_value, reactor_type)

                for channel_name in channels_in_text:
                    if channel_name not in implicit_channels:
                        implicit_channels[channel_name] = {
                            'node_id': f"implicit_{channel_name}",
                            'node_type': 'implicit_channel',
                            'measurement_count': 0,
                            'temporal_data': [],
                            'parent_folder': None,
                            'last_measurement': None,
                            'status': 'referenced',
                            'raw_node_data': {'inferred_from': text_value}
                        }
                        print(f"DEBUG: Found implicit channel {channel_name} in node {node_id}: {text_value}")

        return implicit_channels

    def _extract_channels_from_text(self, text: str, reactor_type: str = None) -> List[str]:
        """Extract channel names from text using reactor-specific patterns"""
        if not isinstance(text, str):
            return []

        valid_channels = []

        if reactor_type and reactor_type in self.reactor_config:
            # Use reactor-specific pattern
            config = self.reactor_config[reactor_type]
            pattern = config.get('regex_pattern', r'^[A-Z]\d{2}$')

            # Modify pattern for text search (remove anchors)
            search_pattern = pattern.replace('^', r'\b').replace('$', r'\b')
            matches = re.findall(search_pattern, text.upper())

            for match in matches:
                if self._is_valid_channel_name(match, reactor_type):
                    valid_channels.append(match)
        else:
            # Try all reactor patterns
            for config in self.reactor_config.values():
                pattern = config.get('regex_pattern', r'^[A-Z]\d{2}$')
                search_pattern = pattern.replace('^', r'\b').replace('$', r'\b')
                matches = re.findall(search_pattern, text.upper())

                for match in matches:
                    if self._is_valid_channel_name(match) and match not in valid_channels:
                        valid_channels.append(match)

        return list(set(valid_channels))  # Remove duplicates

    # Keep all the existing helper methods unchanged
    def _find_channel_nodes(self, graph_manager) -> Dict[str, Dict]:
        """Find all nodes explicitly typed as 'channel'"""
        channel_nodes = {}
        for node_id, node_data in graph_manager.node_data.items():
            if node_data.get('type', '').lower() == 'channel':
                channel_nodes[node_id] = node_data
        return channel_nodes

    def _find_timestamp_nodes(self, graph_manager) -> Dict[str, Dict]:
        """Find all timestamp nodes"""
        timestamp_nodes = {}
        for node_id, node_data in graph_manager.node_data.items():
            if node_data.get('type') == 'Timestamp':
                timestamp_nodes[node_id] = node_data
        return timestamp_nodes

    def _find_folder_nodes(self, graph_manager) -> Dict[str, Dict]:
        """Find all folder nodes"""
        folder_nodes = {}
        for node_id, node_data in graph_manager.node_data.items():
            if node_data.get('type', '').lower() == 'folder':
                folder_nodes[node_id] = node_data
        return folder_nodes

    def _build_channel_info(self, node_id: str, node_data: Dict, graph_manager,
                            timestamp_nodes: Dict, folder_nodes: Dict) -> Dict:
        """Build comprehensive information for a channel"""
        # Find associated timestamps
        timestamps = self._find_channel_timestamps(node_id, graph_manager, timestamp_nodes)

        # Find parent folder
        parent_folder = self._find_channel_parent_folder(node_id, graph_manager, folder_nodes)

        # Extract temporal data
        temporal_data = []
        for ts_id, ts_data in timestamps.items():
            temporal_info = self._extract_temporal_info_from_timestamp(ts_data)
            if temporal_info:
                temporal_data.append({
                    'timestamp_id': ts_id,
                    'datetime': temporal_info['datetime_str'],
                    'components': temporal_info['components']
                })

        # Sort temporal data by datetime
        temporal_data.sort(key=lambda x: x['datetime'])

        # Determine status
        measurement_count = len(temporal_data)
        status = 'active' if measurement_count > 0 else 'inactive'

        return {
            'node_id': node_id,
            'node_type': node_data.get('type', 'channel'),
            'measurement_count': measurement_count,
            'temporal_data': temporal_data,
            'parent_folder': parent_folder,
            'last_measurement': temporal_data[-1]['datetime'] if temporal_data else None,
            'status': status,
            'raw_node_data': node_data
        }

    def _find_channel_timestamps(self, channel_node_id: str, graph_manager, timestamp_nodes: Dict) -> Dict:
        """Find all timestamps associated with a channel"""
        associated_timestamps = {}

        print(f"DEBUG: Looking for timestamps connected to channel {channel_node_id}")

        # Look for INDIRECT connections through intermediate nodes
        intermediate_nodes = set()

        # Find nodes that connect TO the channel
        for (source, target), edge_attrs in graph_manager.edge_data.items():
            if target == channel_node_id:
                intermediate_nodes.add(source)

        # Find nodes that connect FROM the channel
        for (source, target), edge_attrs in graph_manager.edge_data.items():
            if source == channel_node_id:
                intermediate_nodes.add(target)

        # Now check if any intermediate nodes connect to timestamps
        for intermediate in intermediate_nodes:
            for (source, target), edge_attrs in graph_manager.edge_data.items():
                if (source == intermediate and target in timestamp_nodes) or \
                        (target == intermediate and source in timestamp_nodes):
                    timestamp_id = target if source == intermediate else source
                    associated_timestamps[timestamp_id] = timestamp_nodes[timestamp_id]

        print(f"DEBUG: Found {len(associated_timestamps)} timestamps for channel {channel_node_id}")
        return associated_timestamps

    def _find_channel_parent_folder(self, channel_node_id: str, graph_manager, folder_nodes: Dict) -> Optional[str]:
        """Find the parent folder for a channel"""
        for (source, target), edge_attrs in graph_manager.edge_data.items():
            if target == channel_node_id and source in folder_nodes:
                folder_data = folder_nodes[source]
                folder_name = self._extract_folder_name(folder_data)
                return folder_name
        return None

    def _extract_folder_name(self, folder_data: Dict) -> str:
        """Extract meaningful folder name"""
        folder_path = folder_data.get('value', '')
        if folder_path:
            folder_name = folder_path.split('\\')[-1].split('/')[-1]
            return folder_name
        return folder_data.get('name', 'Unknown')

    def _extract_temporal_info_from_timestamp(self, timestamp_data: Dict) -> Optional[Dict]:
        """Extract temporal information from timestamp node"""
        try:
            attrs = timestamp_data.get('attributes', {})

            # Get date/time components
            year = timestamp_data.get('year', attrs.get('year'))
            month = timestamp_data.get('month', attrs.get('month'))
            day = timestamp_data.get('day', attrs.get('day'))
            hour = timestamp_data.get('hour', attrs.get('hour', 0))
            minute = timestamp_data.get('minute', attrs.get('minute', 0))
            second = timestamp_data.get('second', attrs.get('second', 0))

            # Need at least date components
            if year and month and day:
                dt = datetime(
                    year=int(year), month=int(month), day=int(day),
                    hour=int(hour) if hour is not None else 0,
                    minute=int(minute) if minute is not None else 0,
                    second=int(second) if second is not None else 0
                )

                return {
                    'datetime_str': dt.isoformat(),
                    'components': {
                        'year': year, 'month': month, 'day': day,
                        'hour': hour, 'minute': minute, 'second': second
                    }
                }
        except (ValueError, TypeError) as e:
            print(f"DEBUG: Error extracting temporal info: {e}")

        return None