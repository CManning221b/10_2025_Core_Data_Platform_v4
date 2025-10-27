# backend/methods_application/methods/timeline_generation_method.py
from backend.methods_application.method_implementation import MethodImplementation
import json
from datetime import datetime
import re


class TimelineGenerationMethod(MethodImplementation):
    method_id = "TimelineGenerationMethod"
    method_name = "Timeline Generator"
    description = "Creates comprehensive chronological timeline of all timestamped entities and events"

    def execute(self, graph_manager, parameters):
        changes = {
            "nodes_added": 0,
            "edges_added": 0,
            "method_instances_created": 0,
            "timeline_entries": 0,
            "date_range": {},
            "temporal_sources": {}
        }

        print(f" Starting comprehensive timeline generation across all nodes")

        # Only create method instance if there's actual work to do
        method_instance_id = self._create_method_instance(graph_manager, changes)

        # Collect all temporal events from the entire graph
        timeline_entries = []
        processed_nodes = []
        temporal_source_counts = {}

        # Scan every node for temporal information
        for node_id, node_data in graph_manager.node_data.items():
            temporal_events = self._extract_all_temporal_events(node_id, node_data, graph_manager)

            for event in temporal_events:
                timeline_entries.append(event)
                processed_nodes.append(node_id)

                # Track source types
                source_type = event['source_type']
                temporal_source_counts[source_type] = temporal_source_counts.get(source_type, 0) + 1

                print(f" Added {source_type} event for {event['entity_type']} {node_id} at {event['datetime']}")

        print(f" Found {len(timeline_entries)} temporal events from {len(set(processed_nodes))} nodes")
        print(f" Temporal source breakdown: {temporal_source_counts}")

        # Early exit if no temporal events found
        if not timeline_entries:
            print(" No temporal events found - skipping timeline generation")
            return changes

        # Sort by datetime
        timeline_entries.sort(key=lambda x: x['datetime'])

        # Create timeline data structure
        timeline_data = []
        for i, entry in enumerate(timeline_entries):
            timeline_data.append({
                'order': i + 1,
                'datetime': entry['datetime'].isoformat(),
                'entity_id': entry['entity_id'],
                'entity_type': entry['entity_type'],
                'entity_value': entry['entity_value'],
                'source_type': entry['source_type'],
                'source_property': entry['source_property'],
                'temporal_value': entry['temporal_value'],
                'context_properties': entry['context_properties'],
                'description': entry['description']
            })

        # Create Timeline node
        timeline_id = self._get_next_numeric_id(graph_manager)

        try:
            graph_manager.add_node(
                node_id=timeline_id,
                value=f"Comprehensive Timeline of {len(timeline_entries)} events",
                type='Timeline',
                hierarchy='analysis',
                attributes={
                    'timeline_data': json.dumps(timeline_data),
                    'entry_count': len(timeline_entries),
                    'date_range_start': timeline_data[0]['datetime'] if timeline_data else None,
                    'date_range_end': timeline_data[-1]['datetime'] if timeline_data else None,
                    'temporal_sources': json.dumps(temporal_source_counts),
                    'created': datetime.now().isoformat()
                }
            )
            changes["nodes_added"] += 1
            changes["timeline_entries"] = len(timeline_entries)
            changes["temporal_sources"] = temporal_source_counts
            print(f" Created comprehensive Timeline node {timeline_id} with {len(timeline_entries)} events")

            # Connect method instance to the timeline it produced
            self._connect_method_to_output(graph_manager, method_instance_id, timeline_id, changes)

            # Connect method instance to all the nodes it analyzed
            for node_id in set(processed_nodes):
                self._connect_method_to_input(graph_manager, method_instance_id, node_id, changes)

            # Connect Timeline to referenced temporal nodes
            for node_id in set(processed_nodes):
                self._connect_timeline_to_temporal_node(graph_manager, timeline_id, node_id, changes)

            if timeline_data:
                changes["date_range"] = {
                    'start': timeline_data[0]['datetime'],
                    'end': timeline_data[-1]['datetime']
                }

        except KeyError as e:
            print(f" Error creating Timeline node {timeline_id}: {e}")

        print(
            f" TimelineGenerationMethod created comprehensive timeline with {changes['timeline_entries']} events")
        return changes

    def _extract_all_temporal_events(self, node_id, node_data, graph_manager):
        """Extract all temporal events from a single node"""
        events = []

        # 1. Handle archive measurements with timestamps (PRIORITY!)
        if node_data.get('type') == 'measurement':
            timestamp = node_data.get('timestamp')
            if timestamp is not None:
                # Convert float timestamp to datetime if needed
                if isinstance(timestamp, (int, float)):
                    # Assume it's a year for your archive data
                    try:
                        dt = datetime(year=int(timestamp), month=1, day=1)
                        events.append({
                            'datetime': dt,
                            'entity_id': node_id,
                            'entity_type': 'measurement',
                            'entity_value': node_data.get('value', f'Measurement {node_id}'),
                            'source_type': 'measurement_timestamp',
                            'source_property': 'timestamp',
                            'temporal_value': timestamp,
                            'context_properties': {
                                'value_type': node_data.get('value_type', 'unknown'),
                                'location': node_data.get('location', 'unknown')
                            },
                            'description': f"{node_data.get('value_type', 'Measurement')} taken in {timestamp}"
                        })
                    except (ValueError, TypeError):
                        pass

        # 2. Handle archive objects with creation/destruction dates
        if node_data.get('type') == 'object':
            created = node_data.get('created')
            destroyed = node_data.get('destroyed')

            if created is not None:
                try:
                    dt = datetime(year=int(created), month=1, day=1)
                    events.append({
                        'datetime': dt,
                        'entity_id': node_id,
                        'entity_type': 'object_creation',
                        'entity_value': node_data.get('value', f'Object {node_id}'),
                        'source_type': 'object_lifecycle',
                        'source_property': 'created',
                        'temporal_value': created,
                        'context_properties': {
                            'location': node_data.get('location', 'unknown'),
                            'lifespan': f"{created}-{destroyed}" if destroyed else f"{created}-ongoing"
                        },
                        'description': f"{node_data.get('value', 'Object')} created"
                    })
                except (ValueError, TypeError):
                    pass

            if destroyed is not None:
                try:
                    dt = datetime(year=int(destroyed), month=12, day=31)
                    events.append({
                        'datetime': dt,
                        'entity_id': node_id,
                        'entity_type': 'object_destruction',
                        'entity_value': node_data.get('value', f'Object {node_id}'),
                        'source_type': 'object_lifecycle',
                        'source_property': 'destroyed',
                        'temporal_value': destroyed,
                        'context_properties': {
                            'location': node_data.get('location', 'unknown'),
                            'lifespan': f"{created}-{destroyed}" if created else f"unknown-{destroyed}"
                        },
                        'description': f"{node_data.get('value', 'Object')} destroyed"
                    })
                except (ValueError, TypeError):
                    pass

        # 3. Handle validation results (your anachronisms!)
        if node_data.get('type') == 'ValidationResult':
            timestamp = node_data.get('timestamp') or node_data.get('attributes', {}).get('timestamp')
            if timestamp is not None:
                try:
                    dt = datetime(year=int(timestamp), month=6, day=15)  # Mid-year for validation events
                    is_valid = node_data.get('is_valid') or node_data.get('attributes', {}).get('is_valid')
                    events.append({
                        'datetime': dt,
                        'entity_id': node_id,
                        'entity_type': 'validation_event',
                        'entity_value': 'Anachronism' if not is_valid else 'Valid measurement',
                        'source_type': 'validation_result',
                        'source_property': 'timestamp',
                        'temporal_value': timestamp,
                        'context_properties': {
                            'is_valid': is_valid,
                            'validation_reason': node_data.get('validation_reason') or node_data.get('attributes',
                                                                                                     {}).get(
                                'validation_reason', ''),
                            'measurement_id': node_data.get('measurement_id') or node_data.get('attributes', {}).get(
                                'measurement_id'),
                            'object_id': node_data.get('object_id') or node_data.get('attributes', {}).get('object_id')
                        },
                        'description': f"Validation result: {'Valid' if is_valid else 'ANACHRONISM'}"
                    })
                except (ValueError, TypeError):
                    pass

        # 4. Handle explicit Timestamp nodes (for other data types)
        if node_data.get('type') == 'Timestamp':
            dt = self._extract_datetime_from_timestamp(node_data)
            if dt:
                # Find entities linked to this timestamp
                linked_entities = self._find_entities_linked_to_timestamp(node_id, graph_manager)

                for entity_id, entity_data in linked_entities:
                    events.append({
                        'datetime': dt,
                        'entity_id': entity_id,
                        'entity_type': entity_data.get('type', 'unknown'),
                        'entity_value': entity_data.get('value', ''),
                        'source_type': 'timestamp_node',
                        'source_property': 'computed_datetime',
                        'temporal_value': node_data.get('value', ''),
                        'context_properties': self._extract_context_properties(node_data),
                        'description': f"{entity_data.get('type', 'Entity')} temporal reference"
                    })

        # 5. Handle nodes with direct temporal properties (execution times, etc.)
        temporal_properties = self._find_temporal_properties(node_data)
        for prop_name, prop_value in temporal_properties:
            dt = self._parse_datetime_string(prop_value)
            if dt:
                events.append({
                    'datetime': dt,
                    'entity_id': node_id,
                    'entity_type': node_data.get('type', 'unknown'),
                    'entity_value': node_data.get('value', ''),
                    'source_type': 'direct_property',
                    'source_property': prop_name,
                    'temporal_value': prop_value,
                    'context_properties': self._extract_context_properties(node_data),
                    'description': f"{node_data.get('type', 'Node')} {prop_name}"
                })

        # 6. Handle nodes with temporal properties in attributes
        attrs = node_data.get('attributes', {})
        for prop_name, prop_value in self._find_temporal_properties({'attributes': attrs}):
            if prop_name.startswith('attributes.'):
                actual_prop = prop_name.replace('attributes.', '')
                dt = self._parse_datetime_string(prop_value)
                if dt:
                    events.append({
                        'datetime': dt,
                        'entity_id': node_id,
                        'entity_type': node_data.get('type', 'unknown'),
                        'entity_value': node_data.get('value', ''),
                        'source_type': 'attribute_property',
                        'source_property': actual_prop,
                        'temporal_value': prop_value,
                        'context_properties': self._extract_context_properties(node_data),
                        'description': f"{node_data.get('type', 'Node')} {actual_prop}"
                    })

        return events

    def _find_temporal_properties(self, data, prefix=''):
        """Find all properties that contain temporal information"""
        temporal_props = []
        temporal_property_names = {
            'execution_time', 'created', 'last_updated', 'modified', 'timestamp',
            'date', 'time', 'datetime', 'created_at', 'updated_at', 'executed_at'
        }

        def scan_dict(d, current_prefix=''):
            for key, value in d.items():
                full_key = f"{current_prefix}.{key}" if current_prefix else key

                if isinstance(value, dict):
                    scan_dict(value, full_key)
                elif isinstance(value, str):
                    # Check if property name suggests temporal data
                    if key.lower() in temporal_property_names or 'time' in key.lower() or 'date' in key.lower():
                        temporal_props.append((full_key, value))
                    # Check if value looks like a datetime string
                    elif self._looks_like_datetime(value):
                        temporal_props.append((full_key, value))

        scan_dict(data)
        return temporal_props

    def _looks_like_datetime(self, value):
        """Check if a string value looks like a datetime"""
        if not isinstance(value, str) or len(value) < 10:
            return False

        # Common datetime patterns
        datetime_patterns = [
            r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}',  # ISO format
            r'\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}',  # SQL format
            r'\d{2}/\d{2}/\d{4}',  # US date format
            r'\d{4}/\d{2}/\d{2}',  # ISO date format
        ]

        for pattern in datetime_patterns:
            if re.search(pattern, value):
                return True
        return False

    def _parse_datetime_string(self, datetime_str):
        """Parse various datetime string formats"""
        if not isinstance(datetime_str, str):
            return None

        # Try different parsing approaches
        parsing_strategies = [
            # ISO format with timezone
            lambda s: datetime.fromisoformat(s.replace('Z', '+00:00')),
            # ISO format without timezone
            lambda s: datetime.fromisoformat(s),
            # Common formats
            lambda s: datetime.strptime(s, '%Y-%m-%d %H:%M:%S'),
            lambda s: datetime.strptime(s, '%Y-%m-%dT%H:%M:%S'),
            lambda s: datetime.strptime(s, '%Y-%m-%d'),
            lambda s: datetime.strptime(s, '%m/%d/%Y'),
            lambda s: datetime.strptime(s, '%Y/%m/%d'),
        ]

        for strategy in parsing_strategies:
            try:
                return strategy(datetime_str)
            except (ValueError, TypeError):
                continue

        print(f" Could not parse datetime string: '{datetime_str}'")
        return None

    def _find_entities_linked_to_timestamp(self, timestamp_id, graph_manager):
        """Find entities that reference a timestamp node"""
        linked_entities = []

        for (source, target), edge_attrs in graph_manager.edge_data.items():
            if target == timestamp_id and edge_attrs.get('edge_type') == 'has_timestamp':
                entity_node = graph_manager.node_data.get(source)
                if entity_node:
                    linked_entities.append((source, entity_node))

        return linked_entities

    def _extract_datetime_from_timestamp(self, timestamp_data):
        """Extract datetime from timestamp node components"""
        try:
            # Check both direct attributes and attributes dict for compatibility
            attrs = timestamp_data.get('attributes', {})

            # Get date components (try direct first, then attributes)
            year = timestamp_data.get('year', attrs.get('year'))
            month = timestamp_data.get('month', attrs.get('month'))
            day = timestamp_data.get('day', attrs.get('day'))

            # Get time components (try direct first, then attributes)
            hour = timestamp_data.get('hour', attrs.get('hour', 0))
            minute = timestamp_data.get('minute', attrs.get('minute', 0))
            second = timestamp_data.get('second', attrs.get('second', 0))

            # Need at least date components to create a datetime
            if year and month and day:
                return datetime(
                    year=int(year),
                    month=int(month),
                    day=int(day),
                    hour=int(hour) if hour is not None else 0,
                    minute=int(minute) if minute is not None else 0,
                    second=int(second) if second is not None else 0
                )
            else:
                print(f" Insufficient date components: year={year}, month={month}, day={day}")
                return None

        except (ValueError, TypeError) as e:
            print(f" Error constructing datetime: {e}")
            return None

    def _extract_context_properties(self, node_data):
        """Extract relevant context properties from a node"""
        context = {}

        # Common contextual properties to include
        contextual_props = {
            'method_type', 'method_id', 'method_name', 'hierarchy', 'pattern',
            'confidence', 'channel_id', 'property_type', 'entry_count', 'original_text'
        }

        # Get direct properties
        for prop in contextual_props:
            if prop in node_data:
                context[prop] = node_data[prop]

        # Get attribute properties
        attrs = node_data.get('attributes', {})
        for prop in contextual_props:
            if prop in attrs:
                context[prop] = attrs[prop]

        return context

    def _create_method_instance(self, graph_manager, changes):
        """Create a method instance node for execution traceability"""
        method_instance_id = self._get_next_numeric_id(graph_manager)

        try:
            graph_manager.add_node(
                node_id=method_instance_id,
                value=f"TimelineGenerationMethod execution",
                type='TimelineGenerationMethod',
                hierarchy='analysis',
                attributes={
                    'method_type': 'TimelineGenerationMethod',
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

    def _connect_method_to_output(self, graph_manager, method_instance_id, timeline_id, changes):
        """Connect method instance to the timeline it created"""
        print(f" Connecting method {method_instance_id} -> timeline {timeline_id}")

        try:
            graph_manager.add_edge(
                source=method_instance_id,
                target=timeline_id,
                attributes={
                    'edge_type': 'produces',
                    'direction': 'out',
                    'provenance_type': 'generated_by',
                    'created': datetime.now().isoformat()
                }
            )
            changes["edges_added"] += 1
            print(f" Created output edge {method_instance_id} -> {timeline_id}")

        except (KeyError, ValueError) as e:
            print(f" Error creating output edge: {e}")

    def _connect_method_to_input(self, graph_manager, method_instance_id, node_id, changes):
        """Connect method instance to nodes it analyzed"""
        print(f" Connecting method {method_instance_id} -> node {node_id}")

        try:
            graph_manager.add_edge(
                source=method_instance_id,
                target=node_id,
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

    def _connect_timeline_to_temporal_node(self, graph_manager, timeline_id, node_id, changes):
        """Connect timeline to nodes with temporal information"""
        try:
            graph_manager.add_edge(
                source=timeline_id,
                target=node_id,
                attributes={
                    'edge_type': 'references',
                    'direction': 'out',
                    'provenance_type': 'contains_temporal_reference',
                    'created': datetime.now().isoformat()
                }
            )
            changes["edges_added"] += 1

        except (KeyError, ValueError) as e:
            print(f" Error creating timeline-temporal edge: {e}")

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