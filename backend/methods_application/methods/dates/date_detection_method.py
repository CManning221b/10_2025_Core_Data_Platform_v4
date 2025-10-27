# backend/methods_application/methods/date_detection_method.py
from backend.methods_application.method_implementation import MethodImplementation
import re
from datetime import datetime


class DateDetectionMethod(MethodImplementation):
    method_id = "DateDetectionMethod"
    method_name = "Date Detection in Components"
    description = "Searches component text for temporal patterns and creates Timestamp nodes with source-aware merging"

    def execute(self, graph_manager, parameters):
        changes = {
            "nodes_added": 0,
            "edges_added": 0,
            "method_instances_created": 0,
            "components_processed": 0,
            "dates_found": 0,
            "date_patterns": {},
            "timestamps_merged": 0
        }

        # Find all component nodes that might contain temporal info
        component_nodes = []
        for node_id, node_data in graph_manager.node_data.items():
            if node_data.get('type') == 'component':
                text = node_data.get('value', '').strip()
                if text:  # Only process components with actual text
                    component_nodes.append((node_id, node_data))

        print(f" Found {len(component_nodes)} component nodes with text to analyze")

        # Early exit if no work to do
        if not component_nodes:
            print(" No components with text found - skipping method execution")
            return changes

        # Only create method instance if there's actual work to do
        method_instance_id = self._create_method_instance(graph_manager, changes)

        for comp_id, comp_data in component_nodes:
            changes["components_processed"] += 1

            # Extract text from component
            text = comp_data.get('value', '').strip()

            # Get context for better disambiguation
            context = self._get_component_context(graph_manager, comp_id)

            # Try to detect temporal patterns with context
            temporal_info = self._detect_temporal_patterns_with_context(text, context)

            if temporal_info:
                # Find parent entities to check for existing related timestamps
                parent_entities = self._find_parent_entities(graph_manager, comp_id)

                # Look for existing timestamps from the same sources
                related_timestamps = self._find_related_timestamps(graph_manager, parent_entities)

                # Check if we can merge with an existing timestamp
                merge_target = None
                for existing_id in related_timestamps:
                    if self._can_merge_timestamps(graph_manager, existing_id, temporal_info):
                        merge_target = existing_id
                        break

                if merge_target:
                    # Merge with existing timestamp (preserving existing components)
                    self._merge_timestamps_preserving(graph_manager, merge_target, temporal_info, comp_id, changes)
                    self._connect_method_to_input(graph_manager, method_instance_id, comp_id, changes)
                    changes["timestamps_merged"] += 1
                    print(f" Merged temporal info from '{text}' into existing timestamp {merge_target}")
                else:
                    # Create new timestamp node
                    timestamp_id = self._create_timestamp_node(graph_manager, temporal_info, comp_id, changes)

                    # Connect method to output and input
                    self._connect_method_to_output(graph_manager, method_instance_id, timestamp_id, changes)
                    self._connect_method_to_input(graph_manager, method_instance_id, comp_id, changes)

                    # Connect parent entities to the timestamp
                    for parent_id, parent_type in parent_entities:
                        self._connect_entity_to_timestamp(graph_manager, parent_id, parent_type, timestamp_id, changes)

                changes["dates_found"] += 1
                pattern = temporal_info['pattern']
                changes["date_patterns"][pattern] = changes["date_patterns"].get(pattern, 0) + 1

        print(f" DateDetectionMethod processed {changes['components_processed']} components, "
              f"found {changes['dates_found']} temporal patterns, merged {changes['timestamps_merged']} timestamps")
        return changes

    def _get_component_context(self, graph_manager, comp_id):
        """Get context about where this component appears to help with disambiguation"""
        context = {
            'parent_names': [],
            'file_paths': [],
            'folder_patterns': [],
            'other_components': []
        }

        # Find parent entities and their names
        parent_entities = self._find_parent_entities(graph_manager, comp_id)
        for parent_id, parent_type in parent_entities:
            parent_node = graph_manager.node_data.get(parent_id)
            if parent_node:
                parent_value = parent_node.get('value', '')
                context['parent_names'].append(parent_value)

                if parent_type == 'file':
                    context['file_paths'].append(parent_value)
                elif parent_type == 'folder':
                    context['folder_patterns'].append(parent_value)

        # Find other components from the same parents (for pattern context)
        for parent_id, parent_type in parent_entities:
            for (source, target), edge_attrs in graph_manager.edge_data.items():
                if source == parent_id and target != comp_id:
                    target_node = graph_manager.node_data.get(target)
                    if target_node and target_node.get('type') == 'component':
                        other_comp_text = target_node.get('value', '').strip()
                        if other_comp_text and other_comp_text != graph_manager.node_data[comp_id].get('value', ''):
                            context['other_components'].append(other_comp_text)

        return context

    def _detect_temporal_patterns_with_context(self, text, context):
        """Detect temporal patterns in text using context for better disambiguation"""
        patterns = [
            # Date patterns - more specific first
            {
                'regex': r'^(\d{4})(\d{2})(\d{2})$',
                'type': 'date',
                'pattern': 'YYYYMMDD',
                'parser': lambda m: {
                    'year': int(m.group(1)),
                    'month': int(m.group(2)),
                    'day': int(m.group(3))
                }
            },
            # Time patterns - require context clues or formatting
            {
                'regex': r'^(\d{2}):(\d{2}):(\d{2})$',  # HH:MM:SS with colons
                'type': 'time',
                'pattern': 'HH:MM:SS',
                'parser': lambda m: {
                    'hour': int(m.group(1)),
                    'minute': int(m.group(2)),
                    'second': int(m.group(3))
                }
            },
            {
                'regex': r'^(\d{2}):(\d{2})$',  # HH:MM with colons
                'type': 'time',
                'pattern': 'HH:MM',
                'parser': lambda m: {
                    'hour': int(m.group(1)),
                    'minute': int(m.group(2))
                }
            },
            # Year only
            {
                'regex': r'^(20\d{2})$',
                'type': 'date',
                'pattern': 'YYYY',
                'parser': lambda m: {'year': int(m.group(1))}
            }
        ]

        # Check specific patterns first
        for pattern_info in patterns:
            match = re.match(pattern_info['regex'], text)
            if match:
                try:
                    parsed_components = pattern_info['parser'](match)

                    # Validate components
                    if pattern_info['type'] == 'date':
                        year = parsed_components.get('year')
                        month = parsed_components.get('month')
                        day = parsed_components.get('day')

                        if month and (month < 1 or month > 12):
                            continue
                        if day and (day < 1 or day > 31):
                            continue

                    elif pattern_info['type'] == 'time':
                        hour = parsed_components.get('hour')
                        minute = parsed_components.get('minute')
                        second = parsed_components.get('second')

                        if hour and (hour < 0 or hour > 23):
                            continue
                        if minute and (minute < 0 or minute > 59):
                            continue
                        if second and (second < 0 or second > 59):
                            continue

                    return {
                        'type': pattern_info['type'],
                        'pattern': pattern_info['pattern'],
                        'components': parsed_components,
                        'original_text': text
                    }
                except ValueError:
                    continue

        # Handle ambiguous 6-digit pattern with context-aware disambiguation
        six_digit_match = re.match(r'^(\d{2})(\d{2})(\d{2})$', text)
        if six_digit_match:
            # Use context to determine if this should be date or time
            is_likely_date = self._is_likely_date_context(text, context)

            if is_likely_date:
                # Try as date first (YYMMDD)
                try:
                    year = 2000 + int(six_digit_match.group(1))
                    month = int(six_digit_match.group(2))
                    day = int(six_digit_match.group(3))

                    # Validate date components
                    if 1 <= month <= 12 and 1 <= day <= 31:
                        print(f" Context suggests '{text}' is date -> {year}-{month:02d}-{day:02d}")
                        return {
                            'type': 'date',
                            'pattern': 'YYMMDD',
                            'components': {'year': year, 'month': month, 'day': day},
                            'original_text': text
                        }
                except ValueError:
                    pass

            # If date validation failed or context suggests time, try as time (HHMMSS)
            try:
                hour = int(six_digit_match.group(1))
                minute = int(six_digit_match.group(2))
                second = int(six_digit_match.group(3))

                # Validate time components
                if 0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59:
                    print(f" Interpreting '{text}' as time -> {hour:02d}:{minute:02d}:{second:02d}")
                    return {
                        'type': 'time',
                        'pattern': 'HHMMSS',
                        'components': {'hour': hour, 'minute': minute, 'second': second},
                        'original_text': text
                    }
            except ValueError:
                pass

        return None

    def _is_likely_date_context(self, text, context):
        """Use context clues to determine if a 6-digit number is likely a date"""
        # Look for date patterns in parent names/paths
        for parent_name in context['parent_names']:
            # Look for YYYYMMDD patterns in filenames/folders
            if re.search(r'20\d{6}', parent_name):  # YYYYMMDD pattern
                return True
            # Look for date separators suggesting this is a date context
            if re.search(r'\d{2}[-_/]\d{2}[-_/]\d{2,4}', parent_name):
                return True

        # Look at other components for date/time patterns
        date_like_components = 0
        time_like_components = 0

        for other_comp in context['other_components']:
            # Count YYYYMMDD patterns (8 digits starting with 20)
            if re.match(r'^20\d{6}$', other_comp):
                date_like_components += 1
            # Count obvious time patterns (with colons or very early hours)
            elif re.match(r'^\d{2}:\d{2}', other_comp) or (re.match(r'^0\d\d\d\d\d$', other_comp)):
                time_like_components += 1

        # If we see more date-like patterns, assume this is also a date
        if date_like_components > time_like_components:
            return True

        # Check if the number itself is more date-like or time-like
        first_two = int(text[:2])

        # If first two digits suggest a recent year (24 = 2024), it's likely a date
        if first_two >= 20 and first_two <= 30:  # 2020-2030 range
            return True

        # If first two digits are valid hours but month/day validation would fail, it's likely time
        if first_two <= 23:  # Valid hour
            month = int(text[2:4])
            day = int(text[4:6])
            if month > 12 or day > 31:  # Invalid as date
                return False

        # Default to time for ambiguous cases
        return False

    def _can_merge_timestamps(self, graph_manager, existing_id, new_temporal_info):
        """Check if new temporal info can merge with existing timestamp"""
        existing_node = graph_manager.node_data.get(existing_id)
        if not existing_node:
            print(f" Existing node {existing_id} not found")
            return False

        # Check both direct attributes and attributes dict for compatibility
        existing_attrs = existing_node.get('attributes', {})

        # Try direct node attributes first, then attributes dict
        has_date = existing_node.get('has_date', existing_attrs.get('has_date', False))
        has_time = existing_node.get('has_time', existing_attrs.get('has_time', False))
        new_type = new_temporal_info['type']

        print(
            f" Checking merge for timestamp {existing_id}: has_date={has_date}, has_time={has_time}, new_type={new_type}")

        can_merge = (has_date and new_type == 'time') or (has_time and new_type == 'date')

        if can_merge:
            print(f" Can merge {new_type} info into timestamp {existing_id}")
        else:
            print(f" Cannot merge {new_type} info into timestamp {existing_id} "
                  f"(has_date={has_date}, has_time={has_time})")

        return can_merge

    def _merge_timestamps_preserving(self, graph_manager, existing_id, new_temporal_info, source_comp_id, changes):
        """Merge new temporal info into existing timestamp, preserving existing components"""
        existing_node = graph_manager.node_data.get(existing_id)

        # Get existing attributes from both locations for compatibility
        existing_attrs = existing_node.get('attributes', {})

        # Merge all current attributes (both direct and from attributes dict)
        merged_attrs = {}
        merged_attrs.update(existing_attrs)  # Start with attributes dict
        merged_attrs.update({k: v for k, v in existing_node.items()
                             if k not in ['value', 'type', 'hierarchy']})  # Add direct attributes

        # Only add NEW temporal components - don't overwrite existing ones
        new_components = new_temporal_info['components']
        for key, value in new_components.items():
            if key not in merged_attrs or merged_attrs[key] is None:
                merged_attrs[key] = value
                print(f" Adding new component {key}={value} to timestamp {existing_id}")
            else:
                print(f" Preserving existing component {key}={merged_attrs[key]} in timestamp {existing_id}")

        # Update flags
        if new_temporal_info['type'] == 'date':
            merged_attrs['has_date'] = True
        else:
            merged_attrs['has_time'] = True

        # Update source information
        original_sources = merged_attrs.get('source_components', [])
        if isinstance(original_sources, str):
            original_sources = [original_sources]
        if source_comp_id not in original_sources:
            original_sources.append(source_comp_id)
        merged_attrs['source_components'] = original_sources

        # Update merged patterns
        existing_patterns = merged_attrs.get('merged_patterns', [])
        if new_temporal_info['pattern'] not in existing_patterns:
            existing_patterns.append(new_temporal_info['pattern'])
        merged_attrs['merged_patterns'] = existing_patterns

        # Build new display value using preserved components
        display_parts = []
        if merged_attrs.get('has_date'):
            year = merged_attrs.get('year', '????')
            month = merged_attrs.get('month', '??')
            day = merged_attrs.get('day', '??')

            # Format each component properly
            year_str = str(year) if year != '????' else '????'
            month_str = f"{month:02d}" if isinstance(month, int) else str(month)
            day_str = f"{day:02d}" if isinstance(day, int) else str(day)

            display_parts.append(f"{year_str}-{month_str}-{day_str}")

        if merged_attrs.get('has_time'):
            hour = merged_attrs.get('hour', '??')
            minute = merged_attrs.get('minute', '??')
            second = merged_attrs.get('second', '00')

            # Format each component properly
            hour_str = f"{hour:02d}" if isinstance(hour, int) else str(hour)
            minute_str = f"{minute:02d}" if isinstance(minute, int) else str(minute)

            time_str = f"{hour_str}:{minute_str}"
            if second and second != '00':
                second_str = f"{second:02d}" if isinstance(second, int) else str(second)
                time_str += f":{second_str}"
            display_parts.append(time_str)

        new_value = ' '.join(display_parts)

        # Update the node - store attributes directly on the node like graph_manager expects
        for key, value in merged_attrs.items():
            graph_manager.node_data[existing_id][key] = value

        graph_manager.node_data[existing_id]['value'] = new_value

        print(f" Merged timestamp {existing_id} now shows: {new_value}")
        print(
            f" Final components: year={merged_attrs.get('year')}, month={merged_attrs.get('month')}, day={merged_attrs.get('day')}, hour={merged_attrs.get('hour')}, minute={merged_attrs.get('minute')}, second={merged_attrs.get('second')}")

    # [Keep all the other existing methods unchanged - _find_related_timestamps, _create_timestamp_node, etc.]
    def _find_related_timestamps(self, graph_manager, source_parents):
        """Find existing timestamps from the same sources"""
        related_timestamps = []

        print(f" Looking for related timestamps from parents: {source_parents}")

        for parent_id, parent_type in source_parents:
            print(f" Checking parent {parent_id} ({parent_type}) for existing timestamps")

            # Find timestamps connected to the same parent
            found_edges = []
            for (src, tgt), edge_attrs in graph_manager.edge_data.items():
                if src == parent_id:
                    edge_type = edge_attrs.get('edge_type')
                    print(f" Found edge {src}->{tgt} with type '{edge_type}'")

                    if edge_type == 'has_timestamp':
                        target_node = graph_manager.node_data.get(tgt)
                        if target_node and target_node.get('type') == 'Timestamp':
                            related_timestamps.append(tgt)
                            print(f" Found related timestamp {tgt} from same source {parent_id}")
                            found_edges.append(f"{src}->{tgt}")

            if not found_edges:
                print(f" No has_timestamp edges found from parent {parent_id}")

        print(f" Total related timestamps found: {related_timestamps}")
        return list(set(related_timestamps))  # Remove duplicates

    def _create_timestamp_node(self, graph_manager, temporal_info, source_comp_id, changes):
        """Create a new timestamp node with temporal information"""
        timestamp_id = self._get_next_numeric_id(graph_manager)

        # Build attributes with temporal components
        attributes = {
            'original_text': temporal_info['original_text'],
            'pattern': temporal_info['pattern'],
            'temporal_type': temporal_info['type'],
            'source_components': [source_comp_id]
        }

        # Add temporal components and flags
        if temporal_info['type'] == 'date':
            attributes.update({
                'has_date': True,
                'has_time': False,
                **temporal_info['components']  # year, month, day
            })
            comps = temporal_info['components']
            year = comps.get('year', '????')
            month = comps.get('month', '??')
            day = comps.get('day', '??')

            # Format each component properly
            year_str = str(year) if year != '????' else '????'
            month_str = f"{month:02d}" if isinstance(month, int) else str(month)
            day_str = f"{day:02d}" if isinstance(day, int) else str(day)

            display_value = f"{year_str}-{month_str}-{day_str}"

        else:  # time
            attributes.update({
                'has_date': False,
                'has_time': True,
                **temporal_info['components']  # hour, minute, second
            })
            comps = temporal_info['components']
            hour = comps.get('hour', '??')
            minute = comps.get('minute', '??')
            second = comps.get('second')

            # Format each component properly
            hour_str = f"{hour:02d}" if isinstance(hour, int) else str(hour)
            minute_str = f"{minute:02d}" if isinstance(minute, int) else str(minute)

            display_value = f"{hour_str}:{minute_str}"
            if second is not None:
                second_str = f"{second:02d}" if isinstance(second, int) else str(second)
                display_value += f":{second_str}"

        try:
            graph_manager.add_node(
                node_id=timestamp_id,
                value=display_value,
                type='Timestamp',
                hierarchy='temporal',
                attributes=attributes
            )
            changes["nodes_added"] += 1
            print(
                f" Created Timestamp node {timestamp_id} for '{temporal_info['original_text']}' -> {display_value}")

        except KeyError as e:
            print(f" Error creating Timestamp node: {e}")
            return None

        return timestamp_id

    # [Keep all other existing methods unchanged...]
    def _create_method_instance(self, graph_manager, changes):
        """Create a method instance node for execution traceability"""
        method_instance_id = self._get_next_numeric_id(graph_manager)

        try:
            graph_manager.add_node(
                node_id=method_instance_id,
                value=f"DateDetectionMethod execution",
                type='DateDetectionMethod',
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

    def _connect_method_to_output(self, graph_manager, method_instance_id, timestamp_id, changes):
        """Connect method instance to its output Timestamp node"""
        print(f" Connecting method {method_instance_id} -> Timestamp {timestamp_id}")

        try:
            graph_manager.add_edge(
                source=method_instance_id,
                target=timestamp_id,
                attributes={
                    'edge_type': 'produces',
                    'direction': 'out',
                    'provenance_type': 'generated_by',
                    'created': datetime.now().isoformat()
                }
            )
            changes["edges_added"] += 1
            print(f" Created output edge {method_instance_id} -> {timestamp_id}")

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

    def _connect_entity_to_timestamp(self, graph_manager, parent_id, parent_type, timestamp_id, changes):
        """Connect parent entity to the timestamp"""
        print(f" Connecting {parent_type} {parent_id} -> Timestamp {timestamp_id}")

        try:
            graph_manager.add_edge(
                source=parent_id,
                target=timestamp_id,
                attributes={
                    'edge_type': 'has_timestamp',
                    'direction': 'out',
                    'provenance_type': 'contains_temporal_info',
                    'created': datetime.now().isoformat()
                }
            )
            changes["edges_added"] += 1
            print(f" Created timestamp relationship edge {parent_id} -> {timestamp_id}")

        except (KeyError, ValueError) as e:
            print(f" Error creating timestamp relationship edge: {e}")

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