import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
import numpy as np
from backend.core_graph_managers.no2_graphManager.graphManager import GraphManager


class TimeSeriesService:
    """Clean time series visualization service with ontology-driven categorization"""

    def __init__(self, ontology_manager=None):
        self.graph_manager = None
        self.ontology_manager = ontology_manager
        self.ontology_cache = {}
        self.color_palette = {
            'measurements': ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#8E5572'],
            'events': ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7'],
            'objects': ['#6C5CE7', '#FD79A8', '#FDCB6E', '#E84393', '#74B9FF']
        }

    def generate_timeline_data(self, graph_manager: GraphManager, ontology_manager=None,
                               sort_criteria=None, color_strategy=None, filter_options=None,
                               visualization_mode='timeline', category_filter='all') -> dict:
        """Generate categorized timeline data with custom sorting and filtering"""
        self.graph_manager = graph_manager
        self.ontology_manager = ontology_manager or self.ontology_manager
        self.custom_sort_criteria = sort_criteria or {}
        self.custom_color_strategy = color_strategy or {}
        self.custom_filter_options = filter_options or {}
        self.category_filter = category_filter

        if self.ontology_manager:
            self._cache_ontology_patterns()

        temporal_data = self._extract_and_categorize_temporal_data()

        if self.custom_filter_options:
            temporal_data = self._apply_custom_filters(temporal_data)

        has_data = any(temporal_data.values())
        if not has_data:
            return {
                'plot_html': None,
                'has_data': False,
                'error': 'No temporal data found in graph'
            }

        try:
            plot_html = self.generate_timeline_html(temporal_data, visualization_mode=visualization_mode,
                                                    full_html=False)
            return {
                'plot_html': plot_html,
                'has_data': True,
                'temporal_data': temporal_data,
                'error': None
            }
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                'plot_html': None,
                'has_data': False,
                'error': str(e)
            }

    def _apply_custom_filters(self, temporal_data: Dict[str, List]) -> Dict[str, List]:
        """Apply custom filtering options to temporal data"""
        filtered_data = {
            'measurements': [],
            'events': [],
            'objects': []
        }

        allowed_types = self.custom_filter_options.get('node_types', [])

        for category, items in temporal_data.items():
            for item in items:
                item_type = item.get('node_type', 'Unknown')
                if not allowed_types or item_type in allowed_types:
                    filtered_data[category].append(item)

        return filtered_data

    def _apply_category_filter(self, temporal_data: Dict) -> Dict:
        """Filter temporal data based on category selection"""
        if not hasattr(self, 'category_filter') or self.category_filter == 'all':
            return temporal_data

        filtered = {
            'measurements': [],
            'events': [],
            'objects': []
        }

        if self.category_filter == 'measurements':
            filtered['measurements'] = temporal_data['measurements']
        elif self.category_filter == 'events':
            filtered['events'] = temporal_data['events']
        elif self.category_filter == 'objects':
            filtered['objects'] = temporal_data['objects']

        return filtered

    def _generate_empty_plot(self, message: str) -> str:
        """Generate an empty plot with a message"""
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            height=400,
            template='plotly_white',
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig.to_html(include_plotlyjs=True, div_id="temporal-analysis-plot", config={}, full_html=False)

    def _cache_ontology_patterns(self):
        """Cache ontology-derived patterns for efficient lookup"""
        if not self.ontology_manager:
            return

        self.ontology_cache['naming_properties'] = self._discover_naming_properties()
        self.ontology_cache['measurement_properties'] = self._discover_measurement_properties()
        self.ontology_cache['categorization_rules'] = self._discover_categorization_rules()
        self.ontology_cache['hierarchy_patterns'] = self._discover_hierarchy_patterns()

    def _discover_naming_properties(self) -> List[str]:
        """Discover the most common naming properties across all node types"""
        naming_candidates = []
        node_types = self.ontology_manager.get_node_types()

        for node_type, type_info in node_types.items():
            try:
                prop_stats = self.ontology_manager.get_type_property_statistics(node_type, is_node_type=True)

                for prop_name, prop_info in prop_stats['properties'].items():
                    if (prop_info['type_info']['primary_type'] == 'str' and
                            prop_info['presence']['percentage'] > 50):
                        name_score = self._calculate_naming_score_from_stats(prop_name, prop_info)
                        naming_candidates.append((prop_name, name_score, prop_info['presence']['percentage']))

            except Exception:
                continue

        naming_candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
        return [prop[0] for prop in naming_candidates]

    def _calculate_naming_score_from_stats(self, prop_name: str, prop_info: Dict) -> float:
        """Calculate naming score using only statistical properties"""
        score = 0.0

        uniqueness = prop_info.get('cardinality', {}).get('uniqueness', '')
        if uniqueness == 'unique':
            score += 10.0
        elif uniqueness == 'variable':
            score += 5.0

        presence_pct = prop_info.get('presence', {}).get('percentage', 0)
        score += presence_pct * 0.1

        if 'statistics' in prop_info and 'avg_length' in prop_info['statistics']:
            avg_length = prop_info['statistics']['avg_length']
            if 5 <= avg_length <= 50:
                score += 3.0
            elif avg_length > 100:
                score -= 5.0

        if 'cardinality' in prop_info:
            distinct_count = prop_info['cardinality'].get('distinct_count', 0)
            total_count = prop_info['cardinality'].get('total_count', 0)
            if total_count > 0:
                variance_ratio = distinct_count / total_count
                score += variance_ratio * 5.0

        return score

    def _discover_measurement_properties(self) -> List[str]:
        """Discover numeric properties linked to timestamps"""
        measurement_candidates = Counter()
        timestamp_linked_nodes = self._find_all_timestamp_linked_nodes()

        for node_id in timestamp_linked_nodes:
            node_data = self.graph_manager.node_data.get(node_id, {})

            for prop_name, prop_value in node_data.items():
                if isinstance(prop_value, (int, float)) and not self._is_structural_property(prop_name):
                    measurement_candidates[prop_name] += 1

            attrs = node_data.get('attributes', {})
            for prop_name, prop_value in attrs.items():
                if isinstance(prop_value, (int, float)) and not self._is_structural_property(prop_name):
                    measurement_candidates[f"attributes.{prop_name}"] += 1

        return [prop for prop, count in measurement_candidates.most_common()]

    def _find_all_timestamp_linked_nodes(self) -> List[str]:
        """Find all nodes that are linked to timestamp nodes"""
        timestamp_linked = []
        timestamp_nodes = [
            node_id for node_id, node_data in self.graph_manager.node_data.items()
            if node_data.get('type') == 'Timestamp'
        ]

        for timestamp_id in timestamp_nodes:
            linked_entities = self._find_entities_linked_to_timestamp(timestamp_id)
            timestamp_linked.extend([entity_id for entity_id, _ in linked_entities])

        return timestamp_linked

    def _discover_categorization_rules(self) -> Dict[str, Any]:
        """Discover node type categorization using graph structure analysis"""
        rules = {
            'object_types': set(),
            'event_types': set(),
            'measurement_types': set(),
            'reasoning': {}
        }

        node_types = self.ontology_manager.get_node_types()

        for node_type, type_info in node_types.items():
            reasoning = []
            category_scores = {'object': 0, 'event': 0, 'measurement': 0}

            try:
                node_analysis = self.ontology_manager.analyze_node_type(node_type)
                structural_metrics = node_analysis.get('structural_metrics', {})

                connectivity_ratio = structural_metrics.get('connectivity_ratio', 0)
                if connectivity_ratio > 2.0:
                    category_scores['object'] += 2.0
                    reasoning.append(f"High connectivity ({connectivity_ratio:.1f})")

                hierarchy_pos = structural_metrics.get('hierarchical_position', {})
                if hierarchy_pos:
                    root_pct = hierarchy_pos.get('root_percentage', 0)
                    intermediate_pct = hierarchy_pos.get('intermediate_percentage', 0)

                    if root_pct > 30 or intermediate_pct > 30:
                        category_scores['object'] += 1.5
                        reasoning.append(
                            f"Container role (root: {root_pct:.1f}%, intermediate: {intermediate_pct:.1f}%)")

                timestamp_linkage = self._analyze_timestamp_linkage_for_type(node_type)
                if timestamp_linkage['has_direct_timestamp_links']:
                    category_scores['measurement'] += 2.0
                    reasoning.append("Direct timestamp linkage detected")

                numeric_prop_ratio = self._analyze_numeric_properties_for_type(node_type)
                if numeric_prop_ratio > 0.3:
                    category_scores['measurement'] += numeric_prop_ratio * 2.0
                    reasoning.append(f"High numeric property ratio ({numeric_prop_ratio:.2f})")

                relationship_patterns = self._analyze_relationship_patterns_for_type(node_type)
                category_scores['object'] += relationship_patterns['container_score']
                category_scores['event'] += relationship_patterns['temporal_score']

            except Exception:
                pass

            best_category = max(category_scores.items(), key=lambda x: x[1])
            if best_category[1] > 0:
                rules[f'{best_category[0]}_types'].add(node_type)
                rules['reasoning'][node_type] = {
                    'category': best_category[0],
                    'score': best_category[1],
                    'reasoning': reasoning
                }
            else:
                rules['event_types'].add(node_type)
                rules['reasoning'][node_type] = {
                    'category': 'event',
                    'score': 0,
                    'reasoning': ['Default classification']
                }

        return rules

    def _discover_hierarchy_patterns(self) -> Dict[str, Any]:
        """Discover hierarchical relationship patterns"""
        patterns = {
            'parent_child_edges': [],
            'container_edges': [],
            'ownership_edges': [],
            'type_hierarchies': {}
        }

        relationship_patterns = self.ontology_manager.extract_relationship_patterns()

        for pattern in relationship_patterns.get('full_patterns', []):
            edge_type = pattern['edge_type'].lower()
            source_type = pattern['source_type']
            target_type = pattern['target_type']

            if any(term in edge_type for term in ['contain', 'has', 'include', 'part']):
                patterns['container_edges'].append(pattern)
            elif any(term in edge_type for term in ['parent', 'child', 'member', 'belong']):
                patterns['parent_child_edges'].append(pattern)
            elif any(term in edge_type for term in ['own', 'possess', 'hold']):
                patterns['ownership_edges'].append(pattern)

            if source_type not in patterns['type_hierarchies']:
                patterns['type_hierarchies'][source_type] = {'children': set(), 'parents': set()}
            if target_type not in patterns['type_hierarchies']:
                patterns['type_hierarchies'][target_type] = {'children': set(), 'parents': set()}

            patterns['type_hierarchies'][source_type]['children'].add(target_type)
            patterns['type_hierarchies'][target_type]['parents'].add(source_type)

        return patterns

    def _group_objects_by_strategy(self, objects: List[Dict], color_strategy: Dict) -> Dict[str, List]:
        """Group objects based on color strategy"""
        method = color_strategy.get('method', 'node_type')

        if method == 'property_value':
            property_name = color_strategy.get('property')
            if property_name:
                return self._group_by_property_value(objects, property_name)
        elif method == 'parent_type':
            return self._group_by_parent_type(objects)
        elif method == 'measurement_range':
            return self._group_by_measurement_range(objects)

        groups = defaultdict(list)
        for obj in objects:
            groups[obj['node_type']].append(obj)
        return groups

    def _group_by_property_value(self, objects: List[Dict], property_name: str) -> Dict[str, List]:
        """Group objects by a specific property value"""
        groups = defaultdict(list)
        for obj in objects:
            prop_value = obj.get('all_properties', {}).get(property_name, 'Unknown')
            group_key = f"{property_name}: {prop_value}"
            groups[group_key].append(obj)
        return groups

    def _group_by_parent_type(self, objects: List[Dict]) -> Dict[str, List]:
        """Group objects by their parent node types"""
        groups = defaultdict(list)
        for obj in objects:
            parents = self._get_object_parents(obj['node_id'])
            if parents:
                parent_type = parents[0]['type']
                group_key = f"Parent: {parent_type}"
            else:
                group_key = "No Parent"
            groups[group_key].append(obj)
        return groups

    def _group_by_measurement_range(self, objects: List[Dict]) -> Dict[str, List]:
        """Group objects by measurement value ranges"""
        groups = defaultdict(list)

        values = [obj['measurement_value'] for obj in objects if obj.get('measurement_value') is not None]

        if not values:
            groups['No Measurements'] = objects
            return groups

        min_val, max_val = min(values), max(values)
        range_size = (max_val - min_val) / 4 if max_val > min_val else 1

        for obj in objects:
            value = obj.get('measurement_value')
            if value is None:
                group_key = "No Value"
            else:
                range_idx = min(3, int((value - min_val) / range_size)) if range_size > 0 else 0
                group_key = f"Range {range_idx + 1} ({min_val + range_idx * range_size:.2f}-{min_val + (range_idx + 1) * range_size:.2f})"
            groups[group_key].append(obj)

        return groups

    def _extract_and_categorize_temporal_data(self) -> Dict[str, List]:
        """Extract and categorize all temporal data from the graph"""
        measurements = []
        events = []
        objects = []

        timeline_events = self._extract_timeline_events(self.graph_manager)

        if timeline_events:
            for event_data in timeline_events:
                category = self._categorize_timeline_event(event_data)
                if category == 'measurement':
                    measurements.append(event_data)
                elif category == 'object':
                    objects.append(event_data)
                else:
                    events.append(event_data)

            for category_data in [measurements, events, objects]:
                category_data.sort(key=lambda x: x['datetime'])

            return {
                'measurements': measurements,
                'events': events,
                'objects': objects
            }

        for node_id, node_data in self.graph_manager.node_data.items():
            if node_data.get('type') in ['Timeline', 'Timestamp']:
                continue

            temporal_info = self._extract_temporal_info(node_id, node_data)

            if temporal_info:
                data_item = self._create_comprehensive_data_item(node_id, node_data, temporal_info)
                category = self._categorize_temporal_data(node_data, temporal_info)

                if category == 'measurement':
                    measurements.append(data_item)
                elif category == 'event':
                    events.append(data_item)
                elif category == 'object':
                    objects.append(data_item)

        timestamp_linked_data = self._extract_timestamp_linked_entities()

        for data_item in timestamp_linked_data:
            temporal_info_for_categorization = {
                'datetime': data_item['datetime'],
                'value': data_item['measurement_value'],
                'source': 'timestamp_link'
            }

            category = self._categorize_temporal_data(data_item['raw_data'], temporal_info_for_categorization)

            if category == 'measurement':
                measurements.append(data_item)
            elif category == 'event':
                events.append(data_item)
            elif category == 'object':
                objects.append(data_item)

        for category_data in [measurements, events, objects]:
            category_data.sort(key=lambda x: x['datetime'])

        return {
            'measurements': measurements,
            'events': events,
            'objects': objects
        }

    def _categorize_timeline_event(self, event_data: Dict) -> str:
        """Categorize timeline events based on their source type"""
        source_type = event_data.get('temporal_source', '')

        if source_type == 'measurement_timestamp':
            return 'measurement'
        if source_type == 'object_lifecycle':
            return 'object'
        if source_type == 'validation_result':
            return 'event'

        return 'event'

    def _extract_timeline_events(self, graph_manager):
        """Extract events from Timeline nodes"""
        timeline_events = []

        for node_id, node_data in graph_manager.node_data.items():
            if node_data.get('type') != 'Timeline':
                continue

            timeline_data_raw = node_data.get('timeline_data')
            if not timeline_data_raw:
                continue

            try:
                if isinstance(timeline_data_raw, str):
                    events = json.loads(timeline_data_raw)
                else:
                    events = timeline_data_raw

                for event in events:
                    event_datetime = self._parse_datetime(event['datetime'])
                    if event_datetime:
                        timeline_events.append({
                            'node_id': event['entity_id'],
                            'node_type': event['entity_type'],
                            'datetime': event_datetime,
                            'display_name': event['entity_value'],
                            'node_summary': event['description'],
                            'measurement_value': event.get('temporal_value'),
                            'temporal_source': event['source_type'],
                            'temporal_attribute': event['source_property'],
                            'all_properties': event.get('context_properties', {}),
                            'raw_data': event
                        })

            except (json.JSONDecodeError, KeyError, TypeError):
                continue

        return timeline_events

    def _create_comprehensive_data_item(self, node_id: str, node_data: Dict, temporal_info: Dict) -> Dict:
        """Create a comprehensive data item with all node details"""

        display_name = self._get_comprehensive_display_name(node_id, node_data)
        measurement_value = self._extract_measurement_value(node_data)
        node_summary = self._get_node_summary(node_data)
        all_properties = self._extract_all_properties(node_data)

        return {
            'node_id': node_id,
            'node_type': node_data.get('type', 'Unknown'),
            'datetime': temporal_info['datetime'],
            'value': measurement_value,
            'measurement_value': measurement_value,
            'display_name': display_name,
            'node_summary': node_summary,
            'all_properties': all_properties,
            'raw_data': node_data,
            'temporal_source': temporal_info.get('source', 'unknown'),
            'temporal_attribute': temporal_info.get('attribute_name', ''),
        }

    def _get_node_summary(self, node_data: Dict) -> str:
        """Get a comprehensive summary of the node"""
        parts = []

        node_type = node_data.get('type', 'Unknown')
        parts.append(f"Type: {node_type}")

        if 'hierarchy' in node_data:
            parts.append(f"Hierarchy: {node_data['hierarchy']}")

        key_props = ['value', 'name', 'status', 'state', 'level']
        for prop in key_props:
            if prop in node_data and node_data[prop] is not None:
                value = str(node_data[prop])
                if len(value) < 50:
                    parts.append(f"{prop.title()}: {value}")

        attrs = node_data.get('attributes', {})
        for prop in key_props:
            if prop in attrs and attrs[prop] is not None:
                value = str(attrs[prop])
                if len(value) < 50:
                    parts.append(f"{prop.title()}: {value}")

        return " | ".join(parts)

    def _extract_all_properties(self, node_data: Dict) -> Dict[str, Any]:
        """Extract all properties for detailed display"""
        all_props = {}

        skip_keys = {'type', 'hierarchy', 'attributes'}
        for key, value in node_data.items():
            if key not in skip_keys and not key.startswith('_'):
                all_props[key] = value

        attrs = node_data.get('attributes', {})
        for key, value in attrs.items():
            if not key.startswith('_'):
                all_props[f"attr_{key}"] = value

        return all_props

    def _extract_measurement_value(self, node_data: Dict) -> Optional[float]:
        """Extract numeric value using ontology-discovered measurement properties"""

        if self.ontology_cache.get('measurement_properties'):
            for prop_name in self.ontology_cache['measurement_properties']:
                if '.' in prop_name:
                    parts = prop_name.split('.', 1)
                    if parts[0] == 'attributes':
                        attrs = node_data.get('attributes', {})
                        if parts[1] in attrs and isinstance(attrs[parts[1]], (int, float)):
                            return float(attrs[parts[1]])
                else:
                    if prop_name in node_data and isinstance(node_data[prop_name], (int, float)):
                        value = float(node_data[prop_name])
                        if not self._is_likely_structural_value(value, prop_name):
                            return value

        for key, value in node_data.items():
            if (isinstance(value, (int, float)) and
                    not self._is_structural_property(key) and
                    not self._is_likely_structural_value(value, key)):
                return float(value)

        attrs = node_data.get('attributes', {})
        for key, value in attrs.items():
            if (isinstance(value, (int, float)) and
                    not self._is_structural_property(key) and
                    not self._is_likely_structural_value(value, key)):
                return float(value)

        return None

    def _extract_timestamp_linked_entities(self) -> List[Dict]:
        """Extract entities that are linked to Timestamp nodes"""
        linked_entities_data = []

        for timestamp_id, timestamp_data in self.graph_manager.node_data.items():
            if timestamp_data.get('type') != 'Timestamp':
                continue

            dt = self._extract_datetime_from_timestamp_node(timestamp_data)
            if not dt:
                continue

            linked_entities = self._find_entities_linked_to_timestamp(timestamp_id)

            for entity_id, entity_data in linked_entities:
                temporal_info = {'datetime': dt, 'source': 'timestamp_link'}
                data_item = self._create_comprehensive_data_item(entity_id, entity_data, temporal_info)
                data_item['timestamp_source'] = timestamp_id
                linked_entities_data.append(data_item)

        return linked_entities_data

    def _find_entities_linked_to_timestamp(self, timestamp_id: str) -> List[Tuple[str, Dict]]:
        """Find entities that reference a timestamp node"""
        linked_entities = []

        for (source, target), edge_attrs in self.graph_manager.edge_data.items():
            if target == timestamp_id and edge_attrs.get('edge_type') == 'has_timestamp':
                entity_node = self.graph_manager.node_data.get(source)
                if entity_node:
                    linked_entities.append((source, entity_node))

        for (source, target), edge_attrs in self.graph_manager.edge_data.items():
            if source == timestamp_id:
                entity_node = self.graph_manager.node_data.get(target)
                if entity_node:
                    linked_entities.append((target, entity_node))

        return linked_entities

    def _extract_temporal_info(self, node_id: str, node_data: Dict) -> Optional[Dict]:
        """Extract temporal information from a node"""

        temporal_attrs = self._find_temporal_attributes(node_data)
        if temporal_attrs:
            dt = self._parse_datetime(temporal_attrs[0][1])
            if dt:
                return {
                    'datetime': dt,
                    'source': 'temporal_attribute',
                    'attribute_name': temporal_attrs[0][0],
                    'value': self._extract_measurement_value(node_data)
                }

        return None

    def _find_temporal_attributes(self, node_data: Dict) -> List[Tuple[str, Any]]:
        """Find temporal attributes in node data"""
        temporal_attrs = []
        temporal_keywords = ['time', 'date', 'created', 'modified', 'updated', 'timestamp', 'execution_time']

        for key, value in node_data.items():
            if any(keyword in key.lower() for keyword in temporal_keywords):
                temporal_attrs.append((key, value))

        attrs = node_data.get('attributes', {})
        for key, value in attrs.items():
            if any(keyword in key.lower() for keyword in temporal_keywords):
                temporal_attrs.append((f"attributes.{key}", value))

        return temporal_attrs

    def _extract_datetime_from_timestamp_node(self, timestamp_data: Dict) -> Optional[datetime]:
        """Extract datetime from timestamp node components"""
        try:
            attrs = timestamp_data.get('attributes', {})
            year = timestamp_data.get('year', attrs.get('year'))
            month = timestamp_data.get('month', attrs.get('month'))
            day = timestamp_data.get('day', attrs.get('day'))
            hour = timestamp_data.get('hour', attrs.get('hour', 0))
            minute = timestamp_data.get('minute', attrs.get('minute', 0))
            second = timestamp_data.get('second', attrs.get('second', 0))

            if year and month and day:
                return datetime(
                    year=int(year), month=int(month), day=int(day),
                    hour=int(hour) if hour is not None else 0,
                    minute=int(minute) if minute is not None else 0,
                    second=int(second) if second is not None else 0
                )
            return None
        except (ValueError, TypeError):
            return None

    def _parse_datetime(self, value: Any) -> Optional[datetime]:
        """Parse datetime from various formats"""
        if isinstance(value, datetime):
            return value.replace(tzinfo=None) if value.tzinfo else value

        if not isinstance(value, str):
            return None

        formats = [
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
        ]

        for fmt in formats:
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue

        try:
            dt = datetime.fromisoformat(value.replace('Z', '+00:00'))
            return dt.replace(tzinfo=None)
        except ValueError:
            pass

        return None

    def _get_measurement_group_key(self, measurement: Dict) -> str:
        """Get a meaningful group key for measurements"""
        node_id = measurement['node_id']
        node_type = measurement['node_type']

        parent_context = self._find_parent_context(node_id)

        if parent_context:
            return f"{parent_context} - {node_type}"
        else:
            display_name = measurement.get('display_name', node_type)
            return f"{node_type} ({display_name})" if display_name != node_type else node_type

    def _get_comprehensive_display_name(self, node_id: str, node_data: Dict) -> str:
        """Get the best possible display name"""

        if self.ontology_cache.get('naming_properties'):
            name_candidates = self.ontology_cache['naming_properties']
        else:
            name_candidates = [
                'name', 'title', 'label', 'value', 'description',
                'entity_name', 'display_name', 'filename'
            ]

        for candidate in name_candidates:
            if candidate in node_data and node_data[candidate]:
                value = str(node_data[candidate]).strip()
                if value and value.lower() != 'none':
                    return value

        attrs = node_data.get('attributes', {})
        for candidate in name_candidates:
            if candidate in attrs and attrs[candidate]:
                value = str(attrs[candidate]).strip()
                if value and value.lower() != 'none':
                    return value

        return node_id

    def _categorize_temporal_data(self, node_data: Dict, temporal_info: Dict) -> str:
        """Categorize temporal data using ontology-derived rules"""
        node_type = node_data.get('type', '').lower()
        temporal_source = temporal_info.get('source', '')
        is_timestamp_linked = (temporal_source == 'timestamp_link')
        has_measurement_value = temporal_info.get('value') is not None

        if is_timestamp_linked and has_measurement_value:
            return 'measurement'

        if self.ontology_cache.get('categorization_rules'):
            rules = self.ontology_cache['categorization_rules']

            if node_type in rules['object_types']:
                return 'object'
            elif node_type in rules['event_types']:
                return 'event'

        return 'event'

    def _find_parent_context(self, node_id: str) -> Optional[str]:
        """Find parent context using ontology-discovered hierarchy patterns"""

        current_node_data = self.graph_manager.node_data.get(node_id, {})
        current_node_type = current_node_data.get('type', '')

        for (source, target), edge_attrs in self.graph_manager.edge_data.items():
            if target == node_id:
                source_node = self.graph_manager.node_data.get(source)
                if source_node:
                    source_type = source_node.get('type', '')

                    if self._is_hierarchical_relationship(source_type, edge_attrs.get('edge_type', ''),
                                                          current_node_type):
                        parent_name = self._extract_meaningful_name(source_node)
                        if parent_name:
                            return parent_name

                        grandparent = self._find_parent_context(source)
                        if grandparent:
                            return grandparent

        return None

    def _is_hierarchical_relationship(self, source_type: str, edge_type: str, target_type: str) -> bool:
        """Determine if a relationship represents hierarchy using ontology patterns"""
        if not self.ontology_cache.get('hierarchy_patterns'):
            return False

        patterns = self.ontology_cache['hierarchy_patterns']

        for pattern_list in [patterns['parent_child_edges'], patterns['container_edges'], patterns['ownership_edges']]:
            for pattern in pattern_list:
                if (pattern['source_type'] == source_type and
                        pattern['edge_type'] == edge_type and
                        pattern['target_type'] == target_type):
                    return True

        return False

    def _is_structural_property(self, prop_name: str) -> bool:
        """Determine if a property is structural rather than measurement"""
        structural_indicators = [
            'year', 'month', 'day', 'hour', 'minute', 'second',
            'hierarchy', 'level', 'order', 'position', 'index', 'id'
        ]

        prop_lower = prop_name.lower()
        return any(indicator in prop_lower for indicator in structural_indicators)

    def _extract_meaningful_name(self, node_data: Dict) -> Optional[str]:
        """Extract meaningful name from any node type"""

        if self.ontology_cache.get('naming_properties'):
            name_candidates = self.ontology_cache['naming_properties']
        else:
            name_candidates = ['name', 'title', 'label', 'value']

        for candidate in name_candidates:
            if candidate in node_data and node_data[candidate]:
                value = str(node_data[candidate]).strip()
                if value and value.lower() not in ['none', 'unknown', '']:
                    return self._clean_extracted_name(value)

        return None

    def _clean_extracted_name(self, name: str) -> str:
        """Clean extracted names"""
        cleaned = name.split('\\')[-1].split('/')[-1]

        if '.' in cleaned and len(cleaned.split('.')[-1]) <= 4:
            cleaned = cleaned.rsplit('.', 1)[0]

        return cleaned if cleaned else name

    def _analyze_timestamp_linkage_for_type(self, node_type: str) -> Dict[str, Any]:
        """Analyze how often this node type links to timestamps"""
        analysis = {
            'has_direct_timestamp_links': False,
            'timestamp_link_percentage': 0.0
        }

        nodes_of_type = [
            node_id for node_id, node_data in self.graph_manager.node_data.items()
            if node_data.get('type') == node_type
        ]

        if not nodes_of_type:
            return analysis

        timestamp_linked_count = sum(
            1 for node_id in nodes_of_type if self._node_links_to_timestamp(node_id)
        )

        analysis['timestamp_link_percentage'] = timestamp_linked_count / len(nodes_of_type)
        analysis['has_direct_timestamp_links'] = analysis['timestamp_link_percentage'] > 0.5

        return analysis

    def _analyze_numeric_properties_for_type(self, node_type: str) -> float:
        """Calculate ratio of numeric properties for a node type"""
        try:
            prop_stats = self.ontology_manager.get_type_property_statistics(node_type, is_node_type=True)
            total_props = len(prop_stats['properties'])

            if total_props == 0:
                return 0.0

            numeric_props = sum(
                1 for prop_info in prop_stats['properties'].values()
                if prop_info['type_info']['primary_type'] in ['int', 'float']
            )

            return numeric_props / total_props

        except Exception:
            return 0.0

    def _analyze_relationship_patterns_for_type(self, node_type: str) -> Dict[str, float]:
        """Analyze relationship patterns for categorization hints"""
        patterns = {
            'container_score': 0.0,
            'temporal_score': 0.0
        }

        try:
            node_analysis = self.ontology_manager.analyze_node_type(node_type)
            relationships = node_analysis.get('relationships', {})
            outgoing_edges = relationships.get('outgoing', {})

            for edge_type, targets in outgoing_edges.items():
                if len(targets) > 1:
                    patterns['container_score'] += 0.5

                for target_type in targets:
                    if target_type == 'Timestamp':
                        patterns['temporal_score'] += 1.0

        except Exception:
            pass

        return patterns

    def _node_links_to_timestamp(self, node_id: str) -> bool:
        """Check if a specific node links to a timestamp"""
        for (source, target), edge_attrs in self.graph_manager.edge_data.items():
            if source == node_id:
                target_node = self.graph_manager.node_data.get(target, {})
                if target_node.get('type') == 'Timestamp':
                    return True
        return False

    def _is_likely_structural_value(self, value: float, prop_name: str) -> bool:
        """Determine if a numeric value is likely structural"""
        if isinstance(value, int) and 0 <= value <= 10:
            return True

        if isinstance(value, int) and value > 1000000:
            return True

        return False

    def _sort_objects_for_display(self, objects: List[Dict], sort_criteria: Dict = None) -> List[Dict]:
        """Sort objects with multiple strategies"""
        if not objects:
            return objects

        if not sort_criteria:
            sort_criteria = self._auto_detect_sorting_strategy(objects)

        sort_method = sort_criteria.get('method', 'display_name')
        sort_property = sort_criteria.get('property', None)
        reverse_order = sort_criteria.get('reverse', False)

        try:
            if sort_method == 'property_value' and sort_property:
                return self._sort_by_property_value(objects, sort_property, reverse_order)
            elif sort_method == 'parent_relationship':
                return self._sort_by_parent_relationships(objects, reverse_order)
            elif sort_method == 'child_count':
                return self._sort_by_child_count(objects, reverse_order)
            elif sort_method == 'measurement_value':
                return self._sort_by_measurement_value(objects, reverse_order)
            else:
                return sorted(objects, key=lambda x: x['display_name'].lower(), reverse=reverse_order)

        except Exception:
            return sorted(objects, key=lambda x: x['display_name'].lower())

    def _auto_detect_sorting_strategy(self, objects: List[Dict]) -> Dict:
        """Auto-detect the best sorting strategy"""
        if not objects:
            return {'method': 'display_name'}

        property_analysis = self._analyze_group_properties(objects)
        relationship_analysis = self._analyze_group_relationships(objects)

        if property_analysis['best_numeric_property']:
            return {
                'method': 'property_value',
                'property': property_analysis['best_numeric_property'],
                'reverse': False
            }

        if relationship_analysis['has_varied_parents']:
            return {
                'method': 'parent_relationship',
                'reverse': False
            }

        if any(obj.get('measurement_value') is not None for obj in objects):
            return {
                'method': 'measurement_value',
                'reverse': False
            }

        return {'method': 'display_name'}

    def _analyze_group_properties(self, objects: List[Dict]) -> Dict:
        """Analyze properties across a group of objects"""
        analysis = {
            'best_numeric_property': None,
            'best_string_property': None,
            'property_variance': {}
        }

        all_properties = defaultdict(list)
        for obj in objects:
            for prop_name, prop_value in obj.get('all_properties', {}).items():
                if prop_value is not None:
                    all_properties[prop_name].append(prop_value)

        for prop_name, values in all_properties.items():
            if len(values) < len(objects) * 0.7:
                continue

            if all(isinstance(v, (int, float)) for v in values):
                variance = len(set(values)) / len(values)
                if variance > 0.3:
                    analysis['property_variance'][prop_name] = variance
                    if not analysis['best_numeric_property']:
                        analysis['best_numeric_property'] = prop_name

        return analysis

    def _analyze_group_relationships(self, objects: List[Dict]) -> Dict:
        """Analyze relationships across a group of objects"""
        analysis = {
            'has_varied_parents': False,
            'has_varied_children': False,
            'common_parent_types': Counter(),
            'common_child_types': Counter()
        }

        parent_types = []
        for obj in objects:
            parents = self._get_object_parents(obj['node_id'])
            parent_types.extend([p['type'] for p in parents])
            analysis['common_parent_types'].update([p['type'] for p in parents])

        if len(set(parent_types)) > 1 and len(parent_types) > 0:
            analysis['has_varied_parents'] = True

        return analysis

    def _sort_by_property_value(self, objects: List[Dict], property_name: str, reverse: bool = False) -> List[Dict]:
        """Sort objects by a specific property value"""

        def get_property_value(obj):
            all_props = obj.get('all_properties', {})
            value = all_props.get(property_name)

            if value is None:
                return (1, 0) if not reverse else (0, 0)

            if isinstance(value, (int, float)):
                return (0, value) if not reverse else (0, -value)

            return (0, str(value).lower()) if not reverse else (0, str(value).lower())

        return sorted(objects, key=get_property_value, reverse=reverse)

    def _sort_by_parent_relationships(self, objects: List[Dict], reverse: bool = False) -> List[Dict]:
        """Sort objects by their parent relationships"""

        def get_parent_sort_key(obj):
            parents = self._get_object_parents(obj['node_id'])
            if not parents:
                return (1, "")

            primary_parent = parents[0]
            return (0, primary_parent.get('type', ''), primary_parent.get('name', ''))

        return sorted(objects, key=get_parent_sort_key, reverse=reverse)

    def _sort_by_child_count(self, objects: List[Dict], reverse: bool = False) -> List[Dict]:
        """Sort objects by number of children"""

        def get_child_count(obj):
            children = self._get_object_children(obj['node_id'])
            return len(children)

        return sorted(objects, key=get_child_count, reverse=reverse)

    def _sort_by_measurement_value(self, objects: List[Dict], reverse: bool = False) -> List[Dict]:
        """Sort objects by measurement value"""

        def get_measurement_value(obj):
            value = obj.get('measurement_value')
            if value is None:
                return (1, 0) if not reverse else (0, 0)
            return (0, value) if not reverse else (0, -value)

        return sorted(objects, key=get_measurement_value, reverse=reverse)

    def _get_object_parents(self, node_id: str) -> List[Dict]:
        """Get parent nodes for an object"""
        parents = []
        for (source, target), edge_attrs in self.graph_manager.edge_data.items():
            if target == node_id:
                parent_node = self.graph_manager.node_data.get(source)
                if parent_node:
                    parents.append({
                        'id': source,
                        'type': parent_node.get('type', ''),
                        'name': self._get_comprehensive_display_name(source, parent_node),
                        'edge_type': edge_attrs.get('edge_type', '')
                    })
        return parents

    def _get_object_children(self, node_id: str) -> List[Dict]:
        """Get child nodes for an object"""
        children = []
        for (source, target), edge_attrs in self.graph_manager.edge_data.items():
            if source == node_id:
                child_node = self.graph_manager.node_data.get(target)
                if child_node:
                    children.append({
                        'id': target,
                        'type': child_node.get('type', ''),
                        'name': self._get_comprehensive_display_name(target, child_node),
                        'edge_type': edge_attrs.get('edge_type', '')
                    })
        return children

    # ============ VISUALIZATION GENERATION ============

    def generate_timeline_html(self, temporal_data: Dict, visualization_mode: str = 'timeline',
                               full_html: bool = False) -> str:
        """Generate timeline visualization in requested mode"""
        if visualization_mode == 'histogram':
            return self._generate_histogram(temporal_data, full_html)
        else:
            return self._generate_timeline(temporal_data, full_html)

    def _generate_timeline(self, temporal_data: Dict, full_html: bool = False) -> str:
        """Generate timeline visualization with improved visual design"""

        # Filter data based on category selection
        filtered_data = self._apply_category_filter(temporal_data)

        # Count how many categories we're showing
        categories_to_show = sum(1 for v in filtered_data.values() if len(v) > 0)

        if categories_to_show == 0:
            return self._generate_empty_plot("No data available for selected category")

        # Create single plot for selected category
        fig = go.Figure()

        if filtered_data['measurements']:
            self._add_timeline_measurements(fig, filtered_data['measurements'])

        if filtered_data['events']:
            self._add_timeline_events(fig, filtered_data['events'])

        if filtered_data['objects']:
            self._add_timeline_objects(fig, filtered_data['objects'])

        self._update_timeline_layout_single(fig, filtered_data)
        return self._render_html(fig, full_html)

    def _generate_histogram(self, temporal_data: Dict, full_html: bool = False) -> str:
        """Generate histogram visualization with improved scaling"""

        # Filter data based on category selection
        filtered_data = self._apply_category_filter(temporal_data)

        # Count how many categories we're showing
        categories_to_show = sum(1 for v in filtered_data.values() if len(v) > 0)

        if categories_to_show == 0:
            return self._generate_empty_plot("No data available for selected category")

        # Create single plot
        fig = go.Figure()

        if filtered_data['measurements']:
            self._add_histogram_measurements(fig, filtered_data['measurements'])

        if filtered_data['events']:
            self._add_histogram_events(fig, filtered_data['events'])

        if filtered_data['objects']:
            self._add_histogram_objects(fig, filtered_data['objects'])

        self._update_histogram_layout_single(fig, filtered_data)
        return self._render_html(fig, full_html)

    # ============ TIMELINE VISUALIZATION METHODS ============

    def _add_timeline_events(self, fig: go.Figure, events: List[Dict]):
        """Add events as timeline lanes with improved visual design"""
        if not events:
            return

        event_groups = defaultdict(list)
        for event in events:
            event_groups[event['node_type']].append(event)

        color_idx = 0
        y_position = 0

        for group_name, group_data in event_groups.items():
            color = self.color_palette['events'][color_idx % len(self.color_palette['events'])]
            group_data.sort(key=lambda x: x['datetime'])

            timestamps = [item['datetime'] for item in group_data]
            y_values = [y_position] * len(group_data)

            # Add lane background with gradient
            if timestamps:
                min_time = min(timestamps)
                max_time = max(timestamps)
                duration = (max_time - min_time).total_seconds()

                fig.add_shape(
                    type="rect",
                    x0=min_time, x1=max_time,
                    y0=y_position - 0.4, y1=y_position + 0.4,
                    fillcolor=color, opacity=0.15,
                    line=dict(width=1, color=color, dash='dot'),
                    layer="below"
                )

                # Add connecting line through events
                fig.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=y_values,
                        mode='lines',
                        line=dict(color=color, width=1, dash='dot'),
                        showlegend=False,
                        hoverinfo='skip'
                    )
                )

            # Calculate statistics for this group
            event_count = len(group_data)
            time_span = self._format_time_span(timestamps) if len(timestamps) > 1 else "Single event"

            hover_texts = [
                f"<b>{group_name}</b><br>"
                f"<b>{item['display_name']}</b><br>"
                f"Time: {item['datetime'].strftime('%Y-%m-%d %H:%M:%S')}<br>"
                f"Event {idx + 1} of {event_count}<br>"
                f"<i>Span: {time_span}</i>"
                for idx, item in enumerate(group_data)
            ]

            # Add markers with varying sizes based on position
            marker_sizes = self._calculate_marker_sizes(group_data, event_count)

            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=y_values,
                    mode='markers+text',
                    name=f"{group_name} ({event_count})",
                    marker=dict(
                        size=marker_sizes,
                        color=color,
                        symbol='diamond',
                        line=dict(color='white', width=2),
                        opacity=0.9
                    ),
                    text=[f"{i + 1}" if event_count <= 20 else "" for i in range(event_count)],
                    textposition="middle center",
                    textfont=dict(size=8, color='white', family='Arial Black'),
                    hovertext=hover_texts,
                    hovertemplate='%{hovertext}<extra></extra>',
                    showlegend=True
                )
            )

            # Add start and end annotations
            if len(timestamps) > 1:
                # Start marker
                fig.add_annotation(
                    x=timestamps[0],
                    y=y_position,
                    text="START",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor=color,
                    ax=-40,
                    ay=-30,
                    font=dict(size=9, color=color, family='Arial'),
                    bgcolor='white',
                    bordercolor=color,
                    borderwidth=1,
                    borderpad=2
                )

                # End marker
                fig.add_annotation(
                    x=timestamps[-1],
                    y=y_position,
                    text="END",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor=color,
                    ax=40,
                    ay=-30,
                    font=dict(size=9, color=color, family='Arial'),
                    bgcolor='white',
                    bordercolor=color,
                    borderwidth=1,
                    borderpad=2
                )

            y_position += 1
            color_idx += 1

        # Update y-axis to show lane labels with counts
        lane_labels = [f"{name}" for name in event_groups.keys()]
        fig.update_yaxes(
            ticktext=lane_labels,
            tickvals=list(range(len(event_groups))),
            showgrid=False,
            side='left'
        )

    def _add_timeline_measurements(self, fig: go.Figure, measurements: List[Dict]):
        """Add measurements with improved timeline visualization"""
        if not measurements:
            return

        measurement_groups = defaultdict(list)
        for measurement in measurements:
            group_key = self._get_measurement_group_key(measurement)
            measurement_groups[group_key].append(measurement)

        color_idx = 0
        y_position = 0

        for group_name, group_data in measurement_groups.items():
            valid_data = [item for item in group_data if item.get('measurement_value') is not None]
            if not valid_data:
                continue

            color = self.color_palette['measurements'][color_idx % len(self.color_palette['measurements'])]
            valid_data.sort(key=lambda x: x['datetime'])

            timestamps = [item['datetime'] for item in valid_data]
            values = [item['measurement_value'] for item in valid_data]
            y_values = [y_position] * len(valid_data)

            # Add lane background
            if timestamps:
                min_time = min(timestamps)
                max_time = max(timestamps)

                fig.add_shape(
                    type="rect",
                    x0=min_time, x1=max_time,
                    y0=y_position - 0.4, y1=y_position + 0.4,
                    fillcolor=color, opacity=0.15,
                    line=dict(width=1, color=color, dash='dot'),
                    layer="below"
                )

                # Add connecting line
                fig.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=y_values,
                        mode='lines',
                        line=dict(color=color, width=1, dash='dot'),
                        showlegend=False,
                        hoverinfo='skip'
                    )
                )

            # Calculate statistics
            measurement_count = len(valid_data)
            time_span = self._format_time_span(timestamps) if len(timestamps) > 1 else "Single measurement"
            avg_value = sum(values) / len(values) if values else 0
            min_value = min(values) if values else 0
            max_value = max(values) if values else 0

            hover_texts = [
                f"<b>{group_name}</b><br>"
                f"<b>{item['display_name']}</b><br>"
                f"Value: <b>{item['measurement_value']:.2f}</b><br>"
                f"Time: {item['datetime'].strftime('%Y-%m-%d %H:%M:%S')}<br>"
                f"Measurement {idx + 1} of {measurement_count}<br>"
                f"Avg: {avg_value:.2f} | Min: {min_value:.2f} | Max: {max_value:.2f}<br>"
                f"<i>Span: {time_span}</i>"
                for idx, item in enumerate(valid_data)
            ]

            # Add markers with size based on value
            normalized_sizes = self._normalize_marker_sizes_by_value(values)

            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=y_values,
                    mode='markers+text',
                    name=f"{group_name} ({measurement_count})",
                    marker=dict(
                        size=normalized_sizes,
                        color=values,
                        colorscale='Viridis',
                        symbol='circle',
                        line=dict(color='white', width=2),
                        opacity=0.9,
                        showscale=True,
                        colorbar=dict(
                            title="Value",
                            x=1.15,
                            len=0.3,
                            y=y_position / max(1, len(measurement_groups))
                        )
                    ),
                    text=[f"{v:.1f}" if measurement_count <= 15 else "" for v in values],
                    textposition="middle center",
                    textfont=dict(size=8, color='white', family='Arial Black'),
                    hovertext=hover_texts,
                    hovertemplate='%{hovertext}<extra></extra>',
                    showlegend=True
                )
            )

            # Add trend line if enough data points
            if len(valid_data) >= 3:
                self._add_trend_line(fig, timestamps, y_values, values, color, y_position)

            y_position += 1
            color_idx += 1

        # Update y-axis
        lane_labels = [f"{name}" for name in measurement_groups.keys()]
        fig.update_yaxes(
            ticktext=lane_labels,
            tickvals=list(range(len(measurement_groups))),
            showgrid=False,
            side='left'
        )

    def _add_timeline_objects(self, fig: go.Figure, objects: List[Dict]):
        """Add objects with improved timeline visualization"""
        if not objects:
            return

        if hasattr(self, 'custom_sort_criteria') and self.custom_sort_criteria:
            sorted_objects = self._sort_objects_for_display(objects, self.custom_sort_criteria)
        else:
            sorted_objects = self._sort_objects_for_display(objects)

        if hasattr(self, 'custom_color_strategy') and self.custom_color_strategy:
            object_groups = self._group_objects_by_strategy(sorted_objects, self.custom_color_strategy)
        else:
            object_groups = defaultdict(list)
            for obj in sorted_objects:
                object_groups[obj['node_type']].append(obj)

        color_idx = 0
        y_position = 0

        for group_name, group_data in object_groups.items():
            color = self.color_palette['objects'][color_idx % len(self.color_palette['objects'])]
            group_data.sort(key=lambda x: x['datetime'])

            timestamps = [item['datetime'] for item in group_data]
            y_values = [y_position] * len(group_data)

            # Add lane background
            if timestamps:
                min_time = min(timestamps)
                max_time = max(timestamps)

                fig.add_shape(
                    type="rect",
                    x0=min_time, x1=max_time,
                    y0=y_position - 0.4, y1=y_position + 0.4,
                    fillcolor=color, opacity=0.15,
                    line=dict(width=1, color=color, dash='dot'),
                    layer="below"
                )

                # Add connecting line
                fig.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=y_values,
                        mode='lines',
                        line=dict(color=color, width=1, dash='dot'),
                        showlegend=False,
                        hoverinfo='skip'
                    )
                )

            # Calculate statistics
            object_count = len(group_data)
            time_span = self._format_time_span(timestamps) if len(timestamps) > 1 else "Single occurrence"

            hover_texts = [
                f"<b>{group_name}</b><br>"
                f"<b>{item['display_name']}</b><br>"
                f"Time: {item['datetime'].strftime('%Y-%m-%d %H:%M:%S')}<br>"
                f"Object {idx + 1} of {object_count}<br>"
                f"<i>Span: {time_span}</i>"
                for idx, item in enumerate(group_data)
            ]

            # Add markers
            marker_sizes = self._calculate_marker_sizes(group_data, object_count)

            fig.add_trace(
                go.Scatter(
                    x=timestamps,
                    y=y_values,
                    mode='markers+text',
                    name=f"{group_name} ({object_count})",
                    marker=dict(
                        size=marker_sizes,
                        color=color,
                        symbol='square',
                        line=dict(color='white', width=2),
                        opacity=0.9
                    ),
                    text=[f"{i + 1}" if object_count <= 20 else "" for i in range(object_count)],
                    textposition="middle center",
                    textfont=dict(size=8, color='white', family='Arial Black'),
                    hovertext=hover_texts,
                    hovertemplate='%{hovertext}<extra></extra>',
                    showlegend=True
                )
            )

            y_position += 1
            color_idx += 1

        # Update y-axis
        lane_labels = [f"{name}" for name in object_groups.keys()]
        fig.update_yaxes(
            ticktext=lane_labels,
            tickvals=list(range(len(object_groups))),
            showgrid=False,
            side='left'
        )

    # ============ HELPER METHODS FOR TIMELINE ENHANCEMENTS ============

    def _format_time_span(self, timestamps: List[datetime]) -> str:
        """Format the time span between first and last timestamp"""
        if len(timestamps) < 2:
            return "N/A"

        time_diff = timestamps[-1] - timestamps[0]

        days = time_diff.days
        hours = time_diff.seconds // 3600
        minutes = (time_diff.seconds % 3600) // 60

        if days > 365:
            years = days / 365
            return f"{years:.1f} years"
        elif days > 30:
            months = days / 30
            return f"{months:.1f} months"
        elif days > 0:
            return f"{days} days, {hours} hours"
        elif hours > 0:
            return f"{hours} hours, {minutes} min"
        else:
            return f"{minutes} minutes"

    def _calculate_marker_sizes(self, data: List[Dict], total_count: int) -> List[int]:
        """Calculate marker sizes based on position and total count"""
        if total_count <= 10:
            # Larger markers for fewer items
            base_size = 16
            return [base_size] * total_count  # Consistent size
        elif total_count <= 50:
            # Medium markers
            base_size = 14
            return [base_size] * total_count  # Consistent size
        else:
            # Smaller markers for many items
            base_size = 12
            return [base_size] * total_count  # Consistent size

    def _normalize_marker_sizes_by_value(self, values: List[float]) -> List[int]:
        """Normalize marker sizes based on measurement values"""
        if not values:
            return [14]

        min_val = min(values)
        max_val = max(values)

        if max_val == min_val:
            return [14] * len(values)

        # Normalize to size range 10-20
        normalized = []
        for val in values:
            normalized_size = 10 + ((val - min_val) / (max_val - min_val)) * 10
            normalized.append(int(normalized_size))

        return normalized

    def _add_trend_line(self, fig: go.Figure, timestamps: List[datetime], y_values: List[float],
                        values: List[float], color: str, y_position: int):
        """Add a trend line for measurement data"""
        # Convert timestamps to numeric values for regression
        x_numeric = [(t - timestamps[0]).total_seconds() for t in timestamps]

        # Simple linear regression
        n = len(x_numeric)
        sum_x = sum(x_numeric)
        sum_y = sum(values)
        sum_xy = sum(x * y for x, y in zip(x_numeric, values))
        sum_x2 = sum(x * x for x in x_numeric)

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x) if (n * sum_x2 - sum_x * sum_x) != 0 else 0
        intercept = (sum_y - slope * sum_x) / n

        # Calculate trend line values
        trend_y = [slope * x + intercept for x in x_numeric]

        # Add as a subtle line above the data
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=[y_position + 0.15] * len(timestamps),
                mode='lines',
                line=dict(color=color, width=2, dash='dash'),
                showlegend=False,
                hoverinfo='skip',
                opacity=0.4
            )
        )

    def _update_timeline_layout_single(self, fig: go.Figure, filtered_data: Dict):
        """Update layout for single-category timeline"""
        # Determine which category is being shown
        category_name = ''
        total_items = 0
        if filtered_data['measurements']:
            category_name = 'Measurements'
            total_items = len(filtered_data['measurements'])
        elif filtered_data['events']:
            category_name = 'Events'
            total_items = len(filtered_data['events'])
        elif filtered_data['objects']:
            category_name = 'Objects'
            total_items = len(filtered_data['objects'])

        # Calculate time range for subtitle
        all_items = filtered_data['measurements'] + filtered_data['events'] + filtered_data['objects']
        if all_items:
            all_timestamps = [item['datetime'] for item in all_items]
            min_time = min(all_timestamps)
            max_time = max(all_timestamps)
            time_range = f"{min_time.strftime('%Y-%m-%d')} to {max_time.strftime('%Y-%m-%d')}"
            total_span = self._format_time_span([min_time, max_time])
        else:
            time_range = "No data"
            total_span = "N/A"

        # Count unique groups (lanes) to determine height
        unique_groups = 0
        if filtered_data['measurements']:
            measurement_groups = defaultdict(list)
            for m in filtered_data['measurements']:
                measurement_groups[self._get_measurement_group_key(m)].append(m)
            unique_groups = len(measurement_groups)
        elif filtered_data['events']:
            event_groups = defaultdict(list)
            for e in filtered_data['events']:
                event_groups[e['node_type']].append(e)
            unique_groups = len(event_groups)
        elif filtered_data['objects']:
            object_groups = defaultdict(list)
            for obj in filtered_data['objects']:
                object_groups[obj['node_type']].append(obj)
            unique_groups = len(object_groups)

        # Calculate sensible height: 80px per lane + 200px for margins/title
        calculated_height = min(900, max(400, unique_groups * 80 + 200))

        fig.update_layout(
            title=dict(
                text=f'<b>Timeline View - {category_name}</b><br>'
                     f'<sub>{total_items} items | {time_range} | Duration: {total_span}</sub>',
                x=0.5,
                font=dict(size=20)
            ),
            height=calculated_height,
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
                font=dict(size=10)
            ),
            hovermode='closest',
            template='plotly_white',
            plot_bgcolor='rgba(250,250,250,1)',
            xaxis=dict(
                title='<b>Time</b>',
                type='date',
                showgrid=True,
                gridcolor='rgba(200,200,200,0.3)',
                gridwidth=1,
                zeroline=False,
                tickformat='%Y-%m-%d<br>%H:%M:%S'
            ),
            yaxis=dict(
                title='',
                showgrid=False,
                zeroline=False,
                fixedrange=True
            ),
            margin=dict(l=150, r=200, t=100, b=80)
        )

    # ============ HISTOGRAM METHODS ============

    def _add_histogram_events(self, fig: go.Figure, events: List[Dict]):
        """Add event frequency histogram with improved scaling"""
        if not events:
            return

        event_groups = defaultdict(list)
        for event in events:
            event_groups[event['node_type']].append(event)

        color_idx = 0

        for group_name, group_data in event_groups.items():
            frequency_data = self._create_frequency_buckets(group_data)
            if not frequency_data:
                continue

            bucket_times = [item['bucket_time'] for item in frequency_data]
            bucket_counts = [item['count'] for item in frequency_data]

            color = self.color_palette['events'][color_idx % len(self.color_palette['events'])]

            fig.add_trace(
                go.Bar(
                    x=bucket_times,
                    y=bucket_counts,
                    name=group_name,
                    marker=dict(
                        color=color,
                        line=dict(color=color, width=1),
                        opacity=0.8
                    ),
                    hovertemplate='<b>%{fullData.name}</b><br>Time: %{x}<br>Count: %{y}<extra></extra>',
                    showlegend=True
                )
            )

            color_idx += 1

    def _add_histogram_measurements(self, fig: go.Figure, measurements: List[Dict]):
        """Add measurement frequency histogram"""
        if not measurements:
            return

        measurement_groups = defaultdict(list)
        for measurement in measurements:
            group_key = self._get_measurement_group_key(measurement)
            measurement_groups[group_key].append(measurement)

        color_idx = 0

        for group_name, group_data in measurement_groups.items():
            valid_data = [item for item in group_data if item.get('measurement_value') is not None]
            if not valid_data:
                continue

            frequency_data = self._create_frequency_buckets(valid_data)
            if not frequency_data:
                continue

            bucket_times = [item['bucket_time'] for item in frequency_data]
            bucket_counts = [item['count'] for item in frequency_data]

            color = self.color_palette['measurements'][color_idx % len(self.color_palette['measurements'])]

            fig.add_trace(
                go.Bar(
                    x=bucket_times,
                    y=bucket_counts,
                    name=group_name,
                    marker=dict(
                        color=color,
                        line=dict(color=color, width=1),
                        opacity=0.8
                    ),
                    hovertemplate='<b>%{fullData.name}</b><br>Time: %{x}<br>Count: %{y}<extra></extra>',
                    showlegend=True
                )
            )

            color_idx += 1

    def _add_histogram_objects(self, fig: go.Figure, objects: List[Dict]):
        """Add object frequency histogram"""
        if not objects:
            return

        if hasattr(self, 'custom_sort_criteria') and self.custom_sort_criteria:
            sorted_objects = self._sort_objects_for_display(objects, self.custom_sort_criteria)
        else:
            sorted_objects = self._sort_objects_for_display(objects)

        if hasattr(self, 'custom_color_strategy') and self.custom_color_strategy:
            object_groups = self._group_objects_by_strategy(sorted_objects, self.custom_color_strategy)
        else:
            object_groups = defaultdict(list)
            for obj in sorted_objects:
                object_groups[obj['node_type']].append(obj)

        color_idx = 0

        for group_name, group_data in object_groups.items():
            frequency_data = self._create_frequency_buckets(group_data)
            if not frequency_data:
                continue

            bucket_times = [item['bucket_time'] for item in frequency_data]
            bucket_counts = [item['count'] for item in frequency_data]

            color = self.color_palette['objects'][color_idx % len(self.color_palette['objects'])]

            fig.add_trace(
                go.Bar(
                    x=bucket_times,
                    y=bucket_counts,
                    name=group_name,
                    marker=dict(
                        color=color,
                        line=dict(color=color, width=1),
                        opacity=0.8
                    ),
                    hovertemplate='<b>%{fullData.name}</b><br>Time: %{x}<br>Count: %{y}<extra></extra>',
                    showlegend=True
                )
            )

            color_idx += 1

    def _update_histogram_layout_single(self, fig: go.Figure, filtered_data: Dict):
        """Update layout for single-category histogram with proper scaling"""
        # Determine which category is being shown
        category_name = ''
        if filtered_data['measurements']:
            category_name = 'Measurements'
        elif filtered_data['events']:
            category_name = 'Events'
        elif filtered_data['objects']:
            category_name = 'Objects'

        fig.update_layout(
            title=dict(
                text=f'Frequency Distribution - {category_name}',
                x=0.5,
                font=dict(size=20)
            ),
            height=600,
            width=1400,
            showlegend=True,
            hovermode='x unified',
            template='plotly_white',
            barmode='stack',
            plot_bgcolor='rgba(240,240,240,0.5)',
            paper_bgcolor='white',
            xaxis=dict(
                title='Time',
                type='date',
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                tickformat='%Y-%m-%d %H:%M:%S'
            ),
            yaxis=dict(
                title='Frequency (Count)',
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                zeroline=True,
                rangemode='tozero'
            )
        )

    def _create_frequency_buckets(self, items: List[Dict]) -> List[Dict]:
        """Create time-based frequency buckets"""
        if not items:
            return []

        timestamp_groups = defaultdict(list)
        for item in items:
            rounded_timestamp = item['datetime'].replace(microsecond=0)
            timestamp_groups[rounded_timestamp].append(item)

        frequency_data = []
        for bucket_time, items_in_bucket in sorted(timestamp_groups.items()):
            frequency_data.append({
                'bucket_time': bucket_time,
                'count': len(items_in_bucket),
                'items': items_in_bucket
            })

        return frequency_data

    def _render_html(self, fig: go.Figure, full_html: bool) -> str:
        """Render figure as HTML"""
        config = {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': ['lasso2d'],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'temporal_analysis',
                'height': 900,
                'width': 1400,
                'scale': 2
            }
        }

        if full_html:
            return fig.to_html(
                include_plotlyjs='cdn',
                div_id="temporal-analysis-plot",
                config=config
            )
        else:
            return fig.to_html(
                include_plotlyjs=True,
                div_id="temporal-analysis-plot",
                config=config,
                full_html=False
            )

    def get_available_options(self, graph_manager: GraphManager, ontology_manager=None) -> dict:
        """Get available sorting and filtering options"""

        self.graph_manager = graph_manager
        self.ontology_manager = ontology_manager

        if self.ontology_manager:
            self._cache_ontology_patterns()

        node_types = list(set(node_data.get('type', 'Unknown')
                              for node_data in graph_manager.node_data.values()))

        options = {
            'sortable_properties': [],
            'colorable_properties': [],
            'node_types': node_types,
            'measurement_properties': self.ontology_cache.get('measurement_properties', []),
            'naming_properties': self.ontology_cache.get('naming_properties', [])
        }

        all_properties = defaultdict(set)
        property_examples = defaultdict(list)

        for node_id, node_data in graph_manager.node_data.items():
            all_props = {}

            for key, value in node_data.items():
                if key not in ['type', 'hierarchy', 'attributes'] and not key.startswith('_'):
                    if value is not None:
                        all_props[key] = value

            attrs = node_data.get('attributes', {})
            for key, value in attrs.items():
                if not key.startswith('_') and value is not None:
                    all_props[f"attributes.{key}"] = value

            for prop_name, prop_value in all_props.items():
                all_properties[prop_name].add(type(prop_value).__name__)
                if len(property_examples[prop_name]) < 3:
                    property_examples[prop_name].append(str(prop_value)[:50])

        for prop_name, prop_types in all_properties.items():
            if len(prop_types) == 1:
                prop_type = list(prop_types)[0]
                display_name = prop_name.replace('_', ' ').replace('attributes.', '').title()

                property_info = {
                    'property_name': prop_name,
                    'display_name': display_name,
                    'type': prop_type,
                    'examples': property_examples[prop_name]
                }

                if prop_type in ['int', 'float']:
                    options['sortable_properties'].append(property_info)

                options['colorable_properties'].append(property_info)

        return options