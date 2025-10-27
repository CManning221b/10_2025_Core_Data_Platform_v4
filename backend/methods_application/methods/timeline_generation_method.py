# backend/methods_application/methods/timeline_generation_method.py
from backend.methods_application.method_implementation import MethodImplementation
import json
from datetime import datetime


class TimelineGenerationMethod(MethodImplementation):
    method_id = "TimelineGenerationMethod"
    method_name = "Timeline Generator"
    description = "Creates chronological timeline of all dated entities"

    def execute(self, graph_manager, parameters):
        changes = {
            "nodes_added": 0,
            "edges_added": 0,
            "timeline_entries": 0,
            "date_range": {}
        }

        # Find all Date nodes
        date_nodes = []
        for node_id, node_data in graph_manager.node_data.items():
            if node_data.get('type') == 'Date':
                date_nodes.append((node_id, node_data))

        if not date_nodes:
            return changes

        # Create Timeline node with consistent numeric ID
        timeline_id = self._get_next_numeric_id(graph_manager)

        # Collect all dated entities
        timeline_entries = []

        for date_id, date_data in date_nodes:
            try:
                date_str = date_data.get('parsed_datetime')
                if date_str:
                    dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))

                    # Find entities linked to this date
                    for edge_id, edge_attrs in graph_manager.edge_data.items():
                        if '_' in edge_id:
                            source, target = edge_id.split('_', 1)
                            if (target == date_id and edge_attrs.get('edge_type') == 'has_date'):
                                entity_node = graph_manager.node_data.get(source)
                                if entity_node:
                                    timeline_entries.append({
                                        'datetime': dt,
                                        'date_id': date_id,
                                        'entity_id': source,
                                        'entity_type': entity_node.get('type'),
                                        'entity_value': entity_node.get('value', ''),
                                        'date_text': date_data.get('original_text', ''),
                                        'pattern': date_data.get('pattern', '')
                                    })
            except (ValueError, TypeError):
                continue

        # Sort by datetime
        timeline_entries.sort(key=lambda x: x['datetime'])

        # Create timeline data
        timeline_data = []
        for i, entry in enumerate(timeline_entries):
            timeline_data.append({
                'order': i + 1,
                'date': entry['datetime'].isoformat(),
                'entity_id': entry['entity_id'],
                'entity_type': entry['entity_type'],
                'entity_value': entry['entity_value'],
                'date_text': entry['date_text'],
                'pattern': entry['pattern']
            })

        # Create Timeline node
        graph_manager.node_data[timeline_id] = {
            'type': 'Timeline',
            'value': f"Timeline of {len(timeline_entries)} entities",
            'hierarchy': 'analysis',
            'timeline_data': json.dumps(timeline_data),
            'entry_count': len(timeline_entries),
            'date_range_start': timeline_data[0]['date'] if timeline_data else None,
            'date_range_end': timeline_data[-1]['date'] if timeline_data else None,
            'created': datetime.now().isoformat()
        }
        changes["nodes_added"] += 1
        changes["timeline_entries"] = len(timeline_entries)

        if timeline_data:
            changes["date_range"] = {
                'start': timeline_data[0]['date'],
                'end': timeline_data[-1]['date']
            }

        return changes

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