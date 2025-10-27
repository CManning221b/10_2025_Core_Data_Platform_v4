# backend/methods_application/methods/ObjectLifespanAnalysisMethod.py
from backend.methods_application.method_implementation import MethodImplementation
from datetime import datetime
import json


class ObjectLifespanAnalysisMethod(MethodImplementation):
    method_id = "ObjectLifespanAnalysisMethod"
    method_name = "Object Lifespan Analysis Method"
    description = "Analyzes temporal spans and lifecycle patterns of archive objects"

    def execute(self, graph_manager, parameters):
        changes = {
            "nodes_added": 0,
            "edges_added": 0,
            "method_instances_created": 0,
            "objects_analyzed": 0,
            "analysis_summary": {}
        }

        # Create ONE method instance for the execution
        method_instance_id = self._create_method_instance(graph_manager, changes)

        # Find ALL objects with temporal data
        objects_with_lifespans = []
        for node_id, node_data in graph_manager.node_data.items():
            if node_data.get('type') == 'object':
                created = node_data.get('created')
                destroyed = node_data.get('destroyed')
                if created is not None and destroyed is not None:
                    objects_with_lifespans.append((node_id, node_data, created, destroyed))

        # Create ONE LifespanAnalysis node per individual object
        for object_id, object_data, created, destroyed in objects_with_lifespans:
            analysis_data = self._analyze_single_object_lifespan(
                object_id, object_data, created, destroyed
            )

            # Create individual LifespanAnalysis node for THIS object
            analysis_id = self._create_lifespan_analysis_node(
                graph_manager, analysis_data, changes
            )

            # Connect method to this specific analysis
            self._connect_method_to_input(graph_manager, method_instance_id, object_id, changes)
            self._connect_method_to_output(graph_manager, method_instance_id, analysis_id, changes)

            changes["objects_analyzed"] += 1

        return changes

    def _analyze_single_object_lifespan(self, object_id, object_data, created, destroyed):
        """Analyze lifespan of a single object"""
        duration = destroyed - created

        return {
            'object_id': object_id,
            'object_name': object_data.get('value', 'unknown'),
            'created': created,
            'destroyed': destroyed,
            'duration': duration,
            'lifespan_category': self._categorize_single_lifespan(duration)
        }

    def _categorize_single_lifespan(self, duration):
        """Categorize a single object's lifespan"""
        if duration <= 5:
            return 'short'
        elif duration <= 15:
            return 'medium'
        else:
            return 'long'

    def _analyze_object_lifespans(self, objects_with_lifespans):
        """Perform comprehensive lifespan analysis"""
        lifespans = []
        overlap_periods = []

        for object_id, object_data, created, destroyed in objects_with_lifespans:
            duration = destroyed - created
            lifespans.append({
                'object_id': object_id,
                'object_name': object_data.get('value', 'unknown'),
                'created': created,
                'destroyed': destroyed,
                'duration': duration
            })

        # Calculate summary statistics
        durations = [ls['duration'] for ls in lifespans]

        analysis = {
            'total_objects': len(lifespans),
            'average_lifespan': sum(durations) / len(durations) if durations else 0,
            'min_lifespan': min(durations) if durations else 0,
            'max_lifespan': max(durations) if durations else 0,
            'earliest_creation': min(ls['created'] for ls in lifespans) if lifespans else None,
            'latest_destruction': max(ls['destroyed'] for ls in lifespans) if lifespans else None,
            'archive_timespan': None,
            'temporal_overlaps': self._find_temporal_overlaps(lifespans),
            'lifespan_distribution': self._categorize_lifespans(durations),
            'individual_lifespans': lifespans
        }

        if analysis['earliest_creation'] and analysis['latest_destruction']:
            analysis['archive_timespan'] = analysis['latest_destruction'] - analysis['earliest_creation']

        return analysis

    def _find_temporal_overlaps(self, lifespans):
        """Find objects with overlapping lifespans"""
        overlaps = []

        for i, ls1 in enumerate(lifespans):
            for j, ls2 in enumerate(lifespans[i + 1:], i + 1):
                # Check if lifespans overlap
                overlap_start = max(ls1['created'], ls2['created'])
                overlap_end = min(ls1['destroyed'], ls2['destroyed'])

                if overlap_start < overlap_end:
                    overlap_duration = overlap_end - overlap_start
                    overlaps.append({
                        'object1_id': ls1['object_id'],
                        'object1_name': ls1['object_name'],
                        'object2_id': ls2['object_id'],
                        'object2_name': ls2['object_name'],
                        'overlap_start': overlap_start,
                        'overlap_end': overlap_end,
                        'overlap_duration': overlap_duration
                    })

        return overlaps

    def _categorize_lifespans(self, durations):
        """Categorize lifespans into short/medium/long"""
        if not durations:
            return {'short': 0, 'medium': 0, 'long': 0}

        avg_duration = sum(durations) / len(durations)

        categories = {'short': 0, 'medium': 0, 'long': 0}

        for duration in durations:
            if duration < avg_duration * 0.7:
                categories['short'] += 1
            elif duration > avg_duration * 1.3:
                categories['long'] += 1
            else:
                categories['medium'] += 1

        return categories

    def _create_lifespan_analysis_node(self, graph_manager, analysis_data, changes):
        """Create a LifespanAnalysis node"""
        analysis_id = self._get_next_numeric_id(graph_manager)

        try:
            graph_manager.add_node(
                node_id=analysis_id,
                value=f"Lifespan Analysis of {analysis_data['total_objects']} objects",
                type='LifespanAnalysis',
                hierarchy='analysis',
                attributes={
                    'total_objects': analysis_data['total_objects'],
                    'average_lifespan': analysis_data['average_lifespan'],
                    'min_lifespan': analysis_data['min_lifespan'],
                    'max_lifespan': analysis_data['max_lifespan'],
                    'earliest_creation': analysis_data['earliest_creation'],
                    'latest_destruction': analysis_data['latest_destruction'],
                    'archive_timespan': analysis_data['archive_timespan'],
                    'temporal_overlaps_count': len(analysis_data['temporal_overlaps']),
                    'lifespan_distribution': json.dumps(analysis_data['lifespan_distribution']),
                    'analysis_data': json.dumps(analysis_data, default=str),
                    'created': datetime.now().isoformat()
                }
            )
            changes["nodes_added"] += 1
            print(f" Created LifespanAnalysis node {analysis_id}")

        except KeyError as e:
            print(f" LifespanAnalysis node {analysis_id} already exists")

        return analysis_id

    def _create_method_instance(self, graph_manager, changes):
        """Create a method instance node for execution traceability"""
        method_instance_id = self._get_next_numeric_id(graph_manager)

        try:
            graph_manager.add_node(
                node_id=method_instance_id,
                value=f"ObjectLifespanAnalysisMethod execution",
                type='ObjectLifespanAnalysisMethod',
                hierarchy='analysis',
                attributes={
                    'method_type': 'ObjectLifespanAnalysisMethod',
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

    def _connect_method_to_input(self, graph_manager, method_instance_id, input_id, changes):
        """Connect method instance to input nodes"""
        try:
            graph_manager.add_edge(
                source=method_instance_id,
                target=input_id,
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

    def _connect_method_to_output(self, graph_manager, method_instance_id, output_id, changes):
        """Connect method instance to output nodes"""
        try:
            graph_manager.add_edge(
                source=method_instance_id,
                target=output_id,
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