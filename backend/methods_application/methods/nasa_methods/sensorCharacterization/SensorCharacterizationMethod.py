# backend/methods_application/methods/sensor_characterization_method.py
from backend.methods_application.method_implementation import MethodImplementation
import subprocess
import json
import os
import math
from datetime import datetime


class SensorCharacterizationMethod(MethodImplementation):
    method_id = "SensorCharacterizationMethod"
    method_name = "Sensor Trend Analysis"
    description = "Analyzes sensor trends and identifies evaluation patterns"

    # HARDCODED BOOLEAN - Set to True for JavaScript, False for Python
    USE_JAVASCRIPT = False  # This was missing!

    def execute(self, graph_manager, parameters):
        changes = {
            "nodes_added": 0,
            "edges_added": 0,
            "sensors_analyzed": 0,
            "degrading_sensors": 0,
            "stable_sensors": 0,
            "high_variance_sensors": 0
        }

        # Create method instance for traceability
        method_instance_id = self._create_method_instance(graph_manager, changes)

        # Find all measurement sequences (these have the trend data we need)
        measurement_sequences = self._find_measurement_sequences(graph_manager)
        print(f"DEBUG: Found {len(measurement_sequences)} measurement sequences to analyze")

        # Group by sensor type for summary analysis
        sensor_summaries = {}

        for seq_id, seq_data in measurement_sequences.items():
            # Analyze this measurement sequence
            analysis = self._analyze_measurement_sequence(seq_data)

            if analysis:
                # Create individual result node
                result_id = self._create_trend_result_node(
                    graph_manager, seq_id, analysis, changes
                )

                if result_id:  # Only proceed if node creation succeeded
                    # Connect to method
                    self._connect_method_to_input(graph_manager, method_instance_id, seq_id, changes)
                    self._connect_method_to_output(graph_manager, method_instance_id, result_id, changes)

                    # Update counters
                    changes["sensors_analyzed"] += 1
                    if analysis['category'] == 'degrading':
                        changes["degrading_sensors"] += 1
                    elif analysis['category'] == 'high_variance':
                        changes["high_variance_sensors"] += 1
                    else:
                        changes["stable_sensors"] += 1

                    # Collect for sensor type summary
                    sensor_type = analysis['sensor_type']
                    if sensor_type not in sensor_summaries:
                        sensor_summaries[sensor_type] = {
                            'total': 0, 'degrading': 0, 'stable': 0, 'high_variance': 0
                        }
                    sensor_summaries[sensor_type]['total'] += 1
                    sensor_summaries[sensor_type][analysis['category']] += 1

        # Create sensor type summary nodes
        for sensor_type, summary in sensor_summaries.items():
            summary_id = self._create_sensor_summary_node(
                graph_manager, sensor_type, summary, changes
            )
            if summary_id:
                self._connect_method_to_output(graph_manager, method_instance_id, summary_id, changes)

        return changes

    def _find_measurement_sequences(self, graph_manager):
        """Find all measurement sequence nodes"""
        sequences = {}
        for node_id, node_data in graph_manager.node_data.items():
            if node_data.get('type') == 'measurement_sequence':
                sequences[node_id] = node_data
        return sequences

    def _analyze_measurement_sequence(self, seq_data):
        """Analyze a measurement sequence for trends and patterns"""
        values = seq_data.get('values', [])
        sensor_type = seq_data.get('sensor_type', 'unknown')
        engine_id = seq_data.get('engine_id', 'unknown')

        if len(values) < 10:  # Need minimum data
            return None

        # Get existing trend from sequence_stats if available
        sequence_stats = seq_data.get('sequence_stats', {})
        existing_trend = sequence_stats.get('trend', 'stable')

        # Calculate basic statistics
        mean_val = sum(values) / len(values)
        variance = sum((x - mean_val) ** 2 for x in values) / len(values)
        std_dev = variance ** 0.5
        coefficient_of_variation = std_dev / abs(mean_val) if mean_val != 0 else 0

        # Calculate simple linear trend
        n = len(values)
        x_vals = list(range(n))
        slope = self._calculate_slope(x_vals, values)

        # Determine trend strength
        trend_strength = abs(slope) * n  # Scale by number of points

        # Categorize the sensor behavior
        if 'increasing' in existing_trend.lower() or 'decreasing' in existing_trend.lower():
            if trend_strength > 0.1:
                category = 'degrading'
                severity = 'high' if trend_strength > 1.0 else 'moderate'
            else:
                category = 'stable'
                severity = 'low'
        elif coefficient_of_variation > 0.1:  # High variability
            category = 'high_variance'
            severity = 'high' if coefficient_of_variation > 0.2 else 'moderate'
        else:
            category = 'stable'
            severity = 'low'

        return {
            'sensor_type': sensor_type,
            'engine_id': engine_id,
            'category': category,
            'severity': severity,
            'trend_direction': 'increasing' if slope > 0 else 'decreasing' if slope < 0 else 'stable',
            'slope': round(slope, 6),
            'trend_strength': round(trend_strength, 3),
            'coefficient_of_variation': round(coefficient_of_variation, 3),
            'data_points': len(values),
            'mean_value': round(mean_val, 2),
            'std_deviation': round(std_dev, 3),
            'existing_trend': existing_trend
        }

    def _calculate_slope(self, x_vals, y_vals):
        """Calculate linear regression slope"""
        n = len(x_vals)
        sum_x = sum(x_vals)
        sum_y = sum(y_vals)
        sum_xy = sum(x * y for x, y in zip(x_vals, y_vals))
        sum_x2 = sum(x * x for x in x_vals)

        denominator = n * sum_x2 - sum_x * sum_x
        if denominator == 0:
            return 0

        return (n * sum_xy - sum_x * sum_y) / denominator

    def _create_trend_result_node(self, graph_manager, sequence_id, analysis, changes):
        """Create a trend analysis result node"""
        result_id = self._get_next_numeric_id(graph_manager)

        # Create display value based on analysis
        category = analysis['category']
        sensor_type = analysis['sensor_type']
        engine_id = analysis['engine_id']
        severity = analysis['severity']

        if category == 'degrading':
            icon = "📈" if analysis['trend_direction'] == 'increasing' else "📉"
            display_value = f"{icon} {sensor_type} (Engine {engine_id}): {category} trend - {severity} severity"
        elif category == 'high_variance':
            icon = "📊"
            display_value = f"{icon} {sensor_type} (Engine {engine_id}): High variability detected"
        else:
            icon = "✅"
            display_value = f"{icon} {sensor_type} (Engine {engine_id}): Stable behavior"

        try:
            graph_manager.add_node(
                node_id=result_id,
                value=display_value,
                type='SensorTrendAnalysis',
                hierarchy='analysis',
                attributes={
                    'sequence_id': sequence_id,
                    'sensor_type': sensor_type,
                    'engine_id': engine_id,
                    'trend_category': category,
                    'severity': severity,
                    'trend_direction': analysis['trend_direction'],
                    'slope': analysis['slope'],
                    'trend_strength': analysis['trend_strength'],
                    'coefficient_of_variation': analysis['coefficient_of_variation'],
                    'data_points_analyzed': analysis['data_points'],
                    'mean_value': analysis['mean_value'],
                    'std_deviation': analysis['std_deviation'],
                    'created': datetime.now().isoformat()
                }
            )
            changes["nodes_added"] += 1
            print(f"DEBUG: Created trend analysis node {result_id}: {display_value}")
            return result_id

        except Exception as e:
            print(f"DEBUG: Error creating trend result node: {e}")
            return None

    def _create_sensor_summary_node(self, graph_manager, sensor_type, summary, changes):
        """Create a sensor type summary node"""
        result_id = self._get_next_numeric_id(graph_manager)

        total = summary['total']
        degrading = summary['degrading']
        stable = summary['stable']
        high_variance = summary['high_variance']

        degrading_pct = (degrading / total * 100) if total > 0 else 0

        if degrading_pct > 20:
            icon = "⚠️"
            status = "High Risk"
        elif degrading_pct > 5:
            icon = "⚡"
            status = "Moderate Risk"
        else:
            icon = "✅"
            status = "Low Risk"

        display_value = f"{icon} {sensor_type} Summary: {status} ({degrading}/{total} degrading, {degrading_pct:.1f}%)"

        try:
            graph_manager.add_node(
                node_id=result_id,
                value=display_value,
                type='SensorTypeSummary',
                hierarchy='analysis',
                attributes={
                    'sensor_type': sensor_type,
                    'risk_status': status,
                    'total_sensors': total,
                    'degrading_count': degrading,
                    'stable_count': stable,
                    'high_variance_count': high_variance,
                    'degrading_percentage': round(degrading_pct, 1),
                    'created': datetime.now().isoformat()
                }
            )
            changes["nodes_added"] += 1
            print(f"DEBUG: Created sensor summary node {result_id}: {display_value}")
            return result_id

        except Exception as e:
            print(f"DEBUG: Error creating summary node: {e}")
            return None

    def _create_method_instance(self, graph_manager, changes):
        """Create a method instance node for execution traceability"""
        method_instance_id = self._get_next_numeric_id(graph_manager)

        try:
            graph_manager.add_node(
                node_id=method_instance_id,
                value=f"SensorCharacterizationMethod execution",
                type='SensorCharacterizationMethod',
                hierarchy='analysis',
                attributes={
                    'method_type': 'SensorCharacterizationMethod',
                    'execution_time': datetime.now().isoformat(),
                    'method_id': self.method_id,
                    'method_name': self.method_name,
                    'language_used': "javascript" if self.USE_JAVASCRIPT else "python"
                }
            )
            changes["nodes_added"] += 1
            print(f"DEBUG: Created method_instance node {method_instance_id}")
            return method_instance_id

        except KeyError as e:
            print(f"DEBUG: Method instance node {method_instance_id} already exists")
            return method_instance_id

    def _connect_method_to_input(self, graph_manager, method_instance_id, node_id, changes):
        """Connect method instance to node it analyzed"""
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
            print(f"DEBUG: Error creating input edge: {e}")

    def _connect_method_to_output(self, graph_manager, method_instance_id, result_id, changes):
        """Connect method instance to result it created"""
        try:
            graph_manager.add_edge(
                source=method_instance_id,
                target=result_id,
                attributes={
                    'edge_type': 'produces',
                    'direction': 'out',
                    'provenance_type': 'generated_by',
                    'created': datetime.now().isoformat()
                }
            )
            changes["edges_added"] += 1

        except (KeyError, ValueError) as e:
            print(f"DEBUG: Error creating output edge: {e}")

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