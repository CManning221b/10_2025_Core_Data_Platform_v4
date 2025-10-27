# backend/methods_application/methods/sensor_bounds_check_method.py
from backend.methods_application.method_implementation import MethodImplementation
import subprocess
import json
import os
from datetime import datetime


class SensorBoundsCheckMethod(MethodImplementation):
    method_id = "SensorBoundsCheckMethod"
    method_name = "Universal Sensor Bounds Validation"
    description = "Validates sensor readings against ontology-defined bounds in multiple languages"

    # HARDCODED BOOLEAN - Set to True for JavaScript, False for Python
    USE_JAVASCRIPT = False  # Change this to switch implementation

    def execute(self, graph_manager, parameters):
        changes = {
            "nodes_added": 0,
            "edges_added": 0,
            "sensors_checked": 0,
            "measurements_analyzed": 0,
            "bounds_violations": 0,
            "sequence_violations": 0,
            "language_used": "javascript" if self.USE_JAVASCRIPT else "python",
            "bounds_sources": []
        }

        # Create method instance for traceability
        method_instance_id = self._create_method_instance(graph_manager, changes)

        # Find all sensor nodes
        sensor_nodes = self._find_sensor_nodes(graph_manager)
        print(f"DEBUG: Found {len(sensor_nodes)} sensor nodes to check")

        for sensor_id, sensor_data in sensor_nodes.items():
            print(f"DEBUG: Checking sensor {sensor_id} of type {sensor_data.get('type')}")

            # Get sensor bounds from ontology
            bounds = self._get_sensor_bounds_from_ontology(sensor_data)
            if not bounds:
                print(f"DEBUG: No bounds found for sensor {sensor_id}")
                continue

            print(f"DEBUG: Found bounds for {sensor_id}: {bounds}")

            # Get all measurement nodes connected to this sensor
            measurement_nodes = self._get_measurement_nodes_for_sensor(graph_manager, sensor_id)

            if not measurement_nodes:
                print(f"DEBUG: No measurements found for sensor {sensor_id}")
                continue

            print(f"DEBUG: Found {len(measurement_nodes)} measurement nodes for sensor {sensor_id}")

            # Process each measurement node (individual or sequence)
            for measurement_id, measurement_data in measurement_nodes.items():
                result = self._analyze_measurement_bounds(measurement_data, bounds)

                if result:
                    # Create result node
                    result_id = self._create_bounds_result_node(
                        graph_manager, sensor_id, measurement_id, result, changes
                    )

                    # Connect method to inputs and outputs
                    self._connect_method_to_input(graph_manager, method_instance_id, measurement_id, changes)
                    self._connect_method_to_output(graph_manager, method_instance_id, result_id, changes)

                    changes["measurements_analyzed"] += 1
                    if not result.get('is_valid', True):
                        if result.get('measurement_type') == 'sequence':
                            changes["sequence_violations"] += 1
                        else:
                            changes["bounds_violations"] += 1

            changes["sensors_checked"] += 1
            changes["bounds_sources"].append(bounds['source'])

        return changes

    def _find_sensor_nodes(self, graph_manager):
        """Find all sensor nodes in the graph"""
        # Updated to include all sensor types from your ontology
        sensor_types = [
            'temperature_sensor', 'pressure_sensor', 'speed_sensor',
            'ratio_sensor', 'flow_sensor', 'bleed_sensor', 'vibration_sensor'
        ]
        sensor_nodes = {}

        for node_id, node_data in graph_manager.node_data.items():
            node_type = node_data.get('type', '').lower()
            if any(sensor_type in node_type for sensor_type in sensor_types):
                sensor_nodes[node_id] = node_data

        return sensor_nodes

    def _get_measurement_nodes_for_sensor(self, graph_manager, sensor_id):
        """Get all measurement nodes connected to this sensor"""
        measurement_nodes = {}

        # Look for measurement nodes connected to this sensor
        for (source, target), edge_data in graph_manager.edge_data.items():
            if source == sensor_id and edge_data.get('edge_type') == 'produces_measurement':
                measurement_node = graph_manager.node_data.get(target)
                if measurement_node:
                    node_type = measurement_node.get('type', '')
                    if node_type in ['measurement', 'measurement_sequence']:
                        measurement_nodes[target] = measurement_node

        return measurement_nodes

    def _analyze_measurement_bounds(self, measurement_data, bounds):
        """Analyze bounds for either individual measurement or measurement sequence"""

        measurement_type = measurement_data.get('type')

        if measurement_type == 'measurement':
            # Handle individual measurement
            return self._analyze_individual_measurement(measurement_data, bounds)

        elif measurement_type == 'measurement_sequence':
            # Handle measurement sequence
            return self._analyze_measurement_sequence(measurement_data, bounds)

        else:
            print(f"DEBUG: Unknown measurement type: {measurement_type}")
            return None

    def _analyze_individual_measurement(self, measurement_data, bounds):
        """Analyze bounds for a single measurement"""

        value = measurement_data.get('value') or measurement_data.get('current_value')
        if value is None:
            return None

        try:
            value = float(value)
        except (ValueError, TypeError):
            return None

        # Execute bounds check using hardcoded language choice
        if self.USE_JAVASCRIPT:
            result = self._javascript_bounds_check(value, bounds)
        else:
            result = self._python_bounds_check(value, bounds)

        result['measurement_type'] = 'individual'
        result['is_valid'] = result['valid']
        result['analysis_summary'] = f"Single measurement: {result['status']}"

        return result

    def _analyze_measurement_sequence(self, measurement_data, bounds):
        """Analyze bounds for a measurement sequence"""

        values = measurement_data.get('values', [])
        if not values:
            return None

        # Analyze each value in the sequence
        results = []
        violations = []

        for i, value in enumerate(values):
            try:
                value = float(value)

                if self.USE_JAVASCRIPT:
                    check_result = self._javascript_bounds_check(value, bounds)
                else:
                    check_result = self._python_bounds_check(value, bounds)

                results.append(check_result)

                if not check_result['valid']:
                    cycle_num = measurement_data.get('cycle_numbers', [i])[i] if i < len(
                        measurement_data.get('cycle_numbers', [])) else i
                    violations.append({
                        'cycle': cycle_num,
                        'value': value,
                        'status': check_result['status'],
                        'violation_amount': check_result.get('violation_amount', 0)
                    })

            except (ValueError, TypeError):
                continue

        # Summarize sequence results
        total_measurements = len(results)
        valid_measurements = sum(1 for r in results if r['valid'])
        violation_count = len(violations)

        is_valid = violation_count == 0
        violation_percentage = (violation_count / total_measurements * 100) if total_measurements > 0 else 0

        # Determine overall sequence status
        if violation_percentage == 0:
            sequence_status = "all_within_range"
        elif violation_percentage < 10:
            sequence_status = "mostly_within_range"
        elif violation_percentage < 50:
            sequence_status = "some_violations"
        else:
            sequence_status = "many_violations"

        return {
            'measurement_type': 'sequence',
            'is_valid': is_valid,
            'sequence_status': sequence_status,
            'total_measurements': total_measurements,
            'valid_measurements': valid_measurements,
            'violation_count': violation_count,
            'violation_percentage': round(violation_percentage, 1),
            'violations': violations[:5],  # Include up to 5 example violations
            'implementation': "javascript" if self.USE_JAVASCRIPT else "python",
            'analysis_summary': f"Sequence: {violation_count}/{total_measurements} violations ({violation_percentage:.1f}%)"
        }

    def _get_sensor_bounds_from_ontology(self, sensor_data):
        """Extract bounds from sensor node properties or lookup ontology specifications"""

        # First, try to get bounds from sensor node properties (if loaded directly)
        min_val = (sensor_data.get('bounds:min_value') or
                   sensor_data.get('min_value') or
                   sensor_data.get('normal_operating_range_min') or
                   sensor_data.get('min_operating_value'))

        max_val = (sensor_data.get('bounds:max_value') or
                   sensor_data.get('max_value') or
                   sensor_data.get('normal_operating_range_max') or
                   sensor_data.get('max_operating_value'))

        if min_val is not None and max_val is not None:
            unit = (sensor_data.get('bounds:unit') or
                    sensor_data.get('unit', 'unknown'))
            sensor_type = sensor_data.get('sensor_type', sensor_data.get('type', 'unknown'))
            return {
                'min_threshold': float(min_val),
                'max_threshold': float(max_val),
                'unit': unit,
                'source': f"ontology_bounds_{sensor_type}"
            }


        return self._lookup_ontology_bounds_specification(sensor_data)

    def _lookup_ontology_bounds_specification(self, sensor_data):
        """Look up bounds from ontology bounds specifications"""

        # Get sensor identifiers
        sensor_id = (sensor_data.get('sensor_id', '') or
                     sensor_data.get('value', '')).upper()
        sensor_type = sensor_data.get('type', '').lower()

        # EXPANDED ontology bounds specifications
        ontology_specifications = {
            # Temperature sensors (corrected)
            'T2': {'min': 515.0, 'max': 525.0, 'unit': '°R', 'source': 'ontology_T2_specification'},
            'T24': {'min': 630.0, 'max': 650.0, 'unit': '°R', 'source': 'ontology_T24_specification'},  # Was 1000-1200

            # Pressure sensors (corrected)
            'P2': {'min': 14.0, 'max': 15.0, 'unit': 'psia', 'source': 'ontology_P2_specification'},
            'P15': {'min': 20.0, 'max': 23.0, 'unit': 'psia', 'source': 'ontology_P15_specification'},  # Was 60-80
            'P30': {'min': 550.0, 'max': 560.0, 'unit': 'psia', 'source': 'ontology_P30_specification'},  # Was 400-500
            'PS30': {'min': 45.0, 'max': 50.0, 'unit': 'psia', 'source': 'ontology_PS30_specification'},  # Was 380-520

            # Speed sensors (corrected)
            'NF': {'min': 2300.0, 'max': 2500.0, 'unit': 'rpm', 'source': 'ontology_Nf_specification'},
            'NC': {'min': 8000.0, 'max': 10000.0, 'unit': 'rpm', 'source': 'ontology_NC_specification'},
            'NRF': {'min': 2380.0, 'max': 2390.0, 'unit': 'rpm', 'source': 'ontology_NRF_specification'},  # Was 500-700
            'NF_DMD': {'min': 2200.0, 'max': 2600.0, 'unit': 'rpm', 'source': 'ontology_NF_DMD_specification'},

            # Ratio sensors (already working)
            'EPR': {'min': 1.2, 'max': 1.8, 'unit': 'ratio', 'source': 'ontology_EPR_specification'},
            'BPR': {'min': 5.0, 'max': 12.0, 'unit': 'ratio', 'source': 'ontology_BPR_specification'},
            'PCNFR_DMD': {'min': 80.0, 'max': 105.0, 'unit': '%', 'source': 'ontology_PCNFR_DMD_specification'},

            # Flow sensors (corrected)
            'W31': {'min': 30.0, 'max': 50.0, 'unit': 'lbm/s', 'source': 'ontology_W31_specification'},
            'FARB': {'min': 0.02, 'max': 0.05, 'unit': 'ratio', 'source': 'ontology_FARB_specification'},

            # Bleed sensors (already working)
            'HTBLEED': {'min': 350.0, 'max': 450.0, 'unit': '°R', 'source': 'ontology_HTBLEED_specification'},

            'T30': {'min': 1580.0, 'max': 1605.0, 'unit': '°R', 'source': 'ontology_T30_specification'},  # Was 1600
            'T50': {'min': 1390.0, 'max': 1425.0, 'unit': '°R', 'source': 'ontology_T50_specification'},  # Was 1420
            'NRC': {'min': 8115.0, 'max': 8150.0, 'unit': 'rpm', 'source': 'ontology_NRC_specification'},  # Was 8120
            'W32': {'min': 23.0, 'max': 24.5, 'unit': 'lbm/s', 'source': 'ontology_W32_specification'},  # Was 24.0
            'PHI': {'min': 520.0, 'max': 525.5, 'unit': 'deg', 'source': 'ontology_PHI_specification'},  # Was 525.0

        }

        # Generic category bounds (from your bounds ontology)
        category_specifications = {
            'temperature': {
                'min': 60.0, 'max': 90.0, 'unit': '°C',
                'source': 'ontology_temperature_generic_bounds'
            },
            'pressure': {
                'min': 1.0, 'max': 10.0, 'unit': 'bar',
                'source': 'ontology_pressure_generic_bounds'
            },
            'speed': {
                'min': 0.0, 'max': 5000.0, 'unit': 'rpm',
                'source': 'ontology_speed_generic_bounds'
            },
        }

        print(f"DEBUG: Looking up bounds for sensor_id='{sensor_id}', sensor_type='{sensor_type}'")

        # Try exact sensor ID match first
        if sensor_id in ontology_specifications:
            spec = ontology_specifications[sensor_id]
            print(f"DEBUG: Found specific ontology bounds for {sensor_id}")
            return {
                'min_threshold': spec['min'],
                'max_threshold': spec['max'],
                'unit': spec['unit'],
                'source': spec['source']
            }

        # Try category-based matching
        for category, bounds in category_specifications.items():
            if category in sensor_type:
                print(f"DEBUG: Found category ontology bounds for {category}")
                return {
                    'min_threshold': bounds['min'],
                    'max_threshold': bounds['max'],
                    'unit': bounds['unit'],
                    'source': bounds['source']
                }

        # If no ontology bounds found, fall back to default bounds
        print(f"DEBUG: No ontology bounds found, falling back to defaults")
        return self._get_default_bounds_by_type(sensor_data)

    def _get_default_bounds_by_type(self, sensor_data):
        """Get default bounds based on sensor type if not in ontology"""
        sensor_type = sensor_data.get('sensor_type', '').upper()
        sensor_node_type = sensor_data.get('type', '').lower()

        # Updated bounds dictionary with all sensor types
        default_bounds = {
            # Specific sensor IDs
            'T2': {'min': 515.0, 'max': 525.0, 'unit': '°R'},
            'P2': {'min': 14.0, 'max': 15.0, 'unit': 'psia'},
            'NF': {'min': 2300.0, 'max': 2500.0, 'unit': 'rpm'},

            # ADD MISSING SENSOR TYPES:
            'RATIO_SENSOR': {'min': 0.5, 'max': 2.0, 'unit': 'ratio'},
            'FLOW_SENSOR': {'min': 10.0, 'max': 100.0, 'unit': 'kg/s'},
            'BLEED_SENSOR': {'min': 0.0, 'max': 5.0, 'unit': 'kg/s'},
            'VIBRATION_SENSOR': {'min': 0.0, 'max': 10.0, 'unit': 'mm/s'},

            # Generic category bounds
            'TEMPERATURE': {'min': 60.0, 'max': 90.0, 'unit': '°C'},
            'PRESSURE': {'min': 1.0, 'max': 10.0, 'unit': 'bar'},
            'SPEED': {'min': 0.0, 'max': 5000.0, 'unit': 'rpm'},
        }

        # Try exact sensor type match first
        if sensor_type in default_bounds:
            bounds_info = default_bounds[sensor_type]
            return {
                'min_threshold': bounds_info['min'],
                'max_threshold': bounds_info['max'],
                'unit': bounds_info['unit'],
                'source': f"default_bounds_{sensor_type}"
            }

        # Try by sensor category (from node type)
        for category in ['TEMPERATURE', 'PRESSURE', 'SPEED', 'RATIO', 'FLOW', 'BLEED', 'VIBRATION']:
            if category.lower() in sensor_node_type:
                category_key = f"{category}_SENSOR" if category in ['RATIO', 'FLOW', 'BLEED', 'VIBRATION'] else category
                if category_key in default_bounds:
                    bounds_info = default_bounds[category_key]
                    return {
                        'min_threshold': bounds_info['min'],
                        'max_threshold': bounds_info['max'],
                        'unit': bounds_info['unit'],
                        'source': f"default_bounds_{category}"
                    }

        return None

    def _python_bounds_check(self, value, bounds):
        """Python implementation - direct execution"""
        min_threshold = bounds['min_threshold']
        max_threshold = bounds['max_threshold']

        if value < min_threshold:
            return {
                "valid": False,
                "status": "below_range",
                "value": value,
                "violation_amount": round(min_threshold - value, 2),
                "threshold_violated": "minimum",
                "implementation": "python"
            }
        elif value > max_threshold:
            return {
                "valid": False,
                "status": "above_range",
                "value": value,
                "violation_amount": round(value - max_threshold, 2),
                "threshold_violated": "maximum",
                "implementation": "python"
            }
        else:
            return {
                "valid": True,
                "status": "within_range",
                "value": value,
                "margin": round(min(value - min_threshold, max_threshold - value), 2),
                "implementation": "python"
            }

    def _javascript_bounds_check(self, value, bounds):
        """JavaScript implementation via external script"""

        js_script_path = os.path.join(
            os.path.dirname(__file__),
            '..', 'javascript_methods',
            'sensor_bounds_check.js'
        )

        try:
            result = subprocess.run(
                ['node', js_script_path, str(value), str(bounds['min_threshold']), str(bounds['max_threshold'])],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.returncode == 0:
                js_result = json.loads(result.stdout.strip())
                js_result['implementation'] = 'javascript'
                return js_result
            else:
                print(f"JavaScript execution failed: {result.stderr}")
                fallback_result = self._python_bounds_check(value, bounds)
                fallback_result['implementation'] = 'python_fallback'
                return fallback_result

        except (subprocess.TimeoutExpired, json.JSONDecodeError, FileNotFoundError) as e:
            print(f"JavaScript execution failed ({e}), falling back to Python")
            fallback_result = self._python_bounds_check(value, bounds)
            fallback_result['implementation'] = 'python_fallback'
            return fallback_result

    def _create_bounds_result_node(self, graph_manager, sensor_id, measurement_id, result, changes):
        """Create a bounds check result node"""
        result_id = self._get_next_numeric_id(graph_manager)

        # Create descriptive value based on result type
        if result['measurement_type'] == 'individual':
            if result['valid']:
                display_value = f"✓ Within range: {result['value']} (margin: {result.get('margin', 'N/A')})"
            else:
                display_value = f"✗ {result['status']}: {result['value']} (violation: {result.get('violation_amount', 'N/A')})"
        else:  # sequence
            status_icon = "✓" if result['is_valid'] else "⚠️"
            display_value = f"{status_icon} Sequence: {result['violation_count']}/{result['total_measurements']} violations ({result['violation_percentage']}%)"

        try:
            graph_manager.add_node(
                node_id=result_id,
                value=display_value,
                type='BoundsResult',
                hierarchy='analysis',
                attributes={
                    'sensor_id': sensor_id,
                    'measurement_id': measurement_id,
                    'measurement_type': result['measurement_type'],
                    'bounds_check_result': result.get('sequence_status', result.get('status')),
                    'is_valid': result['is_valid'],
                    'implementation_used': result.get('implementation', 'unknown'),
                    'analysis_summary': result['analysis_summary'],
                    'created': datetime.now().isoformat(),
                    **{k: v for k, v in result.items() if
                       k not in ['measurement_type', 'is_valid', 'implementation', 'analysis_summary']}
                }
            )
            changes["nodes_added"] += 1
            print(f"DEBUG: Created BoundsResult node {result_id}: {display_value}")

        except KeyError as e:
            print(f"DEBUG: Error creating BoundsResult node: {e}")
            return None

        return result_id

    # [Keep all your existing helper methods exactly as they are]
    def _create_method_instance(self, graph_manager, changes):
        """Create a method instance node for execution traceability"""
        method_instance_id = self._get_next_numeric_id(graph_manager)

        try:
            graph_manager.add_node(
                node_id=method_instance_id,
                value=f"SensorBoundsCheckMethod execution",
                type='SensorBoundsCheckMethod',
                hierarchy='analysis',
                attributes={
                    'method_type': 'SensorBoundsCheckMethod',
                    'execution_time': datetime.now().isoformat(),
                    'method_id': self.method_id,
                    'method_name': self.method_name,
                    'language_used': "javascript" if self.USE_JAVASCRIPT else "python"
                }
            )
            changes["nodes_added"] += 1
            print(f"DEBUG: Created method_instance node {method_instance_id}")

        except KeyError as e:
            print(f"DEBUG: Method instance node {method_instance_id} already exists")

        return method_instance_id

    def _connect_method_to_input(self, graph_manager, method_instance_id, measurement_id, changes):
        """Connect method instance to measurement it analyzed"""
        try:
            graph_manager.add_edge(
                source=method_instance_id,
                target=measurement_id,
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