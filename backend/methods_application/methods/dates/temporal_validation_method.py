# backend/methods_application/methods/temporal_validation_method.py
from backend.methods_application.method_implementation import MethodImplementation
from datetime import datetime
import json


class TemporalValidationMethod(MethodImplementation):
    method_id = "TemporalValidationMethod"
    method_name = "Temporal Validation Method"
    description = "Validates that measurements fall within object lifespans to detect anachronisms"

    def execute(self, graph_manager, parameters):
        changes = {
            "nodes_added": 0,
            "edges_added": 0,
            "method_instances_created": 0,
            "validations_performed": 0,
            "anachronisms_detected": 0,
            "validation_results": []
        }

        # Find ALL measurement → object relationships
        validation_pairs = []
        for (source, target), edge_attrs in graph_manager.edge_data.items():
            if edge_attrs.get('edge_type') == 'measures':
                measurement_node = graph_manager.node_data.get(source)
                object_node = graph_manager.node_data.get(target)

                if measurement_node and object_node:
                    validation_pairs.append((source, target, measurement_node, object_node))

        # Create SEPARATE method instance for EACH measurement-object pair
        for measurement_id, object_id, measurement_data, object_data in validation_pairs:
            # Create individual method instance for this specific validation
            method_instance_id = self._create_method_instance(graph_manager, changes)

            # Connect inputs: measurement and object to method instance
            self._connect_method_to_input(graph_manager, method_instance_id, measurement_id, changes)
            self._connect_method_to_input(graph_manager, method_instance_id, object_id, changes)

            # Perform validation
            result = self._validate_temporal_consistency(
                measurement_id, measurement_data, object_id, object_data
            )

            if result:
                # Create validation result for this pair
                validation_result_id = self._create_validation_result_node(
                    graph_manager, result, changes
                )

                # Connect output: method instance to validation result
                self._connect_method_to_output(graph_manager, method_instance_id, validation_result_id, changes)

                changes["validations_performed"] += 1
                if not result["is_valid"]:
                    changes["anachronisms_detected"] += 1

        return changes

    def _validate_temporal_consistency(self, measurement_id, measurement_data, object_id, object_data):
        """Validate if measurement timestamp falls within object lifespan"""
        try:
            # Extract timestamp from measurement
            timestamp = measurement_data.get('timestamp')
            if timestamp is None:
                return None

            # Extract object lifespan
            created = object_data.get('commissioned')
            destroyed = object_data.get('decommissioned')

            if created is None or destroyed is None:
                return None

            # Perform validation
            is_valid = created <= timestamp <= destroyed

            return {
                "measurement_id": measurement_id,
                "object_id": object_id,
                "timestamp": timestamp,
                "object_lifespan_start": created,
                "object_lifespan_end": destroyed,
                "is_valid": is_valid,
                "validation_reason": self._generate_validation_reason(
                    timestamp, created, destroyed, is_valid, measurement_data, object_data
                )
            }

        except Exception as e:
            print(f" Error validating {measurement_id} -> {object_id}: {e}")
            return None

    def _generate_validation_reason(self, timestamp, created, destroyed, is_valid, measurement_data, object_data):
        """Generate human-readable validation explanation"""
        measurement_name = measurement_data.get('value', 'measurement')
        object_name = object_data.get('value', 'object')

        if is_valid:
            return f"Measurement {measurement_name} (timestamp: {timestamp}) falls within lifespan of {object_name} ({created}-{destroyed})"
        else:
            if timestamp < created:
                return f"Measurement {measurement_name} (timestamp: {timestamp}) predates given start date of {object_name} (: {created}) - ANACHRONISM"
            else:
                return f"Measurement {measurement_name} (timestamp: {timestamp}) postdates given end date of {object_name} (: {destroyed}) - ANACHRONISM"

    def _create_validation_result_node(self, graph_manager, result, changes):
        """Create a ValidationResult node with direct temporal properties"""
        result_id = self._get_next_numeric_id(graph_manager)

        try:
            # Create ValidationResult with all properties in attributes
            timestamp_year = int(result['timestamp'])
            graph_manager.add_node(
                node_id=result_id,
                value=f"Validation: {result['measurement_id']} → {result['object_id']}",
                type='ValidationResult',
                hierarchy='analysis',
                attributes={
                    'measurement_id': result['measurement_id'],
                    'object_id': result['object_id'],
                    'is_valid': result['is_valid'],
                    'timestamp': result['timestamp'],  # This will be picked up by timeline extraction
                    'validation_year': timestamp_year,
                    'validation_date': f"{timestamp_year}-06-15",  # Mid-year date
                    'object_lifespan_start': result['object_lifespan_start'],
                    'object_lifespan_end': result['object_lifespan_end'],
                    'validation_reason': result['validation_reason'],
                    'created': datetime.now().isoformat()
                }
            )
            changes["nodes_added"] += 1
            print(f" Created ValidationResult node {result_id} with temporal properties")

        except KeyError as e:
            print(f" ValidationResult node {result_id} already exists")

        return result_id

    def _create_method_instance(self, graph_manager, changes):
        """Create a method instance node for execution traceability"""
        method_instance_id = self._get_next_numeric_id(graph_manager)

        try:
            graph_manager.add_node(
                node_id=method_instance_id,
                value=f"TemporalValidationMethod execution",
                type='TemporalValidationMethod',
                hierarchy='analysis',
                attributes={
                    'method_type': 'TemporalValidationMethod',
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