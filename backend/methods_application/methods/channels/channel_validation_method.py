# backend/methods_application/methods/reactor/channel_validation_method.py
from backend.methods_application.method_implementation import MethodImplementation
import re
from datetime import datetime
from collections import defaultdict


class ChannelValidationMethod(MethodImplementation):
    method_id = "ChannelValidationMethod"
    method_name = "Channel ID Validation and Compliance Check"
    description = "Validates channel IDs against reactor-specific patterns and rules, identifies anomalies"

    def __init__(self, ontology_manager=None):
        super().__init__()
        self.ontology_manager = ontology_manager
        self.validation_rules = self._load_validation_rules()

    def execute(self, graph_manager, parameters):
        changes = {
            "nodes_added": 0,
            "edges_added": 0,
            "method_instances_created": 0,
            "channels_validated": 0,
            "validation_reports_created": 0,
            "valid_channels": 0,
            "invalid_channels": 0,
            "anomalies_detected": 0,
            "reactor_type": None,
            "validation_summary": {},
            "anomaly_types": defaultdict(int)
        }

        # Load validation rules from ontology if available
        if self.ontology_manager:
            self.validation_rules = self._load_validation_rules_from_ontology()

        # Detect reactor type and get appropriate validation rules
        reactor_type = self._detect_reactor_type_from_graph(graph_manager)
        changes["reactor_type"] = reactor_type

        if not reactor_type:
            print(" No reactor type detected - using generic validation")
            reactor_type = "generic"

        validation_rule = self.validation_rules.get(reactor_type, self.validation_rules["generic"])
        print(f" Using {validation_rule['name']} validation rules for {reactor_type}")

        # Create method instance
        method_instance_id = self._create_method_instance(graph_manager, changes, reactor_type, validation_rule)

        # Find all channel nodes to validate
        channel_nodes = self._find_channel_nodes(graph_manager)
        print(f" Found {len(channel_nodes)} channel nodes to validate")

        # Get existing grid positions and reactor constraints
        grid_constraints = self._get_grid_constraints(graph_manager, reactor_type)

        # Validate each channel
        validation_results = []
        for channel_id, channel_data in channel_nodes.items():
            result = self._validate_channel(channel_id, channel_data, reactor_type, validation_rule, grid_constraints)
            validation_results.append(result)

            # Create validation report node
            validation_report_id = self._create_validation_report_node(
                graph_manager, channel_id, result, reactor_type, changes
            )

            # Connect channel to validation report
            self._connect_channel_to_validation(graph_manager, channel_id, validation_report_id, changes)

            # Connect method to validation report
            self._connect_method_to_output(graph_manager, method_instance_id, validation_report_id, changes)

            # Update channel with validation status
            self._update_channel_validation_status(graph_manager, channel_id, result, changes)

            # Count results
            changes["channels_validated"] += 1
            if result["is_valid"]:
                changes["valid_channels"] += 1
            else:
                changes["invalid_channels"] += 1
                changes["anomalies_detected"] += len(result["anomalies"])
                for anomaly in result["anomalies"]:
                    changes["anomaly_types"][anomaly["type"]] += 1

        # Create comprehensive validation summary
        summary_report_id = self._create_summary_report(
            graph_manager, validation_results, reactor_type, validation_rule, changes
        )

        # Connect method to summary
        self._connect_method_to_output(graph_manager, method_instance_id, summary_report_id, changes)

        # Generate validation summary
        changes["validation_summary"] = self._generate_validation_summary(validation_results, reactor_type)

        print(f" Validated {changes['channels_validated']} channels: "
              f"{changes['valid_channels']} valid, {changes['invalid_channels']} invalid, "
              f"{changes['anomalies_detected']} total anomalies")

        return changes

    def _load_validation_rules(self):
        """Load default validation rules for different reactor types"""
        return {
            'CANDU': {
                'name': 'CANDU_validation_rules',
                'description': 'Validation rules for CANDU reactor channels',
                'pattern': r'^[A-Y](?:0[1-9]|1[0-9]|2[0-5])$',
                'excluded_channels': [
                    'A01', 'A02', 'A23', 'A24', 'Y01', 'Y02', 'Y23', 'Y24'
                ],
                'allowed_letters': 'ABCDEFGHJKLMNPQRSTUVWXY',  # No I or O
                'allowed_numbers': range(1, 26),  # 01-25
                'grid_width': 24,
                'grid_height': 25,
                'format_description': 'Letter (A-Y, no I/O) + 2 digits (01-25)',
                'validation_checks': [
                    'pattern_match',
                    'excluded_list',
                    'letter_validity',
                    'number_range',
                    'grid_bounds',
                    'duplicate_detection'
                ]
            },
            'AGR': {
                'name': 'AGR_validation_rules',
                'description': 'Validation rules for AGR reactor channels',
                'pattern': r'^(?:0[2-9]|[1-4][0-9]|5[2-9]|[6-9][0-9])(?:[0-9][0-9])$',
                'excluded_channels': ['50', '51', '99'],
                'number_constraints': {
                    'first_two_digits': {
                        'range': [(2, 48), (52, 98)],  # 02-48, 52-98
                        'step': 2  # Even numbers only
                    },
                    'last_two_digits': {
                        'range': [(2, 98)],
                        'step': 2  # Even numbers only
                    }
                },
                'grid_width': 49,
                'grid_height': 49,
                'format_description': '4 digits: even numbers 02-48, 52-98',
                'validation_checks': [
                    'pattern_match',
                    'excluded_list',
                    'even_number_constraint',
                    'range_validity',
                    'grid_bounds',
                    'duplicate_detection'
                ]
            },
            'generic': {
                'name': 'generic_validation_rules',
                'description': 'Generic validation rules for unknown reactor types',
                'pattern': r'^[A-Z]?\d{1,4}$',
                'excluded_channels': [],
                'format_description': 'Generic alphanumeric channel identifier',
                'validation_checks': [
                    'pattern_match',
                    'basic_format',
                    'duplicate_detection'
                ]
            }
        }

    def _load_validation_rules_from_ontology(self):
        """Load validation rules from ontology"""
        rules = self._load_validation_rules()  # Start with defaults

        try:
            if hasattr(self.ontology_manager, 'node_types'):
                for class_name, class_data in self.ontology_manager.node_types.items():
                    if 'ChannelValidator' in class_name or 'Validation' in class_name:
                        reactor_type = self._infer_reactor_type_from_method(class_name)
                        if reactor_type and reactor_type in rules:
                            annotations = class_data.get('attributes', {})
                            rule = rules[reactor_type]

                            # Update pattern from ontology
                            if 'hasRegexPattern' in annotations:
                                rule['pattern'] = annotations['hasRegexPattern']

                            # Update excluded channels
                            if 'hasExcludedPositions' in annotations:
                                excluded = str(annotations['hasExcludedPositions'])
                                rule['excluded_channels'] = [
                                    ch.strip() for ch in excluded.split(',') if ch.strip()
                                ]

                            # Update grid dimensions
                            if 'hasGridDimensions' in annotations:
                                dims = annotations['hasGridDimensions']
                                if ',' in str(dims):
                                    width, height = str(dims).split(',')
                                    rule['grid_width'] = int(width.strip())
                                    rule['grid_height'] = int(height.strip())

                            print(f" Updated {reactor_type} validation rules from ontology")

        except Exception as e:
            print(f" Error loading validation rules from ontology: {e}")

        return rules

    def _detect_reactor_type_from_graph(self, graph_manager):
        """Detect reactor type from existing graph nodes"""
        # Look for reactor type nodes
        for node_data in graph_manager.node_data.values():
            if node_data.get('type') == 'reactor_type':
                reactor_type = (node_data.get('reactor_type_code') or
                                node_data.get('attributes', {}).get('reactor_type_code'))
                if reactor_type:
                    return reactor_type

        # Look for validation method instances from previous runs
        for node_data in graph_manager.node_data.values():
            if node_data.get('type') == 'ChannelValidationMethod':
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

        # Test each reactor type's pattern
        for reactor_type, rule in self.validation_rules.items():
            if reactor_type == 'generic':
                continue

            pattern = rule['pattern']
            matches = sum(1 for ch in channel_samples if re.match(pattern, ch))
            confidence = matches / len(channel_samples) if channel_samples else 0.0

            if confidence > 0.7:
                return reactor_type

        return None

    def _find_channel_nodes(self, graph_manager):
        """Find all channel nodes in the graph"""
        channel_nodes = {}
        for node_id, node_data in graph_manager.node_data.items():
            if node_data.get('type') == 'channel':
                channel_nodes[node_id] = node_data
        return channel_nodes

    def _get_grid_constraints(self, graph_manager, reactor_type):
        """Get grid constraints from existing grid layout"""
        constraints = {
            'valid_positions': set(),
            'excluded_positions': set(),
            'grid_bounds': None
        }

        # Look for grid layout node
        for node_data in graph_manager.node_data.values():
            if (node_data.get('type') == 'grid_layout' and
                    node_data.get('attributes', {}).get('reactor_type') == reactor_type):
                attrs = node_data.get('attributes', {})
                constraints['grid_bounds'] = {
                    'width': attrs.get('grid_width'),
                    'height': attrs.get('grid_height')
                }
                break

        # Collect valid positions from grid position nodes
        for node_data in graph_manager.node_data.values():
            if (node_data.get('type') == 'grid_position' and
                    node_data.get('attributes', {}).get('reactor_type') == reactor_type):
                channel_id = node_data.get('attributes', {}).get('channel_id')
                if channel_id:
                    constraints['valid_positions'].add(channel_id)

        return constraints

    def _validate_channel(self, channel_node_id, channel_data, reactor_type, validation_rule, grid_constraints):
        """Validate a single channel against all rules"""
        channel_id = self._extract_channel_identifier(channel_data)

        result = {
            'channel_node_id': channel_node_id,
            'channel_id': channel_id,
            'reactor_type': reactor_type,
            'is_valid': True,
            'validation_score': 1.0,
            'anomalies': [],
            'warnings': [],
            'checks_performed': [],
            'validation_timestamp': datetime.now().isoformat()
        }

        if not channel_id:
            result['is_valid'] = False
            result['validation_score'] = 0.0
            result['anomalies'].append({
                'type': 'missing_identifier',
                'severity': 'critical',
                'message': 'Could not extract channel identifier from node data'
            })
            return result

        # Perform validation checks
        checks = validation_rule.get('validation_checks', [])

        for check in checks:
            check_result = self._perform_validation_check(
                channel_id, check, validation_rule, grid_constraints
            )
            result['checks_performed'].append(check_result)

            if not check_result['passed']:
                if check_result['severity'] == 'critical':
                    result['is_valid'] = False
                    result['anomalies'].append(check_result)
                else:
                    result['warnings'].append(check_result)

        # Calculate overall validation score
        if result['checks_performed']:
            passed_checks = sum(1 for check in result['checks_performed'] if check['passed'])
            result['validation_score'] = passed_checks / len(result['checks_performed'])

        return result

    def _perform_validation_check(self, channel_id, check_type, validation_rule, grid_constraints):
        """Perform a specific validation check"""
        check_result = {
            'check_type': check_type,
            'passed': True,
            'severity': 'info',
            'message': f'{check_type} check passed',
            'details': {}
        }

        try:
            if check_type == 'pattern_match':
                pattern = validation_rule['pattern']
                if not re.match(pattern, channel_id):
                    check_result.update({
                        'passed': False,
                        'severity': 'critical',
                        'type': 'pattern_mismatch',
                        'message': f'Channel ID "{channel_id}" does not match required pattern {pattern}',
                        'details': {'expected_pattern': pattern, 'actual_value': channel_id}
                    })

            elif check_type == 'excluded_list':
                excluded = validation_rule.get('excluded_channels', [])
                if channel_id in excluded:
                    check_result.update({
                        'passed': False,
                        'severity': 'critical',
                        'type': 'excluded_position',
                        'message': f'Channel ID "{channel_id}" is in the excluded positions list',
                        'details': {'excluded_list': excluded}
                    })

            elif check_type == 'letter_validity':
                if hasattr(validation_rule, 'allowed_letters') and channel_id:
                    letter = channel_id[0] if channel_id[0].isalpha() else None
                    allowed = validation_rule.get('allowed_letters', '')
                    if letter and letter not in allowed:
                        check_result.update({
                            'passed': False,
                            'severity': 'critical',
                            'type': 'invalid_letter',
                            'message': f'Letter "{letter}" not allowed in reactor type',
                            'details': {'allowed_letters': allowed, 'used_letter': letter}
                        })

            elif check_type == 'number_range':
                if validation_rule.get('allowed_numbers'):
                    numbers = re.findall(r'\d+', channel_id)
                    if numbers:
                        number = int(numbers[0])
                        allowed_range = validation_rule['allowed_numbers']
                        if number not in allowed_range:
                            check_result.update({
                                'passed': False,
                                'severity': 'critical',
                                'type': 'number_out_of_range',
                                'message': f'Number {number} not in allowed range',
                                'details': {'allowed_range': list(allowed_range), 'used_number': number}
                            })

            elif check_type == 'even_number_constraint':
                if 'number_constraints' in validation_rule:
                    constraints = validation_rule['number_constraints']
                    if len(channel_id) == 4 and channel_id.isdigit():
                        first_two = int(channel_id[:2])
                        last_two = int(channel_id[2:])

                        # Check first two digits
                        first_valid = False
                        for start, end in constraints['first_two_digits']['range']:
                            if start <= first_two <= end and first_two % 2 == 0:
                                first_valid = True
                                break

                        # Check last two digits
                        last_valid = (last_two % 2 == 0 and 2 <= last_two <= 98)

                        if not (first_valid and last_valid):
                            check_result.update({
                                'passed': False,
                                'severity': 'critical',
                                'type': 'even_number_violation',
                                'message': f'Channel ID "{channel_id}" violates even number constraints',
                                'details': {
                                    'first_two_valid': first_valid,
                                    'last_two_valid': last_valid,
                                    'constraints': constraints
                                }
                            })

            elif check_type == 'grid_bounds':
                bounds = grid_constraints.get('grid_bounds')
                if bounds and 'width' in bounds and 'height' in bounds:
                    # This would require position calculation - simplified check
                    if len(channel_id) > 10:  # Unreasonably long
                        check_result.update({
                            'passed': False,
                            'severity': 'warning',
                            'type': 'suspicious_format',
                            'message': f'Channel ID "{channel_id}" has unusual length',
                            'details': {'length': len(channel_id)}
                        })

            elif check_type == 'duplicate_detection':
                # This would be checked at the graph level
                check_result['message'] = 'Duplicate detection requires graph-level analysis'

        except Exception as e:
            check_result.update({
                'passed': False,
                'severity': 'error',
                'type': 'validation_error',
                'message': f'Error performing {check_type} check: {str(e)}',
                'details': {'error': str(e)}
            })

        return check_result

    def _extract_channel_identifier(self, channel_data):
        """Extract channel identifier from channel node data"""
        candidates = [
            channel_data.get('channel_id'),
            channel_data.get('attributes', {}).get('channel_id'),
            channel_data.get('value', ''),
            channel_data.get('name', '')
        ]

        for candidate in candidates:
            if candidate:
                clean_id = str(candidate).upper().strip()
                if clean_id.startswith('CHANNEL '):
                    clean_id = clean_id[8:]
                if clean_id:
                    return clean_id

        return None

    def _create_validation_report_node(self, graph_manager, channel_node_id, validation_result, reactor_type, changes):
        """Create validation report node for a channel"""
        report_id = self._get_next_numeric_id(graph_manager)

        # Calculate summary statistics
        total_checks = len(validation_result['checks_performed'])
        passed_checks = sum(1 for check in validation_result['checks_performed'] if check['passed'])
        critical_anomalies = len(validation_result['anomalies'])
        warnings = len(validation_result['warnings'])

        graph_manager.add_node(
            node_id=report_id,
            value=f"Validation Report for {validation_result['channel_id']}",
            type='channel_validation_report',
            hierarchy='analysis',
            attributes={
                'channel_node_id': channel_node_id,
                'channel_id': validation_result['channel_id'],
                'reactor_type': reactor_type,
                'is_valid': validation_result['is_valid'],
                'validation_score': validation_result['validation_score'],
                'total_checks': total_checks,
                'passed_checks': passed_checks,
                'critical_anomalies': critical_anomalies,
                'warnings': warnings,
                'validation_method': 'ChannelValidationMethod',
                'created': datetime.now().isoformat(),
                'detailed_results': validation_result
            }
        )
        changes["nodes_added"] += 1
        changes["validation_reports_created"] += 1
        return report_id

    def _create_summary_report(self, graph_manager, validation_results, reactor_type, validation_rule, changes):
        """Create comprehensive validation summary report"""
        summary_id = self._get_next_numeric_id(graph_manager)

        # Calculate summary statistics
        total_channels = len(validation_results)
        valid_channels = sum(1 for r in validation_results if r['is_valid'])
        invalid_channels = total_channels - valid_channels

        # Anomaly analysis
        anomaly_types = defaultdict(int)
        severity_counts = defaultdict(int)

        for result in validation_results:
            for anomaly in result['anomalies']:
                anomaly_types[anomaly['type']] += 1
                severity_counts[anomaly['severity']] += 1

        # Calculate overall compliance rate
        compliance_rate = (valid_channels / total_channels) if total_channels > 0 else 0.0

        graph_manager.add_node(
            node_id=summary_id,
            value=f"Channel Validation Summary - {reactor_type}",
            type='validation_summary_report',
            hierarchy='analysis',
            attributes={
                'reactor_type': reactor_type,
                'validation_rule_name': validation_rule['name'],
                'total_channels_validated': total_channels,
                'valid_channels': valid_channels,
                'invalid_channels': invalid_channels,
                'compliance_rate': compliance_rate,
                'anomaly_types': dict(anomaly_types),
                'severity_counts': dict(severity_counts),
                'validation_method': 'ChannelValidationMethod',
                'created': datetime.now().isoformat(),
                'validation_rule_details': validation_rule
            }
        )
        changes["nodes_added"] += 1
        return summary_id

    def _update_channel_validation_status(self, graph_manager, channel_id, validation_result, changes):
        """Update channel node with validation status"""
        channel_data = graph_manager.node_data[channel_id]

        if 'attributes' not in channel_data:
            channel_data['attributes'] = {}

        channel_data['attributes'].update({
            'validation_status': 'valid' if validation_result['is_valid'] else 'invalid',
            'validation_score': validation_result['validation_score'],
            'last_validated': validation_result['validation_timestamp'],
            'validation_method': 'ChannelValidationMethod',
            'anomaly_count': len(validation_result['anomalies']),
            'warning_count': len(validation_result['warnings'])
        })

    def _generate_validation_summary(self, validation_results, reactor_type):
        """Generate human-readable validation summary"""
        total = len(validation_results)
        valid = sum(1 for r in validation_results if r['is_valid'])

        summary = {
            'reactor_type': reactor_type,
            'total_channels': total,
            'valid_channels': valid,
            'invalid_channels': total - valid,
            'compliance_percentage': (valid / total * 100) if total > 0 else 0,
            'most_common_anomalies': [],
            'validation_quality': 'high' if valid / total > 0.9 else 'medium' if valid / total > 0.7 else 'low'
        }

        # Find most common anomaly types
        anomaly_counts = defaultdict(int)
        for result in validation_results:
            for anomaly in result['anomalies']:
                anomaly_counts[anomaly['type']] += 1

        summary['most_common_anomalies'] = sorted(
            anomaly_counts.items(), key=lambda x: x[1], reverse=True
        )[:5]

        return summary

    def _connect_channel_to_validation(self, graph_manager, channel_id, validation_report_id, changes):
        """Connect channel to its validation report"""
        try:
            graph_manager.add_edge(
                source=channel_id,
                target=validation_report_id,
                attributes={
                    'edge_type': 'has_validation_report',
                    'direction': 'out',
                    'created': datetime.now().isoformat()
                }
            )
            changes["edges_added"] += 1
        except (KeyError, ValueError) as e:
            print(f" Error connecting channel to validation report: {e}")

    def _create_method_instance(self, graph_manager, changes, reactor_type, validation_rule):
        """Create method instance node"""
        method_instance_id = self._get_next_numeric_id(graph_manager)

        graph_manager.add_node(
            node_id=method_instance_id,
            value=f"ChannelValidationMethod execution ({reactor_type})",
            type='ChannelValidationMethod',
            hierarchy='analysis',
            attributes={
                'method_type': 'ChannelValidationMethod',
                'reactor_type': reactor_type,
                'validation_rule_name': validation_rule['name'],
                'execution_time': datetime.now().isoformat(),
                'method_id': self.method_id,
                'method_name': self.method_name
            }
        )
        changes["nodes_added"] += 1
        changes["method_instances_created"] += 1
        return method_instance_id

    def _connect_method_to_output(self, graph_manager, method_instance_id, output_node_id, changes):
        """Connect method to its output"""
        try:
            graph_manager.add_edge(
                source=method_instance_id,
                target=output_node_id,
                attributes={
                    'edge_type': 'produces',
                    'direction': 'out',
                    'provenance_type': 'generated_by',
                    'created': datetime.now().isoformat()
                }
            )
            changes["edges_added"] += 1
        except (KeyError, ValueError) as e:
            print(f" Error creating method output edge: {e}")

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