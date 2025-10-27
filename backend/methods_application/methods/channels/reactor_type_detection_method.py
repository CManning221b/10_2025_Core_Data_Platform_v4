# backend/methods_application/methods/reactor/reactor_type_detection_method.py
from backend.methods_application.method_implementation import MethodImplementation
import re
from datetime import datetime
from collections import Counter


class ReactorTypeDetectionMethod(MethodImplementation):
    method_id = "ReactorTypeDetectionMethod"
    method_name = "Reactor Type Detection and Classification"
    description = "Analyzes channel patterns and graph structure to determine reactor type and creates reactor type nodes"

    def __init__(self, ontology_manager=None):
        super().__init__()
        self.ontology_manager = ontology_manager
        self.reactor_signatures = self._load_reactor_signatures()
        self.confidence_threshold = 0.7

    def execute(self, graph_manager, parameters):
        changes = {
            "nodes_added": 0,
            "edges_added": 0,
            "method_instances_created": 0,
            "reactor_type_detected": None,
            "detection_confidence": 0.0,
            "channel_analysis": {},
            "pattern_matches": {},
            "evidence_summary": {}
        }

        # Load reactor signatures from ontology if available
        if self.ontology_manager:
            self.reactor_signatures = self._load_reactor_signatures_from_ontology()
            self.confidence_threshold = self._get_confidence_threshold_from_ontology()

        print(f" Starting reactor type detection with threshold {self.confidence_threshold}")

        # Analyze the graph for reactor evidence
        evidence = self._analyze_reactor_evidence(graph_manager)
        changes["evidence_summary"] = evidence

        # Determine reactor type based on evidence
        reactor_type, confidence = self._determine_reactor_type(evidence)
        changes["reactor_type_detected"] = reactor_type
        changes["detection_confidence"] = confidence

        print(f" Detected reactor type: {reactor_type} (confidence: {confidence:.3f})")

        if reactor_type and confidence >= self.confidence_threshold:
            # Create method instance
            method_instance_id = self._create_method_instance(graph_manager, changes, reactor_type, confidence)

            # Create or update reactor type node
            reactor_type_node_id = self._create_reactor_type_node(graph_manager, reactor_type, confidence, evidence,
                                                                  changes)

            # Create reactor instance node if we have enough evidence
            if confidence > 0.8:
                reactor_instance_id = self._create_reactor_instance_node(graph_manager, reactor_type, evidence, changes)
                self._connect_reactor_instance_to_type(graph_manager, reactor_instance_id, reactor_type_node_id,
                                                       changes)
                self._connect_method_to_output(graph_manager, method_instance_id, reactor_instance_id, changes)

            # Connect method to reactor type
            self._connect_method_to_output(graph_manager, method_instance_id, reactor_type_node_id, changes)

            # Link existing channels to reactor type
            self._link_channels_to_reactor_type(graph_manager, reactor_type_node_id, reactor_type, changes)

            # Update channel nodes with reactor type information
            self._update_channel_reactor_info(graph_manager, reactor_type, changes)

        else:
            print(
                f" Reactor type detection failed - confidence {confidence:.3f} below threshold {self.confidence_threshold}")
            # Still create a method instance to record the attempt
            method_instance_id = self._create_method_instance(graph_manager, changes, "Unknown", confidence)

        return changes

    def _load_reactor_signatures(self):
        """Load default reactor signatures for detection"""
        return {
            'CANDU': {
                'channel_patterns': [
                    r'^[A-Y](?:0[1-9]|1[0-9]|2[0-5])$',
                    r'^[A-Y]-(?:0[1-9]|1[0-9]|2[0-5])$'
                ],
                'grid_indicators': {
                    'width': 24,
                    'height': 25,
                    'total_positions': 600
                },
                'excluded_positions': ['A01', 'A02', 'A23', 'A24', 'Y01', 'Y02', 'Y23', 'Y24'],
                'typical_channels': ['A01', 'B12', 'H13', 'P14', 'Y25'],
                'pattern_confidence': 0.9,
                'keywords': ['candu', 'canadian', 'deuterium', 'uranium', 'heavy water'],
                'channel_count_range': (200, 600)
            },
            'AGR': {
                'channel_patterns': [
                    r'^(?:0[2-9]|[1-4][0-9]|5[2-9]|[6-9][0-9])(?:[0-9][0-9])$',
                    r'^\d{4}$'
                ],
                'grid_indicators': {
                    'width': 49,
                    'height': 49,
                    'total_positions': 2401
                },
                'excluded_positions': ['50', '51', '99'],
                'typical_channels': ['0878', '1270', '5264', '8742'],
                'pattern_confidence': 0.85,
                'keywords': ['agr', 'advanced', 'gas', 'cooled', 'graphite'],
                'channel_count_range': (300, 1000)
            }
        }

    def _load_reactor_signatures_from_ontology(self):
        """Load reactor signatures from ontology if available"""
        signatures = self._load_reactor_signatures()  # Start with defaults

        try:
            if hasattr(self.ontology_manager, 'node_types'):
                for class_name, class_data in self.ontology_manager.node_types.items():
                    if 'ReactorTypeDetection' in class_name or 'GridGeneration' in class_name:
                        reactor_type = self._infer_reactor_type_from_method(class_name)
                        if reactor_type and reactor_type in signatures:
                            annotations = class_data.get('attributes', {})

                            # Update pattern from ontology
                            if 'hasRegexPattern' in annotations:
                                signatures[reactor_type]['channel_patterns'] = [annotations['hasRegexPattern']]
                                print(f" Updated {reactor_type} pattern from ontology")

                            # Update grid dimensions
                            if 'hasGridDimensions' in annotations:
                                dims = annotations['hasGridDimensions']
                                if ',' in str(dims):
                                    width, height = str(dims).split(',')
                                    signatures[reactor_type]['grid_indicators'].update({
                                        'width': int(width.strip()),
                                        'height': int(height.strip())
                                    })

                            # Update excluded positions
                            if 'hasExcludedPositions' in annotations:
                                excluded = str(annotations['hasExcludedPositions'])
                                signatures[reactor_type]['excluded_positions'] = [
                                    ch.strip() for ch in excluded.split(',') if ch.strip()
                                ]

        except Exception as e:
            print(f" Error loading signatures from ontology: {e}")

        return signatures

    def _get_confidence_threshold_from_ontology(self):
        """Get confidence threshold from ontology"""
        try:
            if hasattr(self.ontology_manager, 'node_types'):
                for class_name, class_data in self.ontology_manager.node_types.items():
                    if 'ReactorTypeDetection' in class_name:
                        annotations = class_data.get('attributes', {})
                        if 'hasConfidenceThreshold' in annotations:
                            return float(annotations['hasConfidenceThreshold'])
        except Exception as e:
            print(f" Error loading confidence threshold from ontology: {e}")

        return 0.7  # Default

    def _analyze_reactor_evidence(self, graph_manager):
        """Analyze the graph for reactor type evidence"""
        evidence = {
            'channel_patterns': {},
            'channel_samples': [],
            'grid_evidence': {},
            'keyword_evidence': {},
            'structural_evidence': {},
            'confidence_factors': {}
        }

        # Collect channel samples
        channel_samples = []
        channel_nodes = []

        # From explicit channel nodes
        for node_id, node_data in graph_manager.node_data.items():
            if node_data.get('type') == 'channel':
                channel_id = node_data.get('channel_id') or node_data.get('value', '')
                if channel_id:
                    channel_samples.append(channel_id.upper())
                    channel_nodes.append(node_id)

        # From component text analysis
        for node_id, node_data in graph_manager.node_data.items():
            if node_data.get('type') == 'component':
                text = node_data.get('value', '').upper()
                # Try each reactor pattern to find potential channels
                for reactor_type, signature in self.reactor_signatures.items():
                    for pattern in signature['channel_patterns']:
                        matches = re.findall(pattern, text)
                        channel_samples.extend(matches)

        evidence['channel_samples'] = list(set(channel_samples))  # Remove duplicates
        print(f" Found {len(evidence['channel_samples'])} unique channel samples")

        # Analyze channel patterns against each reactor type
        for reactor_type, signature in self.reactor_signatures.items():
            pattern_matches = 0
            total_samples = len(evidence['channel_samples'])

            if total_samples > 0:
                for channel in evidence['channel_samples']:
                    for pattern in signature['channel_patterns']:
                        if re.match(pattern, channel):
                            pattern_matches += 1
                            break  # Don't double-count the same channel

                pattern_confidence = pattern_matches / total_samples
                evidence['channel_patterns'][reactor_type] = {
                    'matches': pattern_matches,
                    'total_samples': total_samples,
                    'confidence': pattern_confidence,
                    'matching_channels': [ch for ch in evidence['channel_samples']
                                          if any(re.match(p, ch) for p in signature['channel_patterns'])]
                }

        # Analyze grid structure evidence
        evidence['grid_evidence'] = self._analyze_grid_structure(graph_manager, evidence['channel_samples'])

        # Analyze keyword evidence in node values
        evidence['keyword_evidence'] = self._analyze_keyword_evidence(graph_manager)

        # Analyze structural patterns
        evidence['structural_evidence'] = self._analyze_structural_evidence(graph_manager, channel_nodes)

        return evidence

    def _analyze_grid_structure(self, graph_manager, channel_samples):
        """Analyze grid structure evidence"""
        grid_evidence = {}

        for reactor_type, signature in self.reactor_signatures.items():
            grid_info = signature['grid_indicators']
            excluded = signature['excluded_positions']

            # Calculate theoretical channel count
            theoretical_total = grid_info['width'] * grid_info['height']
            theoretical_active = theoretical_total - len(excluded)

            # Analyze actual channel distribution
            actual_count = len(channel_samples)

            # Calculate coverage if we have channels
            coverage_score = 0.0
            if actual_count > 0 and theoretical_active > 0:
                coverage_ratio = actual_count / theoretical_active
                # Ideal coverage is around 0.3-0.8 (realistic for inspection data)
                if 0.1 <= coverage_ratio <= 1.0:
                    coverage_score = min(1.0, coverage_ratio)
                elif coverage_ratio > 1.0:
                    coverage_score = 0.5  # Suspicious - more channels than positions

            # Check if channel count is in expected range
            count_range = signature['channel_count_range']
            count_score = 0.0
            if count_range[0] <= actual_count <= count_range[1]:
                count_score = 1.0
            elif actual_count > 0:
                # Partial credit for being somewhat close
                if actual_count < count_range[0]:
                    count_score = actual_count / count_range[0]
                else:
                    count_score = count_range[1] / actual_count

            grid_evidence[reactor_type] = {
                'theoretical_total': theoretical_total,
                'theoretical_active': theoretical_active,
                'actual_count': actual_count,
                'coverage_ratio': actual_count / theoretical_active if theoretical_active > 0 else 0,
                'coverage_score': coverage_score,
                'count_score': count_score,
                'grid_confidence': (coverage_score + count_score) / 2
            }

        return grid_evidence

    def _analyze_keyword_evidence(self, graph_manager):
        """Analyze keyword evidence in graph"""
        keyword_evidence = {}

        # Collect all text from relevant nodes
        all_text = []
        for node_data in graph_manager.node_data.values():
            if node_data.get('type') in ['file', 'folder', 'filename', 'component']:
                value = str(node_data.get('value', '')).lower()
                all_text.append(value)

        combined_text = ' '.join(all_text)

        for reactor_type, signature in self.reactor_signatures.items():
            keywords = signature['keywords']
            keyword_matches = 0
            matched_keywords = []

            for keyword in keywords:
                if keyword in combined_text:
                    keyword_matches += 1
                    matched_keywords.append(keyword)

            keyword_confidence = keyword_matches / len(keywords) if keywords else 0.0

            keyword_evidence[reactor_type] = {
                'matches': keyword_matches,
                'total_keywords': len(keywords),
                'confidence': keyword_confidence,
                'matched_keywords': matched_keywords
            }

        return keyword_evidence

    def _analyze_structural_evidence(self, graph_manager, channel_nodes):
        """Analyze structural patterns in the graph"""
        structural_evidence = {
            'channel_node_count': len(channel_nodes),
            'has_timestamp_connections': False,
            'has_folder_structure': False,
            'complexity_indicators': {}
        }

        # Check for timestamp connections (indicates measurement data)
        timestamp_connections = 0
        for node_id in channel_nodes:
            for (source, target), edge_attrs in graph_manager.edge_data.items():
                if source == node_id or target == node_id:
                    connected_node_id = target if source == node_id else source
                    connected_node = graph_manager.node_data.get(connected_node_id)
                    if connected_node and connected_node.get('type') == 'Timestamp':
                        timestamp_connections += 1
                        structural_evidence['has_timestamp_connections'] = True

        # Check folder structure complexity
        folder_count = len([n for n in graph_manager.node_data.values() if n.get('type') == 'folder'])
        structural_evidence['has_folder_structure'] = folder_count > 0
        structural_evidence['complexity_indicators'] = {
            'folder_count': folder_count,
            'timestamp_connections': timestamp_connections,
            'total_nodes': len(graph_manager.node_data),
            'total_edges': len(graph_manager.edge_data)
        }

        return structural_evidence

    def _determine_reactor_type(self, evidence):
        """Determine reactor type based on collected evidence"""
        reactor_scores = {}

        for reactor_type in self.reactor_signatures.keys():
            scores = []

            # Channel pattern score (most important)
            if reactor_type in evidence['channel_patterns']:
                pattern_score = evidence['channel_patterns'][reactor_type]['confidence']
                scores.append(('pattern', pattern_score, 0.5))  # 50% weight

            # Grid structure score
            if reactor_type in evidence['grid_evidence']:
                grid_score = evidence['grid_evidence'][reactor_type]['grid_confidence']
                scores.append(('grid', grid_score, 0.3))  # 30% weight

            # Keyword evidence score
            if reactor_type in evidence['keyword_evidence']:
                keyword_score = evidence['keyword_evidence'][reactor_type]['confidence']
                scores.append(('keyword', keyword_score, 0.1))  # 10% weight

            # Structural evidence score
            structural_score = 0.5  # Neutral score for structural evidence
            if evidence['structural_evidence']['has_timestamp_connections']:
                structural_score += 0.2
            if evidence['structural_evidence']['has_folder_structure']:
                structural_score += 0.2
            if evidence['structural_evidence']['channel_node_count'] > 10:
                structural_score += 0.1

            scores.append(('structural', min(1.0, structural_score), 0.1))  # 10% weight

            # Calculate weighted average
            total_weighted_score = sum(score * weight for _, score, weight in scores)
            total_weight = sum(weight for _, _, weight in scores)

            if total_weight > 0:
                final_score = total_weighted_score / total_weight
                reactor_scores[reactor_type] = {
                    'final_score': final_score,
                    'component_scores': scores,
                    'sample_count': len(evidence['channel_samples'])
                }

        # Find best match
        if reactor_scores:
            best_reactor = max(reactor_scores.items(), key=lambda x: x[1]['final_score'])
            return best_reactor[0], best_reactor[1]['final_score']

        return None, 0.0

    def _create_reactor_type_node(self, graph_manager, reactor_type, confidence, evidence, changes):
        """Create or update reactor type node"""
        # Check if reactor type node already exists
        existing_node = None
        for node_id, node_data in graph_manager.node_data.items():
            if (node_data.get('type') == 'reactor_type' and
                    node_data.get('reactor_type_code') == reactor_type):
                existing_node = node_id
                break

        if existing_node:
            # Update existing node with new confidence and evidence
            node_data = graph_manager.node_data[existing_node]
            node_data['attributes'].update({
                'detection_confidence': confidence,
                'last_updated': datetime.now().isoformat(),
                'evidence_summary': evidence['evidence_summary'] if 'evidence_summary' in evidence else {},
                'channel_sample_count': len(evidence['channel_samples'])
            })
            print(f" Updated existing reactor_type node {existing_node}")
            return existing_node
        else:
            # Create new reactor type node
            reactor_type_id = self._get_next_numeric_id(graph_manager)

            graph_manager.add_node(
                node_id=reactor_type_id,
                value=f"{reactor_type} Reactor Type",
                type='reactor_type',
                hierarchy='reactor',
                attributes={
                    'reactor_type_code': reactor_type,
                    'reactor_type_name': self._get_reactor_display_name(reactor_type),
                    'detection_confidence': confidence,
                    'detection_method': 'ReactorTypeDetectionMethod',
                    'created': datetime.now().isoformat(),
                    'channel_sample_count': len(evidence['channel_samples']),
                    'evidence_summary': {
                        'pattern_matches': evidence.get('channel_patterns', {}),
                        'grid_analysis': evidence.get('grid_evidence', {}),
                        'keyword_matches': evidence.get('keyword_evidence', {})
                    }
                }
            )
            changes["nodes_added"] += 1
            print(f" Created reactor_type node {reactor_type_id} for {reactor_type}")
            return reactor_type_id

    def _create_reactor_instance_node(self, graph_manager, reactor_type, evidence, changes):
        """Create a reactor instance node"""
        reactor_instance_id = self._get_next_numeric_id(graph_manager)

        graph_manager.add_node(
            node_id=reactor_instance_id,
            value=f"{reactor_type} Reactor Instance",
            type='reactor',
            hierarchy='reactor',
            attributes={
                'reactor_type': reactor_type,
                'instance_name': f"Detected {reactor_type} Reactor",
                'detection_method': 'ReactorTypeDetectionMethod',
                'created': datetime.now().isoformat(),
                'channel_count': len(evidence['channel_samples']),
                'has_measurement_data': evidence['structural_evidence']['has_timestamp_connections']
            }
        )
        changes["nodes_added"] += 1
        print(f" Created reactor instance node {reactor_instance_id}")
        return reactor_instance_id

    def _get_reactor_display_name(self, reactor_type):
        """Get display name for reactor type"""
        display_names = {
            'CANDU': 'Canadian Deuterium Uranium Reactor',
            'AGR': 'Advanced Gas-cooled Reactor',
            'PWR': 'Pressurized Water Reactor',
            'BWR': 'Boiling Water Reactor'
        }
        return display_names.get(reactor_type, reactor_type)

    def _infer_reactor_type_from_method(self, method_name):
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

    # Helper methods for connections and ID generation
    def _create_method_instance(self, graph_manager, changes, reactor_type, confidence):
        """Create method instance node"""
        method_instance_id = self._get_next_numeric_id(graph_manager)

        graph_manager.add_node(
            node_id=method_instance_id,
            value=f"ReactorTypeDetectionMethod execution",
            type='ReactorTypeDetectionMethod',
            hierarchy='analysis',
            attributes={
                'method_type': 'ReactorTypeDetectionMethod',
                'detected_reactor_type': reactor_type,
                'detection_confidence': confidence,
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
            print(f" Error creating output edge: {e}")

    def _connect_reactor_instance_to_type(self, graph_manager, reactor_instance_id, reactor_type_node_id, changes):
        """Connect reactor instance to its type"""
        try:
            graph_manager.add_edge(
                source=reactor_instance_id,
                target=reactor_type_node_id,
                attributes={
                    'edge_type': 'has_reactor_type',
                    'direction': 'out',
                    'created': datetime.now().isoformat()
                }
            )
            changes["edges_added"] += 1
        except (KeyError, ValueError) as e:
            print(f" Error creating reactor type edge: {e}")

    def _link_channels_to_reactor_type(self, graph_manager, reactor_type_node_id, reactor_type, changes):
        """Link existing channel nodes to the detected reactor type"""
        channel_count = 0
        for node_id, node_data in graph_manager.node_data.items():
            if node_data.get('type') == 'channel':
                try:
                    graph_manager.add_edge(
                        source=node_id,
                        target=reactor_type_node_id,
                        attributes={
                            'edge_type': 'belongs_to_reactor_type',
                            'direction': 'out',
                            'created': datetime.now().isoformat()
                        }
                    )
                    changes["edges_added"] += 1
                    channel_count += 1
                except (KeyError, ValueError):
                    pass  # Edge might already exist

        print(f" Linked {channel_count} channels to {reactor_type} reactor type")

    def _update_channel_reactor_info(self, graph_manager, reactor_type, changes):
        """Update channel nodes with reactor type information"""
        updated_channels = 0
        for node_id, node_data in graph_manager.node_data.items():
            if node_data.get('type') == 'channel':
                if 'attributes' not in node_data:
                    node_data['attributes'] = {}

                node_data['attributes'].update({
                    'reactor_type': reactor_type,
                    'reactor_classification_time': datetime.now().isoformat()
                })
                updated_channels += 1

        print(f" Updated {updated_channels} channel nodes with {reactor_type} reactor type info")

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