# backend/methods_application/methods/SensorBranchingDecisionMethod.py
from backend.methods_application.method_implementation import MethodImplementation
from datetime import datetime


class SensorBranchingDecisionMethod(MethodImplementation):
    method_id = "SensorBranchingDecisionMethod"
    method_name = "Risk Prioritization Analysis"
    description = "Categorizes sensor findings by risk level and determines analysis priorities"

    def execute(self, graph_manager, parameters):
        changes = {
            "nodes_added": 0,
            "edges_added": 0,
            "sensors_analyzed": 0,
            "immediate_action_sensors": 0,
            "near_term_watch_sensors": 0,
            "normal_monitoring_sensors": 0,
            "risk_categories": {}
        }

        # Create method instance
        method_instance_id = self._create_method_instance(graph_manager, changes)

        # Find SensorTrendAnalysis nodes (not TrendResult)
        trend_results = self._find_sensor_trend_analysis_nodes(graph_manager)
        print(f"DEBUG: Found {len(trend_results)} sensor trend analysis results")

        # Also get RUL data for risk assessment context
        rul_data = self._get_rul_context(graph_manager)
        print(f"DEBUG: Found RUL data for {len(rul_data)} engines")

        # Analyze each sensor trend for risk prioritization
        for result_id, result_data in trend_results.items():
            # Extract trend info
            trend_category = result_data.get('trend_category', 'stable')
            severity = result_data.get('severity', 'low')
            engine_id = result_data.get('engine_id', 'unknown')
            sensor_type = result_data.get('sensor_type', 'unknown')

            # Get RUL context for this engine
            engine_rul = rul_data.get(engine_id, {}).get('rul_value', 100)

            # Make risk prioritization decision
            risk_decision = self._assess_risk_priority(
                trend_category, severity, engine_rul, sensor_type
            )

            # Create risk prioritization node
            decision_id = self._create_risk_prioritization_node(
                graph_manager, result_id, risk_decision, changes
            )

            if decision_id:
                # Connect method to inputs and outputs
                self._connect_method_to_input(graph_manager, method_instance_id, result_id, changes)
                self._connect_method_to_output(graph_manager, method_instance_id, decision_id, changes)

                changes["sensors_analyzed"] += 1

                # Update risk category counts
                risk_level = risk_decision['risk_level']
                if risk_level == 'immediate_action':
                    changes["immediate_action_sensors"] += 1
                elif risk_level == 'near_term_watch':
                    changes["near_term_watch_sensors"] += 1
                else:
                    changes["normal_monitoring_sensors"] += 1

                # Track risk categories
                if risk_level not in changes["risk_categories"]:
                    changes["risk_categories"][risk_level] = 0
                changes["risk_categories"][risk_level] += 1

        # Create overall risk summary
        if trend_results:
            summary_id = self._create_risk_summary_node(graph_manager, changes)
            if summary_id:
                self._connect_method_to_output(graph_manager, method_instance_id, summary_id, changes)

        return changes

    def _find_sensor_trend_analysis_nodes(self, graph_manager):
        """Find all SensorTrendAnalysis nodes"""
        trend_results = {}

        for node_id, node_data in graph_manager.node_data.items():
            if node_data.get('type') == 'SensorTrendAnalysis':
                trend_results[node_id] = node_data

        return trend_results

    def _get_rul_context(self, graph_manager):
        """Get RUL data for risk context"""
        rul_data = {}

        for node_id, node_data in graph_manager.node_data.items():
            if node_data.get('type') == 'rul_predictor':
                engine_id = node_data.get('engine_id') or node_data.get('asset_id')
                rul_value = node_data.get('rul_value')
                if engine_id and rul_value:
                    rul_data[engine_id] = {'rul_value': rul_value}

        return rul_data

    def _assess_risk_priority(self, trend_category, severity, engine_rul, sensor_type):
        """Assess risk priority based on multiple factors"""

        # Risk scoring system
        risk_score = 0

        # Trend category risk
        if trend_category == 'degrading':
            risk_score += 3
        elif trend_category == 'high_variance':
            risk_score += 2
        else:  # stable
            risk_score += 0

        # Severity risk
        if severity == 'high':
            risk_score += 3
        elif severity == 'moderate':
            risk_score += 2
        else:  # low
            risk_score += 1

        # RUL-based urgency
        if engine_rul <= 10:
            risk_score += 4  # Critical RUL
        elif engine_rul <= 30:
            risk_score += 2  # Low RUL
        elif engine_rul <= 50:
            risk_score += 1  # Moderate RUL
        else:
            risk_score += 0  # High RUL

        # Sensor criticality (some sensors more important)
        critical_sensors = ['T30', 'T50', 'P30']  # Temperature and pressure
        if sensor_type in critical_sensors:
            risk_score += 1

        # Determine risk level and actions
        if risk_score >= 7:
            risk_level = 'immediate_action'
            urgency = 'critical'
            recommended_action = 'immediate_inspection_required'
            time_frame = '0-10_cycles'
        elif risk_score >= 4:
            risk_level = 'near_term_watch'
            urgency = 'elevated'
            recommended_action = 'schedule_maintenance_window'
            time_frame = '11-30_cycles'
        else:
            risk_level = 'normal_monitoring'
            urgency = 'routine'
            recommended_action = 'continue_monitoring'
            time_frame = '>30_cycles'

        return {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'urgency': urgency,
            'recommended_action': recommended_action,
            'time_frame': time_frame,
            'contributing_factors': {
                'trend_category': trend_category,
                'severity': severity,
                'engine_rul': engine_rul,
                'sensor_type': sensor_type
            }
        }

    def _create_risk_prioritization_node(self, graph_manager, trend_result_id, risk_decision, changes):
        """Create a risk prioritization result node"""
        decision_id = self._get_next_numeric_id(graph_manager)

        risk_level = risk_decision['risk_level']
        urgency = risk_decision['urgency']
        action = risk_decision['recommended_action']
        time_frame = risk_decision['time_frame']

        # Create display value with appropriate urgency icons
        if risk_level == 'immediate_action':
            icon = "🚨"
        elif risk_level == 'near_term_watch':
            icon = "⚠️"
        else:
            icon = "✅"

        display_value = f"{icon} {risk_level.replace('_', ' ').title()}: {action} ({time_frame})"

        try:
            graph_manager.add_node(
                node_id=decision_id,
                value=display_value,
                type='RiskPrioritization',
                hierarchy='analysis',
                attributes={
                    'trend_analysis_id': trend_result_id,
                    'risk_level': risk_level,
                    'risk_score': risk_decision['risk_score'],
                    'urgency': urgency,
                    'recommended_action': action,
                    'time_frame': time_frame,
                    'contributing_factors': risk_decision['contributing_factors'],
                    'created': datetime.now().isoformat()
                }
            )
            changes["nodes_added"] += 1
            print(f"DEBUG: Created RiskPrioritization {decision_id}: {display_value}")
            return decision_id

        except Exception as e:
            print(f"DEBUG: Error creating RiskPrioritization node: {e}")
            return None

    def _create_risk_summary_node(self, graph_manager, changes):
        """Create an overall risk summary node"""
        summary_id = self._get_next_numeric_id(graph_manager)

        immediate = changes["immediate_action_sensors"]
        near_term = changes["near_term_watch_sensors"]
        normal = changes["normal_monitoring_sensors"]
        total = changes["sensors_analyzed"]

        # Calculate percentages
        immediate_pct = (immediate / total * 100) if total > 0 else 0
        near_term_pct = (near_term / total * 100) if total > 0 else 0

        display_value = f"🔍 Fleet Risk Summary: {immediate} immediate ({immediate_pct:.1f}%), {near_term} near-term ({near_term_pct:.1f}%), {normal} normal"

        try:
            graph_manager.add_node(
                node_id=summary_id,
                value=display_value,
                type='FleetRiskSummary',
                hierarchy='analysis',
                attributes={
                    'total_sensors_analyzed': total,
                    'immediate_action_count': immediate,
                    'near_term_watch_count': near_term,
                    'normal_monitoring_count': normal,
                    'immediate_action_percentage': round(immediate_pct, 1),
                    'near_term_watch_percentage': round(near_term_pct, 1),
                    'risk_categories': changes["risk_categories"],
                    'created': datetime.now().isoformat()
                }
            )
            changes["nodes_added"] += 1
            print(f"DEBUG: Created FleetRiskSummary {summary_id}: {display_value}")
            return summary_id

        except Exception as e:
            print(f"DEBUG: Error creating FleetRiskSummary node: {e}")
            return None

    def _create_method_instance(self, graph_manager, changes):
        """Create a method instance node for execution traceability"""
        method_instance_id = self._get_next_numeric_id(graph_manager)

        try:
            graph_manager.add_node(
                node_id=method_instance_id,
                value=f"SensorBranchingDecisionMethod execution",
                type='SensorBranchingDecisionMethod',
                hierarchy='analysis',
                attributes={
                    'method_type': 'SensorBranchingDecisionMethod',
                    'execution_time': datetime.now().isoformat(),
                    'method_id': self.method_id,
                    'method_name': self.method_name
                }
            )
            changes["nodes_added"] += 1
            print(f"DEBUG: Created method_instance node {method_instance_id}")
            return method_instance_id

        except KeyError as e:
            print(f"DEBUG: Method instance node {method_instance_id} already exists")
            return method_instance_id

    def _connect_method_to_input(self, graph_manager, method_instance_id, result_id, changes):
        """Connect method instance to trend result it analyzed"""
        try:
            graph_manager.add_edge(
                source=method_instance_id,
                target=result_id,
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

    def _connect_method_to_output(self, graph_manager, method_instance_id, decision_id, changes):
        """Connect method instance to decision it created"""
        try:
            graph_manager.add_edge(
                source=method_instance_id,
                target=decision_id,
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