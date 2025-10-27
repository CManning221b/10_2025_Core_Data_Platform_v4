# backend/methods_application/methods/system_performance_evaluation_method.py
from backend.methods_application.method_implementation import MethodImplementation
from datetime import datetime
import json


class SystemPerformanceEvaluationMethod(MethodImplementation):
    method_id = "SystemPerformanceEvaluationMethod"
    method_name = "Executive Fleet Analysis Summary"
    description = "Generates comprehensive executive summary by synthesizing all analysis results"

    def execute(self, graph_manager, parameters):
        changes = {
            "nodes_added": 0,
            "edges_added": 0,
            "summaries_generated": 0,
            "data_points_analyzed": 0,
            "executive_insights": []
        }

        # Create method instance
        method_instance_id = self._create_method_instance(graph_manager, changes)

        # Collect all analysis results from previous methods
        analysis_data = self._collect_all_analysis_results(graph_manager)
        print(f"DEBUG: Collected analysis data: {len(analysis_data)} result types")

        # Generate comprehensive executive summary
        executive_summary = self._generate_executive_summary(analysis_data)

        # Create executive summary node
        if executive_summary:
            summary_id = self._create_executive_summary_node(
                graph_manager, executive_summary, changes
            )

            if summary_id:
                # Connect to ALL analysis inputs
                for result_type, results in analysis_data.items():
                    for result_id in results.keys():
                        self._connect_method_to_input(graph_manager, method_instance_id, result_id, changes)

                self._connect_method_to_output(graph_manager, method_instance_id, summary_id, changes)
                changes["summaries_generated"] = 1

        return changes

    def _collect_all_analysis_results(self, graph_manager):
        """Collect all analysis results from previous methods"""
        analysis_data = {
            'sensor_trends': {},
            'bounds_results': {},
            'risk_prioritizations': {},
            'degradation_patterns': {},
            'fleet_summaries': {},
            'rul_data': {}
        }

        for node_id, node_data in graph_manager.node_data.items():
            node_type = node_data.get('type', '')

            if node_type == 'SensorTrendAnalysis':
                analysis_data['sensor_trends'][node_id] = node_data
            elif node_type == 'BoundsResult':
                analysis_data['bounds_results'][node_id] = node_data
            elif node_type == 'RiskPrioritization':
                analysis_data['risk_prioritizations'][node_id] = node_data
            elif node_type == 'DegradationPattern':
                analysis_data['degradation_patterns'][node_id] = node_data
            elif node_type == 'FleetPatternSummary':
                analysis_data['fleet_summaries'][node_id] = node_data
            elif node_type == 'rul_predictor':
                engine_id = node_data.get('engine_id') or node_data.get('asset_id')
                if engine_id:
                    analysis_data['rul_data'][engine_id] = node_data

        return analysis_data

    def _generate_executive_summary(self, analysis_data):
        """Generate comprehensive executive summary using rule-based NLP"""

        # Extract key metrics
        metrics = self._extract_key_metrics(analysis_data)

        # Build summary sections
        summary_sections = []

        # 1. Fleet Overview
        overview = self._generate_fleet_overview(metrics)
        summary_sections.append(overview)

        # 2. Critical Findings
        critical_findings = self._generate_critical_findings(metrics, analysis_data)
        summary_sections.append(critical_findings)

        # 3. Risk Assessment
        risk_assessment = self._generate_risk_assessment(metrics, analysis_data)
        summary_sections.append(risk_assessment)

        # 4. Pattern Analysis
        pattern_analysis = self._generate_pattern_analysis(metrics, analysis_data)
        summary_sections.append(pattern_analysis)

        # 5. Recommended Actions
        recommendations = self._generate_recommendations(metrics, analysis_data)
        summary_sections.append(recommendations)

        # Combine all sections
        full_summary = "\n\n".join(summary_sections)

        return {
            'full_text': full_summary,
            'sections': {
                'fleet_overview': overview,
                'critical_findings': critical_findings,
                'risk_assessment': risk_assessment,
                'pattern_analysis': pattern_analysis,
                'recommendations': recommendations
            },
            'metrics': metrics
        }

    def _extract_key_metrics(self, analysis_data):
        """Extract key numerical metrics from all analysis results"""
        metrics = {
            'total_engines': len(analysis_data['rul_data']),
            'total_sensors_analyzed': len(analysis_data['sensor_trends']),
            'bounds_violations': 0,
            'degrading_sensors': 0,
            'stable_sensors': 0,
            'immediate_action_count': 0,
            'near_term_watch_count': 0,
            'normal_monitoring_count': 0,
            'critical_rul_engines': 0,
            'low_rul_engines': 0,
            'high_rul_engines': 0,
            'avg_fleet_rul': 0,
            'min_rul': 999,
            'max_rul': 0,
            'pattern_count': len(analysis_data['degradation_patterns']),
            'critical_pattern_engines': 0,
            'sensor_type_counts': {}
        }

        # Process RUL data
        rul_values = []
        for engine_id, rul_data in analysis_data['rul_data'].items():
            rul_value = rul_data.get('rul_value', 0)
            rul_values.append(rul_value)

            if rul_value <= 20:
                metrics['critical_rul_engines'] += 1
            elif rul_value <= 50:
                metrics['low_rul_engines'] += 1
            else:
                metrics['high_rul_engines'] += 1

            metrics['min_rul'] = min(metrics['min_rul'], rul_value)
            metrics['max_rul'] = max(metrics['max_rul'], rul_value)

        if rul_values:
            metrics['avg_fleet_rul'] = sum(rul_values) / len(rul_values)

        # Process sensor trends
        for trend_id, trend_data in analysis_data['sensor_trends'].items():
            category = trend_data.get('trend_category', 'stable')
            sensor_type = trend_data.get('sensor_type', 'unknown')

            # Count by category
            if category == 'degrading':
                metrics['degrading_sensors'] += 1
            else:
                metrics['stable_sensors'] += 1

            # Count by sensor type
            if sensor_type not in metrics['sensor_type_counts']:
                metrics['sensor_type_counts'][sensor_type] = {'total': 0, 'degrading': 0}
            metrics['sensor_type_counts'][sensor_type]['total'] += 1
            if category == 'degrading':
                metrics['sensor_type_counts'][sensor_type]['degrading'] += 1

        # Process bounds results
        for bounds_id, bounds_data in analysis_data['bounds_results'].items():
            if not bounds_data.get('is_valid', True):
                metrics['bounds_violations'] += 1

        # Process risk prioritizations
        for risk_id, risk_data in analysis_data['risk_prioritizations'].items():
            risk_level = risk_data.get('risk_level', 'normal_monitoring')
            if risk_level == 'immediate_action':
                metrics['immediate_action_count'] += 1
            elif risk_level == 'near_term_watch':
                metrics['near_term_watch_count'] += 1
            else:
                metrics['normal_monitoring_count'] += 1

        # Process degradation patterns
        for pattern_id, pattern_data in analysis_data['degradation_patterns'].items():
            severity = pattern_data.get('pattern_severity', 'low')
            if severity in ['critical', 'high']:
                engine_count = pattern_data.get('engine_count', 0)
                metrics['critical_pattern_engines'] += engine_count

        return metrics

    def _generate_fleet_overview(self, metrics):
        """Generate fleet overview section"""
        total_engines = metrics['total_engines']
        total_sensors = metrics['total_sensors_analyzed']
        avg_rul = metrics['avg_fleet_rul']

        # Fleet health status
        critical_pct = (metrics['critical_rul_engines'] / total_engines * 100) if total_engines > 0 else 0

        if critical_pct > 15:
            fleet_status = "CRITICAL - Multiple engines require immediate attention"
        elif critical_pct > 5:
            fleet_status = "ELEVATED RISK - Several engines showing degradation"
        else:
            fleet_status = "NOMINAL - Fleet operating within normal parameters"

        overview = f"""FLEET ANALYSIS EXECUTIVE SUMMARY

Fleet Status: {fleet_status}

Key Fleet Metrics:
- Total Engines Analyzed: {total_engines}
- Total Sensor Points: {total_sensors}
- Average Fleet RUL: {avg_rul:.1f} cycles
- RUL Range: {metrics['min_rul']:.0f} - {metrics['max_rul']:.0f} cycles
- Engines Requiring Immediate Attention: {metrics['critical_rul_engines']} ({critical_pct:.1f}%)"""

        return overview

    def _generate_critical_findings(self, metrics, analysis_data):
        """Generate critical findings section"""
        findings = ["CRITICAL FINDINGS"]

        # RUL-based findings
        if metrics['critical_rul_engines'] > 0:
            critical_engines = []
            for engine_id, rul_data in analysis_data['rul_data'].items():
                if rul_data.get('rul_value', 0) <= 20:
                    critical_engines.append(f"Engine {engine_id} ({rul_data.get('rul_value', 0):.0f} cycles)")

            findings.append(
                f"• IMMEDIATE ACTION REQUIRED: {metrics['critical_rul_engines']} engines with critically low RUL")
            findings.append(f"  Critical engines: {', '.join(critical_engines[:5])}")  # Show first 5
            if len(critical_engines) > 5:
                findings.append(f"  Additional {len(critical_engines) - 5} engines require assessment")

        # Sensor degradation findings
        degrading_pct = (metrics['degrading_sensors'] / metrics['total_sensors_analyzed'] * 100) if metrics['total_sensors_analyzed'] > 0 else 0
        if degrading_pct > 10:
            findings.append(
                f"• SENSOR DEGRADATION: {metrics['degrading_sensors']} sensors ({degrading_pct:.1f}%) showing degradation trends")

        # Bounds violations
        if metrics['bounds_violations'] > 0:
            findings.append(
                f"• OPERATIONAL LIMITS EXCEEDED: {metrics['bounds_violations']} sensor readings outside acceptable ranges")

        # Pattern-based findings
        if metrics['critical_pattern_engines'] > 0:
            findings.append(
                f"• DEGRADATION PATTERNS: {metrics['critical_pattern_engines']} engines affected by critical degradation patterns")

        if len(findings) == 1:  # Only header
            findings.append("• No critical issues identified - fleet operating normally")

        return "\n".join(findings)

    def _generate_risk_assessment(self, metrics, analysis_data):
        """Generate risk assessment section"""
        total_risks = metrics['immediate_action_count'] + metrics['near_term_watch_count'] + metrics['normal_monitoring_count']

        risk_lines = ["RISK ASSESSMENT"]

        if total_risks > 0:
            immediate_pct = (metrics['immediate_action_count'] / total_risks * 100)
            near_term_pct = (metrics['near_term_watch_count'] / total_risks * 100)

            risk_lines.append(
                f"• Immediate Action Required: {metrics['immediate_action_count']} sensors ({immediate_pct:.1f}%)")
            risk_lines.append(
                f"• Near-term Monitoring: {metrics['near_term_watch_count']} sensors ({near_term_pct:.1f}%)")
            risk_lines.append(f"• Normal Operations: {metrics['normal_monitoring_count']} sensors")

            # Risk distribution insight
            if immediate_pct > 15:
                risk_lines.append("• FLEET RISK LEVEL: HIGH - Multiple systems require immediate intervention")
            elif immediate_pct > 5:
                risk_lines.append("• FLEET RISK LEVEL: MODERATE - Enhanced monitoring recommended")
            else:
                risk_lines.append("• FLEET RISK LEVEL: LOW - Standard maintenance protocols sufficient")

        return "\n".join(risk_lines)

    def _generate_pattern_analysis(self, metrics, analysis_data):
        """Generate pattern analysis section"""
        pattern_lines = ["DEGRADATION PATTERN ANALYSIS"]

        if metrics['pattern_count'] > 0:
            pattern_lines.append(f"• {metrics['pattern_count']} distinct degradation patterns identified")

            # Analyze specific patterns
            for pattern_id, pattern_data in analysis_data['degradation_patterns'].items():
                pattern_type = pattern_data.get('pattern_type', 'unknown')
                engine_count = pattern_data.get('engine_count', 0)
                avg_rul = pattern_data.get('average_rul', 0)
                severity = pattern_data.get('pattern_severity', 'low')

                severity_label = severity.upper()

                # Convert pattern type to readable format
                readable_type = pattern_type.replace('_', ' ').title()
                pattern_lines.append(
                    f"  [{severity_label}] {readable_type}: {engine_count} engines (avg RUL: {avg_rul:.1f})")

                # Add cause if available
                likely_cause = pattern_data.get('likely_cause', '')
                if likely_cause:
                    readable_cause = likely_cause.replace('_', ' ').replace('or', '/').title()
                    pattern_lines.append(f"            Likely cause: {readable_cause}")
        else:
            pattern_lines.append("• No significant degradation patterns detected")

        return "\n".join(pattern_lines)

    def _generate_recommendations(self, metrics, analysis_data):
        """Generate actionable recommendations"""
        rec_lines = ["RECOMMENDED ACTIONS"]

        recommendations = []

        # Critical RUL recommendations
        if metrics['critical_rul_engines'] > 0:
            critical_engines = [engine_id for engine_id, rul_data in analysis_data['rul_data'].items()
                                if rul_data.get('rul_value', 0) <= 20]
            recommendations.append(f"IMMEDIATE: Schedule maintenance for engines {', '.join(critical_engines[:3])}")
            if len(critical_engines) > 3:
                recommendations.append(
                    f"STRATEGIC: Develop replacement strategy for {len(critical_engines)} critical engines")

        # Sensor-based recommendations
        worst_sensor_types = []
        for sensor_type, counts in metrics['sensor_type_counts'].items():
            if counts['total'] > 0:
                degrading_pct = (counts['degrading'] / counts['total'] * 100)
                if degrading_pct > 20:
                    worst_sensor_types.append((sensor_type, degrading_pct))

        if worst_sensor_types:
            worst_sensor_types.sort(key=lambda x: x[1], reverse=True)
            top_sensor = worst_sensor_types[0]
            recommendations.append(
                f"HIGH PRIORITY: Investigate {top_sensor[0]} sensors ({top_sensor[1]:.1f}% degrading)")

        # Pattern-based recommendations
        for pattern_id, pattern_data in analysis_data['degradation_patterns'].items():
            if pattern_data.get('pattern_severity') in ['critical', 'high']:
                action = pattern_data.get('recommended_action', '').replace('_', ' ').title()
                if action:
                    recommendations.append(f"PATTERN-BASED: {action}")

        # Bounds violations
        if metrics['bounds_violations'] > 10:
            recommendations.append(
                f"CALIBRATION: Review sensor calibration procedures - {metrics['bounds_violations']} bounds violations detected")

        # Default recommendation if none critical
        if not recommendations:
            recommendations.append("ROUTINE: Continue standard maintenance protocols - no critical issues identified")
            recommendations.append("MONITORING: Maintain current sensor monitoring schedule")

        # Add numbered recommendations
        for i, rec in enumerate(recommendations, 1):
            rec_lines.append(f"{i}. {rec}")

        return "\n".join(rec_lines)

    def _create_executive_summary_node(self, graph_manager, summary_data, changes):
        """Create executive summary node"""
        summary_id = self._get_next_numeric_id(graph_manager)

        # Create concise display value
        metrics = summary_data['metrics']
        critical_count = metrics['critical_rul_engines']
        pattern_count = metrics['pattern_count']

        display_value = f"Executive Summary: {metrics['total_engines']} engines analyzed, {critical_count} critical, {pattern_count} patterns identified"

        try:
            graph_manager.add_node(
                node_id=summary_id,
                value=display_value,
                type='ExecutiveSummary',
                hierarchy='analysis',
                attributes={
                    'full_summary_text': summary_data['full_text'],
                    'fleet_overview': summary_data['sections']['fleet_overview'],
                    'critical_findings': summary_data['sections']['critical_findings'],
                    'risk_assessment': summary_data['sections']['risk_assessment'],
                    'pattern_analysis': summary_data['sections']['pattern_analysis'],
                    'recommendations': summary_data['sections']['recommendations'],
                    'key_metrics': summary_data['metrics'],
                    'total_engines_analyzed': metrics['total_engines'],
                    'critical_engines_count': metrics['critical_rul_engines'],
                    'degradation_patterns_count': metrics['pattern_count'],
                    'immediate_action_sensors': metrics['immediate_action_count'],
                    'created': datetime.now().isoformat()
                }
            )
            changes["nodes_added"] += 1
            changes["data_points_analyzed"] = metrics['total_sensors_analyzed']
            print(f"DEBUG: Created ExecutiveSummary {summary_id}: {display_value}")
            return summary_id

        except Exception as e:
            print(f"DEBUG: Error creating ExecutiveSummary node: {e}")
            return None

    # [Keep all existing helper methods unchanged]
    def _create_method_instance(self, graph_manager, changes):
        method_instance_id = self._get_next_numeric_id(graph_manager)
        try:
            graph_manager.add_node(
                node_id=method_instance_id,
                value=f"SystemPerformanceEvaluationMethod execution",
                type='SystemPerformanceEvaluationMethod',
                hierarchy='analysis',
                attributes={
                    'method_type': 'SystemPerformanceEvaluationMethod',
                    'execution_time': datetime.now().isoformat(),
                    'method_id': self.method_id,
                    'method_name': self.method_name
                }
            )
            changes["nodes_added"] += 1
        except KeyError:
            pass
        return method_instance_id

    def _connect_method_to_input(self, graph_manager, method_instance_id, input_id, changes):
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
        except (KeyError, ValueError):
            pass

    def _connect_method_to_output(self, graph_manager, method_instance_id, output_id, changes):
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
        except (KeyError, ValueError):
            pass

    def _get_next_numeric_id(self, graph_manager):
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