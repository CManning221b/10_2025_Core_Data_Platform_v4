import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

class AdaptiveMeasurementGrouper:
    """
    Intelligently groups measurements based on GLOBAL data density and temporal distribution
    to optimize total graph complexity while preserving temporal patterns.

    Added support for regularly sampled data to evenly partition sequences
    based on global node density constraints.
    """

    def __init__(self, max_total_nodes=2000, min_nodes_per_stream=5, regular_sampling=False):
        """
        Initialize with global constraints instead of per-sensor targets

        Args:
            max_total_nodes: Maximum total measurement nodes across entire graph
            min_nodes_per_stream: Minimum nodes per sensor stream to preserve patterns
            regular_sampling: If True, treat data as evenly sampled and partition by count instead of time
        """
        self.max_total_nodes = max_total_nodes
        self.min_nodes_per_stream = min_nodes_per_stream
        self.regular_sampling = regular_sampling

        # Temporal grouping hierarchy (finest to coarsest)
        self.temporal_levels = [
            ('individual', timedelta(seconds=0)),  # No grouping
            ('minute', timedelta(minutes=1)),
            ('hour', timedelta(hours=1)),
            ('day', timedelta(days=1)),
            ('week', timedelta(weeks=1)),
            ('month', timedelta(days=30)),
            ('quarter', timedelta(days=90))
        ]

    def calculate_global_density_targets(self, total_engines: int, total_sensors: int,
                                         total_measurements: int) -> Tuple[int, int]:
        """Calculate target nodes per sensor stream based on global constraints"""
        sensor_streams = total_engines * total_sensors

        if sensor_streams == 0:
            return (10, 50)  # Fallback

        avg_nodes_per_stream = self.max_total_nodes // sensor_streams
        target_nodes = max(self.min_nodes_per_stream, avg_nodes_per_stream)
        target_min = max(3, target_nodes // 2)
        target_max = target_nodes * 2

        print(f"Global density calculation:")
        print(f"  - {sensor_streams} sensor streams ({total_engines} engines × {total_sensors} sensors)")
        print(f"  - {total_measurements} total measurements")
        print(f"  - Target nodes per stream: {target_min}-{target_max}")
        print(f"  - Estimated total nodes: {sensor_streams * target_nodes}")

        return (target_min, target_max)

    def analyze_temporal_distribution(self, timestamps: List[datetime],
                                      global_targets: Tuple[int, int] = None) -> Dict:
        """Analyze temporal distribution with global density awareness"""
        if not timestamps:
            return {}

        timestamps = sorted(timestamps)
        measurement_count = len(timestamps)
        target_min, target_max = global_targets if global_targets else (10, 50)

        # ✅ NEW MODE: Regularly sampled data
        if self.regular_sampling:
            # Uniform chunks mode, ignoring gaps
            ideal_nodes = max(target_min, min(target_max, measurement_count))
            chunk_size = max(1, measurement_count // ideal_nodes)

            temporal_stats = {
                'uniform_chunks': {
                    'measurements_per_unit': chunk_size,
                    'total_units': measurement_count / chunk_size,
                    'estimated_nodes': measurement_count / chunk_size,
                    'density_score': 100,  # perfect fit
                    'fits_global_target': True
                }
            }

            return {
                'total_measurements': measurement_count,
                'time_span': timestamps[-1] - timestamps[0],
                'avg_gap_seconds': None,
                'temporal_stats': temporal_stats,
                'global_targets': (target_min, target_max)
            }

        # Existing irregular mode
        total_span = timestamps[-1] - timestamps[0]
        gaps = [(timestamps[i + 1] - timestamps[i]).total_seconds()
                for i in range(len(timestamps) - 1)]
        avg_gap_seconds = sum(gaps) / len(gaps) if gaps else 0

        temporal_stats = {}
        for level_name, level_delta in self.temporal_levels:
            if level_name == 'individual':
                estimated_nodes = measurement_count
                measurements_per_unit = 1
                total_units = measurement_count
            else:
                level_seconds = level_delta.total_seconds()
                if total_span.total_seconds() > 0:
                    total_units = max(1, total_span.total_seconds() / level_seconds)
                    measurements_per_unit = measurement_count / total_units
                    estimated_nodes = total_units
                else:
                    total_units = 1
                    measurements_per_unit = measurement_count
                    estimated_nodes = 1

            density_score = self._calculate_density_score(
                estimated_nodes, measurements_per_unit, target_min, target_max
            )

            temporal_stats[level_name] = {
                'measurements_per_unit': measurements_per_unit,
                'total_units': total_units,
                'estimated_nodes': estimated_nodes,
                'density_score': density_score,
                'fits_global_target': target_min <= estimated_nodes <= target_max
            }

        return {
            'total_measurements': measurement_count,
            'time_span': total_span,
            'avg_gap_seconds': avg_gap_seconds,
            'temporal_stats': temporal_stats,
            'global_targets': (target_min, target_max)
        }

    def _calculate_density_score(self, estimated_nodes: float, measurements_per_unit: float,
                                 target_min: int, target_max: int) -> float:
        """Calculate density score based on global targets"""
        if estimated_nodes < target_min:
            fit_score = -(target_min - estimated_nodes) * 2
        elif estimated_nodes > target_max:
            fit_score = -(estimated_nodes - target_max) * 1.5
        else:
            target_center = (target_min + target_max) / 2
            fit_score = 100 - abs(estimated_nodes - target_center)
        density_bonus = min(50, measurements_per_unit * 0.5)
        return fit_score + density_bonus

    def select_optimal_grouping(self, temporal_analysis: Dict) -> Tuple[str, timedelta]:
        """Select the best temporal grouping level based on global density"""
        if self.regular_sampling:
            return ('uniform_chunks', timedelta(seconds=0))

        temporal_stats = temporal_analysis['temporal_stats']
        best_level = None
        best_score = float('-inf')

        for level_name, _ in self.temporal_levels:
            if level_name not in temporal_stats:
                continue
            score = temporal_stats[level_name]['density_score']
            if score > best_score:
                best_score = score
                best_level = level_name

        delta = dict(self.temporal_levels).get(best_level, timedelta(days=1))
        return (best_level, delta)

    def create_temporal_groups(self, sensor_data: pd.DataFrame, grouping_level: str,
                               grouping_delta: timedelta, target_nodes: int = None) -> List[Dict]:
        """Group measurements according to selected temporal level"""
        if len(sensor_data) == 0:
            return []

        # ✅ NEW: Uniform chunks mode
        if self.regular_sampling and grouping_level == 'uniform_chunks':
            groups = []
            total_measurements = len(sensor_data)
            # --- NEW: enforce global cap ---
            if target_nodes is None or target_nodes <= 0:
                target_nodes = self.min_nodes_per_stream

            # If global_targets exists, shrink chunks accordingly
            max_possible_nodes = min(target_nodes, total_measurements)
            chunk_size = max(1, total_measurements // max_possible_nodes)

            for i in range(0, total_measurements, chunk_size):
                window_data = sensor_data.iloc[i:i+chunk_size]
                groups.append(self._make_group('uniform_chunks', window_data))
            return groups

        if grouping_level == 'individual':
            return [self._make_group('individual', pd.DataFrame([row]))
                    for _, row in sensor_data.iterrows()]

        # Original time-based grouping
        groups = []
        sensor_data = sensor_data.sort_values('timestamp')
        current_window_start = self._floor_to_level(sensor_data.iloc[0]['timestamp'],
                                                    grouping_level, grouping_delta)

        for window_start in pd.date_range(current_window_start,
                                          sensor_data.iloc[-1]['timestamp'] + grouping_delta,
                                          freq=self._get_pandas_freq(grouping_level)):
            window_end = window_start + grouping_delta
            window_data = sensor_data[(sensor_data['timestamp'] >= window_start) &
                                      (sensor_data['timestamp'] < window_end)]
            if len(window_data) > 0:
                groups.append(self._make_group(grouping_level, window_data))
        return groups

    def _make_group(self, group_type, window_data):
        # Determine whether timestamp or cycle-based
        if 'timestamp' in window_data.columns:
            start_marker = window_data.iloc[0]['timestamp']
            end_marker = window_data.iloc[-1]['timestamp']
        elif 'time_cycles' in window_data.columns:
            start_marker = int(window_data.iloc[0]['time_cycles'])
            end_marker = int(window_data.iloc[-1]['time_cycles'])
        else:
            start_marker = 0
            end_marker = 0

        return {
            'group_type': group_type,
            'start_time': start_marker,
            'end_time': end_marker,
            'measurements': window_data.to_dict('records'),
            'count': len(window_data),
            'stats': {
                'mean': float(window_data['value'].mean()),
                'std': float(window_data['value'].std()) if len(window_data) > 1 else 0.0,
                'min': float(window_data['value'].min()),
                'max': float(window_data['value'].max()),
                'trend': self._calculate_simple_trend(window_data['value'].tolist()),
                'data_quality': self._assess_data_quality(window_data['value'].tolist())
            }
        }

    def _assess_data_quality(self, values: List[float]) -> Dict:
        if not values:
            return {'quality_score': 0, 'issues': ['no_data']}
        issues = []
        if len(values) > 3:
            mean_val = sum(values) / len(values)
            std_val = (sum((x - mean_val) ** 2 for x in values) / len(values)) ** 0.5
            outliers = [v for v in values if abs(v - mean_val) > 3 * std_val]
            if outliers:
                issues.append(f'{len(outliers)}_outliers')
        if len(set(values)) == 1:
            issues.append('constant_values')
        quality_score = max(0, 100 - len(issues) * 20)
        return {
            'quality_score': quality_score,
            'issues': issues if issues else ['none'],
            'outlier_count': len([v for v in values if abs(v - sum(values) / len(values))
                                  > 2 * (max(values) - min(values)) / 4]) if len(values) > 1 else 0
        }

    def _floor_to_level(self, timestamp: datetime, level: str, delta: timedelta) -> datetime:
        if level == 'individual':
            return timestamp
        elif level == 'minute':
            return timestamp.replace(second=0, microsecond=0)
        elif level == 'hour':
            return timestamp.replace(minute=0, second=0, microsecond=0)
        elif level == 'day':
            return timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
        elif level == 'week':
            days_since_monday = timestamp.weekday()
            return (timestamp - timedelta(days=days_since_monday)).replace(
                hour=0, minute=0, second=0, microsecond=0)
        elif level == 'month':
            return timestamp.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        else:
            return timestamp

    def _get_pandas_freq(self, level: str) -> str:
        return {
            'individual': '1S',
            'minute': '1min',
            'hour': '1H',
            'day': '1D',
            'week': '1W',
            'month': '1ME',
            'quarter': '1QE'
        }.get(level, '1D')

    def _calculate_simple_trend(self, values: List[float]) -> str:
        if len(values) < 2:
            return 'stable'
        elif len(values) == 2:
            change_pct = (values[1] - values[0]) / values[0] * 100 if values[0] != 0 else 0
        else:
            first_third = values[:len(values) // 3] if len(values) >= 3 else [values[0]]
            last_third = values[-len(values) // 3:] if len(values) >= 3 else [values[-1]]
            first_avg = sum(first_third) / len(first_third)
            last_avg = sum(last_third) / len(last_third)
            change_pct = (last_avg - first_avg) / first_avg * 100 if first_avg != 0 else 0

        if change_pct > 10:
            return 'increasing_strong'
        elif change_pct > 2:
            return 'increasing_moderate'
        elif change_pct > 0.5:
            return 'increasing_slight'
        elif change_pct < -10:
            return 'decreasing_strong'
        elif change_pct < -2:
            return 'decreasing_moderate'
        elif change_pct < -0.5:
            return 'decreasing_slight'
        else:
            return 'stable'

    def get_grouping_summary(self, temporal_analysis: Dict, selected_level: str) -> Dict:
        if not temporal_analysis or 'temporal_stats' not in temporal_analysis:
            return {}
        stats = temporal_analysis['temporal_stats']
        selected_stats = stats.get(selected_level, {})
        return {
            'selected_level': selected_level,
            'estimated_nodes': selected_stats.get('estimated_nodes', 0),
            'measurements_per_node': selected_stats.get('measurements_per_unit', 0),
            'total_measurements': temporal_analysis.get('total_measurements', 0),
            'global_targets': temporal_analysis.get('global_targets', (0, 0)),
            'efficiency_ratio': (selected_stats.get('estimated_nodes', 1) /
                                 temporal_analysis.get('total_measurements', 1)),
            'all_options': {level: stats[level]['estimated_nodes'] for level in stats.keys()}
        }
