import pandas as pd
import numpy as np
import json
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from AdaptiveMeasurementGrouper import AdaptiveMeasurementGrouper


class TurbofanGraphGenerator:
    """
    Generates NASA Turbofan dataset graphs at different scales for demonstration and testing.
    Uses real NASA data with adaptive measurement grouping throughout.
    Includes ALL 21 sensors and operational settings for complete representation.
    """

    def __init__(self, data_path: str, rul_path: str = None):
        self.data_path = Path(data_path)
        self.rul_path = Path(rul_path) if rul_path else None
        self.raw_data = self.load_nasa_data()
        self.rul_data = self.load_rul_data() if rul_path else None
        self.node_counter = 0
        self.node_data = {}
        self.edge_data = {}
        self.node_id_map = {}

    def load_nasa_data(self) -> pd.DataFrame:
        """Load NASA Turbofan data from text file"""
        columns = [
            'unit_number', 'time_cycles', 'setting_1', 'setting_2', 'setting_3',
            'T2', 'T24', 'T30', 'T50', 'P2', 'P15', 'P30', 'Nf', 'Nc', 'epr',
            'Ps30', 'phi', 'NRf', 'NRc', 'BPR', 'farB', 'htBleed', 'Nf_dmd',
            'PCNfR_dmd', 'W31', 'W32'
        ]

        try:
            data = pd.read_csv(self.data_path, sep=r'\s+', names=columns, header=None)
            print(f"Loaded {len(data)} rows from {self.data_path}")
            return data
        except Exception as e:
            print(f"Error loading data: {e}")
            return self._create_minimal_synthetic_data()

    def load_rul_data(self) -> Optional[pd.DataFrame]:
        """Load RUL (Remaining Useful Life) data"""
        if not self.rul_path or not self.rul_path.exists():
            print("No RUL data file provided or found")
            return None

        try:
            rul_values = pd.read_csv(self.rul_path, header=None, names=['rul'])
            rul_values['unit_number'] = range(1, len(rul_values) + 1)
            print(f"Loaded RUL data for {len(rul_values)} engines")
            return rul_values
        except Exception as e:
            print(f"Error loading RUL data: {e}")
            return None

    def _create_rul_nodes(self, data: pd.DataFrame, engines: Dict):
        """Create RUL (Remaining Useful Life) nodes for engines with complete cycle data"""
        if self.rul_data is None:
            return

        for unit_num in data['unit_number'].unique():
            engine_name = f"Engine_{int(unit_num)}"
            engine_id = engines[unit_num]

            # Get RUL value for this engine
            rul_row = self.rul_data[self.rul_data['unit_number'] == unit_num]
            if len(rul_row) == 0:
                continue

            rul_value = int(rul_row['rul'].iloc[0])  # ✅ Already converted
            max_cycle = int(data[data['unit_number'] == unit_num]['time_cycles'].max())  # ✅ Convert here
            total_expected_cycles = int(max_cycle + rul_value)  # ✅ Convert here

            # Create RUL node
            rul_node_name = f"{engine_name}_RUL_Predictor"
            rul_id = self._get_or_create_node_id("rul_predictor", rul_node_name)

            self.node_data[rul_id].update({
                "engine_id": engine_name,
                "asset_id": engine_name,
                "current_cycle": max_cycle,  # ✅ Now native int
                "remaining_cycles": rul_value,  # ✅ Already native int
                "total_expected_cycles": total_expected_cycles,  # ✅ Now native int
                "rul_value": rul_value,
                "prediction_type": "ground_truth",
                "unit": "cycles"
            })

            self._add_edge(engine_id, rul_id, "has_rul_prediction")
            print(f"    Added RUL node for {engine_name}: {rul_value} cycles remaining (at cycle {max_cycle})")

    def _create_minimal_synthetic_data(self) -> pd.DataFrame:
        """Create minimal synthetic data only as absolute fallback"""
        print("WARNING: Using synthetic data as fallback - real NASA data preferred")
        np.random.seed(42)
        data = []

        for unit in range(1, 4):  # 3 engines
            for cycle in range(1, 21):  # 20 cycles each
                row = {
                    'unit_number': unit,
                    'time_cycles': cycle,
                    'setting_1': np.random.uniform(-0.007, 0.007),
                    'setting_2': np.random.uniform(-0.005, 0.005),
                    'setting_3': np.random.uniform(100, 110),
                    'T2': np.random.uniform(515, 525),
                    'P2': np.random.uniform(14.0, 15.0),
                    'Nf': np.random.uniform(2350, 2400),
                }
                data.append(row)

        return pd.DataFrame(data)

    def _get_data_subset(self, max_engines: int = None, max_cycles: int = None,
                         max_rows: int = None) -> pd.DataFrame:
        """Get filtered subset of real NASA data"""
        if len(self.raw_data) == 0:
            return self._create_minimal_synthetic_data()

        data = self.raw_data.copy()

        if max_engines:
            data = data[data['unit_number'] <= max_engines]
        if max_cycles:
            data = data[data['time_cycles'] <= max_cycles]
        if max_rows:
            data = data.head(max_rows)

        return data

    def _get_complete_sensor_mappings(self):
        """Complete sensor mappings for all 21 NASA turbofan sensors"""
        return {
            # Temperature sensors (°R - Rankine)
            'T2': ('temperature_sensor', '°R'),
            'T24': ('temperature_sensor', '°R'),
            'T30': ('temperature_sensor', '°R'),
            'T50': ('temperature_sensor', '°R'),

            # Pressure sensors (psia)
            'P2': ('pressure_sensor', 'psia'),
            'P15': ('pressure_sensor', 'psia'),
            'P30': ('pressure_sensor', 'psia'),
            'Ps30': ('pressure_sensor', 'psia'),

            # Speed sensors (rpm)
            'Nf': ('speed_sensor', 'rpm'),  # Physical fan speed
            'Nc': ('speed_sensor', 'rpm'),  # Physical core speed
            'NRf': ('speed_sensor', 'rpm'),  # Corrected fan speed
            'NRc': ('speed_sensor', 'rpm'),  # Corrected core speed
            'Nf_dmd': ('speed_sensor', 'rpm'),  # Demanded fan speed

            # Ratio sensors (dimensionless)
            'epr': ('ratio_sensor', 'ratio'),  # Engine pressure ratio
            'BPR': ('ratio_sensor', 'ratio'),  # Bypass ratio
            'PCNfR_dmd': ('ratio_sensor', 'ratio'),  # Demanded corrected fan speed ratio

            # Flow sensors
            'W31': ('flow_sensor', 'lbm/s'),  # Airflow
            'W32': ('flow_sensor', 'lbm/s'),  # Airflow
            'phi': ('flow_sensor', 'pps'),  # Fuel flow
            'farB': ('flow_sensor', 'ratio'),  # Fuel air ratio

            # Bleed sensor
            'htBleed': ('bleed_sensor', 'lbm/s'),  # Bleed enthalpy
        }

    def _get_operational_setting_mappings(self):
        """Operational setting mappings"""
        return {
            'setting_1': ('operational_setting', 'normalized'),
            'setting_2': ('operational_setting', 'normalized'),
            'setting_3': ('operational_setting', 'normalized'),
        }

    def _get_or_create_node_id(self, node_type: str, value: str) -> int:
        """Create unique node ID for type-value combination"""
        key = (node_type, str(value))
        if key not in self.node_id_map:
            node_id = self.node_counter
            self.node_id_map[key] = node_id
            self.node_data[node_id] = {
                "type": node_type,
                "value": str(value),
                "node_id": node_id
            }
            self.node_counter += 1
        return self.node_id_map[key]

    def _add_edge(self, source_id: int, target_id: int, edge_type: str, **kwargs):
        """Add edge between two nodes"""
        edge_key = (source_id, target_id)
        if edge_key not in self.edge_data:
            self.edge_data[edge_key] = {
                "edge_type": edge_type,
                "source": source_id,
                "target": target_id,
                **kwargs
            }

    def _reset_graph(self):
        """Reset graph data for new generation"""
        self.node_counter = 0
        self.node_data.clear()
        self.edge_data.clear()
        self.node_id_map.clear()

    def _save_graph(self, filename: str):
        """Save graph to JSON file"""
        data = {
            "nodes": self.node_data,
            "edges": {f"{k[0]}_{k[1]}": v for k, v in self.edge_data.items()}
        }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved {filename}: {len(self.node_data)} nodes, {len(self.edge_data)} edges")

    def _create_operational_setting_nodes(self, data: pd.DataFrame, engines: Dict):
        """Create operational setting nodes for each engine"""
        setting_mappings = self._get_operational_setting_mappings()

        for unit_num in data['unit_number'].unique():
            engine_name = f"Engine_{int(unit_num)}"
            engine_id = engines[unit_num]

            # Create setting control nodes for this engine
            setting_nodes = {}
            for setting_col in ['setting_1', 'setting_2', 'setting_3']:
                if setting_col in setting_mappings:
                    setting_type, unit = setting_mappings[setting_col]
                    setting_name = f"{engine_name}_{setting_col}_Controller"
                    setting_id = self._get_or_create_node_id(setting_type, setting_name)

                    self.node_data[setting_id].update({
                        "engine_id": engine_name,
                        "setting_type": setting_col,
                        "asset_id": engine_name,
                        "control_parameter": setting_col,
                        "unit": unit
                    })

                    self._add_edge(engine_id, setting_id, "has_operational_setting")
                    setting_nodes[setting_col] = setting_id

            # Create setting value nodes using adaptive grouping
            for setting_col in ['setting_1', 'setting_2', 'setting_3']:
                if setting_col in setting_nodes:
                    # Get all setting values for this engine
                    setting_data = data[
                        (data['unit_number'] == unit_num) &
                        (data[setting_col].notna())
                        ][['time_cycles', setting_col]].copy()

                    if len(setting_data) > 0:
                        setting_controller_id = setting_nodes[setting_col]

                        # Use adaptive grouping for settings too
                        setting_value_nodes = self._create_adaptive_setting_nodes(
                            engine_name, setting_col, setting_data, setting_controller_id
                        )

    def _create_adaptive_setting_nodes(self, engine_name: str, setting_col: str,
                                       setting_data: pd.DataFrame, setting_controller_id: int) -> List[int]:
        """Create setting value nodes using adaptive grouping"""

        # Prepare values
        setting_data = setting_data.copy()
        setting_data['value'] = setting_data[setting_col]

        # Initialize adaptive grouper
        grouper = AdaptiveMeasurementGrouper(regular_sampling=True)

        # Analyze distribution
        total_measurements = len(setting_data)

        # Scale targets for settings (generally fewer setting changes than sensor readings)
        num_engines = self.raw_data['unit_number'].nunique()
        num_settings = 3  # Always 3 operational settings
        total_setting_streams = num_engines * num_settings

        # Settings change less frequently, so smaller node cap
        SETTING_NODE_CAP = 500
        allowed_per_stream = max(1, SETTING_NODE_CAP // total_setting_streams)

        temporal_analysis = grouper.analyze_temporal_distribution(
            timestamps=[i for i in range(total_measurements)],
            global_targets=(allowed_per_stream, allowed_per_stream)
        )

        print(f"    {setting_col}-{engine_name}: {temporal_analysis['total_measurements']} settings, "
              f"cycles {setting_data['time_cycles'].min()}-{setting_data['time_cycles'].max()}")

        # Select optimal grouping
        grouping_level, grouping_delta = grouper.select_optimal_grouping(temporal_analysis)
        target_nodes = temporal_analysis["global_targets"][1]
        print(f"    Selected setting grouping: {grouping_level} ({target_nodes} target nodes)")

        # Group data
        groups = []
        if grouping_level == 'uniform_chunks':
            chunk_size = max(1, total_measurements // target_nodes)
            for i in range(0, total_measurements, chunk_size):
                window_data = setting_data.iloc[i:i + chunk_size]
                groups.append(grouper._make_group('uniform_chunks', window_data))
        else:
            # fallback: individual nodes
            for _, row in setting_data.iterrows():
                groups.append(grouper._make_group('individual', pd.DataFrame([row])))

        # Create nodes for each group
        setting_nodes = []
        for group in groups:
            if group['group_type'] == 'individual':
                setting = group['measurements'][0]
                cycle_num = setting['time_cycles']
                setting_value = setting['value']

                node_name = f"{engine_name}_{setting_col}_Cycle_{cycle_num}_Value_{setting_value:.4f}"
                node_id = self._get_or_create_node_id("setting_value", node_name)

                self.node_data[node_id].update({
                    "engine_id": engine_name,
                    "setting_type": setting_col,
                    "cycle_number": cycle_num,
                    "value": setting_value,
                    "current_value": setting_value,
                    "asset_id": engine_name,
                    "measurement_type": "individual"
                })

            else:
                settings = group['measurements']
                start_cycle = int(min([s['time_cycles'] for s in settings]))
                end_cycle = int(max([s['time_cycles'] for s in settings]))

                values_list = [s['value'] for s in settings]
                cycles_list = [s['time_cycles'] for s in settings]

                node_name = f"{engine_name}_{setting_col}_{grouping_level}_Cycles_{start_cycle}_to_{end_cycle}"
                node_id = self._get_or_create_node_id("setting_sequence", node_name)

                self.node_data[node_id].update({
                    "engine_id": engine_name,
                    "setting_type": setting_col,
                    "grouping_level": grouping_level,
                    "start_cycle": start_cycle,
                    "end_cycle": end_cycle,
                    "measurement_count": group['count'],
                    "values": values_list,
                    "cycle_numbers": cycles_list,
                    "current_value": values_list[-1],
                    "value": values_list[-1],
                    "sequence_stats": group['stats'],
                    "mean_value": group['stats']['mean'],
                    "trend": group['stats']['trend'],
                    "asset_id": engine_name,
                    "measurement_type": "sequence"
                })

            self._add_edge(setting_controller_id, node_id, "produces_setting_value")
            setting_nodes.append(node_id)

        print(f"    Created {len(setting_nodes)} setting nodes using {grouping_level} grouping")
        return setting_nodes

    def _create_graph_from_data(self, data: pd.DataFrame, sensor_subset: List[str] = None,
                                include_settings: bool = True, include_rul: bool = False,
                                graph_name: str = "unnamed"):
        """Create complete graph structure including settings, sensors, and optionally RUL"""

        if len(data) == 0:
            print(f"No data available for {graph_name}")
            return

        # Use ALL sensors if none specified
        if sensor_subset is None:
            all_sensor_mappings = self._get_complete_sensor_mappings()
            sensor_subset = list(all_sensor_mappings.keys())

        sensor_mappings = self._get_complete_sensor_mappings()

        print(f"\nCreating {graph_name} with {len(data)} measurements")
        print(f"Engines: {sorted(data['unit_number'].unique())}")
        print(f"Cycle range: {data['time_cycles'].min()} to {data['time_cycles'].max()}")
        print(f"Sensors: {len(sensor_subset)} sensors")
        print(f"Include settings: {include_settings}")

        # Create engines
        engines = {}
        for unit_num in data['unit_number'].unique():
            engine_name = f"Engine_{int(unit_num)}"
            engine_id = self._get_or_create_node_id("turbofan_engine", engine_name)
            engines[unit_num] = engine_id

        # ✅ NEW: Create operational setting nodes
        if include_settings:
            self._create_operational_setting_nodes(data, engines)

        # Create sensors for each engine
        sensor_instances = {}
        for unit_num in data['unit_number'].unique():
            engine_name = f"Engine_{int(unit_num)}"
            engine_id = engines[unit_num]
            sensor_instances[unit_num] = {}

            for sensor_col in sensor_subset:
                if sensor_col in sensor_mappings:
                    sensor_type, unit = sensor_mappings[sensor_col]
                    sensor_name = f"{engine_name}_{sensor_col}_Sensor"
                    sensor_id = self._get_or_create_node_id(sensor_type, sensor_name)

                    self.node_data[sensor_id].update({
                        "engine_id": engine_name,
                        "sensor_id": sensor_col,
                        "sensor_type": sensor_col,
                        "unit": unit,
                        "asset_id": engine_name
                    })

                    self._add_edge(engine_id, sensor_id, "has_sensor")
                    sensor_instances[unit_num][sensor_col] = sensor_id

        # Create adaptive measurement nodes
        for unit_num in data['unit_number'].unique():
            engine_name = f"Engine_{int(unit_num)}"

            for sensor_col in sensor_subset:
                if sensor_col in sensor_instances[unit_num]:
                    # Get all measurements for this sensor
                    sensor_data = data[
                        (data['unit_number'] == unit_num) &
                        (data[sensor_col].notna())
                        ][['time_cycles', sensor_col]].copy()

                    if len(sensor_data) > 0:
                        sensor_id = sensor_instances[unit_num][sensor_col]

                        # Use adaptive grouping
                        measurement_nodes = self._create_adaptive_measurement_nodes(
                            engine_name, sensor_col, sensor_data, sensor_id
                        )

        if include_rul and self.rul_data is not None:
            self._create_rul_nodes(data, engines)

    def _create_adaptive_measurement_nodes(self, engine_name: str, sensor_col: str,
                                           sensor_data: pd.DataFrame, sensor_id: int) -> List[int]:
        """Create measurement nodes using adaptive grouping (cycle-based for regular data)"""

        # Prepare values
        sensor_data = sensor_data.copy()
        sensor_data['value'] = sensor_data[sensor_col]

        # Initialize adaptive grouper in regular sampling mode
        grouper = AdaptiveMeasurementGrouper(regular_sampling=True)

        # Analyze cycle distribution
        total_measurements = len(sensor_data)
        # Dynamically scale targets based on dataset size
        num_engines = self.raw_data['unit_number'].nunique()
        all_sensor_mappings = self._get_complete_sensor_mappings()
        num_sensors = len(all_sensor_mappings)  # Use total available sensors
        total_streams = num_engines * num_sensors

        # Define a hard global node cap
        GLOBAL_NODE_CAP = 5000  # Increased for full sensor coverage
        allowed_per_stream = max(2, GLOBAL_NODE_CAP // total_streams)

        temporal_analysis = grouper.analyze_temporal_distribution(
            timestamps=[i for i in range(total_measurements)],
            global_targets=(allowed_per_stream, allowed_per_stream)
        )

        print(f"    {sensor_col}-{engine_name}: {temporal_analysis['total_measurements']} measurements, "
              f"cycles {sensor_data['time_cycles'].min()}-{sensor_data['time_cycles'].max()}")

        # Select optimal grouping
        grouping_level, grouping_delta = grouper.select_optimal_grouping(temporal_analysis)
        target_nodes = temporal_analysis["global_targets"][1]  # upper bound for uniform chunks
        print(f"    Selected grouping: {grouping_level} ({target_nodes} target nodes)")

        # Group data by uniform chunks (no timestamps needed)
        groups = []
        if grouping_level == 'uniform_chunks':
            chunk_size = max(1, total_measurements // target_nodes)
            for i in range(0, total_measurements, chunk_size):
                window_data = sensor_data.iloc[i:i + chunk_size]
                groups.append(grouper._make_group('uniform_chunks', window_data))
        else:
            # fallback: individual nodes
            for _, row in sensor_data.iterrows():
                groups.append(grouper._make_group('individual', pd.DataFrame([row])))

        # Create nodes for each group
        measurement_nodes = []
        for group in groups:
            if group['group_type'] == 'individual':
                measurement = group['measurements'][0]
                cycle_num = measurement['time_cycles']
                measurement_value = measurement['value']

                node_name = f"{engine_name}_{sensor_col}_Cycle_{cycle_num}_Reading_{measurement_value:.2f}"
                node_id = self._get_or_create_node_id("measurement", node_name)

                self.node_data[node_id].update({
                    "engine_id": engine_name,
                    "sensor_id": sensor_col,
                    "sensor_type": sensor_col,
                    "cycle_number": cycle_num,
                    "value": measurement_value,
                    "current_value": measurement_value,
                    "asset_id": engine_name,
                    "measurement_type": "individual"
                })

            else:
                measurements = group['measurements']
                start_cycle = int(min([m['time_cycles'] for m in measurements]))
                end_cycle = int(max([m['time_cycles'] for m in measurements]))

                values_list = [m['value'] for m in measurements]
                cycles_list = [m['time_cycles'] for m in measurements]

                node_name = f"{engine_name}_{sensor_col}_{grouping_level}_Cycles_{start_cycle}_to_{end_cycle}"
                node_id = self._get_or_create_node_id("measurement_sequence", node_name)

                self.node_data[node_id].update({
                    "engine_id": engine_name,
                    "sensor_id": sensor_col,
                    "sensor_type": sensor_col,
                    "grouping_level": grouping_level,
                    "start_cycle": start_cycle,
                    "end_cycle": end_cycle,
                    "measurement_count": group['count'],
                    "values": values_list,
                    "cycle_numbers": cycles_list,
                    "current_value": values_list[-1],
                    "value": values_list[-1],
                    "sequence_stats": group['stats'],
                    "mean_value": group['stats']['mean'],
                    "trend": group['stats']['trend'],
                    "asset_id": engine_name,
                    "measurement_type": "sequence"
                })

            self._add_edge(sensor_id, node_id, "produces_measurement")
            measurement_nodes.append(node_id)

        print(f"    Created {len(measurement_nodes)} nodes using {grouping_level} grouping")
        return measurement_nodes

    # === DATASET GENERATION METHODS ===

    def create_small_dataset(self):
        """Small testable dataset: 2 engines with FULL cycles + ALL sensors + RUL"""
        self._reset_graph()
        # Get complete lifecycle data for 2 engines (no cycle limits for RUL)
        small_data = self._get_data_subset(max_engines=2)
        all_sensors = list(self._get_complete_sensor_mappings().keys())  # ✅ ALL sensors
        self._create_graph_from_data(small_data, sensor_subset=all_sensors,
                                     include_settings=True, include_rul=True,  # ✅ Add RUL
                                     graph_name="Small Dataset with RUL")
        self._save_graph("./turbofan_graphs/turbofan_small_dataset_with_rul.json")

    def create_medium_dataset(self):
        """Medium testable dataset: 5 engines with FULL cycles + ALL sensors + RUL"""
        self._reset_graph()
        # Get complete lifecycle data for 5 engines
        medium_data = self._get_data_subset(max_engines=5)
        all_sensors = list(self._get_complete_sensor_mappings().keys())  # ✅ ALL sensors
        self._create_graph_from_data(medium_data, sensor_subset=all_sensors,
                                     include_settings=True, include_rul=True,  # ✅ Add RUL
                                     graph_name="Medium Dataset with RUL")
        self._save_graph("./turbofan_graphs/turbofan_medium_dataset_with_rul.json")

    def create_large_dataset(self):
        """Large dataset: 10 engines, extended cycles, 15 sensors + settings"""
        self._reset_graph()
        large_data = self._get_data_subset(max_engines=10, max_cycles=50, max_rows=1000)
        key_sensors = ['T2', 'T24', 'T30', 'T50', 'P2', 'P15', 'P30', 'Ps30',
                       'Nf', 'Nc', 'NRf', 'NRc', 'epr', 'BPR', 'phi']
        self._create_graph_from_data(large_data, sensor_subset=key_sensors,
                                     include_settings=True, graph_name="Large Dataset")
        self._save_graph("./turbofan_graphs/turbofan_large_dataset.json")

    def create_full_dataset(self, num_engines: Optional[int] = None,
                            downsample: bool = False,
                            sensor_subset: Optional[List[str]] = None,
                            include_settings: bool = True,
                            include_rul: bool = True):  # ✅ NEW parameter
        """Create scalable full dataset graph with RUL predictions"""
        self._reset_graph()

        data = self.raw_data.copy()

        if num_engines:
            data = data[data['unit_number'] <= num_engines]

        if downsample:
            data = data.groupby('unit_number').apply(
                lambda x: x.iloc[::2] if len(x) > 100 else x
            ).reset_index(drop=True)

        if sensor_subset is None:
            all_sensor_mappings = self._get_complete_sensor_mappings()
            sensor_subset = list(all_sensor_mappings.keys())

        self._create_graph_from_data(data, sensor_subset=sensor_subset,
                                     include_settings=include_settings,
                                     include_rul=include_rul,  # ✅ Pass through
                                     graph_name=f"Full Dataset ({num_engines or 'all'} engines)")

        filename = f"./turbofan_graphs/turbofan_full_dataset_{num_engines or 'all'}_engines"
        if include_rul:
            filename += "_with_rul"
        filename += ".json"

        self._save_graph(filename)

    def create_comprehensive_datasets(self):
        """Create comprehensive datasets with ALL sensors, settings, and RUL"""
        print("Creating comprehensive datasets with ALL 21 sensors + 3 settings + RUL...")

        # Only include RUL in comprehensive datasets where we have complete cycles
        all_sensors = list(self._get_complete_sensor_mappings().keys())

        # Small comprehensive with RUL
        self._reset_graph()
        small_data = self._get_data_subset(max_engines=5, max_cycles=None)  # Full cycles
        self._create_graph_from_data(small_data, sensor_subset=all_sensors,
                                     include_settings=True, include_rul=True,
                                     graph_name="Comprehensive Small Dataset with RUL")
        self._save_graph("./turbofan_graphs/turbofan_comprehensive_small_with_rul.json")

    def create_single_engine_demo(self):
        """Single engine demo with ALL sensors + RUL for complete testing"""
        self._reset_graph()
        # Get full cycle data for one engine (no max_cycles limit for RUL)
        single_engine_data = self._get_data_subset(max_engines=1)
        all_sensors = list(self._get_complete_sensor_mappings().keys())
        self._create_graph_from_data(single_engine_data, sensor_subset=all_sensors,
                                     include_settings=True, include_rul=True,  # ✅ Add RUL
                                     graph_name="Single Engine Demo with RUL")
        self._save_graph("./turbofan_graphs/turbofan_single_engine_with_rul.json")

    def create_multi_engine_demo(self):
        """Multi-engine demo with core sensors + RUL for testing"""
        self._reset_graph()
        # Get full cycle data for 3 engines (no max_cycles limit for RUL)
        multi_engine_data = self._get_data_subset(max_engines=3)
        key_sensors = ['T2', 'T24', 'P2', 'P15', 'Nf', 'Nc']
        self._create_graph_from_data(multi_engine_data, sensor_subset=key_sensors,
                                     include_settings=True, include_rul=True,  # ✅ Add RUL
                                     graph_name="Multi Engine Demo with RUL")
        self._save_graph("./turbofan_graphs/turbofan_multi_engine_with_rul.json")

    def create_concept_demo(self):
        """Create minimal demo showing complete structure including RUL"""
        self._reset_graph()

        # Use real NASA data for turbofan part
        turbofan_data = self._get_data_subset(max_engines=1, max_cycles=1)
        if len(turbofan_data) > 0:
            row_data = turbofan_data.iloc[0]
            t2_value = float(row_data['T2'])
            nf_value = float(row_data['Nf'])
            setting1_value = float(row_data['setting_1'])
        else:
            t2_value, nf_value, setting1_value = 520.5, 2388.0, 0.0025

        # Turbofan Engine
        engine_id = self._get_or_create_node_id("turbofan_engine", "Engine_1")

        # Operational Setting
        setting_controller_id = self._get_or_create_node_id("operational_setting", "Engine_1_setting_1_Controller")
        setting_value_id = self._get_or_create_node_id("setting_value",
                                                       f"Engine_1_setting_1_Cycle_1_Value_{setting1_value}")

        self.node_data[setting_controller_id].update({
            "asset_id": "Engine_1", "setting_type": "setting_1", "control_parameter": "setting_1"
        })
        self.node_data[setting_value_id].update({
            "value": setting1_value, "current_value": setting1_value, "setting_type": "setting_1",
            "cycle": 1, "asset_id": "Engine_1"
        })

        # Temperature Sensor
        temp_sensor_id = self._get_or_create_node_id("temperature_sensor", "Engine_1_T2_Sensor")
        temp_reading_id = self._get_or_create_node_id("measurement", f"Engine_1_T2_Reading_{t2_value}")

        self.node_data[temp_sensor_id].update({
            "asset_id": "Engine_1", "sensor_id": "T2", "sensor_type": "T2", "unit": "°R"
        })
        self.node_data[temp_reading_id].update({
            "value": t2_value, "current_value": t2_value, "sensor_id": "T2", "sensor_type": "T2",
            "cycle": 1, "asset_id": "Engine_1", "measurement_type": "individual"
        })

        # Speed Sensor
        speed_sensor_id = self._get_or_create_node_id("speed_sensor", "Engine_1_Nf_Sensor")
        speed_reading_id = self._get_or_create_node_id("measurement", f"Engine_1_Nf_Reading_{nf_value}")

        self.node_data[speed_sensor_id].update({
            "asset_id": "Engine_1", "sensor_id": "Nf", "sensor_type": "Nf", "unit": "rpm"
        })
        self.node_data[speed_reading_id].update({
            "value": nf_value, "current_value": nf_value, "sensor_id": "Nf", "sensor_type": "Nf",
            "cycle": 1, "asset_id": "Engine_1", "measurement_type": "individual"
        })

        # ✅ ADD RUL Node for concept demo
        if self.rul_data is not None:
            rul_value = int(self.rul_data[self.rul_data['unit_number'] == 1]['rul'].iloc[0])
            rul_id = self._get_or_create_node_id("rul_predictor", "Engine_1_RUL_Predictor")

            self.node_data[rul_id].update({
                "engine_id": "Engine_1",
                "asset_id": "Engine_1",
                "current_cycle": 1,
                "remaining_cycles": rul_value,
                "total_expected_cycles": 1 + rul_value,
                "rul_value": rul_value,
                "prediction_type": "ground_truth",
                "unit": "cycles"
            })

            self._add_edge(engine_id, rul_id, "has_rul_prediction")

        # Connect everything
        self._add_edge(engine_id, setting_controller_id, "has_operational_setting")
        self._add_edge(setting_controller_id, setting_value_id, "produces_setting_value")
        self._add_edge(engine_id, temp_sensor_id, "has_sensor")
        self._add_edge(engine_id, speed_sensor_id, "has_sensor")
        self._add_edge(temp_sensor_id, temp_reading_id, "produces_measurement")
        self._add_edge(speed_sensor_id, speed_reading_id, "produces_measurement")

        self._save_graph("./turbofan_graphs/turbofan_concept_demo_with_rul.json")


    def create_all_sensors_full_data_graphs(self, engine_counts: List[int] = [1, 5, 10, 20]):
        """
        Create additional graph datasets that include ALL sensors and ALL raw measurements
        for an increasing number of engines, with no downsampling or grouping.

        Args:
            engine_counts: List of engine counts to include (default [1,5,10,20])
        """
        if not hasattr(self, 'raw_data') or self.raw_data is None:
            raise ValueError("Raw NASA data not loaded. Expected self.raw_data to be a DataFrame.")

        for count in engine_counts:
            # Filter to N engines
            # Detect engine ID column
            engine_col = None
            for candidate in ['engine_id', 'unit', 'unit_number']:
                if candidate in self.raw_data.columns:
                    engine_col = candidate
                    break
            if engine_col is None:
                raise KeyError("No engine identifier column found in raw_data (expected engine_id/unit/unit_number)")

            selected_engines = self.raw_data[engine_col].unique()[:count]
            subset = self.raw_data[self.raw_data[engine_col].isin(selected_engines)]

            # Keep all sensors and all rows (no grouping/downsampling)
            graph_data = subset.copy()

            # Count sensor columns (excluding metadata columns)
            all_sensor_mappings = self._get_complete_sensor_mappings()
            sensor_columns = list(all_sensor_mappings.keys())
            setting_columns = ['setting_1', 'setting_2', 'setting_3']

            # Store or print stats
            print(f"[ALL-SENSORS-FULL] {count} engines: "
                  f"{graph_data[engine_col].nunique()} engines, "
                  f"{len(sensor_columns)} sensors, "
                  f"{len(setting_columns)} settings, "
                  f"{len(graph_data)} measurements")

            # Save dataset for later plotting
            setattr(self, f"graph_all_sensors_{count}eng", graph_data)

    def create_comprehensive_datasets(self):
        """Create comprehensive datasets with ALL sensors and settings for different scales"""

        print("Creating comprehensive datasets with ALL 21 sensors + 3 operational settings...")

        # Tiny dataset: 1 engine, few cycles, ALL sensors
        self._reset_graph()
        tiny_data = self._get_data_subset(max_engines=1, max_cycles=5, max_rows=20)
        all_sensors = list(self._get_complete_sensor_mappings().keys())
        self._create_graph_from_data(tiny_data, sensor_subset=all_sensors,
                                     include_settings=True, graph_name="Comprehensive Tiny Dataset")
        self._save_graph("./turbofan_graphs/turbofan_comprehensive_tiny.json")

        # Small comprehensive: 2 engines, ALL sensors
        self._reset_graph()
        small_data = self._get_data_subset(max_engines=2, max_cycles=15, max_rows=100)
        self._create_graph_from_data(small_data, sensor_subset=all_sensors,
                                     include_settings=True, graph_name="Comprehensive Small Dataset")
        self._save_graph("./turbofan_graphs/turbofan_comprehensive_small.json")

        # Medium comprehensive: 5 engines, ALL sensors
        self._reset_graph()
        medium_data = self._get_data_subset(max_engines=5, max_cycles=25, max_rows=500)
        self._create_graph_from_data(medium_data, sensor_subset=all_sensors,
                                     include_settings=True, graph_name="Comprehensive Medium Dataset")
        self._save_graph("./turbofan_graphs/turbofan_comprehensive_medium.json")

    def create_test_datasets(self):
        """Create small, fast datasets specifically for testing and development"""
        print("Creating test datasets optimized for development...")

        # Ultra-tiny: 1 engine, ALL sensors, RUL
        self._reset_graph()
        tiny_data = self._get_data_subset(max_engines=1)
        all_sensors = list(self._get_complete_sensor_mappings().keys())
        self._create_graph_from_data(tiny_data, sensor_subset=all_sensors,
                                     include_settings=True, include_rul=True,
                                     graph_name="Test Tiny Dataset")
        self._save_graph("./turbofan_graphs/turbofan_test_tiny.json")

        # Test pair: 2 engines, ALL sensors, RUL
        self._reset_graph()
        pair_data = self._get_data_subset(max_engines=2)
        self._create_graph_from_data(pair_data, sensor_subset=all_sensors,
                                     include_settings=True, include_rul=True,
                                     graph_name="Test Pair Dataset")
        self._save_graph("./turbofan_graphs/turbofan_test_pair.json")

    def generate_all_graphs(self):
        """Generate all demonstration graphs using real NASA data"""
        print("Generating NASA Turbofan graphs using real data...")
        print(f"Source data: {len(self.raw_data)} total measurements")

        # Get stats on available data
        all_sensor_mappings = self._get_complete_sensor_mappings()
        available_sensors = [col for col in all_sensor_mappings.keys() if col in self.raw_data.columns]

        print(f"Available sensors: {len(available_sensors)}/21")
        print(f"Available engines: {self.raw_data['unit_number'].nunique()}")
        print(f"Cycle range: {self.raw_data['time_cycles'].min()} to {self.raw_data['time_cycles'].max()}")

        # ✅ UPDATED: Test datasets first (fast for development)
        self.create_test_datasets()

        # Updated datasets with RUL
        self.create_single_engine_demo()
        self.create_multi_engine_demo()
        self.create_small_dataset()
        self.create_medium_dataset()
        self.create_concept_demo()

        # Original larger datasets
        self.create_large_dataset()
        self.create_full_dataset()

        # Comprehensive datasets
        self.create_comprehensive_datasets()
        self.create_all_sensors_full_data_graphs()

        print("\n✅ All graphs generated successfully using real NASA turbofan data!")
        print("📊 Generated graph types:")
        print("   - Test datasets (1-2 engines, ALL sensors, RUL) - Perfect for development")
        print("   - Single/Multi engine demos (with RUL)")
        print("   - Small/Medium datasets (full cycles + ALL sensors + RUL)")
        print("   - Large/Full datasets (progressive scaling)")
        print("   - Comprehensive datasets (ALL sensors for different scales)")


    def get_dataset_summary(self):
        """Print summary of all available sensors and settings"""
        print("\n=== NASA TURBOFAN DATASET SUMMARY ===")

        sensor_mappings = self._get_complete_sensor_mappings()
        setting_mappings = self._get_operational_setting_mappings()

        print(f"\n📡 SENSORS ({len(sensor_mappings)} total):")
        for sensor, (sensor_type, unit) in sensor_mappings.items():
            print(f"  {sensor:12} - {sensor_type:20} ({unit})")

        print(f"\n⚙️  OPERATIONAL SETTINGS ({len(setting_mappings)} total):")
        for setting, (setting_type, unit) in setting_mappings.items():
            print(f"  {setting:12} - {setting_type:20} ({unit})")

        if hasattr(self, 'raw_data') and len(self.raw_data) > 0:
            print(f"\n📈 DATA STATISTICS:")
            print(f"  Total measurements: {len(self.raw_data):,}")
            print(f"  Engines: {self.raw_data['unit_number'].nunique()}")
            print(f"  Cycles per engine: {self.raw_data.groupby('unit_number')['time_cycles'].count().describe()}")
            print(f"  Cycle range: {self.raw_data['time_cycles'].min()}-{self.raw_data['time_cycles'].max()}")

# Usage example
if __name__ == "__main__":
    # ✅ Pass both data and RUL files
    generator = TurbofanGraphGenerator("CMaps/train_FD001.txt", "CMaps/RUL_FD001.txt")
    generator.get_dataset_summary()
    generator.generate_all_graphs()