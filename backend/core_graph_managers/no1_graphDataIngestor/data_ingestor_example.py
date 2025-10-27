from graphDataIngestion import *


paths = [
    "C:/Users/callu/Documents/Johnny Tomatoes.com",
    "C:/Users/callu/Documents/Jimmy Carrot.pdf",
    "C:/Users/callu/Documents/Meat_folder/Harry Ham.html"
]

data_frame = {
    'employee': ['Alice', 'Bob', 'Charlie', 'David'],
    'department': ['Sales', 'Sales', 'Engineering', 'Engineering'],
    'manager': ['Eve', 'Eve', 'Frank', 'Frank']
}

data_frame = pd.DataFrame.from_dict(data_frame)


metadata = pd.DataFrame([
    {'column': 'employee', 'node_type': 'Person', 'edge_type': 'works_in', 'direction': 'out', 'hierarchy_level': 1, 'connects_to': 'department'},
    {'column': 'department', 'node_type': 'Department', 'edge_type': 'managed_by', 'direction': 'out', 'hierarchy_level': 0, 'connects_to': 'manager'},
    {'column': 'manager', 'node_type': 'Person', 'edge_type': 'manages', 'direction': 'in', 'hierarchy_level': 2, 'connects_to': 'department'}
])

json_example = {
    "projects": [
        {
            "name": "Project A",
            "files": ["main.py", "utils.py"],
            "config": {
                "debug": True,
                "version": "1.0"
            },
            "team": [
                {"name": "Alice", "role": "Developer"},
                {"name": "Bob", "role": "Manager"}
            ]
        },
        {
            "name": "Project B",
            "files": ["main_2.py", "utils.py"],
            "config": {
                "debug": True,
                "version": "1.3"
            },
            "team": [
                {"name": "John", "role": "Developer"},
                {"name": "Dave", "role": "Manager"}
            ]
        }
    ]
}

filepath = '../Data/'
with open('../Data/station_quadrant_data_test.pkl', 'rb') as f:
    dataframe_pkl = pickle.load(f)

dataframe_pkl = dataframe_pkl[:10]

metadata_pkl = pd.DataFrame([
    {'column': 'start_datetime',              'node_type': 'timestamp',        'edge_type': 'at_timestamp',             'direction': 'in',  'hierarchy_level': -1, 'connects_to': 'dp_kpa'},
    {'column': 'dp_kpa',                      'node_type': 'Datapoint',        'edge_type': 'at_location',              'direction': 'out', 'hierarchy_level': 0,  'connects_to': 'location'},
    {'column': 'filter_change',               'node_type': 'event',            'edge_type': 'describes',                'direction': 'in',  'hierarchy_level': 1,  'connects_to': 'dp_kpa'},
    {'column': 'Station',                     'node_type': 'station',          'edge_type': 'in_station',               'direction': 'in',  'hierarchy_level': 2,  'connects_to': 'location'},
    {'column': 'Quadrant',                    'node_type': 'region',           'edge_type': 'in_region',                'direction': 'out',  'hierarchy_level': 3,  'connects_to': 'Station'},
    {'column': 'time_since_last_measurement', 'node_type': 'duration',         'edge_type': 'time_since',               'direction': 'in',  'hierarchy_level': 1,  'connects_to': 'dp_kpa'},
    {'column': 'cumulative_time',             'node_type': 'duration',         'edge_type': 'cumulative_time',          'direction': 'in',  'hierarchy_level': 1,  'connects_to': 'dp_kpa'},
    {'column': 'large_temporal_change',       'node_type': 'boolean_flag',     'edge_type': 'has_temporal_change',      'direction': 'in',  'hierarchy_level': 1,  'connects_to': 'dp_kpa'},
    {'column': 'pressure_change',             'node_type': 'float',            'edge_type': 'delta_pressure',           'direction': 'in',  'hierarchy_level': 1,  'connects_to': 'dp_kpa'},
    {'column': 'large_pressure_change',       'node_type': 'boolean_flag',     'edge_type': 'has_pressure_change',      'direction': 'in',  'hierarchy_level': 1,  'connects_to': 'dp_kpa'},
    {'column': 'month',                       'node_type': 'month',            'edge_type': 'in_month',                 'direction': 'out', 'hierarchy_level': 2,  'connects_to': 'start_datetime'},
    {'column': 'season',                      'node_type': 'season',           'edge_type': 'in_season',                'direction': 'in',  'hierarchy_level': 3,  'connects_to': 'month'},
    {'column': 'hour',                        'node_type': 'hour',             'edge_type': 'at_hour',                  'direction': 'out', 'hierarchy_level': 2,  'connects_to': 'start_datetime'},
    {'column': 'relative_time_of_day',        'node_type': 'time_category',    'edge_type': 'in_part_of_day',           'direction': 'in',  'hierarchy_level': 3,  'connects_to': 'hour'},
    {'column': 'day',                         'node_type': 'day',              'edge_type': 'on_day',                   'direction': 'out', 'hierarchy_level': 2,  'connects_to': 'start_datetime'},
    {'column': 'year',                        'node_type': 'year',             'edge_type': 'in_year',                  'direction': 'out', 'hierarchy_level': 2,  'connects_to': 'start_datetime'},
    {'column': 'stable_window',               'node_type': 'bool_flag',        'edge_type': 'has_stable_window',        'direction': 'in',  'hierarchy_level': 1,  'connects_to': 'dp_kpa'},
    {'column': 'stable_rolling',              'node_type': 'bool_flag',        'edge_type': 'has_stable_rolling',       'direction': 'in',  'hierarchy_level': 1,  'connects_to': 'dp_kpa'},
    {'column': 'smoothed_pressure',           'node_type': 'float',            'edge_type': 'smoothed_pressure_value',  'direction': 'in',  'hierarchy_level': 1,  'connects_to': 'dp_kpa'},
    {'column': 'location',                    'node_type': 'location',         'edge_type': 'at_location',              'direction': 'in',  'hierarchy_level': 1,  'connects_to': 'dp_kpa'},
    {'column': 'cycle',                       'node_type': 'process_cycle',    'edge_type': 'in_cycle',                 'direction': 'in',  'hierarchy_level': 3,  'connects_to': 'location'},
    {'column': 'category',                    'node_type': 'type_category',    'edge_type': 'categorizes',              'direction': 'out', 'hierarchy_level': 4,  'connects_to': 'cycle'},
])

filelistCTSB = ["E:/Training/East/A08_East_241005_025910_LTIT_POST_COND_INSP5.00/3-3040-05-0345-A08E-0.00-01-CTSB IVAAS Inspection.pdf",
"E:/Training/East/A08_East_241005_025910_LTIT_POST_COND_INSP5.00/3-3040-05-0345-A08E-0.00-01-CTSB IVAAS Inspection.pptx"
"E:/Training/East/B10_East_241008_031240_LTIT_POST_COND_INSP5.00/video/Raw.mp4",
"E:/Training/East/B10_East_241008_031340_LTIT_POST_COND_INSP5.00/3-3040-05-0345-B10E-5.00-01-CTSB IVAAS Inspection.pdf",
"E:/Training/West/K08_West_241107_124816_LTIT_POST_COND_INSP5.16/3-3040-05-0345-K08W-5.16-01-CTSB IVAAS Inspection.pdf",
"E:/Training/West/Q17_West_241125_230447_LTIT_POST_COND_INSP5.16/3-3040-05-0345-Q17W-5.16-02-CTSB IVAAS Inspection.pdf",
"E:/Training/West/Q17_West_241125_230447_LTIT_POST_COND_INSP5.16/3-3040-05-0345-Q17W-5.16-02-CTSB IVAAS Inspection.pptx",
"E:/Training/West/V07_West_241204_171534_LTIT_POST_COND_INSP5.16/config.json",
"E:/Training/West/V07_West_241204_171534_LTIT_POST_COND_INSP5.16/indications.json",
"E:/Training/West/V07_West_241204_171534_LTIT_POST_COND_INSP5.16/stitched_images/raw_stitched_0640.png",
"E:/Training/West/V07_West_241204_171534_LTIT_POST_COND_INSP5.16/video/Raw.mp4",
"E:/Validation/West/A11_West_241004_192312_LTIT_POST_COND_INSP0.00/3-3040-05-0345-A11W-0.00-03-CTSB IVAAS Inspection.pptx",
"E:/Validation/West/A11_West_241004_192312_LTIT_POST_COND_INSP0.00/3-3040-05-0345-A11W-0.00-04-CTSB IVAAS Inspection.pdf",
"E:/Validation/West/A11_West_241004_192312_LTIT_POST_COND_INSP0.00/3-3040-05-0345-A11W-0.00-04-CTSB IVAAS Inspection.pptx",
"E:/Validation/West/A11_West_241004_192312_LTIT_POST_COND_INSP0.00/3-3040-05-0345-A11W-0.00-05-CTSB IVAAS Inspection.pdf",
"E:/Validation/West/A11_West_241004_192312_LTIT_POST_COND_INSP0.00/3-3040-05-0345-A11W-0.00-05-CTSB IVAAS Inspection.pptx",
"E:/Validation/West/A11_West_241004_192312_LTIT_POST_COND_INSP0.00/3-3040-05-0345-A11W-0.00-06-CTSB IVAAS Inspection.pdf",
"E:/Validation/West/A11_West_241004_192312_LTIT_POST_COND_INSP0.00/3-3040-05-0345-A11W-0.00-06-CTSB IVAAS Inspection.pptx",
"E:/Validation/West/A11_West_241004_192312_LTIT_POST_COND_INSP0.00/config.json",
"E:/Validation/West/A11_West_241004_192312_LTIT_POST_COND_INSP0.00/indications.json",
"E:/Validation/West/W17_West_241207_120203_LTIT_POST_COND_INSP5.16/indications.json",
"E:/Validation/West/W17_West_241207_120203_LTIT_POST_COND_INSP5.16/stitched_images/raw_stitched_0640.png"]


ingester = GraphDataIngestion()

# ingester.reset()
# ingester.ingest_from_json(cosimmitry_json)
# ingester.visualize(method = "pyvis", filename ='../no2_graphManager/cosimmitry_json.html')
# ingester.dictionary_to_JSON(filepath='../no2_graphManager/cosimmitry_json.json')
#

# ingester.reset()
# ingester.ingest_from_dataframe(dataframe_pkl, metadata=metadata_pkl)
# ingester.dictionary_to_JSON(filepath ="../no2_graphManager/dataframe_pkl.json")
# ingester.visualize(method = "pyvis", filename ='../no2_graphManager/dataframe_pkl_graph.html')
#
# ingester.reset()
# ingester.ingest_from_dataframe(data_frame, metadata=metadata)
# print(ingester.dictionary_to_JSON(filepath ="../no2_graphManager/dataframe_graph.json"))
# ingester.visualize(method = "pyvis", filename ='dataframe_graph.html')
#
# ingester.reset()
# ingester.ingest_from_paths(paths)
# print(ingester.dictionary_to_JSON(filepath ="../no2_graphManager/filePath_graph.json"))
# ingester.visualize(method = "pyvis", filename ='filePath_graph.html')

# ingester.reset()
# ingester.ingest_from_json(json_example)
# print(ingester.dictionary_to_JSON(filepath ="../no2_graphManager/Json_graph.json"))
# ingester.visualize(method = "pyvis", filename ='Json_graph.html')

# ingester.reset()
# ingester.ingest_from_folder(filepath)
# print(ingester.dictionary_to_JSON(filepath ="../no2_graphManager/fromFolder_graph.json"))
# ingester.visualize(method = "pyvis", filename ='fromFolder_graph.html')