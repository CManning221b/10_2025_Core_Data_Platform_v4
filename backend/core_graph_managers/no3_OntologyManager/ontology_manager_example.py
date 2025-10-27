
# Initialize the ontology manager
import os
import sys

# Simplified imports
from no2_graphManager.graphManager import GraphManager
from no3_OntologyManager.OntologyManager import *


# Create output directory
os.makedirs("test_output", exist_ok=True)

# Load the graph
print(f"\n{'=' * 20} {"LOADING GRAPH DATA"} {'=' * 20}")

manager = GraphManager()

# Try different possible paths for the JSON file
potential_paths = [
    "../no2_graphManager/dataframe_pkl.json",
    "../no2_graphManager/dataframe_graph.json",
    "../no2_graphManager/filePath_graph.json",
    "../no2_graphManager/fromFolder_graph.json",
    "../no2_graphManager/Json_graph.json",
    "../no2_graphManager/filelistCTSB.json",
]

manager.load_from_json(potential_paths[5])
manager.visualize(method="pyvis", path="./test_output/graph.html")

# Initialize the ontology manager with the graph
print(f"\n{'=' * 20} EXTRACTING ONTOLOGY {'=' * 20}")
ontology_manager = OntologyManager(manager)

# Extract the ontology
ontology_summary = ontology_manager.extract_ontology()
print("Ontology extraction complete:")
print(f"- Node types: {ontology_summary['node_types']}")
print(f"- Edge types: {ontology_summary['edge_types']}")
print(f"- Relationship patterns: {ontology_summary['relationship_patterns']}")

# Get node and edge types
print(f"\n{'=' * 20} NODE TYPES {'=' * 20}")
node_types = ontology_manager.get_node_types()
for type_name, type_info in node_types.items():
    print(f"Node type: {type_name}")
    print(f"  Count: {type_info['count']}")
    print(f"  Coverage: {type_info['coverage']:.2%}")
    print(f"  Properties: {', '.join(type_info['properties'])}")
    print()

print(f"\n{'=' * 20} EDGE TYPES {'=' * 20}")
edge_types = ontology_manager.get_edge_types()
for type_name, type_info in edge_types.items():
    print(f"Edge type: {type_name}")
    print(f"  Count: {type_info['count']}")
    print(f"  Coverage: {type_info['coverage']:.2%}")
    print(f"  Properties: {', '.join(type_info['properties'])}")
    print()

# Analyze a specific node type (pick the first one for the example)
if node_types:
    first_node_type = next(iter(node_types.keys()))
    print(f"\n{'=' * 20} ANALYZING NODE TYPE: {first_node_type} {'=' * 20}")
    node_analysis = ontology_manager.analyze_node_type(first_node_type)
    print(f"Properties: {len(node_analysis['properties'])}")
    print(f"Relationships:")
    print(f"  Outgoing: {len(node_analysis['relationships']['outgoing'])}")
    print(f"  Incoming: {len(node_analysis['relationships']['incoming'])}")

# Analyze a specific edge type (pick the first one for the example)
if edge_types:
    first_edge_type = next(iter(edge_types.keys()))
    print(f"\n{'=' * 20} ANALYZING EDGE TYPE: {first_edge_type} {'=' * 20}")
    edge_analysis = ontology_manager.analyze_edge_type(first_edge_type)
    print(f"Properties: {len(edge_analysis['properties'])}")
    print(f"Connects:")
    print(f"  Source types: {list(edge_analysis['connectivity']['source_types'].keys())}")
    print(f"  Target types: {list(edge_analysis['connectivity']['target_types'].keys())}")

# Extract relationship patterns
print(f"\n{'=' * 20} RELATIONSHIP PATTERNS {'=' * 20}")
relationships = ontology_manager.extract_relationship_patterns()
print(f"Total patterns: {relationships['counts']['total_relationships']}")
print("Common patterns:")
for pattern in relationships.get('common_patterns', [])[:5]:  # Show top 5
    print(f"  {pattern['source_type']} → {pattern['edge_type']} → {pattern['target_type']}: {pattern['count']} instances")

# After the relationship patterns section, add this new section:
print(f"\n{'=' * 20} RELATIONSHIP CARDINALITY {'=' * 20}")

# Get a few relationship patterns to analyze
if relationships and 'common_patterns' in relationships:
    # Take the top 3 most common relationships for analysis
    for pattern in relationships['common_patterns'][:3]:
        src_type = pattern['source_type']
        edge_type = pattern['edge_type']
        tgt_type = pattern['target_type']

        print(f"\nAnalyzing cardinality for: {src_type} → {edge_type} → {tgt_type}")

        try:
            cardinality_info = ontology_manager.analyze_relationship_cardinality(src_type, edge_type, tgt_type)

            # Print basic cardinality information
            card_type = cardinality_info['cardinality']['type']
            print(f"  Cardinality type: {card_type}")
            print(f"  Description: {cardinality_info['cardinality']['description']}")

            # Print outgoing distribution (source → targets)
            out_stats = cardinality_info['cardinality']['outgoing']
            print(f"  Outgoing connections (source → targets):")
            print(f"    Min: {out_stats['min']}, Max: {out_stats['max']}, Median: {out_stats['median']:.2f}")
            print(f"    Sources with multiple targets: {out_stats['sources_with_multiple_targets']}")

            # Print incoming distribution (target ← sources)
            in_stats = cardinality_info['cardinality']['incoming']
            print(f"  Incoming connections (target ← sources):")
            print(f"    Min: {in_stats['min']}, Max: {in_stats['max']}, Median: {in_stats['median']:.2f}")
            print(f"    Targets with multiple sources: {in_stats['targets_with_multiple_sources']}")

            # Print examples if available
            if 'examples' in cardinality_info:
                print("  Examples:")
                for example_type, example_data in cardinality_info['examples'].items():
                    if example_type == 'one_to_one':
                        print(f"    One-to-one: {example_data['source_value']} → {example_data['target_value']}")
                    elif example_type == 'many_sources_one_target':
                        print(
                            f"    Many-to-one: {example_data['total_sources']} sources → {example_data['target_value']}")
                    elif example_type == 'one_source_many_targets':
                        print(
                            f"    One-to-many: {example_data['source_value']} → {example_data['total_targets']} targets")

            # Print constraint metrics if available
            if 'constraint_metrics' in cardinality_info['cardinality']:
                metrics = cardinality_info['cardinality']['constraint_metrics']
                print("  Constraint metrics:")
                print(f"    Source coverage: {metrics['source_coverage']:.2f}%")
                print(f"    Target coverage: {metrics['target_coverage']:.2f}%")
                print(f"    Edge density: {metrics['edge_density']:.2f}%")

        except Exception as e:
            print(f"  Error analyzing cardinality: {str(e)}")

# Add this section after the cardinality section
print(f"\n{'=' * 20} PROPERTY PATTERN ANALYSIS {'=' * 20}")

# Take the first node type and its first property for demonstration
if node_types:
    first_node_type = next(iter(node_types.keys()))
    if node_types[first_node_type]['properties']:
        first_property = node_types[first_node_type]['properties'][0]

        print(f"Analyzing patterns for {first_node_type}.{first_property}")

        try:
            property_patterns = ontology_manager.discover_property_patterns(first_node_type, first_property)

            # Print basic information
            print(f"  Property type: {property_patterns['type_info']['primary_type']}")
            print(
                f"  Presence: {property_patterns['presence']['count']} instances ({property_patterns['presence']['percentage']:.2f}%)")
            print(
                f"  Unique values: {property_patterns['unique_values']['count']} ({property_patterns['unique_values']['percentage']:.2f}%)")

            # Print pattern details
            if 'pattern_details' in property_patterns:
                print("  Pattern details:")
                for pattern_key, pattern_value in property_patterns['pattern_details'].items():
                    if isinstance(pattern_value, dict):
                        print(f"    {pattern_key}: {pattern_value}")
                    else:
                        print(f"    {pattern_key}: {pattern_value}")

            # Print regex pattern if available
            if 'advanced_patterns' in property_patterns and 'regex' in property_patterns['advanced_patterns']:
                print(f"  Regex pattern: {property_patterns['advanced_patterns']['regex']}")

            # Print value samples
            if 'value_samples' in property_patterns:
                sample_values = property_patterns['value_samples']
                print(f"  Sample values: {sample_values[:5]}")

            # Print validation suggestions
            if 'validation_suggestions' in property_patterns:
                print("  Validation suggestions:")
                for suggestion in property_patterns['validation_suggestions']:
                    print(f"    - {suggestion['description']}")

        except Exception as e:
            print(f"  Error analyzing property patterns: {str(e)}")

# Export ontology to different formats
print(f"\n{'=' * 20} EXPORTING ONTOLOGY {'=' * 20}")
json_path = os.path.join("test_output", "ontology_schema.json")
owl_path = os.path.join("test_output", "ontology.owl")

schema_path = ontology_manager.export_to_json_schema(json_path)
owl_path = ontology_manager.export_to_owl(owl_path)

print(f"JSON Schema exported to: {schema_path}")
print(f"OWL Ontology exported to: {owl_path}")

# Generate validation rules
print(f"\n{'=' * 20} VALIDATION RULES {'=' * 20}")
validation_rules = ontology_manager.generate_validation_rules()
print(f"Generated {len(validation_rules['node_type_rules'])} node type rules")
print(f"Generated {len(validation_rules['edge_type_rules'])} edge type rules")
print(f"Generated {len(validation_rules['relationship_rules'])} relationship rules")

# Get ontology summary
print(f"\n{'=' * 20} ONTOLOGY SUMMARY {'=' * 20}")
summary = ontology_manager.get_ontology_summary()
print(summary['overview']['description'])

# Find inconsistencies
print(f"\n{'=' * 20} FINDING INCONSISTENCIES {'=' * 20}")
inconsistencies = ontology_manager.find_inconsistencies()
print(inconsistencies['summary'])

# Suggest improvements
print(f"\n{'=' * 20} SUGGESTED IMPROVEMENTS {'=' * 20}")
suggestions = ontology_manager.suggest_ontology_improvements()
print(suggestions['summary'])
print("\nTop improvements:")
for improvement in suggestions['improvements'][:5]:  # Show top 5
    print(f"- {improvement['description']}")

# Update the visualization section to include your new function
print(f"\n{'=' * 20} VISUALISE ONTOLOGY {'=' * 20}")
ont_graph = ontology_manager.export_to_graph_manager()
ont_graph.visualize(method="pyvis", path="test_output/ontology_manager.html")

# Add this line to use your new function (assuming it's been added to the no3_OntologyManager class)
try:
    ont_graph_v2 = ontology_manager.export_ontology_schema_graph()
    ont_graph_v2.visualize(method="pyvis", path="test_output/ontology_manager_v2.html")
    print("Enriched ontology graph created and visualized.")
except AttributeError:
    print("Note: export_ontology_enriched_graph method not found in no3_OntologyManager class.")

print("\nTest completed successfully!")

#