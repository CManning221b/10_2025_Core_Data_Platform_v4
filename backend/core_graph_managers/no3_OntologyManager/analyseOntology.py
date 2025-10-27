import argparse
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import networkx as nx
from matplotlib.colors import LinearSegmentedColormap
from no2_graphManager.graphManager import GraphManager
from no3_OntologyManager.OntologyManager import OntologyManager




# Define numeric property visualization function
def visualize_numeric_property(prop_patterns, node_type, prop_name, output_dir):
    """Visualize patterns for numeric properties"""

    # Create output filename base
    filename_base = f"{node_type}_{prop_name}".replace(" ", "_").lower()

    # Check if we have range information
    if 'pattern_details' in prop_patterns and 'range' in prop_patterns['pattern_details']:
        range_info = prop_patterns['pattern_details']['range']

        # Create range visualization
        plt.figure(figsize=(10, 6))

        # Create boxplot-like visualization
        min_val = range_info['min']
        max_val = range_info['max']

        # Draw the boxplot manually
        bp_pos = [0]  # Position
        bp_width = 0.5

        # Box and whiskers
        plt.plot([min_val, max_val], [0, 0], 'b-', linewidth=2)  # Whiskers line

        # Show median/average as vertical line
        if 'median' in range_info:
            plt.axvline(x=range_info['median'], ymin=0.3, ymax=0.7,
                        color='orange', linewidth=2, label=f"Median: {range_info['median']}")

        if 'avg' in range_info:
            plt.axvline(x=range_info['avg'], ymin=0.3, ymax=0.7,
                        color='green', linewidth=2, label=f"Average: {range_info['avg']}")

        # Create labels showing min and max
        plt.text(min_val, 0.1, f"{min_val}", ha='center', va='bottom', fontweight='bold')
        plt.text(max_val, 0.1, f"{max_val}", ha='center', va='bottom', fontweight='bold')

        # Add a title and clean up the plot
        plt.title(f'Value Range for {node_type}.{prop_name}', fontsize=14)
        plt.xlabel('Value', fontsize=12)
        plt.yticks([])  # Hide y-axis

        if 'median' in range_info or 'avg' in range_info:
            plt.legend()

        plt.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{filename_base}_range.png'), dpi=300)
        plt.close()

        # If we have distribution data, create histogram
        if 'value_distribution' in prop_patterns and 'histogram' in prop_patterns['value_distribution']:
            histogram_data = prop_patterns['value_distribution']['histogram']

            if histogram_data and len(histogram_data) > 0:
                plt.figure(figsize=(10, 6))

                # Extract bin ranges and counts
                bin_ranges = [item.get('range', str(i)) for i, item in enumerate(histogram_data)]
                counts = [item.get('count', 0) for item in histogram_data]

                # Create histogram
                bars = plt.bar(bin_ranges, counts, color=sns.color_palette("viridis", len(bin_ranges)))

                # Add value labels
                for bar in bars:
                    height = bar.get_height()
                    if height > 0:
                        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                                 f'{int(height)}', ha='center', va='bottom', fontsize=10)

                plt.title(f'Value Distribution for {node_type}.{prop_name}', fontsize=14)
                plt.xlabel('Value Range', fontsize=12)
                plt.ylabel('Count', fontsize=12)
                plt.xticks(rotation=45, ha='right')
                plt.grid(axis='y', alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{filename_base}_histogram.png'), dpi=300)
                plt.close()


# Define boolean property visualization function
def visualize_boolean_property(prop_patterns, node_type, prop_name, output_dir):
    """Visualize patterns for boolean properties"""

    # Create output filename base
    filename_base = f"{node_type}_{prop_name}".replace(" ", "_").lower()

    # Check if we have distribution information
    if ('pattern_details' in prop_patterns and
            'distribution' in prop_patterns['pattern_details']):

        dist_info = prop_patterns['pattern_details']['distribution']

        # Create pie chart visualization
        plt.figure(figsize=(10, 6))

        # Get true/false counts
        true_count = dist_info.get('true_count', 0)
        false_count = dist_info.get('false_count', 0)

        # Only proceed if we have valid data
        if true_count > 0 or false_count > 0:
            labels = ['True', 'False']
            sizes = [true_count, false_count]
            colors = ['#1f77b4', '#ff7f0e']
            explode = (0.1, 0)  # Explode the True slice

            # Create pie chart
            plt.pie(sizes, explode=explode, labels=labels, colors=colors,
                    autopct='%1.1f%%', shadow=True, startangle=90)

            # Add total counts annotation
            plt.annotate(f'Total: {true_count + false_count}\nTrue: {true_count}\nFalse: {false_count}',
                         xy=(0, 0), xycoords='figure fraction',
                         xytext=(0.1, 0.1), textcoords='figure fraction',
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8))

            plt.axis('equal')  # Equal aspect ratio ensures the pie is circular
            plt.title(f'Boolean Distribution for {node_type}.{prop_name}', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{filename_base}_boolean_dist.png'), dpi=300)
            plt.close()





# Function to load from GraphManager and OntologyManager
def analyze_ontology(graph_manager, output_dir="ontology_analysis"):
    """
    Analyze an ontology using the OntologyManager and generate visualizations

    Args:
        graph_manager: GraphManager instance_graph with loaded graph data
        output_dir: Directory to save output visualizations
    """
    try:
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)

        # Create OntologyManager instance_graph
        ontology_manager = OntologyManager(graph_manager)

        # Extract the ontology
        print("Extracting ontology...")
        ontology_summary = ontology_manager.extract_ontology()
        print(f"Extracted {ontology_summary['node_types']} node types and {ontology_summary['edge_types']} edge types")

        # Get node and edge types
        node_types = ontology_manager.get_node_types()
        edge_types = ontology_manager.get_edge_types()

        # Generate visualizations
        print("Generating visualizations...")
        visualize_node_type_distribution(node_types, output_dir)
        visualize_edge_type_distribution(edge_types, output_dir)
        visualize_node_property_distribution(node_types, ontology_manager, output_dir)
        visualize_relationship_patterns(ontology_manager, output_dir)
        visualize_property_value_patterns(ontology_manager, node_types, output_dir)

        # These functions are not defined in your shared code, so commenting them out
        # Uncomment and implement them if needed
        # visualize_cardinality_distribution(ontology_manager, output_dir)
        # visualize_ontology_structure(ontology_manager, output_dir)
        # visualize_ontology_metrics(ontology_manager, output_dir)
        # generate_summary_report(ontology_manager, output_dir)

        print(f"Analysis complete. Visualizations saved to {output_dir} directory.")
        return True
    except Exception as e:
        print(f"Error analyzing ontology: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def visualize_node_type_distribution(node_types, output_dir):
    """Generate visualizations of node type distribution"""

    # Extract data for plotting
    type_names = list(node_types.keys())
    counts = [info['count'] for info in node_types.values()]

    # Sort by count (descending)
    sorted_indices = np.argsort(counts)[::-1]
    type_names = [type_names[i] for i in sorted_indices]
    counts = [counts[i] for i in sorted_indices]

    # 1. Bar chart of node types by count
    plt.figure(figsize=(12, 8))
    bars = plt.bar(type_names, counts, color=sns.color_palette("viridis", len(type_names)))

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                 f'{int(height)}', ha='center', va='bottom', fontweight='bold')

    plt.title('Distribution of Node Types', fontsize=16)
    plt.xlabel('Node Type', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'node_type_distribution.png'), dpi=300)
    plt.close()

    # 2. Pie chart of node type distribution
    plt.figure(figsize=(12, 9))

    # If there are many categories, combine small ones
    if len(type_names) > 10:
        # Keep top 9 categories, group the rest as "Other"
        other_sum = sum(counts[9:])
        main_types = type_names[:9]
        main_counts = counts[:9] + [other_sum]
        labels = main_types + ["Other"]

        # Create custom colors with "Other" in gray
        colors = sns.color_palette("viridis", 9) + [(0.7, 0.7, 0.7)]
    else:
        labels = type_names
        main_counts = counts
        colors = sns.color_palette("viridis", len(type_names))

    # Create pie chart with custom settings
    plt.pie(main_counts, labels=labels, autopct='%1.1f%%',
            startangle=90, counterclock=False,
            colors=colors, wedgeprops={'linewidth': 1, 'edgecolor': 'white'})

    plt.title('Proportion of Node Types', fontsize=16)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'node_type_pie_chart.png'), dpi=300)
    plt.close()

    # 3. Treemap of node types
    try:
        import squarify

        plt.figure(figsize=(12, 8))
        # Normalize sizes for better visualization
        sizes = np.array(counts)
        sizes = 100 * sizes / sizes.sum()

        # Create a custom colormap that varies by size
        norm = plt.Normalize(sizes.min(), sizes.max())
        colors = plt.cm.viridis(norm(sizes))

        # Create tree map
        squarify.plot(sizes=sizes, label=[f"{t}\n({c})" for t, c in zip(type_names, counts)],
                      color=colors, alpha=0.8, pad=True)

        plt.axis('off')
        plt.title('Node Types Treemap (size represents frequency)', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'node_type_treemap.png'), dpi=300)
        plt.close()
    except ImportError:
        print("squarify package not found, skipping treemap visualization")


def visualize_edge_type_distribution(edge_types, output_dir):
    """Generate visualizations of edge type distribution"""

    # Extract data for plotting
    type_names = list(edge_types.keys())
    counts = [info['count'] for info in edge_types.values()]

    # Sort by count (descending)
    sorted_indices = np.argsort(counts)[::-1]
    type_names = [type_names[i] for i in sorted_indices]
    counts = [counts[i] for i in sorted_indices]

    # 1. Bar chart of edge types by count
    plt.figure(figsize=(12, 8))
    bars = plt.bar(type_names, counts, color=sns.color_palette("mako", len(type_names)))

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                 f'{int(height)}', ha='center', va='bottom', fontweight='bold')

    plt.title('Distribution of Edge Types', fontsize=16)
    plt.xlabel('Edge Type', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'edge_type_distribution.png'), dpi=300)
    plt.close()

    # 2. Pie chart of edge type distribution
    plt.figure(figsize=(12, 9))

    # If there are many categories, combine small ones
    if len(type_names) > 10:
        # Keep top 9 categories, group the rest as "Other"
        other_sum = sum(counts[9:])
        main_types = type_names[:9]
        main_counts = counts[:9] + [other_sum]
        labels = main_types + ["Other"]

        # Create custom colors with "Other" in gray
        colors = sns.color_palette("mako", 9) + [(0.7, 0.7, 0.7)]
    else:
        labels = type_names
        main_counts = counts
        colors = sns.color_palette("mako", len(type_names))

    # Create pie chart with custom settings
    plt.pie(main_counts, labels=labels, autopct='%1.1f%%',
            startangle=90, counterclock=False,
            colors=colors, wedgeprops={'linewidth': 1, 'edgecolor': 'white'})

    plt.title('Proportion of Edge Types', fontsize=16)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'edge_type_pie_chart.png'), dpi=300)
    plt.close()


def visualize_node_property_distribution(node_types, ontology_manager, output_dir):
    """Generate visualizations of property distributions across node types"""

    # Aggregate property statistics
    property_counts = defaultdict(int)
    property_by_type = defaultdict(list)

    for node_type, info in node_types.items():
        properties = info['properties']
        for prop in properties:
            property_counts[prop] += 1
            property_by_type[node_type].append(prop)

    # Create a dataframe for easier visualization
    property_df = pd.DataFrame({
        'property': list(property_counts.keys()),
        'occurrence': list(property_counts.values())
    })
    property_df = property_df.sort_values('occurrence', ascending=False)

    # 1. Most common properties bar chart
    plt.figure(figsize=(12, 8))
    top_n = min(15, len(property_df))  # Show top 15 or all if fewer

    bars = plt.bar(property_df['property'][:top_n], property_df['occurrence'][:top_n],
                   color=sns.color_palette("viridis", top_n))

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                 f'{int(height)}', ha='center', va='bottom', fontweight='bold')

    plt.title('Most Common Node Properties', fontsize=16)
    plt.xlabel('Property Name', fontsize=14)
    plt.ylabel('Number of Node Types', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'common_node_properties.png'), dpi=300)
    plt.close()

    # 2. Property occurrence heatmap
    # Identify most common properties (top 20) and all node types
    top_properties = property_df['property'][:min(20, len(property_df))]

    # Build occurrence matrix
    node_type_list = list(node_types.keys())
    occurrence_matrix = np.zeros((len(node_type_list), len(top_properties)))

    for i, node_type in enumerate(node_type_list):
        for j, prop in enumerate(top_properties):
            if prop in property_by_type[node_type]:
                occurrence_matrix[i, j] = 1

    # Create heatmap
    plt.figure(figsize=(14, 10))
    ax = sns.heatmap(occurrence_matrix, cmap='viridis', cbar=False,
                     yticklabels=node_type_list, xticklabels=top_properties,
                     linewidths=0.5)

    # Customize heatmap
    plt.title('Property Occurrence by Node Type', fontsize=16)
    plt.ylabel('Node Type', fontsize=14)
    plt.xlabel('Property Name', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'property_occurrence_heatmap.png'), dpi=300)
    plt.close()

    # 3. Property count distribution
    # Count how many properties each node type has
    property_counts_by_type = {node_type: len(props) for node_type, props in property_by_type.items()}
    sorted_types = sorted(property_counts_by_type.items(), key=lambda x: x[1], reverse=True)

    plt.figure(figsize=(12, 8))
    types, counts = zip(*sorted_types)

    bars = plt.bar(types, counts, color=sns.color_palette("viridis", len(types)))

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                 f'{int(height)}', ha='center', va='bottom', fontweight='bold')

    plt.title('Number of Properties by Node Type', fontsize=16)
    plt.xlabel('Node Type', fontsize=14)
    plt.ylabel('Number of Properties', fontsize=14)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'property_count_by_type.png'), dpi=300)
    plt.close()

    # 4. Property type distribution (data types)
    try:
        # Collect property type information
        property_types = defaultdict(int)
        total_properties = 0

        for node_type, info in node_types.items():
            if hasattr(ontology_manager, 'node_property_stats'):
                if node_type in ontology_manager.node_property_stats:
                    for prop_name, stats in ontology_manager.node_property_stats[node_type].items():
                        primary_type = stats.get('primary_type', 'unknown')
                        property_types[primary_type] += 1
                        total_properties += 1

        if total_properties > 0:
            # Convert to percentages
            property_types = {k: (v / total_properties) * 100 for k, v in property_types.items()}

            # Sort by percentage
            sorted_types = sorted(property_types.items(), key=lambda x: x[1], reverse=True)
            type_names, percentages = zip(*sorted_types)

            # Create pie chart
            plt.figure(figsize=(10, 8))
            plt.pie(percentages, labels=type_names, autopct='%1.1f%%',
                    startangle=90, colors=sns.color_palette("tab10", len(type_names)),
                    wedgeprops={'linewidth': 1, 'edgecolor': 'white'})

            plt.title('Distribution of Property Data Types', fontsize=16)
            plt.axis('equal')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'property_type_distribution.png'), dpi=300)
            plt.close()
    except Exception as e:
        print(f"Error generating property type distribution: {str(e)}")


def visualize_relationship_patterns(ontology_manager, output_dir):
    """Generate visualizations of relationship patterns in the ontology"""

    try:
        # Extract relationship patterns
        relationships = ontology_manager.extract_relationship_patterns()

        # 1. Common relationship patterns bar chart
        if 'common_patterns' in relationships and relationships['common_patterns']:
            common_patterns = relationships['common_patterns']

            # Prepare data
            pattern_labels = [f"{p['source_type']} → {p['edge_type']} → {p['target_type']}"
                              for p in common_patterns]
            pattern_counts = [p['count'] for p in common_patterns]

            # Limit to top 15 patterns for readability
            if len(pattern_labels) > 15:
                pattern_labels = pattern_labels[:15]
                pattern_counts = pattern_counts[:15]

            # Create bar chart
            plt.figure(figsize=(14, 8))
            bars = plt.bar(pattern_labels, pattern_counts,
                           color=sns.color_palette("mako", len(pattern_labels)))

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                         f'{int(height)}', ha='center', va='bottom', fontweight='bold')

            plt.title('Most Common Relationship Patterns', fontsize=16)
            plt.xlabel('Relationship Pattern', fontsize=14)
            plt.ylabel('Count', fontsize=14)
            plt.xticks(rotation=45, ha='right', fontsize=12)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'common_relationship_patterns.png'), dpi=300)
            plt.close()

        # 2. Edge type usage in relationships
        if 'by_edge_type' in relationships:
            edge_types = relationships['by_edge_type']

            # Count source and target types for each edge type
            edge_data = []
            for edge_type, info in edge_types.items():
                source_count = len(info['source_types']) if isinstance(info['source_types'], list) else 0
                target_count = len(info['target_types']) if isinstance(info['target_types'], list) else 0
                pattern_count = len(info['patterns']) if 'patterns' in info else 0

                edge_data.append({
                    'edge_type': edge_type,
                    'source_types': source_count,
                    'target_types': target_count,
                    'patterns': pattern_count
                })

            # Convert to DataFrame and sort
            edge_df = pd.DataFrame(edge_data)
            edge_df = edge_df.sort_values('patterns', ascending=False)

            # Take top 15 edge types
            top_edge_df = edge_df.head(15)

            # Create stacked bar chart
            plt.figure(figsize=(14, 8))

            # Create the stacked bars
            bar_width = 0.5
            r = np.arange(len(top_edge_df))

            p1 = plt.bar(r, top_edge_df['source_types'], bar_width, color='#5975a4', label='Source Types')
            p2 = plt.bar(r, top_edge_df['target_types'], bar_width,
                         bottom=top_edge_df['source_types'], color='#5f9e6e', label='Target Types')

            # Add labels and titles
            plt.title('Edge Type Connectivity', fontsize=16)
            plt.xlabel('Edge Type', fontsize=14)
            plt.ylabel('Number of Connected Node Types', fontsize=14)
            plt.xticks(r, top_edge_df['edge_type'], rotation=45, ha='right', fontsize=12)
            plt.legend(loc='upper right')

            # Add value annotations for patterns
            for i, pattern_count in enumerate(top_edge_df['patterns']):
                plt.annotate(f'Patterns: {pattern_count}',
                             xy=(i, top_edge_df['source_types'].iloc[i] + top_edge_df['target_types'].iloc[i] + 0.5),
                             ha='center', va='bottom', fontsize=10, fontweight='bold')

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'edge_type_connectivity.png'), dpi=300)
            plt.close()

        # 3. Node type connectivity visualization
        if 'by_source_type' in relationships and 'by_target_type' in relationships:
            source_types = relationships['by_source_type']
            target_types = relationships['by_target_type']

            # Collect data on node type connectivity
            node_types = set()
            for s_type in source_types:
                node_types.add(s_type)
            for t_type in target_types:
                node_types.add(t_type)

            node_types = list(node_types)

            # Count outgoing and incoming connections
            conn_data = []
            for node_type in node_types:
                outgoing = len(source_types[node_type]['outgoing_edge_types']) if node_type in source_types else 0
                incoming = len(target_types[node_type]['incoming_edge_types']) if node_type in target_types else 0
                total = outgoing + incoming

                conn_data.append({
                    'node_type': node_type,
                    'outgoing': outgoing,
                    'incoming': incoming,
                    'total': total
                })

            # Convert to DataFrame and sort
            conn_df = pd.DataFrame(conn_data)
            conn_df = conn_df.sort_values('total', ascending=False)

            # Take top 15 node types
            top_conn_df = conn_df.head(15)

            # Create stacked bar chart
            plt.figure(figsize=(14, 8))

            bar_width = 0.5
            r = np.arange(len(top_conn_df))

            p1 = plt.bar(r, top_conn_df['outgoing'], bar_width, color='#E24A33', label='Outgoing Connections')
            p2 = plt.bar(r, top_conn_df['incoming'], bar_width,
                         bottom=top_conn_df['outgoing'], color='#348ABD', label='Incoming Connections')

            # Add labels and titles
            plt.title('Node Type Connectivity', fontsize=16)
            plt.xlabel('Node Type', fontsize=14)
            plt.ylabel('Number of Connection Types', fontsize=14)
            plt.xticks(r, top_conn_df['node_type'], rotation=45, ha='right', fontsize=12)
            plt.legend(loc='upper right')

            # Add total value annotations
            for i, total in enumerate(top_conn_df['total']):
                plt.annotate(f'Total: {total}',
                             xy=(i, top_conn_df['outgoing'].iloc[i] + top_conn_df['incoming'].iloc[i] + 0.2),
                             ha='center', va='bottom', fontsize=10, fontweight='bold')

            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'node_type_connectivity.png'), dpi=300)
            plt.close()

    except Exception as e:
        print(f"Error visualizing relationship patterns: {str(e)}")
        import traceback
        traceback.print_exc()


def visualize_property_value_patterns(ontology_manager, node_types, output_dir):
    """Generate visualizations of property value patterns"""

    try:
        # Select a few node types to analyze (top 3 by frequency)
        sorted_types = sorted(node_types.items(), key=lambda x: x[1]['count'], reverse=True)
        top_node_types = [t[0] for t in sorted_types[:min(3, len(sorted_types))]]

        for node_type in top_node_types:
            # Select a property to analyze
            properties = node_types[node_type]['properties']
            if not properties:
                continue

            # Try to find an interesting property (non-id, non-type)
            selected_prop = next((p for p in properties if p not in ('id', 'type', '_id')), None)
            if not selected_prop and properties:
                selected_prop = properties[0]

            if not selected_prop:
                continue

            try:
                # Get property pattern information
                prop_patterns = ontology_manager.discover_property_patterns(node_type, selected_prop)

                # Skip if missing key information
                if not prop_patterns or 'type_info' not in prop_patterns:
                    continue

                # Create property pattern visualization based on type
                prop_type = prop_patterns['type_info'].get('primary_type', 'unknown')

                # Handle different property types differently
                if prop_type in ('str', 'string'):
                    # Visualize string property
                    visualize_string_property(prop_patterns, node_type, selected_prop, output_dir)
                elif prop_type in ('int', 'float', 'number', 'decimal'):
                    # Visualize numeric property
                    # If you have this function defined, uncomment this line
                    # visualize_numeric_property(prop_patterns, node_type, selected_prop, output_dir)
                    pass
                elif prop_type == 'bool':
                    # Visualize boolean property
                    # If you have this function defined, uncomment this line
                    # visualize_boolean_property(prop_patterns, node_type, selected_prop, output_dir)
                    pass

            except Exception as e:
                print(f"Error analyzing property {node_type}.{selected_prop}: {str(e)}")
                continue

    except Exception as e:
        print(f"Error in property value pattern visualization: {str(e)}")
        import traceback
        traceback.print_exc()


def visualize_string_property(prop_patterns, node_type, prop_name, output_dir):
    """Visualize patterns for string properties"""

    # Create output filename base
    filename_base = f"{node_type}_{prop_name}".replace(" ", "_").lower()

    # 1. String length distribution if available
    if 'pattern_details' in prop_patterns and 'length' in prop_patterns['pattern_details']:
        length_info = prop_patterns['pattern_details']['length']

        # Only create visualization if we have meaningful min/max difference
        if length_info['max'] > length_info['min']:
            plt.figure(figsize=(10, 6))

            # Create histogram-like visualization
            bins = range(int(length_info['min']), int(length_info['max']) + 2)
            plt.hist([length_info['min'], length_info['max']], bins=bins,
                     alpha=0.7, color='skyblue', edgecolor='black')

            # Add vertical line for average
            if 'avg' in length_info:
                plt.axvline(x=length_info['avg'], color='red', linestyle='--',
                            label=f"Avg Length: {length_info['avg']:.1f}")
                plt.legend()

            plt.title(f'Length Distribution for {node_type}.{prop_name}', fontsize=14)
            plt.xlabel('String Length', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{filename_base}_length_dist.png'), dpi=300)
            plt.close()

    # 2. Pattern type visualization
    pattern_type = prop_patterns['pattern_details'].get('pattern_type', 'unknown')

    # If it's an enumeration, visualize the values
    if pattern_type == 'enumeration' and 'allowed_values' in prop_patterns['pattern_details']:
        values = prop_patterns['pattern_details']['allowed_values']

        if values and len(values) <= 15:  # Only for reasonable number of values
            plt.figure(figsize=(10, 6))

            # Count occurrences if available, otherwise use equal weights
            if isinstance(values, dict):
                labels = list(values.keys())
                counts = list(values.values())
            else:
                labels = values
                counts = [1] * len(values)

            # Create bar chart
            bars = plt.bar(labels, counts, color=sns.color_palette("viridis", len(labels)))

            plt.title(f'Enumeration Values for {node_type}.{prop_name}', fontsize=14)
            plt.xlabel('Value', fontsize=12)
            plt.ylabel('Occurrence', fontsize=12)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{filename_base}_enum_values.png'), dpi=300)
            plt.close()

    # 3. Value patterns visualization
    if 'advanced_patterns' in prop_patterns:
        adv_patterns = prop_patterns['advanced_patterns']

        # Create a summary figure
        plt.figure(figsize=(10, 6))
        pattern_text = []

        if 'regex' in adv_patterns:
            pattern_text.append(f"Regex Pattern: {adv_patterns['regex']}")

        if 'common_prefix' in adv_patterns:
            prefix = adv_patterns['common_prefix']['prefix']
            pct = adv_patterns['common_prefix']['percentage']
            pattern_text.append(f"Common Prefix: '{prefix}' ({pct:.1f}%)")

        if 'common_suffix' in adv_patterns:
            suffix = adv_patterns['common_suffix']['suffix']
            pct = adv_patterns['common_suffix']['percentage']
            pattern_text.append(f"Common Suffix: '{suffix}' ({pct:.1f}%)")

        if 'capitalization' in adv_patterns:
            cap_pattern = adv_patterns['capitalization']['pattern']
            pct = adv_patterns['capitalization']['percentage']
            pattern_text.append(f"Capitalization: {cap_pattern} ({pct:.1f}%)")

        if 'character_set' in adv_patterns:
            char_set = adv_patterns['character_set']['pattern']
            pct = adv_patterns['character_set']['percentage']
            pattern_text.append(f"Character Set: {char_set} ({pct:.1f}%)")

        if pattern_text:
            # Create text-based visualization
            plt.axis('off')
            plt.text(0.5, 0.5, "\n".join(pattern_text), ha='center', va='center',
                     fontsize=14, wrap=True)
            plt.title(f'Advanced Patterns for {node_type}.{prop_name}', fontsize=16)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{filename_base}_adv_patterns.png'), dpi=300)
            plt.close()


# Main entry point when script is run directly
if __name__ == "__main__":
    # Create output directory
    os.makedirs("ontology_analysis", exist_ok=True)

    try:
        # Load your graph data - update this path to your actual JSON file
        print("Loading graph data...")
        file_path = "../no2_graphManager/filelistCTSB.json"

        manager = GraphManager()
        manager.load_from_json(file_path)
        print(f"Loaded graph with {len(manager.node_data)} nodes and {len(manager.edge_data)} edges")

        # Analyze ontology and generate visualizations
        print("Analyzing ontology...")
        analyze_ontology(manager, output_dir="ontology_analysis")
        print("Analysis complete!")

    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback

        traceback.print_exc()