import json

from graphDataIngestor.graphDataIngestion import *
from no2_graphManager.graphManager import *


manager = GraphManager()
manager.load_from_json("filelistCTSB.json")
print(manager._get_summary_stats())
manager.visualize()

nkGraph = manager.build_networkit_graph()
nk.overview(nkGraph)

# Now with igraph
igGraph = manager.build_igraph_graph()

# Get basic graph properties similar to NetworkIt overview
print("\nNetwork Properties (igraph):")
print(f"nodes, edges\t\t\t{igGraph.vcount()}, {igGraph.ecount()}")
print(f"directed?\t\t\t{igGraph.is_directed()}")
print(f"weighted?\t\t\t{'weight' in igGraph.edge_attributes()}")
isolated_count = len([v for v in igGraph.vs if igGraph.degree(v.index, mode="in") == 0 and
                                                igGraph.degree(v.index, mode="out") == 0])
print(f"isolated nodes\t\t\t{isolated_count}")

# For degree statistics - specify to use total degree (in + out)
degrees = igGraph.degree(mode="all")  # "all" means both in and out
min_degree = min(degrees)
max_degree = max(degrees)
avg_degree = sum(degrees) / len(degrees)
print(f"min/max/avg degree\t\t{min_degree}, {max_degree}, {avg_degree:.6f}")
print(f"self-loops\t\t\t{len(igGraph.es.select(_is_loop=True))}")

# Calculate density
density = igGraph.density()
print(f"density\t\t\t\t{density:.6f}")

# Calculate clustering coefficient (transitivity)
clustering = igGraph.transitivity_undirected()
print(f"clustering coefficient\t\t{clustering:.6f}")

# Calculate assortativity
assortativity = igGraph.assortativity_degree()
print(f"degree assortativity\t\t{assortativity:.6f}")

# Calculate connected components
components = igGraph.components()
num_components = len(components)
largest_component_size = max(len(comp) for comp in components)
largest_component_percentage = 100 * largest_component_size / igGraph.vcount()
print(f"number of connected components\t{num_components}")
print(f"size of largest component\t{largest_component_size} ({largest_component_percentage:.2f} %)")


