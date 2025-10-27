from rdflib import Graph, RDF, RDFS, OWL, XSD, Literal
from collections import defaultdict
from backend.core_graph_managers.no2_graphManager.graphManager import GraphManager
from backend.core_graph_managers.no3_OntologyManager.OntologyManager import OntologyManager
from rdflib.namespace import split_uri
from rdflib.namespace import Namespace

CORE = Namespace("file:./core_ontology.ttl#")


class OntologyBridge:
    def __init__(self, ontology_paths):
        # Parse the OWL files using rdflib
        self.raw_graph = Graph()
        for path in ontology_paths:
            self.raw_graph.parse(path, format="turtle")

        # Initialize collections to store ontology data
        self.classes = set()
        self.object_properties = []
        self.datatype_properties = []
        self.annotations = defaultdict(dict)

        # Create the graph manager
        self.graph_manager = GraphManager(preload=False)

        # Extract ontology elements
        self._extract_ontology()

    def _extract_ontology(self):
        """Extract all ontology elements from the RDF graph"""
        # Add classes as nodes
        for c in self.raw_graph.subjects(RDF.type, OWL.Class):
            class_name = self._shorten(c)
            self.classes.add(class_name)
            comment = self._get_literal(c, RDFS.comment) or ""

            # Add class as a node
            self.graph_manager.add_node(
                node_id=class_name,
                value=class_name,
                type="Class",
                hierarchy="OntologyClass",
                attributes={"comment": comment}
            )

        # Add XSD primitives as nodes
        xsd_primitives = ["string", "integer", "float", "boolean", "dateTime"]
        for ptype in xsd_primitives:
            ptype_id = f"xsd:{ptype}"
            self.graph_manager.add_node(
                node_id=ptype_id,
                value=ptype_id,
                type="Primitive",
                hierarchy="XSDType",
                attributes={"comment": f"XSD Primitive Type: {ptype}"}
            )

        # Extract object properties as edges
        for prop in self.raw_graph.subjects(RDF.type, OWL.ObjectProperty):
            name = self._shorten(prop)
            domain = self._shorten(self.raw_graph.value(prop, RDFS.domain))
            range_ = self._shorten(self.raw_graph.value(prop, RDFS.range))

            if domain and range_:
                # Make sure nodes exist
                if domain not in self.graph_manager.node_data:
                    self.graph_manager.add_node(
                        node_id=domain,
                        value=domain,
                        type="ImpliedClass",
                        hierarchy="OntologyClass",
                        attributes={}
                    )

                if range_ not in self.graph_manager.node_data:
                    self.graph_manager.add_node(
                        node_id=range_,
                        value=range_,
                        type="ImpliedClass",
                        hierarchy="OntologyClass",
                        attributes={}
                    )

                # Add the edge with a simple edge_type
                try:
                    # THIS IS THE KEY CHANGE: Just use edge_type as the main identifier
                    self.graph_manager.add_edge(
                        source=domain,
                        target=range_,
                        attributes={"edge_type": name}
                    )

                    # Store in our object properties list
                    self.object_properties.append((domain, name, range_))
                except ValueError:
                    # Edge already exists - get the existing edge
                    edge_key = (domain, range_)
                    if edge_key in self.graph_manager.edge_data:
                        # Instead of adding complicated attributes, handle duplicates more simply
                        edge_type = self.graph_manager.edge_data[edge_key]["edge_type"]
                        if isinstance(edge_type, str) and edge_type != name:
                            # If it's a different property, combine them with a separator
                            self.graph_manager.edge_data[edge_key]["edge_type"] = f"{edge_type}, {name}"
                except Exception as e:
                    print(f"Error adding object property {name}: {e}")

        # Extract datatype properties in a similar way
        for prop in self.raw_graph.subjects(RDF.type, OWL.DatatypeProperty):
            name = self._shorten(prop)
            domain = self._shorten(self.raw_graph.value(prop, RDFS.domain))
            range_uri = self.raw_graph.value(prop, RDFS.range)

            if domain and range_uri:
                try:
                    _, range_local = split_uri(range_uri)
                    if str(range_uri).startswith(str(XSD)):
                        range_name = f"xsd:{range_local}"
                    else:
                        range_name = range_local
                except Exception:
                    range_name = str(range_uri)

                # Make sure nodes exist
                if domain not in self.graph_manager.node_data:
                    self.graph_manager.add_node(
                        node_id=domain,
                        value=domain,
                        type="ImpliedClass",
                        hierarchy="OntologyClass",
                        attributes={}
                    )

                if range_name not in self.graph_manager.node_data:
                    self.graph_manager.add_node(
                        node_id=range_name,
                        value=range_name,
                        type="Primitive",
                        hierarchy="XSDType",
                        attributes={}
                    )

                # Add the edge with a simple edge_type
                try:
                    # ALSO SIMPLIFIED: Just use edge_type as the main identifier
                    self.graph_manager.add_edge(
                        source=domain,
                        target=range_name,
                        attributes={"edge_type": name}
                    )

                    # Store in our datatype properties list
                    self.datatype_properties.append((domain, name, range_name))
                except ValueError:
                    # Edge already exists - get the existing edge
                    edge_key = (domain, range_name)
                    if edge_key in self.graph_manager.edge_data:
                        # Instead of adding complicated attributes, handle duplicates more simply
                        edge_type = self.graph_manager.edge_data[edge_key]["edge_type"]
                        if isinstance(edge_type, str) and edge_type != name:
                            # If it's a different property, combine them with a separator
                            self.graph_manager.edge_data[edge_key]["edge_type"] = f"{edge_type}, {name}"
                except Exception as e:
                    print(f"Error adding datatype property {name}: {e}")

        # Extract annotations
        for s, p, o in self.raw_graph.triples((None, None, None)):
            subj = self._shorten(s)
            if str(p).startswith("file:./core_ontology.ttl#"):  # manually check core
                pred = self._shorten(p)
                self.annotations[subj][pred] = self._convert_literal(o)

                # Add annotation to node if it exists
                if subj in self.graph_manager.node_data:
                    if 'attributes' not in self.graph_manager.node_data[subj]:
                        self.graph_manager.node_data[subj]['attributes'] = {}
                    self.graph_manager.node_data[subj]['attributes'][pred] = self._convert_literal(o)

    # Helper methods_application (unchanged)
    def _get_literal(self, subject, pred):
        val = self.raw_graph.value(subject, pred)
        return str(val) if val else None

    def _convert_literal(self, lit):
        if isinstance(lit, Literal):
            return lit.toPython()
        return str(lit)

    def _shorten(self, uri):
        try:
            _, name = split_uri(uri)
            if str(uri).startswith(str(XSD)):
                return f"xsd:{name}"
            return name
        except Exception:
            return str(uri)

    # Public API methods_application (mostly unchanged)
    def get_all_classes(self):
        return sorted(self.classes)

    def get_all_properties(self):
        return self.object_properties + self.datatype_properties

    def get_annotations_for(self, class_name):
        return self.annotations.get(class_name, {})

    def get_subproperties_of(self, superprop_name):
        return [
            (subj, pred, obj)
            for (subj, pred, obj) in self.raw_graph.triples((None, RDFS.subPropertyOf, None))
            if self._shorten(obj) == superprop_name
        ]

    def get_ontology_graph(self):
        return self.graph_manager

    def visualize(self, output_path, highlighted_nodes=None, fuzzy_matches=None, methods=None):
        if not self.graph_manager:
            return None

        # Apply highlighting
        self._apply_highlighting(highlighted_nodes, fuzzy_matches, methods)

        # Render the visualization
        return self.graph_manager.render_pyvis(
            path=output_path,
            height="700px",
            width="100%",
            physics=True,
            spacing_factor=1.2
        )

    def _apply_highlighting(self, highlighted_nodes, fuzzy_matches, methods):
        # Add color to highlighted nodes
        if highlighted_nodes:
            for node_id in highlighted_nodes:
                if node_id in self.graph_manager.node_data:
                    self.graph_manager.node_data[node_id]['color'] = '#ff5733'  # Bright orange
                    self.graph_manager.node_data[node_id]['borderWidth'] = 3

                    # Also track in attributes
                    if 'attributes' not in self.graph_manager.node_data[node_id]:
                        self.graph_manager.node_data[node_id]['attributes'] = {}
                    self.graph_manager.node_data[node_id]['attributes']['highlight_type'] = 'matching'

        # Add color to fuzzy matches
        if fuzzy_matches:
            for loaded_class, _ in fuzzy_matches:
                if loaded_class in self.graph_manager.node_data:
                    self.graph_manager.node_data[loaded_class]['color'] = '#33a1ff'  # Light blue
                    self.graph_manager.node_data[loaded_class]['borderWidth'] = 3

                    # Also track in attributes
                    if 'attributes' not in self.graph_manager.node_data[loaded_class]:
                        self.graph_manager.node_data[loaded_class]['attributes'] = {}
                    self.graph_manager.node_data[loaded_class]['attributes']['highlight_type'] = 'fuzzy'

        # Add color to method nodes
        if methods:
            for node_id in methods:
                if node_id in self.graph_manager.node_data:
                    self.graph_manager.node_data[node_id]['color'] = '#33ff57'  # Light green
                    self.graph_manager.node_data[node_id]['borderWidth'] = 3

                    # Also track in attributes
                    if 'attributes' not in self.graph_manager.node_data[node_id]:
                        self.graph_manager.node_data[node_id]['attributes'] = {}
                    self.graph_manager.node_data[node_id]['attributes']['highlight_type'] = 'method'

    def to_ontology_manager(self):
        """Convert to OntologyManager instance"""
        # Create the OntologyManager using our graph
        ontology_mgr = OntologyManager(self.graph_manager)

        # Extract ontology patterns
        ontology_mgr.extract_ontology()

        return ontology_mgr

