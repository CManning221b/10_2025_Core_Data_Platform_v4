import difflib
import re
from nltk.stem import PorterStemmer
from collections import defaultdict

class FuzzyOntologyMatcher:
    def __init__(self, loaded_classes, inferred_classes,
                 loaded_node_types=None, inferred_node_types=None,
                 loaded_graph_manager=None, inferred_graph_manager=None,
                 loaded_relationships=None, inferred_relationships=None):
        """
        Args:
            loaded_classes (list[str]): list of loaded class names
            inferred_classes (list[str]): list of inferred class names
            loaded_node_types (dict): optional, node_type -> node info for attributes/structure
            inferred_node_types (dict): optional
            loaded_graph_manager: optional graph manager for structural similarity
            inferred_graph_manager: optional graph manager for structural similarity
            loaded_relationships (list[tuple]): optional list of relationships
            inferred_relationships (list[tuple]): optional
        """
        self.loaded_classes = loaded_classes
        self.inferred_classes = inferred_classes
        self.ps = PorterStemmer()

        # Optional detailed info
        self.loaded_node_types = loaded_node_types or {}
        self.inferred_node_types = inferred_node_types or {}
        self.loaded_graph_manager = loaded_graph_manager
        self.inferred_graph_manager = inferred_graph_manager
        self.loaded_relationships = loaded_relationships or []
        self.inferred_relationships = inferred_relationships or []

    # -------------------------------------------------------
    # PUBLIC INTERFACE
    # -------------------------------------------------------
    def find_fuzzy_matches(self, threshold=0.5, top_n=3,
                           use_attributes=True, use_structure=True, use_relations=True):
        """
        Returns a dictionary of loaded_class -> list of (inferred_class, score, reason)
        """
        matches = defaultdict(list)

        for lc in self.loaded_classes:
            for ic in self.inferred_classes:
                score, reason = self._progressive_match(
                    lc, ic, threshold, use_attributes, use_structure, use_relations
                )
                if score >= threshold:
                    matches[lc].append((ic, round(score, 3), reason))

            matches[lc] = sorted(matches[lc], key=lambda x: x[1], reverse=True)[:top_n]

        return {k: v for k, v in matches.items() if v}

    # -------------------------------------------------------
    # PROGRESSIVE MATCHING LOGIC
    # -------------------------------------------------------
    def _progressive_match(self, a, b, threshold, use_attributes, use_structure, use_relations):
        # Normalize
        a_clean = re.sub(r'[^a-zA-Z]', '', a).lower()
        b_clean = re.sub(r'[^a-zA-Z]', '', b).lower()

        # --- 1. Exact match (case-insensitive)
        if a_clean == b_clean:
            return 1.0, "exact"

        # --- 2. Stem match
        if self.ps.stem(a_clean) == self.ps.stem(b_clean):
            return 0.9, "stem"

        # --- 3. Token overlap
        tokens_a = self._split_tokens(a)
        tokens_b = self._split_tokens(b)
        token_overlap = len(set(tokens_a) & set(tokens_b)) / max(len(set(tokens_a) | set(tokens_b)), 1)
        if token_overlap > 0.6:
            return 0.8, "token_overlap"

        # --- 4. Fuzzy string similarity
        fuzzy_ratio = difflib.SequenceMatcher(None, a_clean, b_clean).ratio()
        if fuzzy_ratio > 0.75:
            return fuzzy_ratio, "fuzzy_name"

        # --- 5. Attribute similarity (optional)
        if use_attributes and self.loaded_node_types and self.inferred_node_types:
            attr_sim = self._attribute_similarity(a, b)
            if attr_sim > 0.4:
                return attr_sim, "attribute"

        # --- 6. Structural similarity (optional)
        if use_structure and self.loaded_graph_manager and self.inferred_graph_manager:
            struct_sim = self._structural_similarity(a, b)
            if struct_sim > 0.4:
                return struct_sim, "structure"

        # --- 7. Relationship similarity (optional)
        if use_relations and self.loaded_relationships and self.inferred_relationships:
            rel_sim = self._relationship_similarity(a, b)
            if rel_sim > 0.3:
                return rel_sim, "relationship"

        return 0.0, None

    # -------------------------------------------------------
    # HELPERS
    # -------------------------------------------------------
    def _split_tokens(self, name):
        tokens = re.findall(r'[A-Z]?[a-z]+|[A-Z]+(?=[A-Z][a-z]|$)', name)
        return [t.lower() for t in tokens] or [name.lower()]

    def _attribute_similarity(self, lc, ic):
        props_a = set(self.loaded_node_types.get(lc, {}).get("properties", []))
        props_b = set(self.inferred_node_types.get(ic, {}).get("properties", []))
        if not props_a or not props_b:
            return 0.0
        return len(props_a & props_b) / len(props_a | props_b)

    def _structural_similarity(self, lc, ic):
        gm_a, gm_b = self.loaded_graph_manager, self.inferred_graph_manager
        nodes_a = self.loaded_node_types.get(lc, {}).get("nodes", [])
        nodes_b = self.inferred_node_types.get(ic, {}).get("nodes", [])
        if not nodes_a or not nodes_b:
            return 0.0

        def summary(gm, nodes):
            degrees, neighbor_types = [], set()
            for n in list(nodes)[:5]:
                neigh = gm.neighborhood_cache.get(n) or gm.lazy_neighborhood_cache(n)
                degrees.append(len(neigh['neighbors']))
                for nn in neigh['neighbors']:
                    neighbor_types.add(gm.node_data.get(nn, {}).get('type'))
            return (sum(degrees)/len(degrees) if degrees else 0), len(neighbor_types)

        deg_a, div_a = summary(gm_a, nodes_a)
        deg_b, div_b = summary(gm_b, nodes_b)

        deg_sim = 1 - (abs(deg_a - deg_b) / (max(deg_a, deg_b) + 1e-6))
        div_sim = 1 - (abs(div_a - div_b) / (max(div_a, div_b) + 1e-6))
        return (deg_sim + div_sim) / 2

    def _relationship_similarity(self, lc, ic):
        rel_a = {r for r in self.loaded_relationships if lc in r}
        rel_b = {r for r in self.inferred_relationships if ic in r}
        if not rel_a or not rel_b:
            return 0.0
        sig_a = {tuple(sorted((r[0], r[2]))) for r in rel_a}
        sig_b = {tuple(sorted((r[0], r[2]))) for r in rel_b}
        return len(sig_a & sig_b) / len(sig_a | sig_b)
