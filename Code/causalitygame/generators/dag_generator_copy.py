# Math
import numpy as np

# Graph
import networkx as nx

# DAG
from causalitygame.scm.dag import DAG

# Abstract
from typing import Set, Tuple
from .base import AbstractSCMGenerator


class DAGGenerator(AbstractSCMGenerator):
    """
    A class to generate Directed Acyclic Graphs (DAGs) with configurable structure.

    DAGs are composed of:
    - Roots: Nodes with no incoming edges
    - Leaves: Nodes with no outgoing edges
    - Intermediates: Nodes with both incoming and outgoing edges

    The class supports constraints on:
    - Path length between roots and leaves
    - Edge density
    - Node degree (in/out)
    - Random reproducibility via a NumPy Generator

    Output is a DAG object wrapping a NetworkX graph.
    """

    def __init__(
        self,
        num_nodes: int,
        num_roots: int,
        num_leaves: int,
        edge_density: float,
        max_in_degree: int,
        max_out_degree: int,
        min_path_length: int,
        max_path_length: int,
        random_state: np.random.RandomState,
    ):
        # === Validations ===
        assert num_nodes > 0, "Number of nodes must be positive."
        assert num_roots > 0, "Number of roots must be positive."
        assert num_leaves > 0, "Number of leaves must be positive."
        assert num_nodes > (
            num_roots + num_leaves
        ), "Must have at least one intermediate node."
        assert 0 <= edge_density <= 1, "Edge density must be in [0, 1]."
        assert (
            max_in_degree > 0 and max_out_degree > 0
        ), "In/out degrees must be positive."
        assert (
            min_path_length >= 0 and max_path_length >= 0
        ), "Path lengths must be non-negative."
        assert (
            min_path_length <= max_path_length
        ), "Min path length must be <= max path length."

        # === Core Parameters ===
        self.num_nodes = num_nodes
        self.num_roots = num_roots
        self.num_leaves = num_leaves
        self.edge_density = edge_density
        self.max_in_degree = max_in_degree
        self.max_out_degree = max_out_degree
        self.min_path_length = min_path_length
        self.max_path_length = max_path_length
        self.random_state = random_state

        # === Graph Setup ===
        self.graph = nx.DiGraph()
        self.topological_order = [f"X{i+1}" for i in range(num_nodes)]

        # Precompute node roles
        self.roots = set(self.topological_order[:num_roots])
        self.leaves = set(self.topological_order[-num_leaves:])
        self.intermediates = set(self.topological_order) - self.roots - self.leaves

    def generate(self) -> DAG:
        """
        Main method to generate a DAG.

        Returns:
            DAG: an instance of the DAG class wrapping a NetworkX DiGraph.
        """
        self._build_backbone()
        self._refine_edges()
        self._adjust_path_lengths()
        self._validate()
        return DAG(self.graph)

    def _get_node_types(self) -> Tuple[Set[str], Set[str], Set[str]]:
        """Returns root, leaf, and intermediate node sets."""
        return self.roots, self.leaves, self.intermediates

    def _build_backbone(self):
        """
        Constructs the initial skeleton of the DAG, ensuring:
        - Roots only have outgoing edges
        - Leaves only have incoming edges
        - Intermediates are connected both ways
        """
        for node in self.topological_order:
            self.graph.add_node(node)

        # Ensure roots connect forward
        for root in self.roots:
            possible_targets = [
                t
                for t in (self.intermediates | self.leaves)
                if self.topological_order.index(t) > self.topological_order.index(root)
            ]
            if possible_targets:
                target = self.random_state.choice(possible_targets)
                self.graph.add_edge(root, target)

        # Ensure intermediates have outgoing edges
        for node in self.intermediates:
            possible_targets = [
                t
                for t in (self.intermediates | self.leaves)
                if self.topological_order.index(t) > self.topological_order.index(node)
            ]
            if possible_targets:
                target = self.random_state.choice(possible_targets)
                self.graph.add_edge(node, target)

        # Ensure leaves have incoming edges
        for leaf in self.leaves:
            possible_parents = [
                p
                for p in (self.roots | self.intermediates)
                if self.topological_order.index(p) < self.topological_order.index(leaf)
            ]
            if possible_parents:
                parent = self.random_state.choice(possible_parents)
                self.graph.add_edge(parent, leaf)

        # Ensure intermediates have incoming edges
        for node in self.intermediates:
            if self.graph.in_degree(node) == 0:
                possible_parents = [
                    p
                    for p in (self.roots | self.intermediates - {node})
                    if self.topological_order.index(p)
                    < self.topological_order.index(node)
                ]
                if possible_parents:
                    parent = self.random_state.choice(possible_parents)
                    self.graph.add_edge(parent, node)

    def _refine_edges(self):
        """
        Adds extra edges probabilistically while respecting:
        - Edge density
        - In/out degree limits
        - DAG acyclicity
        """
        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                u = self.topological_order[i]
                v = self.topological_order[j]
                if u in self.leaves or v in self.roots:
                    continue
                if self.random_state.random() < self.edge_density:
                    if (
                        self.graph.in_degree(v) < self.max_in_degree
                        and self.graph.out_degree(u) < self.max_out_degree
                        and not self.graph.has_edge(u, v)
                    ):
                        self.graph.add_edge(u, v)

    def longest_path_bruteforce(self, source, target):
        """
        Finds the longest path between two nodes using all simple paths (slow).

        Returns:
            Tuple[path, length]: Longest path and its length in edges.
        """
        longest = []
        max_len = 0
        for path in nx.all_simple_paths(self.graph, source, target):
            if len(path) > max_len:
                longest = path
                max_len = len(path)
        return longest, max_len - 1

    def _adjust_path_lengths(self):
        """
        Ensures all root-to-leaf paths respect the configured min/max length.

        Shortens long paths using shortcut edges or node bypassing.
        Extends short paths by inserting unused intermediate nodes.
        """
        for root in self.roots:
            for leaf in self.leaves:
                try:
                    # Shortest and longest path
                    shortest = len(nx.shortest_path(self.graph, root, leaf)) - 1
                    longest_path, longest = self.longest_path_bruteforce(root, leaf)

                    # --- Extend too short ---
                    for _ in range(10):
                        if shortest >= self.min_path_length:
                            break
                        # Find a mid node between root and first step
                        path = nx.shortest_path(self.graph, root, leaf)
                        if len(path) < 2:
                            break
                        first = path[1]
                        mid_candidates = [
                            node
                            for node in self.intermediates
                            if node not in path
                            and self.topological_order.index(root)
                            < self.topological_order.index(node)
                            < self.topological_order.index(first)
                        ]
                        if not mid_candidates:
                            break
                        mid = self.random_state.choice(mid_candidates)
                        self.graph.remove_edge(root, first)
                        self.graph.add_edge(root, mid)
                        self.graph.add_edge(mid, first)
                        shortest = len(nx.shortest_path(self.graph, root, leaf)) - 1

                    # --- Shorten too long ---
                    for _ in range(10):
                        if longest <= self.max_path_length:
                            break
                        updated = False
                        for i in range(len(longest_path) - 1):
                            for j in range(i + 2, len(longest_path)):
                                u, v = longest_path[i], longest_path[j]
                                if not self.graph.has_edge(u, v):
                                    if (
                                        self.graph.out_degree(u) < self.max_out_degree
                                        and self.graph.in_degree(v) < self.max_in_degree
                                    ):
                                        self.graph.add_edge(u, v)
                                        updated = True
                                        break
                            if updated:
                                break
                        longest_path, longest = self.longest_path_bruteforce(root, leaf)

                except nx.NetworkXNoPath:
                    continue

    def _validate(self):
        """
        Final validation to ensure DAG structure is valid:
        - Acyclic
        - All node types present
        - No isolated or missing nodes
        """
        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError("Generated graph is not a DAG.")

        if not self.roots:
            raise ValueError("No roots found.")
        if not self.leaves:
            raise ValueError("No leaves found.")
        if not self.intermediates:
            raise ValueError("No intermediates found.")

        isolated = list(nx.isolates(self.graph))
        if isolated:
            raise ValueError(f"Isolated nodes found: {isolated}")

        missing = set(self.topological_order) - set(self.graph.nodes())
        if missing:
            raise ValueError(f"Missing nodes: {missing}")
