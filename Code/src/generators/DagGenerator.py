import random
import networkx as nx
from typing import Set, Tuple
from src.lib.models.abstract.BaseDagGenerator import BaseDAGGenerator
import logging


class DAGGenerator(BaseDAGGenerator):
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
    ):
        # Assertions
        assert num_nodes > 0, "Number of nodes must be positive."
        assert num_roots > 0, "Number of roots must be positive."
        assert num_leaves >= 0, "Number of leaves must be non-negative."
        assert 0 <= edge_density <= 1, "Edge density must be in [0, 1]."
        assert max_in_degree > 0, "Maximum in-degree must be positive."
        assert max_out_degree > 0, "Maximum out-degree must be positive."
        assert min_path_length >= 0, "Minimum path length must be non-negative."
        assert max_path_length >= 0, "Maximum path length must be non-negative."
        assert (
            min_path_length <= max_path_length
        ), "Minimum path length must be less than or equal to maximum path length."

        # Assignments
        self.num_nodes = num_nodes
        self.num_roots = num_roots
        self.num_leaves = num_leaves
        self.edge_density = edge_density
        self.max_in_degree = max_in_degree
        self.max_out_degree = max_out_degree
        self.min_path_length = min_path_length
        self.max_path_length = max_path_length

        # Create topological order with variable names "X1", "X2", ..., "Xn"
        self.topological_order = ["X" + str(i + 1) for i in range(num_nodes)]
        self.graph = nx.DiGraph()

    def generate(self) -> nx.DiGraph:
        # Build the backbone DAG.
        self._build_backbone()
        # Refine edges based on edge density and degree constraints.
        self._refine_edges()
        # Adjust path lengths between roots and leaves.
        self._adjust_path_lengths()
        # Validate the generated DAG.
        self._validate()
        return self.graph

    def _get_node_types(self) -> Tuple[Set[str], Set[str], Set[str]]:
        """
        Partitions the nodes into root, leaf, and intermediate nodes.
        """
        roots = set(self.topological_order[: self.num_roots])
        leaves = (
            set(self.topological_order[-self.num_leaves :])
            if self.num_leaves > 0
            else set()
        )
        intermediates = set(self.topological_order) - (roots | leaves)
        return roots, leaves, intermediates

    def _build_backbone(self):
        """
        Constructs the backbone DAG ensuring:
        - Roots only point to intermediates or leaves.
        - Leaves only receive edges.
        - Intermediate nodes always have at least one outgoing edge.
        """
        roots, leaves, intermediates = self._get_node_types()

        # Step 1: Add all nodes first using their variable names.
        for node in self.topological_order:
            self.graph.add_node(node)

        # Step 2: Ensure each root has at least one outgoing edge.
        for root in roots:
            idx_root = self.topological_order.index(root)
            possible_targets = list(intermediates | leaves)
            possible_targets = [
                t
                for t in possible_targets
                if self.topological_order.index(t) > idx_root
            ]
            if possible_targets:
                target = random.choice(possible_targets)
                self.graph.add_edge(root, target)

        # Step 3: Ensure each intermediate has at least one outgoing edge.
        for inter in intermediates:
            idx_inter = self.topological_order.index(inter)
            possible_targets = list((intermediates | leaves) - {inter})
            possible_targets = [
                t
                for t in possible_targets
                if self.topological_order.index(t) > idx_inter
            ]
            if possible_targets:
                target = random.choice(possible_targets)
                self.graph.add_edge(inter, target)

        # Step 4: Ensure each leaf has at least one incoming edge.
        for leaf in leaves:
            idx_leaf = self.topological_order.index(leaf)
            possible_parents = list(roots | intermediates)
            possible_parents = [
                p
                for p in possible_parents
                if self.topological_order.index(p) < idx_leaf
            ]
            if possible_parents:
                parent = random.choice(possible_parents)
                self.graph.add_edge(parent, leaf)

        # Step 5: Ensure all intermediate nodes have at least one incoming edge.
        for inter in intermediates:
            idx_inter = self.topological_order.index(inter)
            possible_parents = list((roots | intermediates) - {inter})
            possible_parents = [
                p
                for p in possible_parents
                if self.topological_order.index(p) < idx_inter
            ]
            if possible_parents and self.graph.in_degree(inter) == 0:
                parent = random.choice(possible_parents)
                self.graph.add_edge(parent, inter)

    def _refine_edges(self):
        """
        Adds extra edges based on probability while ensuring:
        - No cycles are introduced.
        - Roots do not receive incoming edges.
        - Leaves do not send outgoing edges.
        """
        roots, leaves, _ = self._get_node_types()
        possible_edges = []
        for i, node_i in enumerate(self.topological_order):
            for j in range(i + 1, self.num_nodes):
                node_j = self.topological_order[j]
                if node_i not in leaves and node_j not in roots:
                    possible_edges.append((node_i, node_j))
        random.shuffle(possible_edges)
        for i, j in possible_edges:
            if (
                random.random() < self.edge_density
                and self.graph.in_degree(j) < self.max_in_degree
                and self.graph.out_degree(i) < self.max_out_degree
                and not (i in roots and j in roots)
                and not (i in leaves and j in leaves)
            ):
                self.graph.add_edge(i, j)

    def longest_path_bruteforce(self, source, target):
        longest_path = []
        max_length = 0
        for path in nx.all_simple_paths(self.graph, source, target):
            if len(path) > max_length:
                longest_path = path
                max_length = len(path)
        return longest_path, max_length - 1  # -1 to count edges, not nodes

    def _adjust_path_lengths(self):
        """
        Ensures all root-to-leaf paths adhere to the min/max path length constraints.
        """
        roots, leaves, intermediates = self._get_node_types()
        for root in roots:
            for leaf in leaves:
                try:
                    shortest_path_nodes = nx.shortest_path(
                        self.graph, source=root, target=leaf
                    )
                    shortest_path = len(shortest_path_nodes) - 1
                    longest_path_nodes, longest_path = self.longest_path_bruteforce(
                        root, leaf
                    )

                    # Extend paths if too short by inserting an intermediate node.
                    while shortest_path < self.min_path_length:
                        path_nodes = nx.shortest_path(self.graph, root, leaf)
                        if len(path_nodes) < 2:
                            break
                        first_node = path_nodes[1]
                        available_nodes = list(intermediates - set(path_nodes))
                        if not available_nodes:
                            break
                        mid_node = random.choice(available_nodes)
                        self.graph.remove_edge(root, first_node)
                        self.graph.add_edge(root, mid_node)
                        self.graph.add_edge(mid_node, first_node)
                        shortest_path = (
                            len(nx.shortest_path(self.graph, source=root, target=leaf))
                            - 1
                        )

                    # Shorten paths if too long by removing unnecessary intermediate nodes.
                    while longest_path > self.max_path_length:
                        path_nodes = longest_path_nodes[1:-1]  # Exclude root and leaf.
                        if not path_nodes:
                            break
                        remove_node = random.choice(path_nodes)
                        predecessors = list(self.graph.predecessors(remove_node))
                        successors = list(self.graph.successors(remove_node))
                        for pred in predecessors:
                            for succ in successors:
                                if pred != succ and not self.graph.has_edge(pred, succ):
                                    self.graph.add_edge(pred, succ)
                        self.graph.remove_node(remove_node)
                        longest_path_nodes, longest_path = self.longest_path_bruteforce(
                            root, leaf
                        )
                except nx.NetworkXNoPath:
                    continue  # Skip if no path exists.

    def _validate(self):
        """
        Ensures the DAG remains valid and acyclic.
        """
        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError("Generated graph is not acyclic!")

        roots, leaves, intermediates = self._get_node_types()
        if not roots:
            raise ValueError("No root nodes found!")
        if not leaves:
            raise ValueError("No leaf nodes found!")
        if not intermediates:
            raise ValueError("No intermediate nodes found!")

        isolated_nodes = list(nx.isolates(self.graph))
        if isolated_nodes:
            raise ValueError(f"Isolated nodes found: {isolated_nodes}")

        missing_nodes = set(self.topological_order) - set(self.graph.nodes())
        if missing_nodes:
            raise ValueError(f"Missing nodes found: {missing_nodes}")

        logging.info("DAG validation successful!")
