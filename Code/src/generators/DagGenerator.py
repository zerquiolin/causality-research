import numpy as np
import networkx as nx
import logging
from typing import Set, Tuple
from src.lib.models.abstract.BaseDagGenerator import BaseDAGGenerator
from src.lib.models.scm.DAG import DAG


# class DAGGenerator(BaseDAGGenerator):
#     def __init__(
#         self,
#         num_nodes: int,
#         num_roots: int,
#         num_leaves: int,
#         edge_density: float,
#         max_in_degree: int,
#         max_out_degree: int,
#         min_path_length: int,
#         max_path_length: int,
#         random_state: np.random.Generator,
#     ):
#         # Assertions
#         assert num_nodes > 0, "Number of nodes must be positive."
#         assert num_roots > 0, "Number of roots must be positive."
#         assert num_leaves > 0, "Number of leaves must be positive."
#         # Ensure at least one intermediate exists.
#         assert num_nodes > (
#             num_roots + num_leaves
#         ), "There must be at least one intermediate node."
#         assert 0 <= edge_density <= 1, "Edge density must be in [0, 1]."
#         assert max_in_degree > 0, "Maximum in-degree must be positive."
#         assert max_out_degree > 0, "Maximum out-degree must be positive."
#         assert min_path_length >= 0, "Minimum path length must be non-negative."
#         assert max_path_length >= 0, "Maximum path length must be non-negative."
#         assert (
#             min_path_length <= max_path_length
#         ), "Minimum path length must be ≤ maximum path length."

#         # Assignments
#         self.num_nodes = num_nodes
#         self.num_roots = num_roots
#         self.num_leaves = num_leaves
#         self.edge_density = edge_density
#         self.max_in_degree = max_in_degree
#         self.max_out_degree = max_out_degree
#         self.min_path_length = min_path_length
#         self.max_path_length = max_path_length

#         # Create topological order with variable names "X1", "X2", ..., "Xn"
#         self.topological_order = ["X" + str(i + 1) for i in range(num_nodes)]
#         self.graph = nx.DiGraph()

#         # Set random state
#         self.random_state = random_state

#     def generate(self) -> DAG:
#         # Build the backbone DAG.
#         self._build_backbone()
#         # Refine edges based on edge density and degree constraints.
#         self._refine_edges()
#         # Adjust path lengths between roots and leaves.
#         self._adjust_path_lengths()
#         # Validate the generated DAG.
#         self._validate()
#         # Create and return the DAG instance.
#         dag = DAG(self.graph)
#         return dag

#     def _get_node_types(self) -> Tuple[Set[str], Set[str], Set[str]]:
#         """
#         Partitions the nodes into root, leaf, and intermediate nodes.
#         """
#         roots = set(self.topological_order[: self.num_roots])
#         leaves = (
#             set(self.topological_order[-self.num_leaves :])
#             if self.num_leaves > 0
#             else set()
#         )
#         intermediates = set(self.topological_order) - (roots | leaves)
#         return roots, leaves, intermediates

#     def _build_backbone(self):
#         """
#         Constructs the backbone DAG ensuring:
#         - Roots only point to intermediates or leaves.
#         - Leaves only receive edges.
#         - Intermediate nodes always have at least one outgoing edge.
#         """
#         roots, leaves, intermediates = self._get_node_types()

#         # Step 1: Add all nodes first using their variable names.
#         for node in self.topological_order:
#             self.graph.add_node(node)

#         # Step 2: Ensure each root has at least one outgoing edge.
#         for root in roots:
#             idx_root = self.topological_order.index(root)
#             possible_targets = list(intermediates | leaves)
#             possible_targets = [
#                 t
#                 for t in possible_targets
#                 if self.topological_order.index(t) > idx_root
#             ]
#             if possible_targets:
#                 target = self.random_state.choice(possible_targets)
#                 self.graph.add_edge(root, target)

#         # Step 3: Ensure each intermediate has at least one outgoing edge.
#         for inter in intermediates:
#             idx_inter = self.topological_order.index(inter)
#             possible_targets = list((intermediates | leaves) - {inter})
#             possible_targets = [
#                 t
#                 for t in possible_targets
#                 if self.topological_order.index(t) > idx_inter
#             ]
#             if possible_targets:
#                 target = self.random_state.choice(possible_targets)
#                 self.graph.add_edge(inter, target)

#         # Step 4: Ensure each leaf has at least one incoming edge.
#         for leaf in leaves:
#             idx_leaf = self.topological_order.index(leaf)
#             possible_parents = list(roots | intermediates)
#             possible_parents = [
#                 p
#                 for p in possible_parents
#                 if self.topological_order.index(p) < idx_leaf
#             ]
#             if possible_parents:
#                 parent = self.random_state.choice(possible_parents)
#                 self.graph.add_edge(parent, leaf)

#         # Step 5: Ensure all intermediate nodes have at least one incoming edge.
#         for inter in intermediates:
#             idx_inter = self.topological_order.index(inter)
#             possible_parents = list((roots | intermediates) - {inter})
#             possible_parents = [
#                 p
#                 for p in possible_parents
#                 if self.topological_order.index(p) < idx_inter
#             ]
#             if possible_parents and self.graph.in_degree(inter) == 0:
#                 parent = self.random_state.choice(possible_parents)
#                 self.graph.add_edge(parent, inter)

#     def _refine_edges(self):
#         """
#         Adds extra edges based on probability while ensuring:
#         - No cycles are introduced.
#         - Roots do not receive incoming edges.
#         - Leaves do not send outgoing edges.
#         """
#         roots, leaves, _ = self._get_node_types()
#         possible_edges = []
#         for i, node_i in enumerate(self.topological_order):
#             for j in range(i + 1, self.num_nodes):
#                 node_j = self.topological_order[j]
#                 if node_i not in leaves and node_j not in roots:
#                     possible_edges.append((node_i, node_j))
#         self.random_state.shuffle(possible_edges)
#         for i, j in possible_edges:
#             if (
#                 self.random_state.random() < self.edge_density
#                 and self.graph.in_degree(j) < self.max_in_degree
#                 and self.graph.out_degree(i) < self.max_out_degree
#                 and not (i in roots and j in roots)
#                 and not (i in leaves and j in leaves)
#             ):
#                 self.graph.add_edge(i, j)

#     def longest_path_bruteforce(self, source, target):
#         longest_path = []
#         max_length = 0
#         for path in nx.all_simple_paths(self.graph, source, target):
#             if len(path) > max_length:
#                 longest_path = path
#                 max_length = len(path)
#         return longest_path, max_length - 1  # -1 to count edges, not nodes

#     def _adjust_path_lengths(self):
#         """
#         Ensures all root-to-leaf paths adhere to the min/max path length constraints.
#         """
#         roots, leaves, intermediates = self._get_node_types()
#         for root in roots:
#             for leaf in leaves:
#                 try:
#                     shortest_path_nodes = nx.shortest_path(
#                         self.graph, source=root, target=leaf
#                     )
#                     shortest_path = len(shortest_path_nodes) - 1
#                     longest_path_nodes, longest_path = self.longest_path_bruteforce(
#                         root, leaf
#                     )

#                     # Extend paths if too short by inserting an intermediate node.
#                     while shortest_path < self.min_path_length:
#                         path_nodes = nx.shortest_path(self.graph, root, leaf)
#                         if len(path_nodes) < 2:
#                             break
#                         first_node = path_nodes[1]
#                         available_nodes = list(intermediates - set(path_nodes))
#                         if not available_nodes:
#                             break
#                         mid_node = self.random_state.choice(available_nodes)
#                         self.graph.remove_edge(root, first_node)
#                         self.graph.add_edge(root, mid_node)
#                         self.graph.add_edge(mid_node, first_node)
#                         shortest_path = (
#                             len(nx.shortest_path(self.graph, source=root, target=leaf))
#                             - 1
#                         )

#                     # Shorten paths if too long by removing unnecessary intermediate nodes.
#                     while longest_path > self.max_path_length:
#                         path_nodes = longest_path_nodes[1:-1]  # Exclude root and leaf.
#                         if not path_nodes:
#                             break
#                         remove_node = self.random_state.choice(path_nodes)
#                         predecessors = list(self.graph.predecessors(remove_node))
#                         successors = list(self.graph.successors(remove_node))
#                         for pred in predecessors:
#                             for succ in successors:
#                                 if pred != succ and not self.graph.has_edge(pred, succ):
#                                     self.graph.add_edge(pred, succ)
#                         self.graph.remove_node(remove_node)
#                         longest_path_nodes, longest_path = self.longest_path_bruteforce(
#                             root, leaf
#                         )
#                 except nx.NetworkXNoPath:
#                     continue  # Skip if no path exists.

#     def _validate(self):
#         """
#         Ensures the DAG remains valid and acyclic.
#         """
#         if not nx.is_directed_acyclic_graph(self.graph):
#             raise ValueError("Generated graph is not acyclic!")

#         roots, leaves, intermediates = self._get_node_types()
#         if not roots:
#             raise ValueError("No root nodes found!")
#         if not leaves:
#             raise ValueError("No leaf nodes found!")
#         if not intermediates:
#             raise ValueError("No intermediate nodes found!")

#         isolated_nodes = list(nx.isolates(self.graph))
#         if isolated_nodes:
#             raise ValueError(f"Isolated nodes found: {isolated_nodes}")

#         missing_nodes = set(self.topological_order) - set(self.graph.nodes())
#         if missing_nodes:
#             raise ValueError(f"Missing nodes found: {missing_nodes}")


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
        random_state: np.random.Generator,  # Using numpy Generator type.
    ):
        # Assertions
        assert num_nodes > 0, "Number of nodes must be positive."
        assert num_roots > 0, "Number of roots must be positive."
        assert num_leaves > 0, "Number of leaves must be positive."
        # Ensure at least one intermediate exists.
        assert num_nodes > (
            num_roots + num_leaves
        ), "There must be at least one intermediate node."
        assert 0 <= edge_density <= 1, "Edge density must be in [0, 1]."
        assert max_in_degree > 0, "Maximum in-degree must be positive."
        assert max_out_degree > 0, "Maximum out-degree must be positive."
        assert min_path_length >= 0, "Minimum path length must be non-negative."
        assert max_path_length >= 0, "Maximum path length must be non-negative."
        assert (
            min_path_length <= max_path_length
        ), "Minimum path length must be ≤ maximum path length."

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

        # Set random state
        self.random_state = random_state

    def generate(self) -> DAG:
        # Build the backbone DAG.
        self._build_backbone()
        # Refine edges based on edge density and degree constraints.
        self._refine_edges()
        # Adjust path lengths between roots and leaves.
        self._adjust_path_lengths()
        # Validate the generated DAG.
        self._validate()
        # Create and return the DAG instance.
        dag = DAG(self.graph)
        return dag

    def _get_node_types(self) -> Tuple[Set[str], Set[str], Set[str]]:
        """
        Partitions the nodes into root, leaf, and intermediate nodes.
        """
        roots = set(self.topological_order[: self.num_roots])
        leaves = set(self.topological_order[-self.num_leaves :])
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

        # Step 1: Add all nodes.
        for node in self.topological_order:
            self.graph.add_node(node)

        # Step 2: Ensure each root has at least one outgoing edge.
        for root in roots:
            idx_root = self.topological_order.index(root)
            possible_targets = [
                t
                for t in (intermediates | leaves)
                if self.topological_order.index(t) > idx_root
            ]
            if possible_targets:
                target = self.random_state.choice(possible_targets)
                self.graph.add_edge(root, target)
            else:
                logging.warning(f"No valid target found for root {root}.")

        # Step 3: Ensure each intermediate has at least one outgoing edge.
        for inter in intermediates:
            idx_inter = self.topological_order.index(inter)
            possible_targets = [
                t
                for t in ((intermediates | leaves) - {inter})
                if self.topological_order.index(t) > idx_inter
            ]
            if possible_targets:
                target = self.random_state.choice(possible_targets)
                self.graph.add_edge(inter, target)
            else:
                logging.warning(f"No valid target found for intermediate {inter}.")

        # Step 4: Ensure each leaf has at least one incoming edge.
        for leaf in leaves:
            idx_leaf = self.topological_order.index(leaf)
            possible_parents = [
                p
                for p in (roots | intermediates)
                if self.topological_order.index(p) < idx_leaf
            ]
            if possible_parents:
                parent = self.random_state.choice(possible_parents)
                self.graph.add_edge(parent, leaf)
            else:
                logging.warning(f"No valid parent found for leaf {leaf}.")

        # Step 5: Ensure all intermediate nodes have at least one incoming edge.
        for inter in intermediates:
            idx_inter = self.topological_order.index(inter)
            possible_parents = [
                p
                for p in ((roots | intermediates) - {inter})
                if self.topological_order.index(p) < idx_inter
            ]
            if possible_parents and self.graph.in_degree(inter) == 0:
                parent = self.random_state.choice(possible_parents)
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
        self.random_state.shuffle(possible_edges)
        for i, j in possible_edges:
            if (
                self.random_state.random() < self.edge_density
                and self.graph.in_degree(j) < self.max_in_degree
                and self.graph.out_degree(i) < self.max_out_degree
                and not (i in roots and j in roots)
                and not (i in leaves and j in leaves)
            ):
                self.graph.add_edge(i, j)

    def longest_path_bruteforce(self, source, target):
        """
        Brute-force method to compute the longest path between two nodes.
        Note: This method is not scalable for large graphs.
        """
        longest_path = []
        max_length = 0
        for path in nx.all_simple_paths(self.graph, source, target):
            if len(path) > max_length:
                longest_path = path
                max_length = len(path)
        return longest_path, max_length - 1  # Subtract 1 to count edges, not nodes.

    def _adjust_path_lengths(self):
        """
        Ensures all root-to-leaf paths adhere to the min/max path length constraints.
        Uses iteration counters to avoid infinite loops and a fallback mechanism
        to bypass intermediate nodes when no direct shortcut edge can be added.
        """
        roots, leaves, intermediates = self._get_node_types()
        max_extension_iterations = 10
        max_shortening_iterations = 10

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
                    ext_iter = 0
                    while (
                        shortest_path < self.min_path_length
                        and ext_iter < max_extension_iterations
                    ):
                        path_nodes = nx.shortest_path(
                            self.graph, source=root, target=leaf
                        )
                        if len(path_nodes) < 2:
                            logging.warning(
                                f"Path from {root} to {leaf} is too short and cannot be extended."
                            )
                            break
                        first_node = path_nodes[1]
                        # Only consider nodes with index between root and first_node.
                        available_nodes = [
                            node
                            for node in (intermediates - set(path_nodes))
                            if (
                                self.topological_order.index(root)
                                < self.topological_order.index(node)
                                < self.topological_order.index(first_node)
                            )
                        ]
                        if not available_nodes:
                            logging.warning(
                                f"No available intermediate nodes to extend path from {root} to {leaf}."
                            )
                            break
                        mid_node = self.random_state.choice(available_nodes)
                        self.graph.remove_edge(root, first_node)
                        self.graph.add_edge(root, mid_node)
                        self.graph.add_edge(mid_node, first_node)
                        shortest_path = (
                            len(nx.shortest_path(self.graph, source=root, target=leaf))
                            - 1
                        )
                        ext_iter += 1
                    if ext_iter >= max_extension_iterations:
                        logging.warning(
                            f"Maximum extension iterations reached for path from {root} to {leaf}."
                        )

                    # Shorten paths if too long by adding shortcut edges.
                    short_iter = 0
                    while (
                        longest_path > self.max_path_length
                        and short_iter < max_shortening_iterations
                    ):
                        updated = False
                        # Attempt to add a direct shortcut edge between non-adjacent nodes.
                        for i in range(len(longest_path_nodes) - 1):
                            for j in range(i + 2, len(longest_path_nodes)):
                                source_node = longest_path_nodes[i]
                                target_node = longest_path_nodes[j]
                                if not self.graph.has_edge(source_node, target_node):
                                    if (
                                        self.graph.out_degree(source_node)
                                        < self.max_out_degree
                                        and self.graph.in_degree(target_node)
                                        < self.max_in_degree
                                    ):
                                        self.graph.add_edge(source_node, target_node)
                                        updated = True
                                        break
                            if updated:
                                break
                        # Fallback: try to bypass an intermediate node if no shortcut was added.
                        if not updated:
                            bypassed = False
                            # Iterate over intermediate nodes on the longest path (excluding endpoints).
                            for k in range(1, len(longest_path_nodes) - 1):
                                predecessor = longest_path_nodes[k - 1]
                                intermediate = longest_path_nodes[k]
                                successor = longest_path_nodes[k + 1]
                                if not self.graph.has_edge(predecessor, successor):
                                    # Check if removing edge from predecessor -> intermediate is safe.
                                    safe_to_remove_predecessor = (
                                        self.graph.in_degree(intermediate) > 1
                                    )
                                    # Check if removing edge from intermediate -> successor is safe.
                                    safe_to_remove_successor = (
                                        self.graph.out_degree(intermediate) > 1
                                    )
                                    if (
                                        self.graph.out_degree(predecessor)
                                        < self.max_out_degree
                                        and self.graph.in_degree(successor)
                                        < self.max_in_degree
                                    ):
                                        if safe_to_remove_predecessor:
                                            self.graph.remove_edge(
                                                predecessor, intermediate
                                            )
                                            self.graph.add_edge(predecessor, successor)
                                            bypassed = True
                                            updated = True
                                            break
                                        elif safe_to_remove_successor:
                                            self.graph.remove_edge(
                                                intermediate, successor
                                            )
                                            self.graph.add_edge(predecessor, successor)
                                            bypassed = True
                                            updated = True
                                            break
                            if not bypassed:
                                logging.warning(
                                    f"Unable to bypass intermediate nodes for path from {root} to {leaf} without violating constraints."
                                )
                                break
                        longest_path_nodes, longest_path = self.longest_path_bruteforce(
                            root, leaf
                        )
                        short_iter += 1
                    if short_iter >= max_shortening_iterations:
                        logging.warning(
                            f"Maximum shortening iterations reached for path from {root} to {leaf}."
                        )
                except nx.NetworkXNoPath:
                    logging.warning(
                        f"No path found from {root} to {leaf} during adjustment."
                    )
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
