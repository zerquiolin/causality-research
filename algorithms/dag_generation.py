import random
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Set, Tuple, Dict
import numpy as np


class DAGGenerator:
    """
    Generates a Directed Acyclic Graph (DAG) with user-defined constraints.
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
    ):
        """
        Initializes the DAG generator with user-defined constraints.
        """
        assert (
            0 < num_roots < num_nodes
        ), "Number of roots must be positive and less than total nodes."
        assert (
            0 < num_leaves < num_nodes
        ), "Number of leaves must be positive and less than total nodes."
        assert 0.0 <= edge_density <= 1.0, "Edge density must be between 0 and 1."
        assert (
            min_path_length <= max_path_length
        ), "Min path length cannot be greater than max path length."

        self.num_nodes = num_nodes
        self.num_roots = num_roots
        self.num_leaves = num_leaves
        self.edge_density = edge_density
        self.max_in_degree = max_in_degree
        self.max_out_degree = max_out_degree
        self.min_path_length = min_path_length
        self.max_path_length = max_path_length
        self.graph = nx.DiGraph()
        self.topological_order = list(range(num_nodes))

    def initialize_nodes(self) -> Tuple[Set[int], Set[int], Set[int]]:
        """
        Partitions the nodes into root, leaf, and intermediate nodes.
        """
        roots = set(self.topological_order[: self.num_roots])
        leaves = set(self.topological_order[-self.num_leaves :])
        intermediates = set(self.topological_order) - (roots | leaves)
        print(f"Roots: {roots}, Leaves: {leaves}, Intermediates: {intermediates}")
        return roots, leaves, intermediates

    def build_backbone(self):
        """
        Constructs the backbone DAG ensuring:
        - Roots only point to intermediates or leaves.
        - Leaves only receive edges.
        - Intermediate nodes always have at least one outgoing edge.
        """
        roots, leaves, intermediates = self.initialize_nodes()

        # Step 1: Add all nodes first
        for node in range(self.num_nodes):
            self.graph.add_node(node)

        # Step 2: Ensure each root has at least one outgoing edge
        for root in roots:
            possible_targets = list(intermediates | leaves)
            possible_targets = [
                t for t in possible_targets if t > root
            ]  # Prevent backward edges
            if possible_targets:
                target = random.choice(possible_targets)
                self.graph.add_edge(root, target)

        # Step 3: Ensure each intermediate has at least one outgoing edge
        for inter in intermediates:
            possible_targets = list(intermediates | leaves - {inter})
            possible_targets = [
                t for t in possible_targets if t > inter
            ]  # Prevent backward edges
            if possible_targets:
                target = random.choice(possible_targets)
                self.graph.add_edge(inter, target)

        # Step 4: Ensure each leaf has at least one incoming edge
        for leaf in leaves:
            possible_parents = list(roots | intermediates)
            possible_parents = [
                p for p in possible_parents if p < leaf
            ]  # Prevent backward edges
            if possible_parents:
                parent = random.choice(possible_parents)
                self.graph.add_edge(parent, leaf)

        # Step 5: Ensure all intermediate nodes have at least one incoming edge
        for inter in intermediates:
            possible_parents = list(roots | intermediates - {inter})
            possible_parents = [p for p in possible_parents if p < inter]
            if possible_parents and self.graph.in_degree(inter) == 0:
                parent = random.choice(possible_parents)
                self.graph.add_edge(parent, inter)

    def refine_edges(self):
        """
        Adds extra edges based on probability while ensuring:
        - No cycles are introduced.
        - Roots do not receive incoming edges.
        - Leaves do not send outgoing edges.
        """
        roots, leaves, _ = self.initialize_nodes()

        possible_edges = [
            (i, j)
            for i in range(self.num_nodes)
            for j in range(i + 1, self.num_nodes)  # Ensure topological order (i < j)
            if i not in leaves and j not in roots  # Respect root and leaf constraints
        ]

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

    def adjust_path_lengths(self):
        """
        Ensures all root-to-leaf paths adhere to the min/max path length constraints.
        - If paths are too short, they are forced to go through intermediate nodes.
        - If paths are too long, unnecessary intermediate nodes are removed.
        - Ensures added nodes do not point to another root.
        """
        roots, leaves, intermediates = self.initialize_nodes()

        for root in roots:
            for leaf in leaves:
                try:
                    # Get the shortest and longest paths
                    shortest_path_nodes = nx.shortest_path(
                        self.graph, source=root, target=leaf
                    )
                    shortest_path = len(shortest_path_nodes) - 1
                    longest_path_nodes, longest_path = self.longest_path_bruteforce(
                        root, leaf
                    )

                    print(
                        f"Root: {root}, Leaf: {leaf}, Shortest path: {shortest_path}, Longest path: {longest_path}"
                    )

                    # **Extend paths if too short** by inserting an intermediate node between the root and its next node
                    while shortest_path < self.min_path_length:
                        path_nodes = nx.shortest_path(self.graph, root, leaf)
                        if len(path_nodes) < 2:
                            break  # Path is already too short to modify

                        first_node = path_nodes[
                            1
                        ]  # The node directly connected to the root
                        available_nodes = list(intermediates - set(path_nodes))

                        if not available_nodes:
                            break  # No available intermediate nodes to use

                        mid_node = random.choice(available_nodes)

                        # Remove the direct connection and insert the intermediate node
                        self.graph.remove_edge(root, first_node)
                        self.graph.add_edge(root, mid_node)
                        self.graph.add_edge(mid_node, first_node)

                        # Update shortest path length
                        shortest_path = (
                            len(nx.shortest_path(self.graph, source=root, target=leaf))
                            - 1
                        )

                    # **Shorten paths if too long** by removing unnecessary intermediate nodes
                    while longest_path > self.max_path_length:
                        path_nodes = longest_path_nodes[1:-1]  # Exclude root and leaf
                        if not path_nodes:
                            break  # No intermediates to remove

                        remove_node = random.choice(path_nodes)
                        predecessors = list(self.graph.predecessors(remove_node))
                        successors = list(self.graph.successors(remove_node))

                        # Reconnect predecessors directly to successors
                        for pred in predecessors:
                            for succ in successors:
                                if pred != succ and not self.graph.has_edge(pred, succ):
                                    self.graph.add_edge(pred, succ)

                        self.graph.remove_node(remove_node)  # Remove the node
                        longest_path_nodes, longest_path = self.longest_path_bruteforce(
                            root, leaf
                        )

                except nx.NetworkXNoPath:
                    continue  # Skip if no path exists

    def validate_and_finalize(self):
        """
        Ensures the DAG remains valid and acyclic.
        """
        if not nx.is_directed_acyclic_graph(self.graph):
            raise ValueError("Graph contains cycles; adjustment required.")

        print(
            f"DAG successfully created with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges."
        )
        return self.graph

    def generate_dag(self):
        """
        Executes the complete DAG generation process.
        """
        self.build_backbone()
        self.refine_edges()
        self.adjust_path_lengths()
        return self.validate_and_finalize()

    def draw_dag(
        self,
        dag: nx.DiGraph = None,
        node_types: tuple[List[int], List[int], List[int]] = None,
        layout="spring",
        spacing_factor=2.0,
    ):
        """
        Draws a DAG with improved spacing.
        - `spring_layout`: Uses repulsion to push nodes apart.
        - `dot_layout`: Creates a clean top-down hierarchy.

        Parameters:
        - dag: NetworkX DiGraph (DAG)
        - layout: "spring" for force-directed or "dot" for hierarchical.
        - spacing_factor: Controls vertical spacing (for "dot" layout).
        """
        dag = dag or self.graph
        print(f"Graph nodes: {dag.nodes()}")
        node_types = node_types or self.initialize_nodes()
        print(f"Node types: {node_types}")

        # Detect node roles
        roots, leaves, intermediates = node_types

        # Define node colors & sizes dynamically
        node_colors, node_sizes = [], []
        for node in dag.nodes():
            if node in roots:
                node_colors.append("#81ff00")  # Green for roots
                node_sizes.append(1000)  # Bigger roots
            elif node in leaves:
                node_colors.append("#ff1c00")  # Red for leaves
                node_sizes.append(1000)  # Bigger leaves
            else:
                node_colors.append("#ffffff")  # Yellow for intermediates
                node_sizes.append(800)  # Medium intermediates

        # Choose layout: Spring (force-directed) or Graphviz Hierarchical
        if layout == "dot":
            try:
                pos = nx.nx_agraph.graphviz_layout(
                    dag, prog="dot", args=f"-Granksep={spacing_factor}"
                )
            except ImportError:
                pos = nx.spring_layout(
                    dag, k=0.5, scale=spacing_factor * 10
                )  # Fallback
        else:
            pos = nx.spring_layout(
                dag, k=1.2, scale=spacing_factor * 10, seed=42
            )  # Spread nodes apart

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Draw nodes
        nx.draw_networkx_nodes(
            dag,
            pos,
            node_color=node_colors,
            edgecolors="black",
            node_size=node_sizes,
            ax=ax,
        )

        # Draw labels
        nx.draw_networkx_labels(
            dag,
            pos,
            labels={node: f"H{node}" for node in dag.nodes()},
            font_size=12,
            font_weight="bold",
            ax=ax,
        )

        # Custom arrow placement (fixes overlap issue)
        for u, v in dag.edges():
            start, end = np.array(pos[u]), np.array(pos[v])
            direction = end - start
            norm_dir = direction / np.linalg.norm(direction)  # Normalize vector

            # Offset start and end points to avoid overlapping with nodes
            node_radius = 0.03 * np.linalg.norm(
                list(ax.get_xlim())
            )  # Scale radius based on figure size
            start = start + norm_dir * node_radius
            end = end - norm_dir * node_radius

            # Draw arrow manually
            ax.annotate(
                "",
                xy=end,
                xytext=start,
                arrowprops=dict(
                    arrowstyle="-|>",
                    lw=1.5,
                    color="gray",
                    shrinkA=10,
                    shrinkB=10,
                    clip_on=False,
                    connectionstyle="arc3,rad=0.1",  # Slight curve to improve readability
                ),
            )

        plt.title("DAG Structure", fontsize=16)
        plt.axis("off")  # Hide axis
        plt.show()


# 🔥 Run the DAG Generator
dag_generator = DAGGenerator(9, 3, 3, 0.1, 3, 3, 2, 4)
dag = dag_generator.generate_dag()
dag_generator.draw_dag(dag)
