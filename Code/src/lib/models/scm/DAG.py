# File: causality_game/models/dag.py
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


class DAG:
    def __init__(self, graph: nx.DiGraph):
        """
        :param graph: A NetworkX DiGraph representing the DAG.
        """
        self.graph = graph

    @property
    def nodes(self):
        """Return a list of nodes in the DAG."""
        return list(self.graph.nodes())

    @property
    def edges(self):
        """Return a list of edges in the DAG."""
        return list(self.graph.edges())

    def get_parents(self, node):
        """Return a list of parent nodes for the given node."""
        return list(self.graph.predecessors(node))

    def get_node_types(self):
        """
        Return a list of nodes in the DAG, partitioned into roots, leaves, and intermediates.
        """
        roots, leaves, intermediates = [], [], []

        for node in self.graph.nodes():
            if self.graph.in_degree(node) == 0:
                roots.append(node)
            elif self.graph.out_degree(node) == 0:
                leaves.append(node)
            else:
                intermediates.append(node)

        return roots, leaves, intermediates

    def get_structured_nodes(self):
        """
        Return a dictionary of nodes in the DAG as keys and their corresponding parents as values.
        """
        structured_nodes = {}
        for node in self.graph.nodes():
            structured_nodes[node] = self.get_parents(node)

        return structured_nodes

    def plot(
        self,
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
        # Detect node roles
        roots, leaves, intermediates = self.get_node_types()

        # Define node colors & sizes dynamically
        node_colors, node_sizes = [], []
        for node in self.graph.nodes():
            if node in roots:
                node_colors.append("#81ff00")  # Green for roots
                node_sizes.append(1000)  # Bigger roots
            elif node in leaves:
                node_colors.append("#ff1c00")  # Red for leaves
                node_sizes.append(1000)  # Bigger leaves
            else:
                node_colors.append("#ffffff")  # Yellow for intermediates
                node_sizes.append(800)  # Medium intermediates

        pos = nx.spring_layout(
            self.graph, k=1.2, scale=spacing_factor * 10, seed=42
        )  # Spread nodes apart

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Draw nodes
        nx.draw_networkx_nodes(
            self.graph,
            pos,
            node_color=node_colors,
            edgecolors="black",
            node_size=node_sizes,
            ax=ax,
        )

        # Draw labels
        nx.draw_networkx_labels(
            self.graph,
            pos,
            labels={node: node for node in self.graph.nodes()},
            font_size=12,
            font_weight="bold",
            ax=ax,
        )

        # Custom arrow placement (fixes overlap issue)
        for u, v in self.graph.edges():
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
