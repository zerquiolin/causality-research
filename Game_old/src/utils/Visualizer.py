
import networkx as nx
import matplotlib.pyplot as plt

class Visualizer:
    def __init__(self, bayesian_network):
        self.network = bayesian_network

    def visualize(self):
        """
        Visualize the Bayesian Network using networkx and matplotlib.
        """
        graph = nx.DiGraph(self.network.edges())
        nx.draw(graph, with_labels=True, node_color='skyblue', node_size=2000, font_size=15, font_weight='bold', edge_color='gray')
        plt.show()
