import random
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling


# Define the SCM class
class StructuralCausalModel:
    def __init__(self):
        self.nodes = []
        self.edges = []
        self.model = BayesianNetwork()
        self.node_metadata = {}
        self.continuous_cpds = {}  # Store callables for continuous nodes

    def add_nodes(self, nodes, metadata):
        self.nodes.extend(nodes)
        for node, meta in metadata.items():
            self.node_metadata[node] = meta
        self.model.add_nodes_from(nodes)

    def add_edges(self, edges):
        self.edges.extend(edges)
        self.model.add_edges_from(edges)

    def add_cpds(self, cpds):
        for node, cpd_data in cpds.items():
            if isinstance(cpd_data, TabularCPD):
                self.model.add_cpds(cpd_data)
            elif callable(cpd_data):
                self.continuous_cpds[node] = cpd_data
            else:
                raise ValueError(f"Unsupported CPD format for node {node}.")

    def visualize(self, layout="circular"):
        pos = (
            nx.circular_layout(self.model)
            if layout == "circular"
            else nx.spring_layout(self.model)
        )
        nx.draw(
            self.model,
            pos,
            with_labels=True,
            node_color="lightblue",
            font_weight="bold",
        )
        nx.draw_networkx_edge_labels(
            self.model, pos, edge_labels={(u, v): "" for u, v in self.edges}
        )
        plt.title("Structural Causal Model (SCM)")
        plt.show()


# Step 1: Define Variables
visible_vars = 5
hidden_vars = 3
max_parents = 3
density = 0.8
allowed_distributions = ["Normal", "Categorical"]
noise_distributions = ["Normal", "Uniform"]
noise_level = 0.1
sample_size = 1000
intervenable_ratio = 0.5
seed = 42

random.seed(seed)
np.random.seed(seed)

# Step 2: Create Nodes and Metadata
observable_nodes = [f"V{i}" for i in range(visible_vars)]
hidden_nodes = [f"H{i}" for i in range(hidden_vars)]
all_nodes = observable_nodes + hidden_nodes

random.shuffle(all_nodes)
num_intervenable_nodes = max(1, int(len(observable_nodes) * intervenable_ratio))
intervenable_nodes = random.sample(observable_nodes, num_intervenable_nodes)

node_metadata = {}
for node in all_nodes:
    if node in intervenable_nodes:
        node_metadata[node] = {"type": "intervenable", "domain": [0, 1, 2]}
    elif node in observable_nodes:
        node_metadata[node] = {"type": "observable", "domain": [0, 1, 2]}
    elif node in hidden_nodes:
        node_metadata[node] = {"type": "hidden", "domain": (0, 10)}

# Step 3: Initialize SCM
scm = StructuralCausalModel()
scm.add_nodes(all_nodes, node_metadata)


# Step 4: DAG Construction
def create_edges(nodes, max_parents, density):
    edges = []
    num_nodes = len(nodes)
    for i in range(1, num_nodes):
        parent = random.randint(0, i - 1)
        edges.append((nodes[parent], nodes[i]))
    for i in range(1, num_nodes):
        potential_parents = list(range(i))
        num_parents = min(max_parents, len(potential_parents))
        parents = random.sample(potential_parents, k=num_parents)
        for parent in parents:
            edge = (nodes[parent], nodes[i])
            if edge not in edges and random.random() < density:
                edges.append(edge)
    return edges


edges = create_edges(all_nodes, max_parents, density)
scm.add_edges(edges)


# Step 5: CPD Generation
def generate_cpds(scm, edges, node_metadata, noise_distributions, noise_level):
    cpds = {}
    for node in scm.nodes:
        metadata = node_metadata.get(node, {})
        domain = metadata.get("domain", [0, 1])  # Default to binary if not set
        parents = [edge[0] for edge in edges if edge[1] == node]

        if isinstance(domain, list):  # Discrete domain
            num_states = len(domain)
            evidence_card = [len(node_metadata[parent]["domain"]) for parent in parents]

            num_parent_states = np.prod(evidence_card) if evidence_card else 1
            values = np.random.rand(num_states, num_parent_states)
            values /= values.sum(axis=0, keepdims=True)  # Normalize probabilities

            if parents:
                cpd = TabularCPD(
                    variable=node,
                    variable_card=num_states,
                    values=values.tolist(),
                    evidence=parents,
                    evidence_card=evidence_card,
                )
            else:
                cpd = TabularCPD(
                    variable=node,
                    variable_card=num_states,
                    values=values.tolist(),
                )
            cpds[node] = cpd  # Add the discrete CPD to the dictionary
        elif isinstance(domain, tuple):  # Continuous domain

            def continuous_func(parent_values):
                parent_sum = sum(parent_values) if parent_values else 0
                return (
                    np.random.uniform(domain[0], domain[1])
                    + parent_sum
                    + noise_level * np.random.normal()
                )

            cpds[node] = continuous_func  # Add the callable for continuous nodes
        else:
            raise ValueError(f"Unsupported domain type for node {node}.")

        # Debugging: Print association of CPDs
        print(
            f"Node: {node}, CPD type: {'Continuous' if callable(cpds[node]) else 'Discrete'}"
        )

    return cpds


def add_cpds(scm, cpds):
    for node, cpd in cpds.items():
        if isinstance(cpd, TabularCPD):
            scm.model.add_cpds(cpd)
        elif callable(cpd):
            scm.continuous_cpds[node] = cpd  # For continuous nodes
        else:
            raise ValueError(f"Unsupported CPD format for node {node}.")
    print("CPDs successfully added to SCM.")


def validate_scm(scm):
    for node in scm.nodes:
        if node not in scm.continuous_cpds:
            cpd = scm.model.get_cpds(node)
            if cpd is None:
                raise ValueError(f"No CPD associated with {node}.")
    print("SCM validation passed!")


# Step 6: Generate Synthetic Data
def generate_synthetic_data(scm, edges, sample_size):
    synthetic_data = pd.DataFrame()
    for node in scm.nodes:
        if node in scm.continuous_cpds:
            # Handle continuous node
            parents = [edge[0] for edge in edges if edge[1] == node]
            parent_data = synthetic_data[parents].to_numpy() if parents else []
            synthetic_data[node] = [
                scm.continuous_cpds[node](parent_data[i]) for i in range(sample_size)
            ]
        else:
            # Handle discrete node
            if synthetic_data.empty:
                sampler = BayesianModelSampling(scm.model)
                synthetic_data = sampler.forward_sample(size=sample_size)
    return synthetic_data


cpds = generate_cpds(scm, edges, node_metadata, noise_distributions, noise_level)
add_cpds(scm, cpds)

# Validate SCM to ensure all nodes have CPDs
validate_scm(scm)

# Generate synthetic data
synthetic_data = generate_synthetic_data(scm, edges, sample_size)
print("Synthetic Data:")
print(synthetic_data.head())

# Visualize SCM
scm.visualize()
