import random
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.sampling import BayesianModelSampling
from src.lib.classes.network.SCM import StructuralCausalModel

# Variables
vx = 5  # Visible Variables
hx = 3  # Hidden Variables
mnp = 3  # Maximum Number of Parents
density = 0.8  # Density of the Graph
ad = ["Normal", "Bernoulli", "Exponential", "Poisson"]  # Allowed Distributions
nd = ["Normal", "Bernoulli"]  # Noise Distributions
nl = 0.1  # Noise Level
sample_size = 1000  # Number of samples
inter_nodes = 5  # Number of intervention nodes
seed = 42  # Seed for reproducibility

# Seed: Set seed for reproducibility
random.seed(seed)
np.random.seed(seed)

# Step 1: Create observable and hidden nodes
observable_nodes = [f"V{i}" for i in range(vx)]
hidden_nodes = [f"H{i}" for i in range(hx)]
all_nodes = observable_nodes + hidden_nodes

# Shuffle nodes to mix visible and hidden variables
random.shuffle(all_nodes)  # So the hidden variables are not necessarily at the end

# Check: No duplicate variable names
assert len(all_nodes) == len(set(all_nodes)), "Duplicate variable names detected!"

# Step 2: Initialize the SCM
scm = StructuralCausalModel()
scm.add_nodes(all_nodes)


# Step 3: Dynamically create edges for a DAG
def create_edges(nodes, max_parents, density, seed=42):
    edges = []
    num_nodes = len(nodes)

    # Step 1: Create a spanning tree to ensure connectivity
    for i in range(1, num_nodes):
        parent = random.randint(0, i - 1)
        edges.append((nodes[parent], nodes[i]))

    # Step 2: Add additional edges while maintaining acyclicity
    for i in range(1, num_nodes):
        potential_parents = list(range(i))
        num_parents = min(max_parents, len(potential_parents))
        parents = random.sample(potential_parents, k=num_parents)

        for parent in parents:
            edge = (nodes[parent], nodes[i])
            if edge not in edges and random.random() < density:
                edges.append(edge)

    return edges


edges = create_edges(all_nodes, mnp, density, seed)
scm.add_edges(edges)


# Step 4: Define random node-specific parameters
def generate_random_parameters(node):
    """
    Generate random parameters for a node.

    Parameters:
        node (str): Node name.

    Returns:
        dict: A dictionary of parameters.
    """
    dist_type = random.choice(ad)
    if dist_type == "Normal":
        return {
            "dist": "Normal",
            "mean": random.uniform(-1, 1),
            "std": random.uniform(0.1, 2),
        }
    elif dist_type == "Bernoulli":
        return {"dist": "Bernoulli", "p": random.uniform(0.1, 0.9)}
    elif dist_type == "Exponential":
        return {"dist": "Exponential", "scale": random.uniform(0.1, 2)}
    elif dist_type == "Poisson":
        return {"dist": "Poisson", "lambda": random.uniform(1, 5)}
    else:
        raise ValueError(f"Unsupported distribution type: {dist_type}")


node_params = {node: generate_random_parameters(node) for node in all_nodes}


# Step 5: Assign CPDs using node-specific parameters and separate noise distributions
def add_cdps(scm, nodes, edges, node_params, noise_distributions, nl):
    """
    Adds CPDs to nodes using node-specific parameters and separate noise distributions.

    Parameters:
        scm (StructuralCausalModel): The SCM instance.
        nodes (list): List of nodes.
        edges (list): List of edges.
        node_params (dict): Node-specific parameters.
        noise_distributions (list): Allowed noise distributions.
        nl (float): Noise level.

    Returns:
        None
    """
    for node in nodes:
        parents = [edge[0] for edge in edges if edge[1] == node]
        params = node_params[node]
        dist_type = params["dist"]
        noise_dist = random.choice(noise_distributions)

        print(
            f"Node: {node}, Parents: {parents}, Distribution: {dist_type}, Params: {params}, Noise: {noise_dist}"
        )

        # Define the node function based on the node distribution
        if dist_type == "Normal":
            node_func = lambda x: np.random.normal(
                loc=params["mean"] + np.sum(x), scale=params["std"]
            )
        elif dist_type == "Bernoulli":
            node_func = lambda x: np.random.binomial(
                n=1, p=min(1, max(0, params["p"] + 0.1 * np.sum(x)))
            )
        elif dist_type == "Exponential":
            node_func = lambda x: np.random.exponential(
                scale=params["scale"] + abs(np.sum(x))
            )
        elif dist_type == "Poisson":
            node_func = lambda x: np.random.poisson(
                lam=max(1, params["lambda"] + np.sum(x))
            )
        else:
            raise ValueError(f"Unsupported distribution: {dist_type}")

        # Define the noise function based on the noise distribution
        if noise_dist == "Normal":
            noise_func = lambda: np.random.normal(scale=nl)
        elif noise_dist == "Bernoulli":
            noise_func = lambda: np.random.binomial(n=1, p=0.5)
        else:
            raise ValueError(f"Unsupported noise distribution: {noise_dist}")

        # Precompute CPD values
        if len(parents) == 0:
            # Node without parents
            base_value = node_func([]) + noise_func()
            raw_values = [base_value, 1 - base_value]

            # Shift if negatives exist
            min_val = min(raw_values)
            if min_val < 0:
                raw_values = [v - min_val for v in raw_values]

            # Normalize
            normalized_values = raw_values / np.sum(raw_values)

            cpd = TabularCPD(
                variable=node,
                variable_card=2,  # Binary variable
                values=[[normalized_values[0]], [normalized_values[1]]],
            )
        else:
            # Node with parents
            num_parent_states = 2 ** len(parents)
            raw_cpd_values = np.zeros((2, num_parent_states))

            for i in range(num_parent_states):
                # Parent state combination as binary vector
                parent_state = [int(b) for b in format(i, f"0{len(parents)}b")]
                parent_values = [state * nl for state in parent_state]

                # Compute base_value
                base_value = node_func(parent_values) + noise_func()
                raw_values = [base_value, 1 - base_value]

                # Shift if negatives exist
                min_val = min(raw_values)
                if min_val < 0:
                    raw_values = [v - min_val for v in raw_values]

                # Assign to raw_cpd_values
                raw_cpd_values[0, i] = raw_values[0]
                raw_cpd_values[1, i] = raw_values[1]

            # Normalize probabilities across states
            normalized_cpd_values = raw_cpd_values / np.sum(
                raw_cpd_values, axis=0, keepdims=True
            )

            cpd = TabularCPD(
                variable=node,
                variable_card=2,
                values=normalized_cpd_values.tolist(),
                evidence=parents,
                evidence_card=[2] * len(parents),
            )

        scm.add_cpds({node: cpd})


# add_cpds_with_distributions(scm, all_nodes, edges, ad, nd, nl)
add_cdps(scm, all_nodes, edges, node_params, nd, nl)


# Step 6: Validate SCM
assert scm.is_valid, "SCM is invalid! Check nodes, edges, or CPDs."

# Step 7: Generate Synthetic Data
sampler = BayesianModelSampling(scm.model)
synthetic_data = sampler.forward_sample(size=sample_size)
print("Synthetic Data:")
print(synthetic_data.head())


# Step 8: SCM Validations
def validate_dag(edges):
    graph = nx.DiGraph(edges)
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError("The graph is not a Directed Acyclic Graph (DAG).")
    if not nx.is_weakly_connected(graph):
        raise ValueError("The graph is not fully connected.")
    print("DAG validation passed!")


def validate_cpds(scm):
    for node in scm.nodes:
        cpd = scm.model.get_cpds(node)
        if cpd is None:
            raise ValueError(f"No CPD associated with node {node}.")
        # Validate CPD sums
        total_probabilities = cpd.get_values().sum(axis=0)
        if not np.allclose(total_probabilities, 1):
            raise ValueError(f"CPD for node {node} does not sum to 1.")
    print("CPD validation passed!")


def validate_synthetic_data(synthetic_data, nodes):
    # Check if all nodes are present in the data
    missing_nodes = set(nodes) - set(synthetic_data.columns)
    if missing_nodes:
        raise ValueError(f"Missing nodes in synthetic data: {missing_nodes}")

    # Check value ranges for discrete nodes
    for node in synthetic_data.columns:
        unique_values = synthetic_data[node].unique()
        if len(unique_values) > 2:
            raise ValueError(
                f"Warning: Node {node} has more than 2 unique values. Check CPD logic."
            )
        if not np.all(np.isin(unique_values, [0, 1])):
            raise ValueError(f"Node {node} contains invalid values: {unique_values}")

    print("Synthetic data validation passed!")


def validate_scm(scm, edges, synthetic_data, nodes):
    print("Starting SCM validation...")
    # Validate the DAG structure
    validate_dag(edges)
    # Validate CPDs
    validate_cpds(scm)
    # Validate synthetic data
    validate_synthetic_data(synthetic_data, nodes)
    print("SCM validation completed successfully!")


validate_scm(scm, edges, synthetic_data, all_nodes)

# Step 9: Visualize the SCM
layout = "circular"  # Change to "kamada_kawai", "spring", etc., as needed
scm.visualize(layout)
