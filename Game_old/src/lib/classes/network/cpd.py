import random
import numpy as np
import pandas as pd
import networkx as nx
from pgmpy.factors.discrete import TabularCPD
from src.lib.classes.network.SCM import (
    StructuralCausalModel,
)  # Assuming this is your custom SCM class


def generate_cpds(scm, variables, edges, noise_distributions, noise_level=0.1):
    """
    Generates CPDs for all nodes in the Structural Causal Model.

    Parameters:
        scm (StructuralCausalModel): The SCM object.
        variables (dict): A dictionary containing variable properties:
                          - Type (observable/hidden)
                          - Domain (categories or continuous range)
                          - Distribution parameters.
        edges (list): A list of edges in the DAG.
        noise_distributions (list): Allowed noise distributions (e.g., ["Normal", "Bernoulli"]).
        noise_level (float): The noise level for continuous distributions.

    Returns:
        dict: A dictionary where keys are node names and values are CPDs in TabularCPD format.
    """
    cpds = {}

    for node in scm.nodes:
        parents = [edge[0] for edge in edges if edge[1] == node]
        node_info = variables[node]

        # Node-specific distribution
        dist_type = node_info["dist"]
        params = node_info["params"]
        domain = node_info["domain"]

        # User-specified noise distribution for the current node
        noise_dist = np.random.choice(noise_distributions)

        # Define the noise function dynamically
        if noise_dist == "Normal":
            noise_func = lambda: np.random.normal(scale=noise_level)
        elif noise_dist == "Bernoulli":
            noise_func = lambda: np.random.binomial(n=1, p=0.5)
        else:
            raise ValueError(f"Unsupported noise distribution: {noise_dist}")

        # CPD generation logic
        if dist_type == "Bernoulli":

            def compute_cpd_values(parent_values):
                prob = params["p"] + np.sum(parent_values) * 0.1
                prob = min(max(prob, 0), 1)
                return prob + noise_func()

        elif dist_type == "Normal":

            def compute_cpd_values(parent_values):
                mean = params["mean"] + np.sum(parent_values)
                return np.random.normal(loc=mean, scale=params["std"]) + noise_func()

        elif dist_type == "Poisson":

            def compute_cpd_values(parent_values):
                lam = max(params["lambda"] + np.sum(parent_values), 1)
                return np.random.poisson(lam) + noise_func()

        elif dist_type == "Exponential":

            def compute_cpd_values(parent_values):
                scale = params["scale"] + abs(np.sum(parent_values))
                return np.random.exponential(scale=scale) + noise_func()

        else:
            raise ValueError(f"Unsupported distribution type: {dist_type}")

        # Generate CPD values
        if len(parents) == 0:
            # Independent node
            values = [compute_cpd_values([]), 1 - compute_cpd_values([])]
            cpd = TabularCPD(
                variable=node, variable_card=len(domain), values=[[v] for v in values]
            )
        else:
            # Node with parents
            num_states = len(domain)
            num_parent_states = np.prod(
                [len(variables[parent]["domain"]) for parent in parents]
            )
            cpd_values = np.zeros((num_states, num_parent_states))

            for i in range(num_parent_states):
                parent_state = [int(b) for b in format(i, f"0{len(parents)}b")]
                parent_values = [state * noise_level for state in parent_state]
                raw_value = compute_cpd_values(parent_values)

                cpd_values[0, i] = raw_value
                cpd_values[1, i] = 1 - raw_value

            cpd = TabularCPD(
                variable=node,
                variable_card=num_states,
                values=cpd_values.tolist(),
                evidence=parents,
                evidence_card=[len(variables[parent]["domain"]) for parent in parents],
            )

        cpds[node] = cpd

    return cpds
