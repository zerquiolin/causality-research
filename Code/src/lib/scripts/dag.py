import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy import stats
import itertools


def chi2_test(data, X, Y, alpha=0.05):
    """
    Perform an unconditional chi-square test for independence between two categorical variables.

    This function builds a contingency table for the columns X and Y from the DataFrame 'data'
    and then performs a chi-square test. If either variable has only one category (i.e., no variability),
    the function returns a p-value of 1.0, indicating no evidence to reject independence.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset containing the variables.
    X : str
        Column name for the first variable.
    Y : str
        Column name for the second variable.
    alpha : float, optional
        Significance level for the test (default 0.05). Although not used directly here, it is typically used in decision making.

    Returns
    -------
    float
        The p-value from the chi-square test.
    """
    # Create a contingency table between variables X and Y.
    contingency = pd.crosstab(data[X], data[Y])
    # If one of the variables has only one category, return p-value 1.0.
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return 1.0
    # Perform the chi-square test on the contingency table.
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    return p


def conditional_chi2_test(data, X, Y, cond_vars, alpha=0.05):
    """
    Perform a conditional chi-square test of independence between X and Y given a set of conditioning variables.

    For each unique configuration of the conditioning variables (cond_vars),
    a chi-square test is conducted on the subset of the data. The function collects all the p-values
    and returns the minimum p-value. A low minimum p-value indicates that there is evidence of dependence
    in at least one subgroup.

    Parameters
    ----------
    data : pd.DataFrame
        The dataset containing the variables.
    X : str
        Column name for the first variable.
    Y : str
        Column name for the second variable.
    cond_vars : list of str
        List of column names to condition on.
    alpha : float, optional
        Significance level for the test (default 0.05).

    Returns
    -------
    float
        The minimum p-value across all groups, or 1.0 if no valid tests are conducted.
    """
    # Group the data by the conditioning variables.
    groups = data.groupby(list(cond_vars))
    p_values = []
    # Iterate through each group (each unique combination of cond_vars).
    for _, group in groups:
        # Skip groups with fewer than 5 samples.
        if len(group) < 5:
            continue
        # Build a contingency table for X and Y in the current group.
        contingency = pd.crosstab(group[X], group[Y])
        # If there is no variability, assign p-value 1.0.
        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            p_values.append(1.0)
        else:
            # Perform the chi-square test for the current group.
            chi2, p, dof, expected = stats.chi2_contingency(contingency)
            p_values.append(p)
    # If no valid p-values were calculated, return 1.0.
    if not p_values:
        return 1.0
    # Return the minimum p-value among the groups.
    return min(p_values)


def pc_algorithm(data, alpha=0.05):
    """
    A simple implementation of the PC algorithm for causal discovery.

    The algorithm starts with a complete undirected graph among all variables,
    and then iteratively removes edges based on unconditional and conditional independence tests
    (using chi-square tests). Separating sets are stored for later orientation of edges.
    Finally, the algorithm orients edges based on the separating sets and additional rules,
    producing a directed acyclic graph (DAG) represented as a networkx.DiGraph.

    Parameters
    ----------
    data : pd.DataFrame
        The observational dataset where each column represents a variable.
    alpha : float, optional
        The significance level for the chi-square tests (default 0.05).

    Returns
    -------
    nx.DiGraph
        A directed acyclic graph representing the estimated causal structure.
    """
    # Obtain all variable names.
    nodes = list(data.columns)
    # Initialize a complete undirected graph: each node is connected to all others.
    G = {node: set(n for n in nodes if n != node) for node in nodes}
    # Initialize a dictionary to store the separating set for each pair of nodes.
    sep_set = {frozenset([i, j]): set() for i in nodes for j in nodes if i < j}

    print("Initial undirected graph (skeleton):")
    print(G)

    print("Initial separating sets:")
    print(sep_set)

    # Step 1: Remove edges based on independence tests.
    l = 0  # Start with conditioning sets of size 0 (i.e., unconditional test).
    max_cond_set_size = (
        1  # For a small number of nodes, only need to check sets of size 0 and 1.
    )
    while l <= max_cond_set_size:
        removal_done = False  # Flag to check if any edge was removed at current conditioning set size.
        for i in nodes:
            for j in list(G[i]):
                # Check if there are enough neighbors to form a conditioning set of size l.
                if len(G[i] - {j}) >= l:
                    # For every combination of neighbors of size l (excluding j)
                    for cond_set in itertools.combinations(G[i] - {j}, l):
                        # If l==0, perform an unconditional chi-square test.
                        if l == 0:
                            p_val = chi2_test(data, i, j, alpha)
                            print(
                                f"Unconditional test between {i} and {j}: p-value = {p_val}"
                            )
                        else:
                            # Otherwise, perform a conditional chi-square test given cond_set.
                            p_val = conditional_chi2_test(data, i, j, cond_set, alpha)
                            print(
                                f"Conditional test between {i} and {j} given {cond_set}: p-value = {p_val}"
                            )
                        # If the p-value is high (above alpha), conclude independence and remove the edge.
                        if p_val > alpha:
                            print(
                                f"Removing edge ({i}, {j}) based on independence test."
                            )
                            G[i].remove(j)
                            G[j].remove(i)
                            sep_set[frozenset([i, j])] = set(cond_set)
                            removal_done = True
                            break  # Exit cond_set loop once independence is found.
        if not removal_done:
            l += (
                1  # Increase the size of the conditioning set if no edges were removed.
            )
        else:
            l = 0  # Reset l to 0 after removals to re-check with smaller conditioning sets.

    # Step 2: Orient edges based on the separating sets.
    dag = nx.DiGraph()
    dag.add_nodes_from(nodes)

    # For each pair of non-adjacent nodes, identify common neighbors.
    # For any triple (i, k, j) where i and j are not connected but both connected to k,
    # if k is not in the separating set for (i, j), then orient edges i -> k and j -> k.
    for i in nodes:
        for j in nodes:
            if i == j or j in G[i]:
                continue
            common_neighbors = set(G[i]).intersection(G[j])
            for k in common_neighbors:
                if k not in sep_set[frozenset([i, j])]:
                    print(
                        f"Orienting edges ({i}, {k}) and ({j}, {k}) based on common neighbor {k}."
                    )
                    # Orient edge from i to k if not already reversed.
                    if not dag.has_edge(k, i):
                        dag.add_edge(i, k)
                    # Orient edge from j to k if not already reversed.
                    if not dag.has_edge(k, j):
                        dag.add_edge(j, k)

    # Step 3: For any remaining undirected edges, assign an arbitrary orientation
    # provided that no cycle is introduced.
    for i in nodes:
        for j in G[i]:
            if not dag.has_edge(i, j) and not dag.has_edge(j, i):
                print(
                    f"Orienting edge ({i}, {j}) arbitrarily as no cycle is introduced."
                )
                dag.add_edge(i, j)
                try:
                    # Check for cycles; if a cycle is found, reverse the edge.
                    cycle = nx.find_cycle(dag, orientation="original")
                    dag.remove_edge(i, j)
                    dag.add_edge(j, i)
                    print(
                        f"Cycle found with edge ({i}, {j}), reversing orientation to ({j}, {i})."
                    )
                except nx.NetworkXNoCycle:
                    # If no cycle is found, the orientation is acceptable.
                    print(
                        f"No cycle found with edge ({i}, {j}), orientation remains as ({i}, {j})."
                    )
                    pass

    return dag


def orient_edges_with_intervention(dag, df_obs, interventions, alpha=0.05):
    """
    Use interventional data to orient edges in a causal graph.

    For each variable for which interventional data is available, the function compares the distribution of a target variable
    under observational conditions (df_obs) and under intervention (df_int). A chi-square test is used to determine if the
    intervention on the variable leads to a significant change in the target. If so, the edge is oriented as var -> target.

    Parameters
    ----------
    dag : nx.DiGraph
        The current estimated causal graph.
    df_obs : pd.DataFrame
        The observational dataset.
    interventions : dict
        A dictionary with keys as variable names and values as interventional datasets (pd.DataFrame).
    alpha : float, optional
        Significance level for the chi-square test (default 0.05).

    Returns
    -------
    nx.DiGraph
        The DAG with edges further oriented based on interventional data.
    """
    print("Orienting edges using interventional data...")
    # Loop over each variable for which interventional data is available.
    for var, df_int in interventions.items():
        # Loop over every target variable in the graph.
        for target in dag.nodes:
            if var == target:
                continue  # Skip if the variable is the same as the target.
            # Obtain counts of the target variable from observational data.
            obs_counts = df_obs[target].value_counts()
            print(f"Observational counts for {target}: {obs_counts}")
            # Obtain counts of the target variable from interventional data.
            int_counts = df_int[target].value_counts()
            print(f"Interventional counts for {target}: {int_counts}")
            # Ensure both datasets cover the same categories (assumed here as 0 and 1).
            all_vals = [0, 1]
            obs_vals = [obs_counts.get(val, 0) for val in all_vals]
            int_vals = [int_counts.get(val, 0) for val in all_vals]
            # Perform a chi-square test to compare the two distributions.
            try:
                chi2, p = stats.chisquare(f_obs=int_vals, f_exp=obs_vals)
                print(f"Chi-square test between {var} and {target}: p-value = {p}")
            except ValueError:
                continue  # If test fails, skip this target.
            # If the p-value indicates a significant difference, orient the edge.
            if p < alpha:
                print(
                    f"Intervention on {var} significantly affects {target}, orienting edge as ({var}, {target})."
                )
                # If edge is already oriented as var -> target, continue.
                if dag.has_edge(var, target) and not dag.has_edge(target, var):
                    continue
                # If edge is in the opposite direction, remove it.
                if dag.has_edge(target, var):
                    dag.remove_edge(target, var)
                # Add the edge from var to target.
                dag.add_edge(var, target)
    return dag


def learn_dag(df_obs, interventions, alpha=0.05):
    """
    Learn a causal Directed Acyclic Graph (DAG) from both observational and interventional data.

    This function first estimates the undirected skeleton and partially orients edges using the PC algorithm.
    Then, it refines the edge orientations using interventional data via chi-square tests.

    Parameters
    ----------
    df_obs : pd.DataFrame
        The observational dataset.
    interventions : dict
        A dictionary where keys are variable names and values are interventional datasets (pd.DataFrame).
    alpha : float, optional
        Significance level for the chi-square tests (default 0.05).

    Returns
    -------
    nx.DiGraph
        The learned causal DAG as a directed acyclic graph.
    """
    # Step 1: Use the PC algorithm to learn the skeleton and partial orientations.
    dag = pc_algorithm(df_obs, alpha)
    print("Initial DAG from observational data:")
    print(dag.edges())

    # Step 2: Use interventional data to further orient the edges.
    dag = orient_edges_with_intervention(dag, df_obs, interventions, alpha)
    print("Final DAG after orienting with interventional data:")
    print(dag.edges())
    return dag
