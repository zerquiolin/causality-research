import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy import stats
import itertools


def generate_data(n, intervention=None, intervention_value=None):
    """
    Generate a DataFrame with binary variables A, B, and C.
    Causal structure:
        A -> B
        B -> C
        A -> C
    If intervention is not None, that variable is forced to the given intervention_value.
    """
    df = pd.DataFrame()

    # Generate A (or force intervention)
    if intervention != "A":
        df["A"] = np.random.binomial(1, 0.5, size=n)
    else:
        df["A"] = intervention_value

    # Generate B as a function of A (unless intervened)
    if intervention != "B":
        # P(B=1 | A) = 0.3 + 0.4*A
        prob_B = 0.3 + 0.4 * df["A"]
        df["B"] = np.random.binomial(1, prob_B)
    else:
        df["B"] = intervention_value

    # Generate C as a function of A and B (unless intervened)
    if intervention != "C":
        # P(C=1 | A,B) = 0.3 + 0.3*A + 0.3*B (clipped at 1)
        prob_C = (0.3 + 0.3 * df["A"] + 0.3 * df["B"]).clip(upper=1)
        df["C"] = np.random.binomial(1, prob_C)
    else:
        df["C"] = intervention_value

    return df


def chi2_test(data, X, Y, alpha=0.05):
    """
    Perform an unconditional chi-square test for independence between X and Y.
    Returns the p-value.
    """
    contingency = pd.crosstab(data[X], data[Y])
    # If one variable has only one category, treat it as independent.
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return 1.0
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    return p


def conditional_chi2_test(data, X, Y, cond_vars, alpha=0.05):
    """
    Perform a conditional independence test between X and Y given cond_vars.
    For each configuration of cond_vars, run a chi-square test.
    Returns the minimum p-value across groups (if any group shows dependence, the minimum will be low).
    """
    groups = data.groupby(list(cond_vars))
    p_values = []
    for _, group in groups:
        if len(group) < 5:
            continue
        contingency = pd.crosstab(group[X], group[Y])
        if contingency.shape[0] < 2 or contingency.shape[1] < 2:
            p_values.append(1.0)
        else:
            chi2, p, dof, expected = stats.chi2_contingency(contingency)
            p_values.append(p)
    if not p_values:
        return 1.0
    return min(p_values)


def pc_algorithm(data, alpha=0.05):
    """
    A simple implementation of the PC algorithm for a small set of variables.
    Returns a networkx.DiGraph representing the estimated causal DAG.
    """
    nodes = list(data.columns)
    # Start with a complete undirected graph: dictionary of node: set(neighbors)
    G = {node: set(n for n in nodes if n != node) for node in nodes}
    # Dictionary to store the separating set for each pair
    sep_set = {frozenset([i, j]): set() for i in nodes for j in nodes if i < j}

    # Step 1: Remove edges based on conditional independence tests.
    l = 0
    max_cond_set_size = (
        1  # for three nodes, we only need to check conditioning sets of size 0 and 1
    )
    while l <= max_cond_set_size:
        removal_done = False
        for i in nodes:
            for j in list(G[i]):
                if len(G[i] - {j}) >= l:
                    for cond_set in itertools.combinations(G[i] - {j}, l):
                        # Unconditional test if l==0; otherwise conditional test
                        if l == 0:
                            p_val = chi2_test(data, i, j, alpha)
                        else:
                            p_val = conditional_chi2_test(data, i, j, cond_set, alpha)
                        # If p-value is high, we consider X and Y independent given cond_set.
                        if p_val > alpha:
                            G[i].remove(j)
                            G[j].remove(i)
                            sep_set[frozenset([i, j])] = set(cond_set)
                            removal_done = True
                            break
        if not removal_done:
            l += 1
        else:
            # After removals, reset l to check smaller conditioning sets again
            l = 0

    # Step 2: Orient edges based on the separating sets.
    dag = nx.DiGraph()
    dag.add_nodes_from(nodes)

    # For each triple (i, k, j) where i and j are not connected but both are connected to k,
    # if k is not in the separating set for (i,j), orient i -> k and j -> k.
    for i in nodes:
        for j in nodes:
            if i == j or j in G[i]:
                continue
            common_neighbors = set(G[i]).intersection(G[j])
            for k in common_neighbors:
                if k not in sep_set[frozenset([i, j])]:
                    # Add arrow from i and j into k if not already oriented oppositely
                    if not dag.has_edge(k, i):
                        dag.add_edge(i, k)
                    if not dag.has_edge(k, j):
                        dag.add_edge(j, k)

    # For any remaining undirected edges, orient arbitrarily provided no cycle is introduced.
    for i in nodes:
        for j in G[i]:
            if not dag.has_edge(i, j) and not dag.has_edge(j, i):
                dag.add_edge(i, j)
                try:
                    # Check for cycles; if found, reverse the edge.
                    cycle = nx.find_cycle(dag, orientation="original")
                    dag.remove_edge(i, j)
                    dag.add_edge(j, i)
                except nx.NetworkXNoCycle:
                    pass

    return dag


def orient_edges_with_intervention(dag, df_obs, interventions, alpha=0.05):
    """
    Use interventional data to orient edges: if intervening on var changes target, orient var -> target
    """
    for var, df_int in interventions.items():
        for target in dag.nodes:
            if var == target:
                continue

            # Get value counts (actual counts, not proportions)
            obs_counts = df_obs[target].value_counts()
            int_counts = df_int[target].value_counts()

            # Ensure both have the same categories (0 and 1)
            all_vals = [0, 1]
            obs_vals = [obs_counts.get(val, 0) for val in all_vals]
            int_vals = [int_counts.get(val, 0) for val in all_vals]

            # Now we can use chi-square safely
            try:
                chi2, p = stats.chisquare(f_obs=int_vals, f_exp=obs_vals)
            except ValueError:
                continue  # Skip if something is wrong

            if p < alpha:
                # Intervening on var changed target => var is likely a cause of target
                if dag.has_edge(var, target) and not dag.has_edge(target, var):
                    continue  # Already correctly oriented
                if dag.has_edge(target, var):
                    dag.remove_edge(target, var)
                dag.add_edge(var, target)

    return dag


def plot_dag(dag, title="Learned Causal DAG"):
    """
    Plot the given networkx DAG using matplotlib.
    """
    pos = nx.spring_layout(dag)
    plt.figure(figsize=(6, 4))
    nx.draw(
        dag, pos, with_labels=True, node_color="lightblue", arrows=True, node_size=1500
    )
    plt.title(title)
    plt.show()


def main():
    # Generate observational data
    df_obs = generate_data(1000)

    # Generate interventional data
    df_int_A = generate_data(500, intervention="A", intervention_value=1)
    df_int_B = generate_data(500, intervention="B", intervention_value=1)
    df_int_C = generate_data(500, intervention="C", intervention_value=0)

    # Pack interventional datasets
    interventions = {"A": df_int_A, "B": df_int_B, "C": df_int_C}

    # Step 1: Learn undirected skeleton and partial orientations
    learned_dag = pc_algorithm(df_obs, alpha=0.05)

    # Step 2: Use interventional data to orient more edges
    learned_dag = orient_edges_with_intervention(
        learned_dag, df_obs, interventions, alpha=0.05
    )

    # Step 3: Visualize
    print("Final DAG with interventional orientations:", list(learned_dag.edges()))
    plot_dag(learned_dag)

    return df_obs, df_int_A, df_int_B, df_int_C, learned_dag


if __name__ == "__main__":
    main()
