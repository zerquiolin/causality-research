# import numpy as np
# import pandas as pd
# import networkx as nx
# import matplotlib.pyplot as plt
# from scipy import stats
# import itertools


# def chi2_test(data, X, Y, alpha=0.05):
#     """
#     Perform an unconditional chi-square test for independence between two categorical variables.

#     This function builds a contingency table for the columns X and Y from the DataFrame 'data'
#     and then performs a chi-square test. If either variable has only one category (i.e., no variability),
#     the function returns a p-value of 1.0, indicating no evidence to reject independence.

#     Parameters
#     ----------
#     data : pd.DataFrame
#         The dataset containing the variables.
#     X : str
#         Column name for the first variable.
#     Y : str
#         Column name for the second variable.
#     alpha : float, optional
#         Significance level for the test (default 0.05). Although not used directly here, it is typically used in decision making.

#     Returns
#     -------
#     float
#         The p-value from the chi-square test.
#     """
#     # Create a contingency table between variables X and Y.
#     contingency = pd.crosstab(data[X], data[Y])
#     # If one of the variables has only one category, return p-value 1.0.
#     if contingency.shape[0] < 2 or contingency.shape[1] < 2:
#         return 1.0
#     # Perform the chi-square test on the contingency table.
#     chi2, p, dof, expected = stats.chi2_contingency(contingency)
#     return p


# def conditional_chi2_test(data, X, Y, cond_vars, alpha=0.05):
#     """
#     Perform a conditional chi-square test of independence between X and Y given a set of conditioning variables.

#     For each unique configuration of the conditioning variables (cond_vars),
#     a chi-square test is conducted on the subset of the data. The function collects all the p-values
#     and returns the minimum p-value. A low minimum p-value indicates that there is evidence of dependence
#     in at least one subgroup.

#     Parameters
#     ----------
#     data : pd.DataFrame
#         The dataset containing the variables.
#     X : str
#         Column name for the first variable.
#     Y : str
#         Column name for the second variable.
#     cond_vars : list of str
#         List of column names to condition on.
#     alpha : float, optional
#         Significance level for the test (default 0.05).

#     Returns
#     -------
#     float
#         The minimum p-value across all groups, or 1.0 if no valid tests are conducted.
#     """
#     # Group the data by the conditioning variables.
#     groups = data.groupby(list(cond_vars))
#     p_values = []
#     # Iterate through each group (each unique combination of cond_vars).
#     for _, group in groups:
#         # Skip groups with fewer than 5 samples.
#         if len(group) < 5:
#             continue
#         # Build a contingency table for X and Y in the current group.
#         contingency = pd.crosstab(group[X], group[Y])
#         # If there is no variability, assign p-value 1.0.
#         if contingency.shape[0] < 2 or contingency.shape[1] < 2:
#             p_values.append(1.0)
#         else:
#             # Perform the chi-square test for the current group.
#             chi2, p, dof, expected = stats.chi2_contingency(contingency)
#             p_values.append(p)
#     # If no valid p-values were calculated, return 1.0.
#     if not p_values:
#         return 1.0
#     # Return the minimum p-value among the groups.
#     return min(p_values)


# def pc_algorithm(data, alpha=0.05):
#     """
#     A simple implementation of the PC algorithm for causal discovery.

#     The algorithm starts with a complete undirected graph among all variables,
#     and then iteratively removes edges based on unconditional and conditional independence tests
#     (using chi-square tests). Separating sets are stored for later orientation of edges.
#     Finally, the algorithm orients edges based on the separating sets and additional rules,
#     producing a directed acyclic graph (DAG) represented as a networkx.DiGraph.

#     Parameters
#     ----------
#     data : pd.DataFrame
#         The observational dataset where each column represents a variable.
#     alpha : float, optional
#         The significance level for the chi-square tests (default 0.05).

#     Returns
#     -------
#     nx.DiGraph
#         A directed acyclic graph representing the estimated causal structure.
#     """
#     # Obtain all variable names.
#     nodes = list(data.columns)
#     # Initialize a complete undirected graph: each node is connected to all others.
#     G = {node: set(n for n in nodes if n != node) for node in nodes}
#     # Initialize a dictionary to store the separating set for each pair of nodes.
#     sep_set = {frozenset([i, j]): set() for i in nodes for j in nodes if i < j}

#     print("Initial undirected graph (skeleton):")
#     print(G)

#     print("Initial separating sets:")
#     print(sep_set)

#     # Step 1: Remove edges based on independence tests.
#     l = 0  # Start with conditioning sets of size 0 (i.e., unconditional test).
#     max_cond_set_size = (
#         1  # For a small number of nodes, only need to check sets of size 0 and 1.
#     )
#     while l <= max_cond_set_size:
#         removal_done = False  # Flag to check if any edge was removed at current conditioning set size.
#         for i in nodes:
#             for j in list(G[i]):
#                 # Check if there are enough neighbors to form a conditioning set of size l.
#                 if len(G[i] - {j}) >= l:
#                     # For every combination of neighbors of size l (excluding j)
#                     for cond_set in itertools.combinations(G[i] - {j}, l):
#                         # If l==0, perform an unconditional chi-square test.
#                         if l == 0:
#                             p_val = chi2_test(data, i, j, alpha)
#                             print(
#                                 f"Unconditional test between {i} and {j}: p-value = {p_val}"
#                             )
#                         else:
#                             # Otherwise, perform a conditional chi-square test given cond_set.
#                             p_val = conditional_chi2_test(data, i, j, cond_set, alpha)
#                             print(
#                                 f"Conditional test between {i} and {j} given {cond_set}: p-value = {p_val}"
#                             )
#                         # If the p-value is high (above alpha), conclude independence and remove the edge.
#                         if p_val > alpha:
#                             print(
#                                 f"Removing edge ({i}, {j}) based on independence test."
#                             )
#                             G[i].remove(j)
#                             G[j].remove(i)
#                             sep_set[frozenset([i, j])] = set(cond_set)
#                             removal_done = True
#                             break  # Exit cond_set loop once independence is found.
#         if not removal_done:
#             l += (
#                 1  # Increase the size of the conditioning set if no edges were removed.
#             )
#         else:
#             l = 0  # Reset l to 0 after removals to re-check with smaller conditioning sets.

#     # Step 2: Orient edges based on the separating sets.
#     dag = nx.DiGraph()
#     dag.add_nodes_from(nodes)

#     # For each pair of non-adjacent nodes, identify common neighbors.
#     # For any triple (i, k, j) where i and j are not connected but both connected to k,
#     # if k is not in the separating set for (i, j), then orient edges i -> k and j -> k.
#     for i in nodes:
#         for j in nodes:
#             if i == j or j in G[i]:
#                 continue
#             common_neighbors = set(G[i]).intersection(G[j])
#             for k in common_neighbors:
#                 if k not in sep_set[frozenset([i, j])]:
#                     print(
#                         f"Orienting edges ({i}, {k}) and ({j}, {k}) based on common neighbor {k}."
#                     )
#                     # Orient edge from i to k if not already reversed.
#                     if not dag.has_edge(k, i):
#                         dag.add_edge(i, k)
#                     # Orient edge from j to k if not already reversed.
#                     if not dag.has_edge(k, j):
#                         dag.add_edge(j, k)

#     # Step 3: For any remaining undirected edges, assign an arbitrary orientation
#     # provided that no cycle is introduced.
#     for i in nodes:
#         for j in G[i]:
#             if not dag.has_edge(i, j) and not dag.has_edge(j, i):
#                 print(
#                     f"Orienting edge ({i}, {j}) arbitrarily as no cycle is introduced."
#                 )
#                 dag.add_edge(i, j)
#                 try:
#                     # Check for cycles; if a cycle is found, reverse the edge.
#                     cycle = nx.find_cycle(dag, orientation="original")
#                     dag.remove_edge(i, j)
#                     dag.add_edge(j, i)
#                     print(
#                         f"Cycle found with edge ({i}, {j}), reversing orientation to ({j}, {i})."
#                     )
#                 except nx.NetworkXNoCycle:
#                     # If no cycle is found, the orientation is acceptable.
#                     print(
#                         f"No cycle found with edge ({i}, {j}), orientation remains as ({i}, {j})."
#                     )
#                     pass

#     return dag


# def orient_edges_with_intervention(dag, df_obs, interventions, alpha=0.05):
#     """
#     Use interventional data to orient edges in a causal graph.

#     For each variable for which interventional data is available, the function compares the distribution of a target variable
#     under observational conditions (df_obs) and under intervention (df_int). A chi-square test is used to determine if the
#     intervention on the variable leads to a significant change in the target. If so, the edge is oriented as var -> target.

#     Parameters
#     ----------
#     dag : nx.DiGraph
#         The current estimated causal graph.
#     df_obs : pd.DataFrame
#         The observational dataset.
#     interventions : dict
#         A dictionary with keys as variable names and values as interventional datasets (pd.DataFrame).
#     alpha : float, optional
#         Significance level for the chi-square test (default 0.05).

#     Returns
#     -------
#     nx.DiGraph
#         The DAG with edges further oriented based on interventional data.
#     """
#     print("Orienting edges using interventional data...")
#     # Loop over each variable for which interventional data is available.
#     for var, df_int in interventions.items():
#         # Loop over every target variable in the graph.
#         for target in dag.nodes:
#             if var == target:
#                 continue  # Skip if the variable is the same as the target.
#             # Obtain counts of the target variable from observational data.
#             obs_counts = df_obs[target].value_counts()
#             print(f"Observational counts for {target}: {obs_counts}")
#             # Obtain counts of the target variable from interventional data.
#             int_counts = df_int[target].value_counts()
#             print(f"Interventional counts for {target}: {int_counts}")
#             # Ensure both datasets cover the same categories (assumed here as 0 and 1).
#             all_vals = [0, 1]
#             obs_vals = [obs_counts.get(val, 0) for val in all_vals]
#             int_vals = [int_counts.get(val, 0) for val in all_vals]
#             # Perform a chi-square test to compare the two distributions.
#             try:
#                 chi2, p = stats.chisquare(f_obs=int_vals, f_exp=obs_vals)
#                 print(f"Chi-square test between {var} and {target}: p-value = {p}")
#             except ValueError:
#                 continue  # If test fails, skip this target.
#             # If the p-value indicates a significant difference, orient the edge.
#             if p < alpha:
#                 print(
#                     f"Intervention on {var} significantly affects {target}, orienting edge as ({var}, {target})."
#                 )
#                 # If edge is already oriented as var -> target, continue.
#                 if dag.has_edge(var, target) and not dag.has_edge(target, var):
#                     continue
#                 # If edge is in the opposite direction, remove it.
#                 if dag.has_edge(target, var):
#                     dag.remove_edge(target, var)
#                 # Add the edge from var to target.
#                 dag.add_edge(var, target)
#     return dag


# def learn_dag(df_obs, interventions, alpha=0.05):
#     """
#     Learn a causal Directed Acyclic Graph (DAG) from both observational and interventional data.

#     This function first estimates the undirected skeleton and partially orients edges using the PC algorithm.
#     Then, it refines the edge orientations using interventional data via chi-square tests.

#     Parameters
#     ----------
#     df_obs : pd.DataFrame
#         The observational dataset.
#     interventions : dict
#         A dictionary where keys are variable names and values are interventional datasets (pd.DataFrame).
#     alpha : float, optional
#         Significance level for the chi-square tests (default 0.05).

#     Returns
#     -------
#     nx.DiGraph
#         The learned causal DAG as a directed acyclic graph.
#     """
#     # Step 1: Use the PC algorithm to learn the skeleton and partial orientations.
#     dag = pc_algorithm(df_obs, alpha)
#     print("Initial DAG from observational data:")
#     print(dag.edges())

#     # Step 2: Use interventional data to further orient the edges.
#     dag = orient_edges_with_intervention(dag, df_obs, interventions, alpha)
#     print("Final DAG after orienting with interventional data:")
#     print(dag.edges())
#     return dag


import logging
import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats
import itertools

# Configure logging (DEBUG for detailed output)
logging.basicConfig(
    level=logging.info, format="%(asctime)s - %(levelname)s - %(message)s"
)


def chi2_test(data, X, Y, alpha=0.05):
    """
    Perform an unconditional chi-square test for independence between two categorical variables.

    Constructs a contingency table using columns X and Y from `data` and computes the chi-square test.
    If either variable has only one category, returns a p-value of 1.0.
    """
    contingency = pd.crosstab(data[X], data[Y])
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return 1.0
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    return p


def conditional_chi2_test(data, X, Y, cond_vars, alpha=0.05):
    """
    Perform a conditional chi-square test for independence between X and Y given conditioning variables.

    The data is partitioned into groups defined by the unique values of cond_vars. The function returns the minimum
    p-value across all groups (or 1.0 if insufficient data in all groups).
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


def log_conditional_probabilities(df_obs):
    """
    When exactly three variables are present (say, A, B, and C),
    log for each variable pair (a, b) the conditional probabilities given the remaining variable c.
    For each value of c, this logs:
       - the contingency table for (a, b) given c,
       - the observed conditional joint probability P(a, b | c),
       - the conditional marginals P(a|c) and P(b|c),
       - the expected joint probability under conditional independence: P(a|c)*P(b|c),
       - and the result of a chi-square test.
    """
    nodes = list(df_obs.columns)
    if len(nodes) != 3:
        return

    combos = [
        ((nodes[0], nodes[1]), nodes[2]),
        ((nodes[0], nodes[2]), nodes[1]),
        ((nodes[1], nodes[2]), nodes[0]),
    ]
    for (a, b), c in combos:
        logging.info("=== Analysis for P(%s, %s | %s) ===", a, b, c)
        for c_val in sorted(df_obs[c].unique()):
            subset = df_obs[df_obs[c] == c_val]
            contingency_ab = pd.crosstab(subset[a], subset[b])
            total = contingency_ab.values.sum()
            if total > 0:
                joint_ab = contingency_ab / total
            else:
                joint_ab = contingency_ab.copy()
            marg_a = subset[a].value_counts(normalize=True)
            marg_b = subset[b].value_counts(normalize=True)
            # Build expected joint probability assuming conditional independence.
            expected_joint = pd.DataFrame(
                index=joint_ab.index, columns=joint_ab.columns
            )
            for i in joint_ab.index:
                for j in joint_ab.columns:
                    expected_joint.loc[i, j] = marg_a.get(i, 0) * marg_b.get(j, 0)
            expected_joint = expected_joint.astype(float)
            logging.info("Condition: %s = %s", c, c_val)
            logging.info("Contingency table for %s and %s:\n%s", a, b, contingency_ab)
            logging.info(
                "Observed joint probability P(%s, %s | %s=%s):\n%s",
                a,
                b,
                c,
                c_val,
                joint_ab,
            )
            logging.info("Conditional marginal P(%s|%s=%s):\n%s", a, c, c_val, marg_a)
            logging.info("Conditional marginal P(%s|%s=%s):\n%s", b, c, c_val, marg_b)
            logging.info(
                "Expected joint probability (P(%s|%s)*P(%s|%s)):\n%s",
                a,
                c,
                b,
                c,
                expected_joint,
            )
            try:
                chi2, p, dof, exp = stats.chi2_contingency(contingency_ab)
                logging.info(
                    "Chi-square test for %s and %s given %s=%s yields p-value = %s",
                    a,
                    b,
                    c,
                    c_val,
                    p,
                )
            except Exception as e:
                logging.info(
                    "Chi-square test error for %s and %s given %s=%s: %s",
                    a,
                    b,
                    c,
                    c_val,
                    e,
                )


def apply_meek_rules(dag, skeleton, nodes):
    """
    Iteratively apply Meek's orientation rules (R1–R4) to a partially directed graph.

    Only undirected edges (i.e. those present in the skeleton but not yet oriented in `dag`)
    are considered. Conflicts with existing orientations are logged and the existing
    orientation is retained.
    """
    logging.info(f"Skeleton before Meek's rules: {skeleton}")
    changed = True
    while changed:
        changed = False
        for i in nodes:
            for j in skeleton[i]:
                # Only consider undirected edges: if any directed edge exists between i and j, skip.
                if dag.has_edge(i, j) or dag.has_edge(j, i):
                    continue

                # --- Rule R1 ---
                for X in nodes:
                    if X in (i, j):
                        continue
                    if dag.has_edge(X, i) and (j not in skeleton[X]):
                        if dag.has_edge(j, i):
                            logging.info(
                                "Conflict (R1): Evidence suggests %s -> %s, but existing edge is %s -> %s. Retaining existing orientation.",
                                i,
                                j,
                                j,
                                i,
                            )
                        else:
                            dag.add_edge(i, j)
                            logging.info(
                                "Meek R1: Oriented %s -> %s because %s -> %s and %s and %s are not adjacent.",
                                i,
                                j,
                                X,
                                i,
                                X,
                                j,
                            )
                            changed = True
                        break
                if changed:
                    continue

                # --- Rule R2 ---
                for X in nodes:
                    if X in (i, j):
                        continue
                    if dag.has_edge(i, X) and dag.has_edge(X, j):
                        if dag.has_edge(j, i):
                            logging.info(
                                "Conflict (R2): Evidence suggests %s -> %s, but existing edge is %s -> %s. Retaining orientation.",
                                i,
                                j,
                                j,
                                i,
                            )
                        else:
                            dag.add_edge(i, j)
                            logging.info(
                                "Meek R2: Oriented %s -> %s because %s -> %s and %s -> %s.",
                                i,
                                j,
                                i,
                                X,
                                X,
                                j,
                            )
                            changed = True
                        break
                if changed:
                    continue

                # --- Rule R3 ---
                potential = [X for X in nodes if X not in (i, j) and dag.has_edge(X, i)]
                if len(potential) >= 2:
                    for a, b in itertools.combinations(potential, 2):
                        if (b not in skeleton[a]) and (a not in skeleton[b]):
                            if dag.has_edge(j, i):
                                logging.info(
                                    "Conflict (R3): Evidence suggests %s -> %s, but existing edge is %s -> %s. Retaining orientation.",
                                    i,
                                    j,
                                    j,
                                    i,
                                )
                            else:
                                dag.add_edge(i, j)
                                logging.info(
                                    "Meek R3: Oriented %s -> %s because %s and %s both point to %s and are not adjacent.",
                                    i,
                                    j,
                                    a,
                                    b,
                                    i,
                                )
                                changed = True
                            break
                    if changed:
                        continue

                # --- Rule R4 ---
                if nx.has_path(dag, i, j):
                    if dag.has_edge(j, i):
                        logging.info(
                            "Conflict (R4): Evidence suggests %s -> %s, but existing edge is %s -> %s. Retaining orientation.",
                            i,
                            j,
                            j,
                            i,
                        )
                    else:
                        dag.add_edge(i, j)
                        logging.info(
                            "Meek R4: Oriented %s -> %s because a directed path from %s to %s exists.",
                            i,
                            j,
                            i,
                            j,
                        )
                        changed = True
                    continue
                if nx.has_path(dag, j, i):
                    if dag.has_edge(i, j):
                        logging.info(
                            "Conflict (R4): Evidence suggests %s -> %s, but existing edge is %s -> %s. Retaining orientation.",
                            j,
                            i,
                            i,
                            j,
                        )
                    else:
                        dag.add_edge(j, i)
                        logging.info(
                            "Meek R4: Oriented %s -> %s because a directed path from %s to %s exists.",
                            j,
                            i,
                            j,
                            i,
                        )
                        changed = True
                    continue
    return dag


def pc_algorithm(df_obs, previous_data=None, alpha=0.05):
    """
    Build the DAG structure from observational data using the PC algorithm.
    The function builds the skeleton by performing unconditional and conditional chi-square tests,
    identifies v-structures, and applies Meek's rules to orient as many edges as possible.
    Previously established orientations (if provided via previous_data) are preserved.

    Returns:
      dict: Contains:
            - 'dag': The partially directed acyclic graph (nx.DiGraph).
            - 'skeleton': A dictionary mapping each node to its undirected neighbors.
            - 'sep_set': A dictionary storing the conditioning sets used when an edge was removed.
    """
    nodes = list(df_obs.columns)
    logging.info("Observational PC algorithm: Variables in the dataset: %s", nodes)

    # # Log pairwise probability tables.
    # for var1, var2 in itertools.combinations(nodes, 2):
    #     logging.info("Contingency table for %s and %s:", var1, var2)
    #     contingency = pd.crosstab(df_obs[var1], df_obs[var2])
    #     logging.info("\n%s", contingency)
    #     marginals_var1 = df_obs[var1].value_counts(normalize=True)
    #     marginals_var2 = df_obs[var2].value_counts(normalize=True)
    #     logging.info("Marginal probabilities for %s:\n%s", var1, marginals_var1)
    #     logging.info("Marginal probabilities for %s:\n%s", var2, marginals_var2)
    #     joint_prob = contingency / contingency.values.sum()
    #     logging.info(
    #         "Joint probability table for %s and %s:\n%s", var1, var2, joint_prob
    #     )
    #     cond_prob = contingency.div(contingency.sum(axis=0), axis=1)
    #     logging.info(
    #         "Conditional probabilities of %s given %s:\n%s", var1, var2, cond_prob
    #     )
    #     p_val_debug = chi2_test(df_obs, var1, var2, alpha)
    #     logging.info(
    #         "Unconditional independence test between %s and %s yields p-value = %s",
    #         var1,
    #         var2,
    #         p_val_debug,
    #     )

    log_conditional_probabilities(df_obs)

    # --- Use previous_data if available ---
    if previous_data is not None:
        skeleton = {
            node: set(neighbors)
            for node, neighbors in previous_data.get("skeleton", {}).items()
        }
        sep_set = previous_data.get(
            "sep_set", {frozenset([i, j]): set() for i in nodes for j in nodes if i < j}
        )
        dag = previous_data.get("dag", nx.DiGraph())
        if dag is None:
            dag = nx.DiGraph()
            dag.add_nodes_from(nodes)
        else:
            dag = dag.copy()
            for node in nodes:
                if node not in dag:
                    dag.add_node(node)
        logging.info("PC algorithm: Reusing previous observational data as base.")
    else:
        skeleton = {node: set(n for n in nodes if n != node) for node in nodes}
        sep_set = {frozenset([i, j]): set() for i in nodes for j in nodes if i < j}
        dag = nx.DiGraph()
        dag.add_nodes_from(nodes)
        logging.info("PC algorithm: Starting with a complete graph as skeleton.")

    # --- Step 1: Skeleton Build ---
    max_cond_set_size = 1  # You can adjust this depending on your data.
    l = 0
    while l <= max_cond_set_size:
        removal_done = False
        for i in nodes:
            for j in list(skeleton[i]):
                # If the edge is already oriented, run independence tests for logging conflicts, but do not remove it.
                if dag.has_edge(i, j) or dag.has_edge(j, i):
                    if len(skeleton[i] - {j}) >= l:
                        for cond_set in itertools.combinations(skeleton[i] - {j}, l):
                            if l == 0:
                                p_val = chi2_test(df_obs, i, j, alpha)
                            else:
                                p_val = conditional_chi2_test(
                                    df_obs, i, j, cond_set, alpha
                                )
                            if p_val > alpha:
                                logging.info(
                                    "Conflict: Oriented edge (%s -> %s) remains even though test with cond_set %s yields p-value = %s",
                                    i,
                                    j,
                                    cond_set,
                                    p_val,
                                )
                    continue

                if len(skeleton[i] - {j}) >= l:
                    for cond_set in itertools.combinations(skeleton[i] - {j}, l):
                        if l == 0:
                            p_val = chi2_test(df_obs, i, j, alpha)
                            logging.info(
                                "Unconditional test between %s and %s: p-value = %s",
                                i,
                                j,
                                p_val,
                            )
                        else:
                            p_val = conditional_chi2_test(df_obs, i, j, cond_set, alpha)
                            logging.info(
                                "Conditional test between %s and %s given %s: p-value = %s",
                                i,
                                j,
                                cond_set,
                                p_val,
                            )
                        if p_val > alpha:
                            logging.info(
                                "Removing undirected edge (%s, %s) based on independence test.",
                                i,
                                j,
                            )
                            skeleton[i].remove(j)
                            skeleton[j].remove(i)
                            sep_set[frozenset([i, j])] = set(cond_set)
                            removal_done = True
                            break
            # End inner for j loop.
        if removal_done:
            l = 0
        else:
            l += 1
    logging.info("Skeleton after PC Step 1 (Skeleton Build):")
    for node in skeleton:
        logging.info("%s: %s", node, skeleton[node])

    # --- Step 2: V-Structure Identification ---
    for i in nodes:
        for j in nodes:
            if i == j or j in skeleton[i]:
                continue
            common_neighbors = skeleton[i].intersection(skeleton[j])
            for k in common_neighbors:
                if k not in sep_set[frozenset([i, j])]:
                    if not dag.has_edge(i, k) and not dag.has_edge(k, i):
                        dag.add_edge(i, k)
                        logging.info(
                            "V-Structure: Oriented %s -> %s based on common neighbor %s with sep_set %s",
                            i,
                            k,
                            j,
                            sep_set[frozenset([i, j])],
                        )
                    else:
                        if dag.has_edge(k, i):
                            logging.info(
                                "Conflict in V-Structure: Evidence suggests %s -> %s, but existing orientation is %s -> %s. Retaining.",
                                i,
                                k,
                                k,
                                i,
                            )
                    if not dag.has_edge(j, k) and not dag.has_edge(k, j):
                        dag.add_edge(j, k)
                        logging.info(
                            "V-Structure: Oriented %s -> %s based on common neighbor %s with sep_set %s",
                            j,
                            k,
                            i,
                            sep_set[frozenset([i, j])],
                        )
                    else:
                        if dag.has_edge(k, j):
                            logging.info(
                                "Conflict in V-Structure: Evidence suggests %s -> %s, but existing orientation is %s -> %s. Retaining.",
                                j,
                                k,
                                k,
                                j,
                            )
    logging.info(
        "PC algorithm: DAG after V-Structure Identification: Directed edges: %s",
        list(dag.edges()),
    )

    # --- Step 3: Apply Meek's Orientation Rules ---
    dag = apply_meek_rules(dag, skeleton, nodes)
    logging.info(
        "PC algorithm: Final DAG after applying Meek's rules: Directed edges: %s",
        list(dag.edges()),
    )

    # Log any remaining undirected edges.
    for i in nodes:
        for j in skeleton[i]:
            if not dag.has_edge(i, j) and not dag.has_edge(j, i):
                logging.info("Edge (%s, %s) remains undirected.", i, j)

    return {"dag": dag, "skeleton": skeleton, "sep_set": sep_set}


def orient_edges_with_interventions(dag, df_obs, interventions, alpha=0.05):
    """
    Orient edges in the DAG using interventional data.

    For each variable in the interventions dictionary, compare the distribution of every other variable
    (target) between observational and interventional data. If the distribution changes significantly
    (p-value < alpha), orient the edge from the intervened variable to the target.

    In the chi-square test, the expected frequencies (from observational data) are scaled to have the same total
    count as the interventional frequencies.
    In case of conflict (if an edge is oriented in the reverse direction), the conflicting orientation is removed.
    """
    for var, df_int in interventions.items():
        logging.info("Processing intervention on variable: %s", var)
        for target in dag.nodes:
            if var == target:
                continue
            # Get observed (df_obs) and interventional (df_int) counts for the target.
            obs_counts = df_obs[target].value_counts().sort_index()
            int_counts = df_int[target].value_counts().sort_index()
            logging.info(
                "Observational counts for %s: %s", target, obs_counts.to_dict()
            )
            logging.info(
                "Interventional counts for %s: %s", target, int_counts.to_dict()
            )
            # Ensure both distributions cover the same set of categories.
            categories = set(obs_counts.index).union(set(int_counts.index))
            obs_vals = [obs_counts.get(cat, 0) for cat in categories]
            int_vals = [int_counts.get(cat, 0) for cat in categories]
            total_obs = sum(obs_vals)
            total_int = sum(int_vals)
            # Scale expected frequencies (from observational data) to match the total count of intervention.
            scale = total_int / total_obs if total_obs > 0 else 1
            scaled_obs_vals = [val * scale for val in obs_vals]
            try:
                chi2, p_val = stats.chisquare(f_obs=int_vals, f_exp=scaled_obs_vals)
                logging.info(
                    "Chi-square test for %s (interventional vs. scaled observational): p-value = %s",
                    target,
                    p_val,
                )
            except Exception as e:
                logging.info("Chi-square test error for %s: %s", target, e)
                continue
            if p_val < alpha:
                # If significant change is observed, orient the edge from var to target.
                if dag.has_edge(target, var):
                    logging.info(
                        "Intervention conflict: Removing edge %s -> %s because intervention on %s suggests reverse orientation.",
                        target,
                        var,
                        var,
                    )
                    dag.remove_edge(target, var)
                if not dag.has_edge(var, target):
                    dag.add_edge(var, target)
                    logging.info(
                        "Intervention: Oriented %s -> %s based on interventional data.",
                        var,
                        target,
                    )
    return dag


def learn_dag(df_obs, interventions, previous_data=None, alpha=0.05):
    """
    Learn a causal DAG from both observational and interventional data.

    The function first applies the observational PC algorithm (via pc_algorithm) to build an initial DAG structure.
    Then, it refines edge orientations using interventional data (via orient_edges_with_interventions).
    This separation allows for different logics when processing observational versus interventional data.

    Returns:
      dict: Contains the final 'dag' (nx.DiGraph), 'skeleton', and 'sep_set' for reuse in future iterations.
    """
    # Build the initial DAG using observational data.
    pc_result = pc_algorithm(df_obs, previous_data=previous_data, alpha=alpha)
    dag = pc_result["dag"]

    # Update the DAG using interventional data.
    dag = orient_edges_with_interventions(dag, df_obs, interventions, alpha=alpha)

    # Package and return final results.
    pc_result["dag"] = dag
    return pc_result
