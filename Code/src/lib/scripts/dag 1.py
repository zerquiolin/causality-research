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
    # logging.info(f"Contingency table for {X} and {Y}:\n{contingency}")
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return 1.0
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    return p


def conditional_chi2_test(data, X, Y, cond_vars, alpha=0.05):
    """
    Perform a conditional chi-square test of independence between X and Y given conditioning variables.

    The data is split based on unique configurations of `cond_vars` and a chi-square test is computed on each subgroup.
    Returns the minimum p-value across all subgroups (or 1.0 if no valid test is run).
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


def apply_meek_rules(dag, skeleton, nodes):
    """
    Iteratively apply Meek's orientation rules (R1–R4) to the partially directed graph.

    Only undirected edges (i.e. those present in the skeleton but not yet oriented in `dag`)
    are considered. The rules are applied repeatedly until no further orientations can be made.

    Rules implemented:
      - R1: If X → i and X is not adjacent to j (in the skeleton), then orient i–j as i → j.
      - R2: If i → X and X → j, then orient i–j as i → j.
      - R3: If there exist two distinct vertices X and Y such that X → i and Y → i,
             and X and Y are not adjacent (in the skeleton), then orient i–j as i → j.
      - R4: If a directed path from i to j exists, then orient i–j as i → j.

    In case new evidence conflicts with an already oriented edge, a message is logged and the existing orientation is retained.
    """
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
                        # If already oriented in the opposite way, log a conflict.
                        if dag.has_edge(j, i):
                            logging.info(
                                "Conflict detected (R1): Evidence suggests %s -> %s, "
                                "but existing edge is %s -> %s. Retaining existing orientation.",
                                i,
                                j,
                                j,
                                i,
                            )
                        else:
                            dag.add_edge(i, j)
                            logging.info(
                                "Meek R1: Oriented %s -> %s because %s -> %s and %s & %s are not adjacent.",
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
                                "Conflict detected (R2): Evidence suggests %s -> %s, "
                                "but existing edge is %s -> %s. Retaining existing orientation.",
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
                                    "Conflict detected (R3): Evidence suggests %s -> %s, "
                                    "but existing edge is %s -> %s. Retaining existing orientation.",
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
                            "Conflict detected (R4): Evidence suggests %s -> %s, "
                            "but existing edge is %s -> %s. Retaining existing orientation.",
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
                            "Conflict detected (R4): Evidence suggests %s -> %s, "
                            "but existing edge is %s -> %s. Retaining existing orientation.",
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


def learn_dag(df_obs, previous_data=None, alpha=0.05):
    """
    Learn a causal Directed Acyclic Graph (DAG) from observational data using a PC algorithm–like procedure
    enhanced with Meek's orientation rules. Already oriented edges (from previous_data) are preserved, and
    if new evidence suggests removal of an edge that is already oriented (or that its orientation is inverted),
    a conflict is logged and the existing edge is kept.

    The procedure involves:
      1. Skeleton Build:
         - If previous data is available, reuse its skeleton and separating sets; otherwise, start with a complete graph.
         - Remove edges based on unconditional and conditional independence tests.
         - Before testing/removing an edge, if the edge is already oriented, run the tests and log conflicts but do not remove the edge.
         - Log pairwise probability tables for all variable pairs.
      2. V-Structure Identification:
         - For every pair of non-adjacent nodes sharing a common neighbor, orient the edge toward that neighbor if that neighbor
           is not in the corresponding separating set.
         - In case of conflict with an already oriented edge, log the conflict and retain the existing orientation.
      3. Meek's Edge Orientation:
         - Iteratively apply Meek’s rules (R1–R4) to orient as many remaining undirected edges as possible.

    Additional Debug (for exactly 3 variables):
      For each combination (a, b | c), for each value of c, compute and log:
         • The contingency table for a and b conditioned on c.
         • The observed joint probability P(a, b | c).
         • The conditional marginals P(a | c) and P(b | c).
         • The expected joint probability under conditional independence: P(a | c) · P(b | c).
         • The chi-square test result for conditional independence.

    Returns:
      dict: Contains the updated 'dag' (nx.DiGraph), 'skeleton' (undirected adjacencies), and 'sep_set' (separating sets).
    """
    nodes = list(df_obs.columns)
    logging.info("Variables in the dataset: %s", nodes)

    # # --- Log probability tables for all pairs of variables ---
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

    # --- Additional Debug for exactly 3 variables ---
    # if len(nodes) == 3:
    #     combos = [
    #         ((nodes[0], nodes[1]), nodes[2]),
    #         ((nodes[0], nodes[2]), nodes[1]),
    #         ((nodes[1], nodes[2]), nodes[0]),
    #     ]
    #     for (a, b), c in combos:
    #         logging.info("Analyzing conditional probabilities P(%s, %s | %s):", a, b, c)
    #         for c_val in sorted(df_obs[c].unique()):
    #             subset = df_obs[df_obs[c] == c_val]
    #             contingency_ab = pd.crosstab(subset[a], subset[b])
    #             total = contingency_ab.values.sum()
    #             if total > 0:
    #                 joint_ab = contingency_ab / total
    #             else:
    #                 joint_ab = contingency_ab.copy()
    #             marg_a_given_c = subset[a].value_counts(normalize=True)
    #             marg_b_given_c = subset[b].value_counts(normalize=True)
    #             expected_joint = pd.DataFrame(
    #                 index=joint_ab.index, columns=joint_ab.columns
    #             )
    #             for i in joint_ab.index:
    #                 for j in joint_ab.columns:
    #                     expected_joint.loc[i, j] = marg_a_given_c.get(
    #                         i, 0
    #                     ) * marg_b_given_c.get(j, 0)
    #             expected_joint = expected_joint.astype(float)
    #             logging.info("Condition: %s = %s", c, c_val)
    #             logging.info(
    #                 "Contingency table for %s and %s:\n%s", a, b, contingency_ab
    #             )
    #             logging.info(
    #                 "Observed joint probability P(%s, %s | %s=%s):\n%s",
    #                 a,
    #                 b,
    #                 c,
    #                 c_val,
    #                 joint_ab,
    #             )
    #             logging.info(
    #                 "Conditional marginal P(%s | %s=%s):\n%s",
    #                 a,
    #                 c,
    #                 c_val,
    #                 marg_a_given_c,
    #             )
    #             logging.info(
    #                 "Conditional marginal P(%s | %s=%s):\n%s",
    #                 b,
    #                 c,
    #                 c_val,
    #                 marg_b_given_c,
    #             )
    #             logging.info(
    #                 "Expected joint probability (P(%s|%s)*P(%s|%s)):\n%s",
    #                 a,
    #                 c,
    #                 b,
    #                 c,
    #                 expected_joint,
    #             )
    #             try:
    #                 chi2, p, dof, expected = stats.chi2_contingency(contingency_ab)
    #                 logging.info(
    #                     "Chi-square test for %s and %s given %s=%s yields p-value = %s",
    #                     a,
    #                     b,
    #                     c,
    #                     c_val,
    #                     p,
    #                 )
    #             except Exception as e:
    #                 logging.info(
    #                     "Chi-square test error for %s and %s given %s=%s: %s",
    #                     a,
    #                     b,
    #                     c,
    #                     c_val,
    #                     e,
    #                 )

    # --- Use Previous Data if available ---
    if previous_data is not None:
        # Reuse the previous skeleton and separating sets (copy to avoid side effects)
        skeleton = {
            node: set(neighbors)
            for node, neighbors in previous_data.get("skeleton", {}).items()
        }
        sep_set = previous_data.get(
            "sep_set", {frozenset([i, j]): set() for i in nodes for j in nodes if i < j}
        )
        # Reuse the previously directed DAG.
        dag = previous_data.get("dag", nx.DiGraph())
        if dag is None:
            dag = nx.DiGraph()
            dag.add_nodes_from(nodes)
        else:
            dag = dag.copy()
            for node in nodes:
                if node not in dag:
                    dag.add_node(node)
        logging.info("Using previous calculated data as base.")
    else:
        # Start with a complete graph as the skeleton.
        skeleton = {node: set(n for n in nodes if n != node) for node in nodes}
        sep_set = {frozenset([i, j]): set() for i in nodes for j in nodes if i < j}
        dag = nx.DiGraph()
        dag.add_nodes_from(nodes)
        logging.info("Starting with a complete graph as skeleton.")

    # --- Step 1: Skeleton Build ---
    max_cond_set_size = 1  # Adjust this value as needed.
    l = 0
    while l <= max_cond_set_size:
        removal_done = False
        for i in nodes:
            for j in list(skeleton[i]):
                # If the edge is already oriented, run tests (for logging conflicts) but do not remove it.
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
                                    "Conflict: Oriented edge (%s -> %s) remains despite test with cond_set %s yielding p-value = %s, suggesting independence.",
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
            # End for j in list(skeleton[i])
        if removal_done:
            l = 0  # Reset conditioning set size after a removal.
        else:
            l += 1

    logging.info("Skeleton after Step 1 (Skeleton Build):")
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
                    # If an edge is already oriented, check for conflict and retain orientation.
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
                                "Conflict in V-Structure: Evidence suggests %s -> %s, but existing orientation is %s -> %s. Retaining existing orientation.",
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
                                "Conflict in V-Structure: Evidence suggests %s -> %s, but existing orientation is %s -> %s. Retaining existing orientation.",
                                j,
                                k,
                                k,
                                j,
                            )
    logging.info(
        "DAG after Step 2 (V-Structure Identification): Directed edges: %s",
        list(dag.edges()),
    )

    # --- Step 3: Apply Meek's Orientation Rules ---
    dag = apply_meek_rules(dag, skeleton, nodes)
    logging.info(
        "Final DAG after applying Meek's rules: Directed edges: %s", list(dag.edges())
    )

    # Log any remaining undirected edges.
    for i in nodes:
        for j in skeleton[i]:
            if not dag.has_edge(i, j) and not dag.has_edge(j, i):
                logging.info("Edge (%s, %s) remains undirected.", i, j)

    current_calculated_data = {"dag": dag, "skeleton": skeleton, "sep_set": sep_set}
    return current_calculated_data
