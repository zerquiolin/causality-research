import logging
import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats
import itertools

# Configure logging for detailed output.
logging.basicConfig(
    level=logging.info, format="%(asctime)s - %(levelname)s - %(message)s"
)


def chi2_test(data, X, Y, alpha=0.05):
    """
    Unconditional chi-square test for independence between X and Y.
    Constructs a contingency table from data using columns X and Y.
    Returns the p-value (or 1.0 if one variable has only one category).
    """
    contingency = pd.crosstab(data[X], data[Y])
    if contingency.shape[0] < 2 or contingency.shape[1] < 2:
        return 1.0
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    return p


def conditional_chi2_test(data, X, Y, cond_vars, alpha=0.05):
    """
    Conditional chi-square test for independence between X and Y given conditioning variables.
    Splits data into groups defined by cond_vars and returns the minimum p-value across groups
    (or 1.0 if data is insufficient).
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
    return min(p_values) if p_values else 1.0


def log_conditional_probabilities(df):
    """
    If exactly 3 variables are present (e.g., A, B, C), logs for each pair (a, b)
    the conditional probabilities given the remaining variable c.
    For each value of c, the following are logged:
        - The contingency table for (a, b) given c.
        - The observed conditional joint probability P(a, b | c).
        - The conditional marginals P(a | c) and P(b | c).
        - The expected joint probability under conditional independence: P(a | c) * P(b | c).
        - The result of a chi-square test for conditional independence.
    """
    nodes = list(df.columns)
    if len(nodes) != 3:
        return
    combos = [
        ((nodes[0], nodes[1]), nodes[2]),
        ((nodes[0], nodes[2]), nodes[1]),
        ((nodes[1], nodes[2]), nodes[0]),
    ]
    for (a, b), c in combos:
        logging.info(
            "=== Analyzing conditional probabilities P(%s, %s | %s) ===", a, b, c
        )
        for c_val in sorted(df[c].unique()):
            subset = df[df[c] == c_val]
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
            logging.info("For condition %s = %s:", c, c_val)
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
                chi2, p, dof, expected = stats.chi2_contingency(contingency_ab)
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
    Iteratively apply Meek's orientation rules (R1–R4) to orient undirected edges in the DAG.
    Only edges present in the skeleton (and not yet oriented in dag) are considered.
    If new evidence conflicts with an already established orientation, the previous orientation is preserved.
    """
    changed = True
    while changed:
        changed = False
        for i in nodes:
            for j in skeleton[i]:
                if dag.has_edge(i, j) or dag.has_edge(j, i):
                    continue  # Skip if already oriented.
                # --- Rule R1 ---
                for X in nodes:
                    if X in (i, j):
                        continue
                    if dag.has_edge(X, i) and (j not in skeleton[X]):
                        if dag.has_edge(j, i):
                            logging.info(
                                "Conflict (R1): Evidence suggests %s -> %s, but existing orientation %s -> %s retained.",
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
                                "Conflict (R2): Evidence suggests %s -> %s, but existing orientation %s -> %s retained.",
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
                                    "Conflict (R3): Evidence suggests %s -> %s, but existing orientation %s -> %s retained.",
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
                            "Conflict (R4): Evidence suggests %s -> %s but existing orientation %s -> %s retained.",
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
                            "Conflict (R4): Evidence suggests %s -> %s but existing orientation %s -> %s retained.",
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


def learn_dag(df, previous_data=None, alpha=0.05):
    """
    Learn a causal DAG from a single dataset (which might contain observational and interventional samples),
    using a PC algorithm–like procedure with optional past data.

    The procedure is:
      1. Log triple-based conditional probabilities if exactly 3 variables are present.
      2. Build an undirected skeleton using unconditional and conditional chi-square tests.
         (Edges are removed if the test suggests independence, but already oriented edges are preserved.)
      3. Identify v-structures: For each pair of non-adjacent nodes sharing a common neighbor (and not separated by that neighbor),
         orient the edges toward the common neighbor.
      4. Apply Meek's rules to orient as many remaining undirected edges as possible.

    The function does not use edge confidence scores, nor does it take a separate interventions parameter.
    It returns a dictionary containing:
         - 'dag': the resulting (partially) directed acyclic graph (nx.DiGraph)
         - 'skeleton': a dictionary mapping nodes to their undirected neighbors
         - 'sep_set': a dictionary storing the conditioning sets used when removing edges
    This output can be passed as previous_data in subsequent iterations.
    """
    nodes = list(df.columns)
    logging.info("learn_dag: Variables in the dataset: %s", nodes)

    # --- Log triple-based conditional probabilities (only if exactly 3 variables) ---
    log_conditional_probabilities(df)

    # --- Initialize or reuse previous_data ---
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
        logging.info("learn_dag: Reusing previous data as base.")
    else:
        skeleton = {node: set(n for n in nodes if n != node) for node in nodes}
        sep_set = {frozenset([i, j]): set() for i in nodes for j in nodes if i < j}
        dag = nx.DiGraph()
        dag.add_nodes_from(nodes)
        logging.info("learn_dag: Starting with a complete graph as skeleton.")

    # --- Step 1: Skeleton Build ---
    max_cond_set_size = 1  # Adjust as needed.
    l = 0
    while l <= max_cond_set_size:
        removal_done = False
        for i in nodes:
            for j in list(skeleton[i]):
                # If the edge is already oriented in the DAG, run tests for logging conflicts but do not remove.
                if dag.has_edge(i, j) or dag.has_edge(j, i):
                    if len(skeleton[i] - {j}) >= l:
                        for cond_set in itertools.combinations(skeleton[i] - {j}, l):
                            p_val = (
                                chi2_test(df, i, j, alpha)
                                if l == 0
                                else conditional_chi2_test(df, i, j, cond_set, alpha)
                            )
                            if p_val > alpha:
                                logging.info(
                                    "Conflict: Oriented edge (%s -> %s) remains even though test with cond_set %s yields p-value = %s.",
                                    i,
                                    j,
                                    cond_set,
                                    p_val,
                                )
                    continue

                if len(skeleton[i] - {j}) >= l:
                    for cond_set in itertools.combinations(skeleton[i] - {j}, l):
                        p_val = (
                            chi2_test(df, i, j, alpha)
                            if l == 0
                            else conditional_chi2_test(df, i, j, cond_set, alpha)
                        )
                        logging.info(
                            "Test between %s and %s with cond_set %s yields p-value = %s",
                            i,
                            j,
                            cond_set,
                            p_val,
                        )
                        if p_val > alpha:
                            logging.info(
                                "Removing undirected edge (%s, %s) based on independence test (p-value = %s).",
                                i,
                                j,
                                p_val,
                            )
                            skeleton[i].remove(j)
                            skeleton[j].remove(i)
                            sep_set[frozenset([i, j])] = set(cond_set)
                            removal_done = True
                            break
            # End loop for j in skeleton[i]
        if removal_done:
            l = 0  # Reset conditioning set size if any edge was removed.
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
                    if not dag.has_edge(i, k) and not dag.has_edge(k, i):
                        dag.add_edge(i, k)
                        logging.info(
                            "V-Structure: Oriented %s -> %s based on common neighbor %s (sep_set %s).",
                            i,
                            k,
                            j,
                            sep_set[frozenset([i, j])],
                        )
                    else:
                        if dag.has_edge(k, i):
                            logging.info(
                                "Conflict in V-Structure: Evidence suggests %s -> %s but orientation %s -> %s retained.",
                                i,
                                k,
                                k,
                                i,
                            )
                    if not dag.has_edge(j, k) and not dag.has_edge(k, j):
                        dag.add_edge(j, k)
                        logging.info(
                            "V-Structure: Oriented %s -> %s based on common neighbor %s (sep_set %s).",
                            j,
                            k,
                            i,
                            sep_set[frozenset([i, j])],
                        )
                    else:
                        if dag.has_edge(k, j):
                            logging.info(
                                "Conflict in V-Structure: Evidence suggests %s -> %s but orientation %s -> %s retained.",
                                j,
                                k,
                                k,
                                j,
                            )
    logging.info(
        "DAG after V-Structure Identification: Directed edges: %s", list(dag.edges())
    )

    # --- Step 3: Apply Meek's Orientation Rules ---
    dag = apply_meek_rules(dag, skeleton, nodes)
    logging.info(
        "Final DAG after applying Meek's rules: Directed edges: %s", list(dag.edges())
    )

    for i in nodes:
        for j in skeleton[i]:
            if not dag.has_edge(i, j) and not dag.has_edge(j, i):
                logging.info("Edge (%s, %s) remains undirected.", i, j)

    return {"dag": dag, "skeleton": skeleton, "sep_set": sep_set}
