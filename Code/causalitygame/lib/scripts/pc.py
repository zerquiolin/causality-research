import logging
import itertools
from typing import Dict, List, Set, Tuple
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, combine_pvalues
import pandas as pd
from scipy.stats import pearsonr


# --------------------------------------------------------------------------
# Helper utilities
# --------------------------------------------------------------------------
def _records_to_df(records: List[Dict[str, int]]) -> pd.DataFrame:
    """Convert a list of record‑dicts to a *fresh* ``DataFrame``."""
    if not records:
        return pd.DataFrame()
    return pd.DataFrame.from_records(records).copy()


def ci_test_discrete(df, X, Y, Z, alpha=0.05):
    """Test X⊥Y | Z for discrete variables via stratified chi-square."""
    # No conditioning
    if not Z:
        table = pd.crosstab(df[X], df[Y])
        _, p, _, _ = chi2_contingency(table)
        logging.debug(f"Conditional independence test: {X} _||_ {Y} → p = {p:.4f}")
        return p > alpha

    # Stratify by every combination of Z
    pvals = []
    for _, sub in df.groupby(Z):
        # skip degenerate strata
        if sub[X].nunique() < 2 or sub[Y].nunique() < 2:
            continue
        table = pd.crosstab(sub[X], sub[Y])
        _, p, _, _ = chi2_contingency(table)
        pvals.append(p)

    if not pvals:
        # unable to test (e.g. constant strata)—treat as independent
        return True

    _, p_comb = combine_pvalues(pvals, method="fisher")

    logging.debug(
        f"Conditional independence test: {X} _||_ {Y} | {Z} → p = {p_comb:.4f}"
    )
    return p_comb > alpha


def ci_test(
    df: pd.DataFrame,
    X: str,
    Y: str,
    Z: Tuple[str, ...] | List[str],
    alpha: float = 0.05,
) -> bool:
    """
    Test conditional independence X _||_ Y | Z using (partial) correlation.
    Returns True if independent at level alpha.
    """
    # No conditioning variables: simple Pearson correlation
    if len(Z) == 0:
        r, p = pearsonr(df[X], df[Y])
    else:
        # Regress out Z from X and Y to get residuals
        def residuals(target: str) -> np.ndarray:
            X_mat = df[list(Z)].values
            y_vec = df[target].values
            coef, _, _, _ = np.linalg.lstsq(X_mat, y_vec, rcond=None)
            return y_vec - X_mat.dot(coef)

        rx = residuals(X)
        ry = residuals(Y)
        r, p = pearsonr(rx, ry)

    logging.debug(f"Conditional independence test: {X} _||_ {Y} | {Z} → p = {p:.4f}")
    return p > alpha


def _all_combinations(iterable, r):
    lst = list(iterable)
    if r > len(lst):
        return []
    return itertools.combinations(lst, r)


def _all_permutations(iterable, r):
    lst = list(iterable)
    if r > len(lst):
        return []
    return itertools.permutations(lst, r)


# --------------------------------------------------------------------------
# PC algorithm – edge‑removal + basic orientation
# --------------------------------------------------------------------------


def PC(data: Dict, isNumerical: bool, alpha: float = 0.05):
    """Run the PC algorithm on *observational* rows ``data['empty']``.

    Returns
    -------
    skeleton : ``nx.Graph``
        Undirected graph after conditional‑independence pruning.
    directed_edges : ``set[(tail, head)]``
        Oriented arrows found by v‑structure detection and the first
        Meek propagation rule (R1).
    """
    obs_df = _records_to_df(data["empty"])
    variables = list(obs_df.columns)

    # 1 . start with a complete graph
    G = nx.DiGraph()
    G.add_nodes_from(variables)
    G.add_edges_from(_all_permutations(variables, 2))
    logging.debug(f"PC: initial graph with {list(G.edges())} edges")

    sep_sets: Dict[frozenset, Set[str]] = {}

    # compute the largest possible conditioning‐set size up front
    max_l = (
        max(len(set(G.neighbors(X)) - {Y}) for X, Y in G.edges()) if G.edges() else 0
    )

    # 2. edge-removal
    logging.debug(f"max_l: {max_l}")

    sep_sets = {}  # make sure this is initialized before

    for l in range(max_l + 1):
        removed = True
        # keep going at this l until no more edges can be removed
        while removed:
            removed = False
            # snapshot edges because we'll be mutating G
            for X, Y in list(G.edges()):
                nbrs_X = set(G.neighbors(X)) - {Y}
                if len(nbrs_X) < l:
                    continue
                # test _all_ subsets of size l
                for Z in _all_combinations(nbrs_X, l):
                    conditional = (
                        ci_test_discrete(df=obs_df, X=X, Y=Y, Z=list(Z))
                        if not isNumerical
                        else ci_test(df=obs_df, X=X, Y=Y, Z=list(Z))
                    )
                    if conditional:
                        G.remove_edge(X, Y)
                        sep_sets[frozenset({X, Y})] = set(Z)
                        logging.debug(f"PC: removing edge {X} - {Y} | {Z}")
                        removed = True
                        break  # stop searching subsets for this edge
            # end for edges
        # end while removed at this l
    # end for l

    skeleton = G.copy()

    # 3 . orient v‑structures:  X → Z ← Y (unshielded colliders)
    directed_edges: Set[Tuple[str, str]] = set()
    for Z in variables:
        adj = list(skeleton.neighbors(Z))
        for X, Y in itertools.combinations(adj, 2):
            if skeleton.has_edge(X, Y):
                continue  # shielded triangle
            if Z not in sep_sets.get(frozenset({X, Y}), set()):
                directed_edges.update({(X, Z), (Y, Z)})
                logging.debug(f"PC: orienting v-structure {X} → {Z} ← {Y} (unshielded)")

    # 4 . Meek Rule R1: (X → Y –– Z) and X ∦ Z ⇒ orient Y → Z
    changed = True
    while changed:
        changed = False
        for Y, Z in skeleton.edges():
            if (Y, Z) in directed_edges or (Z, Y) in directed_edges:
                continue
            for X in (
                skeleton.predecessors(Y) if isinstance(skeleton, nx.DiGraph) else []
            ):
                pass  # never executed – placeholder to hush linters
            # simpler explicit search
            for X in skeleton.nodes():
                if (X, Y) in directed_edges and not (
                    skeleton.has_edge(X, Z) or skeleton.has_edge(Z, X)
                ):
                    directed_edges.add((Y, Z))
                    changed = True
                    logging.debug(f"PC: orienting edge {Y} → {Z} (Meek R1)")
                    break

    logging.debug(f"PC: final skeleton {list(skeleton.edges())}")
    logging.debug(f"PC: final directed edges {directed_edges}")

    return skeleton, directed_edges


# --------------------------------------------------------------------------
# Orientation from interventions
# --------------------------------------------------------------------------


def _effect_of_intervention(
    data: Dict, int_var: str, target: str, alpha: float
) -> bool:
    """χ² test whether *do(int_var=⋅)* changes *target*'s marginal."""
    if int_var == "empty" or int_var not in data:
        return False

    base = _records_to_df(data["empty"])
    if base.empty:
        return False
    baseline = base[target].value_counts().reindex([0, 1])

    for inter_value, records in data[int_var].items():
        logging.debug(f"intervention: {int_var} = {inter_value}")
        df_int = _records_to_df(records)
        if df_int.empty:
            continue
        cnt = df_int[target].value_counts().reindex([0, 1])

        logging.debug(f"observational cnts: {baseline.values}")
        logging.debug(f"counts: {cnt.values}")

        a = baseline.values.astype("float64")
        a /= a.sum()
        b = cnt.values.astype("float64")
        b /= b.sum()

        total_variance_distance = abs(a - b).sum() / 2

        logging.debug(
            f"intervention: {int_var} = {inter_value} → {target} p = {total_variance_distance:.4f}"
        )
        if total_variance_distance > alpha:
            return True

    return False


def _create_edge_blacklist_and_path_whitelist(
    data: Dict,
    skeleton: nx.Graph,
    initial_directed_edges: Set[Tuple[str, str]] | None = None,
    alpha: float = 0.05,
) -> Set[Tuple[str, str]]:
    """Orient additional arrows using interventional batches + Meek R1."""
    blacklist = set()
    whitelist = set()
    directed = set(initial_directed_edges) if initial_directed_edges else set()

    for X, Y in skeleton.edges():
        logging.debug(f"orienting edge {X} ↔ {Y} (intervention)")
        if (X, Y) in directed or (Y, X) in directed:
            continue

        x_affects_y = _effect_of_intervention(data, X, Y, alpha)
        y_affects_x = _effect_of_intervention(data, Y, X, alpha)

        logging.debug(
            f"intervention: {X} → {Y} = {x_affects_y}, {Y} → {X} = {y_affects_x}"
        )
        if not x_affects_y:
            blacklist.add((X, Y))
            logging.debug(f"Black listing edge {X} → {Y} (intervention)")
        else:
            whitelist.add((X, Y))
            logging.debug(f"White listing edge {X} → {Y} (intervention)")

        if not y_affects_x:
            blacklist.add((Y, X))
            logging.debug(f"Black listing edge {Y} → {X} (intervention)")
        else:
            whitelist.add((Y, X))
            logging.debug(f"White listing edge {Y} → {X} (intervention)")

    return blacklist


def _clean_skeleton_with_blacklist(
    skeleton: nx.Graph, blacklist: Set[Tuple[str, str]], directed: Set[Tuple[str, str]]
) -> nx.Graph:
    """Remove edges from skeleton that are in the blacklist."""
    G = skeleton.copy()
    for X, Y in blacklist:
        if G.has_edge(X, Y):
            G.remove_edge(X, Y)
            logging.debug(f"Removing edge {X} ↔ {Y} from skeleton (blacklist)")
    return G


# --------------------------------------------------------------------------
# Convenience wrapper – full learning pipeline
# --------------------------------------------------------------------------
def remove_cycles_from_digraph(graph: nx.DiGraph, seed: int) -> nx.DiGraph:
    """
    Removes cycles from a directed graph by deleting one edge per cycle detected.

    Args:
        graph (nx.DiGraph): A potentially cyclic directed graph.
        seed (int): Random seed for reproducibility.

    Returns:
        nx.DiGraph: An acyclic version of the graph.
    """
    rs = np.random.RandomState(seed)
    g = graph.copy()
    try:
        while True:
            cycle = nx.find_cycle(g, orientation="original")
            # Remove one edge from the cycle randomly
            rid = rs.choice(len(cycle))
            edge_to_remove = cycle[rid][:2]  # (source, target)
            logging.info(f"Removing edge {edge_to_remove} to break cycle.")
            g.remove_edge(*edge_to_remove)
    except nx.NetworkXNoCycle:
        pass  # No more cycles

    return g


def learn(
    data: Dict, isNumerical: bool, alpha: float = 0.05, seed: int = 911
) -> nx.DiGraph:
    """End‑to‑end learner: observational PC → interventional orientation.

    Parameters
    ----------
    data : dict
        Master data structure described in the prompt.
    alpha : float, default 0.05
        Significance level for all statistical tests.

    Returns
    -------
    G : ``networkx.DiGraph``
        Directed graph with the edges that could be unequivocally
        oriented.  Ambiguous skeleton edges are **omitted**.  (If you
        prefer to keep them, add them as bidirectional pairs.)
    """
    skeleton, dir_pc = PC(data, alpha, isNumerical)
    blacklist = _create_edge_blacklist_and_path_whitelist(data, skeleton, dir_pc, alpha)
    bl_skeleton = _clean_skeleton_with_blacklist(skeleton, blacklist, dir_pc)

    logging.debug(f"PC directed edges: {dir_pc}")
    logging.debug(f"bl_skeleton: {bl_skeleton.edges()}")
    logging.debug(f"blacklist: {blacklist}")
    logging.debug(f"Oriented skeleton: {list(bl_skeleton.edges()) + list(dir_pc)}")

    G = nx.DiGraph()
    G.add_nodes_from(skeleton.nodes())
    G.add_edges_from(list(bl_skeleton.edges()) + list(dir_pc))

    # todo: fix this
    # Remove cycles from the directed graph (Just temporay solution)
    G = remove_cycles_from_digraph(graph=G, seed=seed)
    return G
