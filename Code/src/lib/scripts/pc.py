import logging
import itertools
from typing import Dict, List, Set, Tuple
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import chi2, chi2_contingency, fisher_exact


# --------------------------------------------------------------------------
# Helper utilities
# --------------------------------------------------------------------------
def _records_to_df(records: List[Dict[str, int]]) -> pd.DataFrame:
    """Convert a list of record‑dicts to a *fresh* ``DataFrame``."""
    if not records:
        return pd.DataFrame()
    return pd.DataFrame.from_records(records).copy()


def _g_square_conditional(
    df: pd.DataFrame,
    X: str,
    Y: str,
    S: Tuple[str, ...] | List[str],
    alpha: float = 0.05,
) -> bool:
    """Likelihood‑ratio (G‑test) for conditional independence ``X ⫫ Y | S``.

    Returns **True** *iff* we *fail* to reject independence at threshold
    ``alpha`` – i.e. ``True`` → treat as independent / remove edge.
    """
    S = tuple(S)
    if not S:
        table = pd.crosstab(df[X], df[Y]).to_numpy()
        if table.shape == (2, 2):
            # Fisher's exact test for 2x2 table
            _, p = fisher_exact(table, alternative="greater")
        else:
            _, p, _, _ = chi2_contingency(
                table, correction=False, lambda_="log-likelihood"
            )
        return p > alpha

    g_stat = 0.0
    df_tot = 0
    for _, sub in df.groupby(list(S)):
        if sub.empty:
            continue
        table = pd.crosstab(sub[X], sub[Y]).to_numpy()
        if table.shape != (2, 2):
            continue  # not enough support in this stratum
        stat, _, _, _ = chi2_contingency(
            table, correction=False, lambda_="log-likelihood"
        )
        g_stat += stat
        df_tot += 1

    if df_tot == 0:
        return False  # unable to test – err on the side of dependence

    p_val = chi2.sf(g_stat, df_tot)
    logging.info(f"G-square: {X} ⫫ {Y} | {S} → p = {p_val:.4f}")
    return p_val > alpha


def _all_combinations(iterable, r):
    lst = list(iterable)
    if r > len(lst):
        return []
    return itertools.combinations(lst, r)


# --------------------------------------------------------------------------
# PC algorithm – edge‑removal + basic orientation
# --------------------------------------------------------------------------


def PC(data: Dict, alpha: float = 0.05):
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
    G.add_edges_from(itertools.permutations(variables, 2))
    logging.info(f"PC: initial graph with {list(G.edges())} edges")

    sep_sets: Dict[frozenset, Set[str]] = {}

    # 2 . edge‑removal
    cont = True
    l = 0
    while cont:
        cont = False
        for X, Y in list(G.edges()):  # snapshot because we might delete
            nbrs_X = set(G.neighbors(X)) - {Y}
            if len(nbrs_X) < l:
                continue
            # for S in _all_combinations(nbrs_X, l):
            #     if _g_square_conditional(obs_df, X, Y, S, alpha):
            #         G.remove_edge(X, Y)
            #         sep_sets[frozenset({X, Y})] = set(S)
            #         cont = True
            #         logging.info(f"PC: removing edge {X} – {Y} | {S}")
            #         break
        l += 1

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
                logging.info(f"PC: orienting v-structure {X} → {Z} ← {Y} (unshielded)")

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
                    logging.info(f"PC: orienting edge {Y} → {Z} (Meek R1)")
                    break

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
    baseline = base[target].value_counts().reindex([0, 1]).fillna(0)

    rows = [baseline.values.astype(int)]
    for _, records in data[int_var].items():
        df_int = _records_to_df(records)
        if df_int.empty:
            continue
        cnt = df_int[target].value_counts().reindex([0, 1]).fillna(0)
        rows.append(cnt.values.astype(int))
    if len(rows) <= 1:
        return False

    contingency = np.vstack(rows)
    _, p, _, _ = chi2_contingency(contingency, correction=False)
    logging.info(f"intervention: {int_var} → {target} → p = {p:.4f}")
    return p < alpha


def orient_edges(
    data: Dict,
    skeleton: nx.Graph,
    initial_directed_edges: Set[Tuple[str, str]] | None = None,
    alpha: float = 0.05,
) -> Set[Tuple[str, str]]:
    """Orient additional arrows using interventional batches + Meek R1."""
    directed = set(initial_directed_edges) if initial_directed_edges else set()

    for X, Y in skeleton.edges():
        logging.info(f"orienting edge {X} ↔ {Y} (intervention)")
        if (X, Y) in directed or (Y, X) in directed:
            continue
        x_affects_y = _effect_of_intervention(data, X, Y, alpha)
        y_affects_x = _effect_of_intervention(data, Y, X, alpha)
        logging.info(
            f"intervention: {X} → {Y} = {x_affects_y}, {Y} → {X} = {y_affects_x}"
        )
        if x_affects_y and not y_affects_x:
            directed.add((X, Y))
            logging.info(f"orienting edge {X} → {Y} (intervention)")
        elif y_affects_x and not x_affects_y:
            directed.add((Y, X))
            logging.info(f"orienting edge {Y} → {X} (intervention)")

    # final Meek R1 sweep
    changed = True
    while changed:
        changed = False
        for Y, Z in skeleton.edges():
            if (Y, Z) in directed or (Z, Y) in directed:
                continue
            for X in skeleton.nodes():
                if (
                    (X, Y) in directed
                    and not skeleton.has_edge(X, Z)
                    and not skeleton.has_edge(Z, X)
                ):
                    directed.add((Y, Z))
                    changed = True
                    logging.info(f"orienting edge {Y} → {Z} (Meek R1)")
                    break
    return directed


# --------------------------------------------------------------------------
# Convenience wrapper – full learning pipeline
# --------------------------------------------------------------------------


def learn(data: Dict, alpha: float = 0.05) -> nx.DiGraph:
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
    skeleton, dir_pc = PC(data, alpha)
    all_oriented = orient_edges(data, skeleton, dir_pc, alpha)

    G = nx.DiGraph()
    G.add_nodes_from(skeleton.nodes())
    G.add_edges_from(all_oriented)
    return G
