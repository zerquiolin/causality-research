import numpy as np
import networkx as nx
import networkx as nx
from causalitygame.scm.dag import DAG
from sklearn.linear_model import LogisticRegression
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional, Set, Union


class DatasetManager:
    """
    Manages loading, validating, and converting observational and interventional datasets.

    Accepts nested structure:
      {
        'empty': [ {col: val, ...}, ... ],
        'A': {'0': [...], '1': [...]},
        ...
      }
    Converts list-of-dicts into numpy arrays, inferring consistent column order from 'empty'.
    """

    def __init__(
        self,
        env_data: Dict[
            str,
            Union[
                List[Dict[str, Union[int, np.integer]]],
                Dict[str, List[Dict[str, Union[int, np.integer]]]],
            ],
        ],
        merge_values: bool = False,
    ):
        flat_env: Dict[str, List[Dict[str, Union[int, np.integer]]]] = {}
        for key, val in env_data.items():
            if key == "empty":
                flat_env["empty"] = val if isinstance(val, list) else []
            elif isinstance(val, dict):
                for v, recs in val.items():
                    flat_env[f"{key}={v}"] = recs
            else:
                flat_env[key] = val
        first = flat_env.get("empty")
        if not first:
            raise ValueError("Observational 'empty' environment is required.")
        self.var_names = list(first[0].keys())
        self.env_data: Dict[str, np.ndarray] = {}
        for env, records in flat_env.items():
            arr = np.zeros((len(records), len(self.var_names)), dtype=int)
            for i, rec in enumerate(records):
                for j, col in enumerate(self.var_names):
                    if col not in rec:
                        raise KeyError(f"Missing column '{col}' in env '{env}' record.")
                    arr[i, j] = int(rec[col])
            self.env_data[env] = arr
        if merge_values:
            self._merge()
        self._validate()

    def _merge(self):
        merged = {"empty": [self.env_data["empty"]]}
        for env, arr in self.env_data.items():
            if env == "empty" or "=" not in env:
                continue
            var = env.split("=")[0]
            merged.setdefault(var, []).append(arr)
        new_env = {"empty": np.vstack(merged["empty"])}
        for var, arrs in merged.items():
            if var == "empty":
                continue
            new_env[var] = np.vstack(arrs)
        self.env_data = new_env

    def _validate(self):
        sizes = {arr.shape[1] for arr in self.env_data.values()}
        if len(sizes) != 1:
            raise ValueError("All environments must share same number of variables.")
        for env, arr in self.env_data.items():
            if not np.array_equal(arr, arr.astype(bool)):
                raise ValueError(f"Non-binary values in environment '{env}'.")

    def get_envs(self) -> List[str]:
        return sorted(self.env_data.keys())

    def get_data(self, env: str) -> np.ndarray:
        return self.env_data[env]

    def get_var_index(self, var: str) -> int:
        return self.var_names.index(var)


class ScoreCalculator:
    """
    Computes local and global interventional BIC scores with caching.
    """

    def __init__(self, datasets: DatasetManager, penalty_weight: float = 1.0):
        self.datasets = datasets
        self.penalty_weight = penalty_weight
        self._cache: Dict[Tuple[int, Tuple[int, ...], str], float] = {}

    def _env_target(self, env: str) -> Optional[int]:
        if env == "empty" or "=" not in env:
            return None
        var = env.split("=")[0]
        return self.datasets.get_var_index(var)

    def local_score(self, child: int, parents: Tuple[int, ...], env: str) -> float:
        key = (child, parents, env)
        if key in self._cache:
            return self._cache[key]
        data = self.datasets.get_data(env)
        target = self._env_target(env)
        if target == child:
            score = 0.0
        else:
            y = data[:, child]
            if not parents:
                p = np.clip(np.mean(y), 1e-6, 1 - 1e-6)
                loglik = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
            else:
                X = data[:, parents]
                model = LogisticRegression(
                    penalty="l2", C=1e12, solver="lbfgs", max_iter=100
                )
                model.fit(X, y)
                lp = model.predict_log_proba(X)
                loglik = np.sum(y * lp[:, 1] + (1 - y) * lp[:, 0])
            penalty = (len(parents) / 2) * np.log(data.shape[0]) * self.penalty_weight
            score = loglik - penalty
        self._cache[key] = score
        return score

    def score_gain_insertion(self, u: int, v: int, graph: "CausalGraph") -> float:
        gain = 0.0
        for env in self.datasets.get_envs():
            if self._env_target(env) == v:
                continue
            old = tuple(sorted(graph.parents_of(v)))
            new = tuple(sorted(old + (u,)))
            gain += self.local_score(v, new, env) - self.local_score(v, old, env)
        return gain

    def score_gain_deletion(self, u: int, v: int, graph: "CausalGraph") -> float:
        gain = 0.0
        for env in self.datasets.get_envs():
            if self._env_target(env) == v:
                continue
            old = tuple(sorted(graph.parents_of(v)))
            new = tuple(p for p in old if p != u)
            gain += self.local_score(v, new, env) - self.local_score(v, old, env)
        return gain

    def score_gain_reversal(self, u: int, v: int, graph: "CausalGraph") -> float:
        return self.score_gain_deletion(u, v, graph) + self.score_gain_insertion(
            v, u, graph
        )


class CausalGraph:
    """
    Directed graph with cycle utilities.
    """

    def __init__(self, num_vars: int):
        self._g = nx.DiGraph()
        self._g.add_nodes_from(range(num_vars))

    def nodes(self) -> List[int]:
        return list(self._g.nodes)

    def edges(self) -> List[Tuple[int, int]]:
        return list(self._g.edges)

    def parents_of(self, node: int) -> Set[int]:
        return set(self._g.predecessors(node))

    def has_edge(self, u: int, v: int) -> bool:
        return self._g.has_edge(u, v)

    def add_edge(self, u: int, v: int):
        self._g.add_edge(u, v)

    def remove_edge(self, u: int, v: int):
        self._g.remove_edge(u, v)

    def reverse_edge(self, u: int, v: int):
        self.remove_edge(u, v)
        self.add_edge(v, u)

    def creates_cycle_if_added(self, u: int, v: int) -> bool:
        t = self._g.copy()
        t.add_edge(u, v)
        return not nx.is_directed_acyclic_graph(t)

    def creates_cycle_if_reversed(self, u: int, v: int) -> bool:
        t = self._g.copy()
        t.remove_edge(u, v)
        t.add_edge(v, u)
        return not nx.is_directed_acyclic_graph(t)


class GIESLearner:
    """
    Greedy Interventional Equivalence Search implementation.
    """

    def __init__(
        self,
        num_vars: int,
        datasets: DatasetManager,
        penalty_weight: float = 1.0,
        max_workers: int = 4,
        skeleton: Optional[Set[frozenset]] = None,
    ):
        self.num_vars = num_vars
        self.datasets = datasets
        self.score_calc = ScoreCalculator(datasets, penalty_weight)
        self.graph = CausalGraph(num_vars)
        self.max_workers = max_workers
        self.skeleton = skeleton

    def fit(self):
        improved = True
        while improved:
            improved = any(
                phase()
                for phase in (
                    self._forward_phase,
                    self._backward_phase,
                    self._turning_phase,
                )
            )

    def _forward_phase(self) -> bool:
        return self._phase_move(self.score_calc.score_gain_insertion, "add")

    def _backward_phase(self) -> bool:
        return self._phase_move(self.score_calc.score_gain_deletion, "del")

    def _turning_phase(self) -> bool:
        return self._phase_move(self.score_calc.score_gain_reversal, "rev")

    def _phase_move(self, score_fn, move_type: str) -> bool:
        if move_type == "add":
            candidates = [
                (u, v)
                for u in range(self.num_vars)
                for v in range(self.num_vars)
                if u != v
                and not self.graph.has_edge(u, v)
                and (not self.skeleton or frozenset({u, v}) in self.skeleton)
                and not self.graph.creates_cycle_if_added(u, v)
            ]
        elif move_type == "del":
            candidates = list(self.graph.edges())
        elif move_type == "rev":
            candidates = [
                (u, v)
                for (u, v) in self.graph.edges()
                if not self.graph.creates_cycle_if_reversed(u, v)
            ]
        else:
            return False
        best_gain = 0.0
        best_move = None
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            futures = {
                ex.submit(score_fn, u, v, self.graph): (u, v) for u, v in candidates
            }
            for fut in as_completed(futures):
                gain = fut.result()
                if gain > best_gain:
                    best_gain, best_move = gain, futures[fut]
        if best_gain > 0 and best_move:
            u, v = best_move
            if move_type == "add":
                self.graph.add_edge(u, v)
            elif move_type == "del":
                self.graph.remove_edge(u, v)
            else:
                self.graph.reverse_edge(u, v)
            return True
        return False

    def get_estimated_dag(self) -> nx.DiGraph:
        return self.graph._g.copy()


def learn(envs):
    dm = DatasetManager(envs, merge_values=True)
    learner = GIESLearner(num_vars=3, datasets=dm, max_workers=2)
    learner.fit()
    est = learner.get_estimated_dag()
    print("Estimated edges:", est.edges())
    # Rename the nodes to match the original variable names
    est = nx.relabel_nodes(est, {i: f"X{i+1}" for i in range(3)})
    # Convert to a DiGraph
    graph = nx.DiGraph(est)
    return graph
