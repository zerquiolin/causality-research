import random
import networkx as nx
from itertools import permutations
from scipy.stats import pearsonr
from src.lib.models.abstract.BaseAgent import BaseAgent


class CausalRLAgent(BaseAgent):
    def __init__(self, threshold=0.1, patience=3):
        self.threshold = threshold
        self.patience = patience
        self.current_dag = nx.DiGraph()
        self.no_improvement_count = 0
        self.best_score = float("inf")
        self.previous_edges = set()

    def learn_dag_from_state(self, state):
        """
        Learns the DAG by aggregating all data in the state.
        """
        import pandas as pd

        combined_data = []

        # Combine all observational and interventional data
        for key, val in state.items():
            if key == "empty":
                combined_data.extend(val)
            else:
                for sublist in val.values():
                    combined_data.extend(sublist)

        if not combined_data:
            return nx.DiGraph()

        full_data = pd.concat(combined_data)
        nodes = full_data.columns
        dag = nx.DiGraph()
        dag.add_nodes_from(nodes)

        # todo: Check if both directions are being considered
        for a, b in permutations(nodes, 2):
            # todo: implement partial correlation to evaluate for confounding variables
            corr, _ = pearsonr(full_data[a], full_data[b])
            if abs(corr) >= self.threshold:
                dag.add_edge(a, b)

        return dag

    def compute_shd(self, new_edges):
        """
        Computes a basic distance metric based on edge changes.
        """
        added = len(new_edges - self.previous_edges)
        removed = len(self.previous_edges - new_edges)
        return added + removed

    def choose_action(self, state: dict):
        """
        Select the next variable to intervene on based on least coverage.
        """
        self.current_dag = self.learn_dag_from_state(state)
        new_edges = set(self.current_dag.edges)
        shd = self.compute_shd(new_edges)

        if shd < self.best_score:
            self.best_score = shd
            self.no_improvement_count = 0
        else:
            self.no_improvement_count += 1

        self.previous_edges = new_edges

        if self.no_improvement_count >= self.patience:
            return None  # Stop intervening

        # Choose the node with the fewest interventions
        coverage = {
            node: sum(len(v) for v in state.get(node, {}).values())
            for node in state
            if node != "empty"
        }

        if not coverage:
            return random.choice(list(state.keys() - {"empty"}))

        return min(coverage, key=coverage.get)
