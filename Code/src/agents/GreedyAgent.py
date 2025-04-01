import re
import numpy as np
import pandas as pd
import networkx as nx
from pgmpy.estimators import PC
from pgmpy.models import BayesianModel

from .base import BaseAgent


class GreedyAgent(BaseAgent):
    _is_first_round = True

    def choose_action(self, samples, actions, num_rounds):
        # Check if this is the first round
        if self._is_first_round:
            self._is_first_round = False
        else:
            self._analyze_dataset(samples)
            return "stop_with_answer", None

        # Define the treatment list
        treatments = []
        # Iterate over all possible actions
        for node in actions.keys():
            # Skip the stop_with_answer action
            if node == "stop_with_answer":
                continue

            # Generate observation data
            if node == "observe":
                treatments.append(("observe", 1000))
                continue

            # Get the domain of the action
            domain = actions[node]
            # Check if the domain is categorical
            if isinstance(domain, list):
                for value in domain:
                    # Add all possible values to the treatment list
                    treatments.append(({node: value}, 500))
            else:
                for i in np.linspace(domain[0], domain[1], 10):
                    # Add 10 values to the treatment list
                    treatments.append(({node: i}, 500))

        return "experiment", treatments

    def _merge_graphs_with_conflict_resolution(self, graphs):
        """
        Merges multiple DAGs into a single graph while resolving conflicts.

        For conflicting edges (u,v) and (v,u), this implementation chooses the
        orientation based on the numeric value extracted from the node names,
        ensuring that the edge is directed from the lower-numbered node to the higher-numbered one.

        Args:
            graphs (list): List of networkx.DiGraph instances.

        Returns:
            networkx.DiGraph: The merged, conflict-resolved DAG.
        """
        merged_graph = nx.DiGraph()
        directed_edges = set()

        def get_numeric(node):
            # Assumes node names are like 'X1', 'X2', etc.
            try:
                return int(node.lstrip("X"))
            except ValueError:
                return float("inf")  # Fallback if the conversion fails

        for g in graphs:
            for u, v in g.edges():
                if (v, u) in directed_edges:
                    # Conflict detected: choose edge based on numeric order.
                    chosen_edge = (u, v) if get_numeric(u) < get_numeric(v) else (v, u)
                    # Remove both conflicting edges and add the chosen edge.
                    directed_edges.discard((v, u))
                    directed_edges.discard((u, v))
                    directed_edges.add(chosen_edge)
                else:
                    directed_edges.add((u, v))

        # Add resolved edges to the merged graph
        for u, v in directed_edges:
            merged_graph.add_edge(u, v, direction="resolved")
        return merged_graph

    def _analyze_dataset(self, samples):
        """
        Analyze the dataset and update the learned DAG.

        This method separates the DAG discovery into two phases:
        1. Build an initial DAG using only observational data.
        2. Refine edge orientations using interventional datasets.
        """

        def gen_dag(sample_list):
            # Extract only the columns that match the pattern "X1", "X2", ... etc.
            columns = [str(key) for key in sample_list[0].keys()]
            df = pd.DataFrame(data=sample_list, columns=columns)
            df.to_csv("data.csv", index=False)  # for debugging

            # Run the PC algorithm to estimate the DAG skeleton
            pc = PC(data=df)
            model: BayesianModel = pc.estimate(return_type="dag")

            # Create a networkx DiGraph from the BayesianModel
            nodes = list(model.nodes())
            edges = list(model.edges())
            graph = nx.DiGraph()
            graph.add_nodes_from(nodes)
            graph.add_edges_from(edges)
            return graph

        # === Phase 1: Observational DAG ===
        obs_graph = None
        if samples.get("empty") and len(samples["empty"]) > 0:
            obs_graph = gen_dag(samples["empty"])
        else:
            # If no observational data, fall back to merging interventional graphs
            obs_graph = nx.DiGraph()

        # === Phase 2: Orientation Refinement Using Interventional Data ===
        intervention_graphs = []
        for key, interventions in samples.items():
            if key == "empty":
                continue
            for intervention_samples in interventions.values():
                # Generate a DAG for the interventional sample
                intervention_graphs.append(gen_dag(intervention_samples))

        # Merge the observational DAG with the interventional DAGs.
        # Your _merge_graphs_with_conflict_resolution method can help fix edge conflicts.
        refined_graph = self._merge_graphs_with_conflict_resolution(
            [obs_graph] + intervention_graphs
        )

        print("Edges in refined graph:", refined_graph.edges())
        print("Nodes in refined graph:", refined_graph.nodes())

        # Save the refined graph as the learned DAG.
        self._learned_graph = refined_graph

    def submit_answer(self):
        """
        Returns a placeholder final answer for evaluation.
        """

        return self._learned_graph
