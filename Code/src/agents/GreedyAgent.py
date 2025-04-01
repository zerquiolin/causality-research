import numpy as np
import pandas as pd
import networkx as nx
from pgmpy.estimators import PC
from pgmpy.models import BayesianModel

from src.lib.models.abstract.BaseAgent import BaseAgent


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
                treatments.append(("observe", 10000))
                continue

            # Get the domain of the action
            domain = actions[node]
            # Check if the domain is categorical
            if isinstance(domain, list):
                for value in domain:
                    # Add all possible values to the treatment list
                    treatments.append(({node: value}, 10000))
            else:
                for i in np.linspace(domain[0], domain[1], 10):
                    # Add 10 values to the treatment list
                    treatments.append(({node: i}, 10000))

        return "experiment", treatments

    def _merge_graphs_with_conflict_resolution(self, graphs):
        merged_graph = nx.DiGraph()
        directed_edges = set()
        undirected_edges = set()

        for g in graphs:
            for u, v in g.edges():
                if (v, u) in directed_edges:
                    # Conflict detected — remove both directed edges and store as undirected
                    directed_edges.discard((v, u))
                    undirected_edges.add(frozenset((u, v)))
                elif frozenset((u, v)) not in undirected_edges:
                    directed_edges.add((u, v))

        # Add directed edges
        for u, v in directed_edges:
            merged_graph.add_edge(u, v, direction="directed")

        # Add undirected edges (use both directions, but tag them)
        for edge in undirected_edges:
            u, v = tuple(edge)
            merged_graph.add_edge(u, v, direction="undirected")
            merged_graph.add_edge(v, u, direction="undirected")

        return merged_graph

    def _analyze_dataset(self, samples):
        """
        Analyze the dataset and decide whether to stop or continue.
        """
        # Structure of samples:
        # {
        #     "X1": {0: [{X1: 0, X2: 1, ..., Xn: 1  }]},
        #     "X2": {0: [{X1: 0, X2: 1, ..., Xn: 1  }]},
        #     ...
        #     "Xn": {0: [{X1: 0, X2: 1, ..., Xn: 1  }]},
        # }

        print(samples.keys())

        def gen_dag(samples):
            print("Generating DAG")
            # Create columns for the DataFrame
            columns = sorted(
                [str(key) for key in samples[0].keys()],
                key=lambda x: int(x.replace("X", "")),
            )
            # Create a DataFrame
            df = pd.DataFrame(data=samples, columns=columns)
            print(df)
            # Save the DataFrame to a CSV file
            df.to_csv("data.csv", index=False)

            # Run the PC algorithm
            pc = PC(data=df)

            # Estimate the DAG using the PC algorithm
            model: BayesianModel = pc.estimate(return_type="dag")

            # Extract nodes and edges
            nodes = list(model.nodes())
            edges = list(model.edges())

            # Create networkx graph
            graph = nx.DiGraph()
            graph.add_nodes_from(nodes)
            graph.add_edges_from(edges)

            return graph

        print(samples.keys())

        print([key for key, node_samples in samples.items() if key != "empty"])

        # Generate a list of DAGs from the samples
        sample_dags = []
        # sample_dags = [
        #     gen_dag(value_samples)
        #     for key, node_samples in samples.items()
        #     for value_samples in node_samples.values()
        #     if key != "empty"
        # ]
        # # Generate a dag from all samples
        # sample_dags.append(
        #     gen_dag(
        #         [
        #             samples
        #             for key, node_samples in samples.items()
        #             for value_samples in node_samples.values()
        #             for samples in value_samples
        #             if key != "empty"
        #         ]
        #     )
        # )
        # Generate a dag from observational samples
        if samples["empty"] and len(samples["empty"]) > 0:
            print("Generating observational DAG")
            sample_dags.append(gen_dag(samples["empty"]))

        # Assuming each graph is a pgmpy BayesianModel or networkx.DiGraph
        all_edges = set()
        all_nodes = set()

        for g in sample_dags:
            print("Graph edges:", g.edges())
            all_edges.update(g.edges())
            all_nodes.update(g.nodes())

        # Create a new merged graph
        merged_graph = nx.DiGraph()
        merged_graph.add_nodes_from(all_nodes)
        merged_graph.add_edges_from(all_edges)

        # Save the result to the class variable
        self._learned_graph = merged_graph

    def submit_answer(self):
        """
        Returns a placeholder final answer for evaluation.
        """

        return self._learned_graph
