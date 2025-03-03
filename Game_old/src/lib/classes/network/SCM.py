from pgmpy.models import BayesianNetwork as BN
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.sampling import BayesianModelSampling
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from typing import List, Tuple, Dict
import numpy as np
import itertools


class StructuralCausalModel:
    """
    A class for managing and performing operations on Bayesian Networks.
    Provides methods for adding nodes, edges, performing interventions (do-calculus),
    querying, and learning from data.
    """

    def __init__(self):
        """
        Initializes the Bayesian Network model.
        """
        self._model: BN = BN()
        self.custom_functions = (
            {}
        )  # Store custom sampling functions for continuous variables
        self._metadata = {}  # Store metadata for each node

    @property
    def metadata(self) -> BN:
        """
        Returns the directed acyclic graph (DAG) structure of the Bayesian Network.
        """
        return self._metadata

    @property
    def model(self) -> BN:
        """
        Returns the directed acyclic graph (DAG) structure of the Bayesian Network.
        """
        return self._model

    @property
    def nodes(self) -> List[str]:
        """
        Returns the nodes of the Bayesian Network.
        """
        return self._model.nodes

    @property
    def edges(self) -> List[Tuple[str, str]]:
        """
        Returns the edges of the Bayesian Network.
        """
        return self._model.edges

    @property
    def roots(self) -> List[str]:
        """
        Returns the root nodes of the Bayesian Network.
        """
        return self._model.get_roots()

    @property
    def leaves(self) -> List[str]:
        """
        Returns the leaf nodes of the Bayesian Network.
        """
        return self._model.get_leaves()

    @property
    def immoralities(self) -> List[Tuple[str, str]]:
        """
        Returns the immoralities in the Bayesian Network.
        Immoralities occur when two parents share a common child but are not connected.
        """
        return self._model.get_immoralities()

    @property
    def independencies(self):
        """
        Returns the independencies implied by the Bayesian Network structure.
        """
        return self._model.get_independencies()

    @property
    def is_valid(self) -> bool:
        """
        Checks if the Bayesian Network is correctly defined with valid CPDs.
        """
        return self._model.check_model()

    def set_metadata(self, metadata: Dict[str, dict]) -> None:
        self._metadata = metadata

    def add_nodes(self, nodes: List[str]) -> None:
        """
        Adds nodes to the Bayesian Network.

        Parameters:
        nodes (List[str]): A list of node names to be added.
        """
        self._model.add_nodes_from(nodes)

    def add_edges(self, edges: List[Tuple[str, str]]) -> None:
        """
        Adds edges to the Bayesian Network.

        Parameters:
        edges (List[Tuple[str, str]]): A list of edge tuples where each tuple represents a directed edge.
        """
        self._model.add_edges_from(edges)

    def remove_edge(self, edge: Tuple[str, str]) -> None:
        """
        Removes an edge from the Bayesian Network.

        Parameters:
        edge (Tuple[str, str]): The edge to be removed, in the form (start, end).
        """
        self._model.remove_edge(*edge)

    def get_parents(self, node: str) -> List[str]:
        """
        Returns the parents of a specified node.

        Parameters:
        node (str): The name of the node.

        Returns:
        List[str]: A list of parent nodes.
        """
        return self._model.get_parents(node)

    def get_children(self, node: str) -> List[str]:
        """
        Returns the children of a specified node.

        Parameters:
        node (str): The name of the node.

        Returns:
        List[str]: A list of child nodes.
        """
        return self._model.get_children(node)

    def is_d_connected(self, start: str, end: str, observed: List[str]) -> bool:
        """
        Checks if two nodes are d-connected given a set of observed nodes.

        Parameters:
        start (str): The starting node.
        end (str): The target node.
        observed (List[str]): A list of observed nodes.

        Returns:
        bool: True if the nodes are d-connected, False otherwise.
        """
        return self._model.is_dconnected(start, end, observed)

    def is_equivalent(self, model: BN) -> bool:
        """
        Checks if the current Bayesian Network is I-equivalent to another Bayesian Network.

        Parameters:
        other_bn (BayesianNetwork): Another Bayesian Network to compare.

        Returns:
        bool: True if the two networks are I-equivalent, False otherwise.
        """
        return self._model.is_iequivalent(model)

    def do_intervention(self, interventions: List[str]) -> BN:
        """
        Performs a 'do' intervention by fixing the values of the given variables
        and removes the dependencies on their parents.

        Parameters:
        interventions (List[str]): A List of variables to intervene.

        Returns:
        BayesianNetwork: A modified Bayesian Network after intervention.
        """
        return self._model.do(interventions, inplace=False)

    def add_cpds(self, cpds: dict):
        """
        Adds Conditional Probability Distributions (CPDs) to the Bayesian Network.

        Parameters:
        cpds (dict): A dictionary where the keys are node names and values are CPDs (either in pandas DataFrame or TabularCPD format).
        """
        for node, cpd_data in cpds.items():
            if isinstance(cpd_data, TabularCPD):
                # If the CPD is already in TabularCPD format, add it directly.
                self._model.add_cpds(cpd_data)
            elif isinstance(cpd_data, pd.DataFrame):
                # Convert the DataFrame to TabularCPD and add it
                self._add_cpd_from_dataframe(node, cpd_data)
            else:
                raise ValueError(
                    f"Unsupported CPD format for node {node}. Expected a pandas DataFrame or TabularCPD."
                )

    # def add_custom_cpds(self, cpds: dict):
    #     """
    #     Adds custom CPDs to the Bayesian Network.

    #     Parameters:
    #     cpds (dict): A dictionary where the keys are node names and values are function and evidence tuples.
    #     """
    #     for node, (handler, evidence) in cpds.items():
    #         self._model.add_cpds(CustomCPD(node, handler, evidence))

    def infer_probability(
        self,
        evidences: Dict[str, int] = None,
        interventions: Dict[str, int] = None,
        output_dataframe: bool = False,
    ) -> pd.Series:
        """
        Performs inference to calculate the probability distribution of a Bayesian network given some interventions.

        Parameters:
        - evidences (Dict[str, int]): A dictionary representing the evidence where keys are variable names and values are observed values.
        - interventions (Dict[str, int]): A dictionary representing the interventions where keys are variable names and values are intervention values.
        - output_dataframe (bool): Flag indicating whether to output the result as a pandas DataFrame.

        Returns:
        - pd.Series: The inferred probability distribution for the query variable.
        - pd.DataFrame: The inferred probability distribution for the query variable as a pandas DataFrame if `output_dataframe` is True.
        """

        intervention = VariableElimination(
            self.do_intervention(interventions.keys() if interventions else [])
        )
        evidence = evidences.copy() if evidences else dict()
        evidence.update(interventions if interventions else dict())

        inference = intervention.query(
            variables=[
                node for node in self._model.nodes if node not in evidence.keys()
            ],
            evidence=evidence,
        )

        if not output_dataframe:
            return inference

        return pd.DataFrame(
            np.column_stack(
                (
                    np.array(
                        list(itertools.product(*[[0, 1] for _ in inference.variables]))
                    ),
                    np.array(inference.values).flatten(),
                )
            ),
            columns=inference.variables + ["P"],
        )

    def _add_cpd_from_dataframe(self, node, df):
        """
        Converts a pandas DataFrame to a TabularCPD and adds it to the network.

        Parameters:
        node (str): The node for which the CPD is defined.
        df (pd.DataFrame): A DataFrame representing the CPD.
        """
        if "P" not in df.columns:
            raise ValueError(
                "DataFrame must contain a 'P' column representing probabilities."
            )

        # Extract the unique parent columns
        evidence = [col for col in df.columns if col not in ["P", node]]
        evidence_card = [df[col].nunique() for col in evidence]
        variable_card = df[node].nunique()

        # Reshape the probability column into the required format for TabularCPD
        values = df.pivot_table(index=evidence, columns=node, values="P").values.T

        # Create the TabularCPD
        cpd = TabularCPD(
            variable=node,
            variable_card=variable_card,
            values=values,
            evidence=evidence if evidence else None,
            evidence_card=evidence_card if evidence else None,
        )

        # Add the CPD to the Bayesian Network
        self._model.add_cpds(cpd)

    def estimate_cpds(
        self, data: pd.DataFrame, estimator=MaximumLikelihoodEstimator
    ) -> None:
        """
        Estimates the Conditional Probability Distributions (CPDs) for the Bayesian Network using the provided data.

        Parameters:
        data (pd.DataFrame): A pandas DataFrame containing the data.
        estimator: The estimator to use for learning CPDs (default is MaximumLikelihoodEstimator).
        """
        self._model.fit(data, estimator=estimator)

    def sample(self, n_samples: int) -> pd.DataFrame:
        """
        Samples from the Bayesian Network using the Gibbs sampling algorithm.

        Parameters:
        n_samples (int): The number of samples to generate.

        Returns:
        pd.DataFrame: A DataFrame containing the sampled data.
        """
        return BayesianModelSampling(self._model).forward_sample(size=n_samples)

    def generate_data(self, sample_size):
        """
        Generates synthetic data, applying custom functions for continuous variables.
        """
        sampler = BayesianModelSampling(self.model)
        discrete_data = sampler.forward_sample(size=sample_size)

        # Apply custom functions for continuous variables
        for node, func in self.custom_functions.items():
            parents = list(self.model.get_parents(node))
            discrete_data[node] = discrete_data[parents].apply(
                lambda row: func(*row), axis=1
            )

        return discrete_data

    def visualize(self, layout="kamada_kawai"):
        """
        Visualizes the DAG with visible (green) and hidden (red) nodes.

        Parameters:
            layout (str): The layout algorithm to use ('spring', 'planar', 'circular', 'shell', 'kamada_kawai').
        """
        nx_graph = nx.DiGraph(self._model.edges())

        # Assign colors based on node type
        node_colors = [
            "green" if node.startswith("V") else "red" for node in nx_graph.nodes()
        ]

        # Choose layout
        pos = getattr(nx, f"{layout}_layout")(nx_graph)

        # Draw the graph
        nx.draw(
            nx_graph,
            pos,
            with_labels=True,
            node_size=1000,
            node_color=node_colors,
            font_size=11,
            font_color="black",
            edge_color="gray",
            arrowsize=20,
            width=1,
        )
        plt.title("DAG Visualization: Green (Visible) | Red (Hidden)")
        plt.show()
