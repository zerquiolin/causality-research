import pandas as pd
import numpy as np
from typing import Dict, Any, List
from causalitygame.scm.dag import DAG
from causalitygame.scm.scm import SCM
from causalitygame.scm.nodes import BaseSCMNode
from causalitygame.scm.base import ACCESSIBILITY_CONTROLLABLE

# from base import AbstractSCMGenerator


class DatabaseSCMNode(BaseSCMNode):
    def __init__(self, name: str, samples_df: pd.DataFrame, parents: List[str]):
        super().__init__(
            name=name,
            accessibility=ACCESSIBILITY_CONTROLLABLE,
            evaluation=None,
            domain=samples_df[name].unique().tolist(),
            noise_distribution=None,
            parents=parents,
            parent_mappings=None,
            random_state=None,
        )
        self.name = name
        self.df = samples_df
        self.parents = parents

    def generate_value(
        self, parent_values: Dict[str, Any], random_state: np.random.RandomState = None
    ):
        # Filter the dataset for rows matching parent values
        filtered = self.df
        for parent in self.parents:
            if parent in parent_values:
                filtered = filtered[filtered[parent] == parent_values[parent]]

        # If no match found, fallback to full column
        if filtered.empty:
            return random_state.choice(self.df[self.name].dropna().tolist())

        return random_state.choice(filtered[self.name].dropna().tolist())


class DatabaseSCMGenerator:
    def __init__(
        self,
        dataframe: pd.DataFrame,
        dag: DAG,
        variable_types: Dict[str, str],
        random_state: np.random.RandomState = np.random.RandomState(42),
    ):
        self.df = dataframe
        self.dag = dag
        self.variable_types = variable_types
        self.random_state = random_state

    def generate(self) -> SCM:
        nodes = []
        topological_order = list(nx.topological_sort(self.dag.graph))

        for node_name in topological_order:
            parents = list(self.dag.graph.predecessors(node_name))
            node = DatabaseSCMNode(name=node_name, samples_df=self.df, parents=parents)
            nodes.append(node)

        return SCM(self.dag, nodes, self.random_state)
