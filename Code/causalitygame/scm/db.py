import numpy as np
import pandas as pd
import itertools as it

from causalitygame.scm.base import SCM
from causalitygame.scm.dag import DAG

from causalitygame.scm.node.db import DatabaseDefinedSCMNode
from causalitygame.scm.node.base import (
    ACCESSIBILITY_CONTROLLABLE,
    ACCESSIBILITY_LATENT,
    ACCESSIBILITY_OBSERVABLE
)

from typing import Dict

import networkx as nx


class DatabaseSCM(SCM):
    def __init__(
        self,
        df: pd.DataFrame,
        outcome_generators: dict,
        controllable_variables_with_domains: dict,
        random_state: np.random.RandomState = np.random.RandomState(42),
        name=None
    ):
        """_summary_

        Args:
            df (pd.DataFrame):
                the dataset that contains all covariate values.
            
            outcome_generators (dict):
                a dictionary that contains outcome generators, one for each outcome variable
                
            random_state (np.random.RandomState, optional): _description_. Defaults to np.random.RandomState(42).
            name (_type_, optional): _description_. Defaults to None.
        """

        # check that both dataframes have the same columns
        assert "revealed" in df.columns, "No column called `revealed` found"
        assert df["revealed"].dtype == bool

        special_cols = ["revealed"]

        # get union of all controllable variables


        # extract controllable variables and outcome variables
        self._controllable_vars = set()
        self._outcome_vars = []
        for outcome_variable, outcome_generator in outcome_generators.items():
            self._outcome_vars.append(outcome_variable)
            self._controllable_vars.update(set(outcome_generator.required_treatments))
        print(self._controllable_vars)

        
        self._covariates = [c for c in df.columns if c not in special_cols]

        # extract treatment domains
        
        # determine all possible treatments
        possible_treatments = np.array(list(it.product(*[
            sorted(pd.unique(df[c]))
            for c in self._controllable_vars
        ])))

        # check that we have all covariate-treatment combinations covered
        for ind, df_ind in df.groupby("individual"):
            for treatment in possible_treatments:
                assert np.any(np.all(df_ind == treatment, axis=1)), f"Incomplete dataset. No treatment {treatment} for individual {ind}"

        # create DAG
        self.df = df
        df_without_factual_col = df.drop(columns=[c for c in special_cols if c in df.columns])
        self.var_names = list(df_without_factual_col.columns)
        nodes = [
            DatabaseDefinedSCMNode(
                name=name,
                df=df.drop(columns=[c for c in special_cols if c in df.columns]),
                revealed_to_agent=df["revealed"],
                accessibility=ACCESSIBILITY_CONTROLLABLE if name in self._controllable_vars else ACCESSIBILITY_OBSERVABLE,
                random_state=random_state
            )
            for name in self.var_names
        ]

        # Create a directed graph
        dag = nx.DiGraph()
        dag.add_nodes_from(self.var_names)
        dag.add_edges_from([
            (v1, v2) for i, v1 in enumerate(self.var_names) for v2 in self.var_names[i + 1:]
        ])
        
        # create the SCM
        super().__init__(dag=DAG(dag), nodes=nodes, name=name, random_state=random_state)

    def to_dict(self) -> Dict:
        """
        Serializes the SCM to a dictionary format.

        Returns:
            Dict: A dictionary representing the SCM's structure and state.
        """

        # Serialize the random state
        state = self.random_state.get_state()
        state_dict = {
            "state": state[0],
            "keys": state[1].tolist(),
            "pos": state[2],
            "has_gauss": state[3],
            "cached_gaussian": state[4],
        }

        return {
            "class": f"{__class__.__module__}.{__class__.__name__}",
            "data": self.df.to_json(orient="records", double_precision=15),
            "covariates_before_intervention": self.covariates_before_intervention,
            "random_state": state_dict,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "SCM":
        """
        Deserializes an SCM instance from a dictionary.

        Args:
            data (Dict): Dictionary containing SCM structure and state.

        Returns:
            SCM: A new SCM instance.
        """
        df = pd.read_json(data["data"], orient="records", precise_float=True)
        covariates_before_intervention = data["covariates_before_intervention"]

        # Reconstruct the random state
        random_state = np.random.RandomState(911)
        if "random_state" in data:
            state_tuple = (
                str(data["random_state"]["state"]),
                np.array(data["random_state"]["keys"], dtype=np.uint32),
                int(data["random_state"]["pos"]),
                int(data["random_state"]["has_gauss"]),
                float(data["random_state"]["cached_gaussian"]),
            )
            random_state.set_state(state_tuple)

        return cls(df, covariates_before_intervention, random_state)
