import numpy as np
import pandas as pd
import itertools as it

from causalitygame.lib.utils.random_state_serialization import random_state_to_json, random_state_from_json

from causalitygame.generators.outcome.base import ComplementaryOutcomeGenerator, OutcomeGenerator
from causalitygame.scm.base import SCM
from causalitygame.scm.dag import DAG

from causalitygame.scm.node.db import DatabaseDefinedCategoricSCMNode, DatabaseDefinedNumericSCMNode
from causalitygame.scm.node.computed import ComputedNumericSCMNode, ComputedCategoricSCMNode
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
        factual_df: pd.DataFrame,
        intervention_variables: list,
        outcome_generators: dict,
        overwrite_factual_outcomes: bool = True,
        only_factual_outcomes_in_interventionless_sampling: bool = True,
        allow_duplicate_interventions_per_covariate_combination: bool = True,
        train_size=0.7,
        revealed_mask=None,
        name=None,
        random_state: np.random.RandomState = np.random.RandomState(42)
    ):
        """

        Creates an SCM where variables will assume values related to a given data frame.

        Since there is an inherent logic of factuals and counter-factuals, the set of intervention variables must be defined.
        

        Args:
            df (pd.DataFrame):
                the dataset that contains all covariate values.
            
            outcome_generators (dict):
                a dictionary that contains outcome generators, one for each outcome variable
                
            random_state (np.random.RandomState, optional): _description_. Defaults to np.random.RandomState(42).
            name (_type_, optional): _description_. Defaults to None.
        """

        # config vars
        self.intervention_variables = intervention_variables
        self._outcome_vars = sorted(outcome_generators.keys())
        self.covariates = [c for c in factual_df.columns if c not in self.intervention_variables and c not in self._outcome_vars]
        self.outcome_generators = outcome_generators
        self.overwrite_factual_outcomes = overwrite_factual_outcomes
        self.only_factual_outcomes_in_interventionless_sampling = only_factual_outcomes_in_interventionless_sampling
        self.allow_duplicate_interventions_per_covariate_combination = allow_duplicate_interventions_per_covariate_combination
        self.train_size = train_size
        self.revealed_mask = revealed_mask

        # save the dataframe
        self.df = factual_df[self.covariates + self.intervention_variables + self._outcome_vars].copy()

        # fit the outcome generator on the factual data
        fitted_generators = {}
        for outcome_var, generator in self.outcome_generators.items():
            gen = generator if self.overwrite_factual_outcomes else ComplementaryOutcomeGenerator(generator)
            gen.fit(
                x=factual_df[self.covariates + self.intervention_variables],
                y=factual_df[outcome_var]
            )
            fitted_generators[outcome_var] = gen
        
        # generate random subset of revealed instances
        if revealed_mask is None:
            train_indices = random_state.choice(
                range(len(factual_df)),
                size=int(self.train_size * len(factual_df)),
                replace=False
            )
            self.revealed_mask = np.array([False] * len(self.df))
            self.revealed_mask[train_indices] = True

        # extract controllable variables and outcome variables
        self._outcome_vars = sorted(outcome_generators.keys())        
        self.var_names = [c for c in self.df.columns]
        
        # create DAG
        nodes = []
        for name in self.var_names:
            if self.df[name].dtype in [float, int, np.number] and len(pd.unique(self.df[name])) > 10:
                if name in self._outcome_vars:
                    node = ComputedNumericSCMNode(
                        name=name,
                        value_computer=fitted_generators[name],
                        accessibility=ACCESSIBILITY_OBSERVABLE,
                        random_state=random_state
                    )
                else:
                    node = DatabaseDefinedNumericSCMNode(
                        name=name,
                        df=self.df,
                        revealed_to_agent=self.revealed_mask,
                        accessibility=ACCESSIBILITY_CONTROLLABLE if name in self.intervention_variables else ACCESSIBILITY_OBSERVABLE,
                        random_state=random_state
                    )
            else:
                if name in self._outcome_vars:
                    node = ComputedCategoricSCMNode(
                        name=name,
                        value_computer=fitted_generators[name],
                        accessibility=ACCESSIBILITY_OBSERVABLE,
                        random_state=random_state
                    )
                else:
                    node = DatabaseDefinedCategoricSCMNode(
                        name=name,
                        df=self.df,
                        revealed_to_agent=self.revealed_mask,
                        accessibility=ACCESSIBILITY_CONTROLLABLE if name in self.intervention_variables else ACCESSIBILITY_OBSERVABLE,
                        random_state=random_state
                    )
            nodes.append(node)

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

        return {
            "class": f"{__class__.__module__}.{__class__.__name__}",
            "factual_df": self.df.to_json(orient="records", double_precision=15),
            "intervention_variables": self.intervention_variables,
            "outcome_generators": {k: ocg.to_dict() for k, ocg in self.outcome_generators.items()},
            "overwrite_factual_outcomes": self.overwrite_factual_outcomes,
            "revealed_mask": [bool(v) for v in self.revealed_mask],
            "random_state": random_state_to_json(self.random_state) if self.random_state is not None else None,
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
        data = data.copy()
        data["factual_df"] = pd.read_json(data["factual_df"], orient="records", precise_float=True)
        data["random_state"] = random_state_from_json(data["random_state"])
        data["overwrite_factual_outcomes"] = data["overwrite_factual_outcomes"]
        data["revealed_mask"] = np.array(data["revealed_mask"])
        data["outcome_generators"] = {k: OutcomeGenerator.from_dict(v) for k, v in data["outcome_generators"].items()}
        return cls(**data)
