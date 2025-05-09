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
        covariates_before_intervention: bool,
        only_factual_outcomes: bool,
        random_state: np.random.RandomState = np.random.RandomState(42),
        name=None
    ):
        """_summary_

        Args:
            df (pd.DataFrame):
                the dataset that contains all factual and counter-factual observations.
                It must contain a boolean column `factual` the indicates whether the observation is factual or counter-factural.
                By convention, every column except `factual` must either start with `t:`, `c:`, or `o:` to indicate treatment, covariate, and output variables, respectively.
            covariates_before_intervention (bool): whether the covariate values are fixed before intervention.
                If so, the player can assign alternative actual treatments to the instances that were not observed in the original data.
                If this is not desired, the value should be set to False, which
            only_factual_outcomes (bool): states whether all data sampled from this SCM must be factual (this can only be guaranteed if covariates come after treatment)
            random_state (np.random.RandomState, optional): _description_. Defaults to np.random.RandomState(42).
            name (_type_, optional): _description_. Defaults to None.

            For any covariate combination, there must be *exactly* one observation with `factual` being `True`.
        """

        # check that both dataframes have the same columns
        assert "factual" in df.columns, "No column called `factual` found"
        self._controllable_columns = []
        self._output_columns = []
        self._covariate_columns = []
        for c in df.columns:
            if c != "factual":
                if c.startswith("t:"):
                    self._controllable_columns.append(c)
                elif c.startswith("c:"):
                    self._covariate_columns.append(c)
                elif c.startswith("o:"):
                    self._output_columns.append(c)
                else: 
                    assert False, f"Column {c} doesn't start with t:, c:, or o:"
        
        # sort columns
        if covariates_before_intervention:
            col_list = self._covariate_columns + self._controllable_columns + self._output_columns
        else:
            col_list = self._controllable_columns + self._covariate_columns + self._output_columns
        df = df[["factual"] + col_list]
        df["factual"] = df["factual"].astype(bool)
        
        # determine all possible treatments
        possible_treatments = list(it.product(*[
            sorted(pd.unique(df[c]))
            for c in self._controllable_columns
        ]))

        mask_fact = df["factual"]
        df_fact = df[mask_fact]
        df_counterfact = df[~mask_fact]

        # check that we have factual and counterfactual data for all covariates
        for cov, df_cov_fact in df_fact.groupby(self._covariate_columns):

            # check for which counterfactual entries these covariates apply
            cov_mask_in_counter_factuals = np.ones(len(df_counterfact)).astype(bool)
            for covariate_name, covariate_val in zip(self._covariate_columns, cov):
                cov_mask_in_counter_factuals &= df_counterfact[covariate_name] == covariate_val
            
            # for every factual treatment, make sure that we also have counter-factual treatments
            for treatment, df_treatment in df_cov_fact.groupby(self._controllable_columns):
                num_instances_with_these_covariates_and_this_factual_treatment = len(df_treatment)

                # check for all other treatments
                for alternative_treatment in possible_treatments:
                    if alternative_treatment != treatment:
                        mask_for_alternative_treatment = np.ones(len(df_counterfact)).astype(bool)
                        for treatment_var, treatment_val in zip(self._controllable_columns, alternative_treatment):
                            mask_for_alternative_treatment &= df_counterfact[treatment_var] == treatment_val
                        num_instances_with_these_covariates_and_this_counterfactual_treatment = np.count_nonzero(mask_for_alternative_treatment & cov_mask_in_counter_factuals)
                        assert num_instances_with_these_covariates_and_this_counterfactual_treatment == num_instances_with_these_covariates_and_this_factual_treatment, f"Expected to see {num_instances_with_these_covariates_and_this_factual_treatment} with alternative treatment {alternative_treatment} in covariate {cov} but saw {num_instances_with_these_covariates_and_this_counterfactual_treatment}"
        
        # create DAG
        if only_factual_outcomes:
            df = df[df["factual"]]
        self.df = df
        df_without_factual_col = df.drop(columns="factual")
        self.var_names = list(df_without_factual_col.columns)
        nodes = [
            DatabaseDefinedSCMNode(
                name=name,
                df=df_without_factual_col,
                accessibility=ACCESSIBILITY_CONTROLLABLE if name in self._controllable_columns else ACCESSIBILITY_OBSERVABLE,
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

        # store the dataframe
        self.df = df
