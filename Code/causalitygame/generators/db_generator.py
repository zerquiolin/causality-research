import numpy as np
import pandas as pd
import itertools as it

from causalitygame.generators.scm_generator import AbstractSCMGenerator
from causalitygame.scm.db import DatabaseSCM

class FactualDatabaseDrivenSCMGenerator(AbstractSCMGenerator):
    """
    This class takes a dataframe with factual observations (but without counter-factuals) and derives an SCM based on an outcome generator

    Args:
        AbstractSCMGenerator (_type_): _description_
    """

    def __init__(
            self,
            factual_df,
            outcome_generators,
            random_state=None,
            overwrite_factual_outcomes: bool = True,
            covariates_before_intervention: bool = True,
            reveal_only_outcomes_of_originally_factual_covariates: bool = False,
            train_size=0.7
        ):
        self._controllable_vars = [c for c in factual_df.columns if c.startswith("t:")]
        self._outcome_vars = [c for c in factual_df.columns if c.startswith("o:")]
        self._covariates = [c for c in factual_df.columns if c.startswith("c:")]
        self.outcome_generators = outcome_generators  # must implement a mapping from covariates + treatment variables to output variables
        self.random_state = np.random.RandomState() if random_state is None else random_state
        self.overwrite_factual_outcomes = overwrite_factual_outcomes
        self.covariates_before_intervention = covariates_before_intervention
        self.reveal_only_outcomes_of_originally_factual_covariates = reveal_only_outcomes_of_originally_factual_covariates # note that this can be confusing if factual outcomes are overwritten by potential outcomes
        self.train_size = train_size

        # reorganize column names
        self.factual_df = factual_df[self._covariates + self._controllable_vars + self._outcome_vars]

        # state vars
        self._full_dataframe_with_potential_outcomes = None
    
    def generate(self):
        
        if self._full_dataframe_with_potential_outcomes is None:

            # fit the outcome generator on the factual data
            for outcome_var, gen_descriptor in self.outcome_generators.items():
                gen_descriptor["generator"].fit(
                    x=self.factual_df[gen_descriptor["input_covariates"]],
                    t=self.factual_df[gen_descriptor["input_treatments"]],
                    y=self.factual_df[outcome_var]
                )
            
            # generate potential outcomes
            """
            rows = []
            for cov, df_cov in self.factual_df.groupby(self._covariates):
        
                # get lookup table for factual covariates and treatments
                lookup = df_cov[self._covariates + self._controllable_vars].values

                for t in possible_treatments:
                    x = np.array(list(cov) + list(t))
                    mask_for_match = np.all(lookup == x, axis=1)
                    is_factual_combination = np.any(mask_for_match)
                    y_entries = [self.outcome_generator.generate(np.array(cov).reshape(1, -1), np.array(t).reshape(-1))] if self.overwrite_factual_outcomes or not is_factual_combination else df_cov[mask_for_match][self._outcome_vars].values
                    
                    # for an x, there *can* be several y's in the data for the same treatment (not in the generated case), so we add a row for each of them
                    may_be_revealed = not self.reveal_only_outcomes_of_originally_factual_covariates or is_factual_combination
                    for y in y_entries:    
                        rows.append([is_factual_combination] + list(x) + list(y) + [may_be_revealed])
            
            # create dataframe that is the basis for the Database SCM
            self._full_dataframe_with_potential_outcomes = pd.DataFrame(rows, columns=["factual_covariate_treatment_combination"] + self._covariates + self._controllable_vars + self._outcome_vars + ["revealable"])
            """
        
        # generate random subset of revealed instances
        population = np.where(self.factual_df["revealable"])[0]
        train_indices = self.random_state.choice(
            population,
            size=int(self.train_size * len(population)),
            replace=False
        )
        #df = self._full_dataframe_with_potential_outcomes.copy().drop(columns="revealable")
        #df["revealed"] = False
        #df.loc[train_indices, "revealed"] = True
        return DatabaseSCM(
            self.factual_df[self._covariates],
            outcome_generators=self.outcome_generators
        )
    
class FullDatabaseDrivenSCMGenerator(AbstractSCMGenerator):
    """
    This class takes a dataset that contains actual outcomes for all possible interventions and derives an SCM from it.

    Args:
        AbstractSCMGenerator (_type_): _description_
    """

    def __init__(
            self,
            full_df,
            random_state=None,
            overwrite_factual_outcomes: bool = True,
            covariates_before_intervention: bool = True,
            reveal_only_outcomes_of_originally_factual_covariates: bool = False,
            train_size=0.7
        ):
        self._controllable_vars = [c for c in full_df.columns if c.startswith("t:")]
        self._outcome_vars = [c for c in full_df.columns if c.startswith("o:")]
        self._covariates = [c for c in full_df.columns if c.startswith("c:")]
        self.random_state = np.random.RandomState() if random_state is None else random_state
        self.overwrite_factual_outcomes = overwrite_factual_outcomes
        self.covariates_before_intervention = covariates_before_intervention
        self.reveal_only_outcomes_of_originally_factual_covariates = reveal_only_outcomes_of_originally_factual_covariates # note that this can be confusing if factual outcomes are overwritten by potential outcomes
        self.train_size = train_size

        # reorganize column names
        self.full_df = full_df
    
    def generate(self):
        
        if self._full_dataframe_with_potential_outcomes is None:

            # fit the outcome generator on the factual data
            self.outcome_generator.fit(
                x=self.factual_df[self._covariates],
                t=self.factual_df[self._controllable_vars],
                y=self.factual_df[self._outcome_vars]
            )

            # determine all possible treatments
            possible_treatments = list(it.product(*[
                sorted(pd.unique(self.factual_df[c]))
                for c in self._controllable_vars
            ]))

            # generate potential outcomes
            rows = []
            for cov, df_cov in self.factual_df.groupby(self._covariates):
        
                # get lookup table for factual covariates and treatments
                lookup = df_cov[self._covariates + self._controllable_vars].values

                for t in possible_treatments:
                    x = np.array(list(cov) + list(t))
                    mask_for_match = np.all(lookup == x, axis=1)
                    is_factual_combination = np.any(mask_for_match)
                    y_entries = [self.outcome_generator.generate(np.array(cov).reshape(1, -1), np.array(t).reshape(-1))] if self.overwrite_factual_outcomes or not is_factual_combination else df_cov[mask_for_match][self._outcome_vars].values
                    
                    # for an x, there *can* be several y's in the data for the same treatment (not in the generated case), so we add a row for each of them
                    may_be_revealed = not self.reveal_only_outcomes_of_originally_factual_covariates or is_factual_combination
                    for y in y_entries:    
                        rows.append([is_factual_combination] + list(x) + list(y) + [may_be_revealed])
            
            # create dataframe that is the basis for the Database SCM
            self._full_dataframe_with_potential_outcomes = pd.DataFrame(rows, columns=["factual_covariate_treatment_combination"] + self._covariates + self._controllable_vars + self._outcome_vars + ["revealable"])
        
        # generate random subset of revealed instances
        population = np.where(self._full_dataframe_with_potential_outcomes["revealable"])[0]
        train_indices = self.random_state.choice(
            population,
            size=int(self.train_size * len(population)),
            replace=False
        )
        df = self._full_dataframe_with_potential_outcomes.copy().drop(columns="revealable")
        df["revealed"] = False
        df.loc[train_indices, "revealed"] = True
        return DatabaseSCM(
            df,
            covariates_before_intervention=self.covariates_before_intervention
        )