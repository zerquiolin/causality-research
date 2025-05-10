import pandas as pd
import numpy as np
from causalitygame.generators.db_generator import DatabaseDrivenSCMGenerator
from causalitygame.scm.db import DatabaseSCM
import pytest

import networkx as nx
import logging

# define stream handler
ch = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
ch.setLevel(logging.DEBUG)

# configure logger for tester
logger = logging.getLogger("tester")
logger.handlers.clear()
logger.addHandler(ch)
logger.setLevel(logging.DEBUG)

@pytest.mark.parametrize(
    "factual_df", [
        (pd.read_csv("causalitygame/data/scm/ihdp_prepared.csv"))
    ])
def test_db_scm_with_pre_treatment_covariates_allowing_counterfactuals(factual_df):

    scm = DatabaseDrivenSCMGenerator(
        factual_df=factual_df,
        covariates_before_intervention=True,
        reveal_only_outcomes_of_originally_factual_covariates=False,
        random_state=np.random.RandomState(0),
        overwrite_factual_outcomes=False,
        outcome_generator=lambda x: [2] # always create a value of 2
    ).generate()

    # identify controlled vars
    treatment_vars = [c for c in factual_df.columns if c.startswith("t:")]

    # check that topological ordering is correct
    assert len(scm.dag.nodes) == factual_df.shape[1]
    seen_treatments = []
    seen_outputs = []
    for n in nx.topological_sort(scm.dag.graph):
        if n.startswith("t:"):
            seen_treatments.append(n)
        if n.startswith("o:"):
            seen_outputs.append(n)
        if n.startswith("c:"):
            assert not seen_treatments, f"incorrect topological ordering. The covariate {n} comes after treatments {treatment_vars}"
            assert not seen_outputs, f"incorrect topological ordering. The covariate {n} comes after outputs {seen_outputs}"

    # check that the controllable variable is indeed controllable
    assert treatment_vars == scm.controllable_vars

    # check whether all data points are contained in the original database
    num_factuals = 0
    num_counter_factuals = 0
    for s in scm.generate_samples(num_samples=10**1):
        row = np.array([s[v] for v in scm.var_names])
        is_factual = np.any([np.all(row == orig_row) for orig_row in factual_df[scm.var_names].values])
        if is_factual:
            num_factuals += 1
        else:
            num_counter_factuals += 1

    # check that we have both factual entries (from the original data) and counter-factual entries
    assert num_counter_factuals >= 3, "There should be at least 3 counterfactual entries even though factuals are not overwritten."
    assert num_factuals >= 3, "There should be at least 3 factual entries since original entries are preserved."


@pytest.mark.parametrize(
    "factual_df", [
        (pd.read_csv("causalitygame/data/scm/ihdp_prepared.csv"))
    ])
def test_db_scm_with_pre_treatment_covariates_prohibiting_counterfactuals(factual_df):

    scm = DatabaseDrivenSCMGenerator(
        factual_df=factual_df,
        covariates_before_intervention=True,
        reveal_only_outcomes_of_originally_factual_covariates=True,
        random_state=np.random.RandomState(0),
        overwrite_factual_outcomes=False,
        outcome_generator=lambda x: [2] # always create a value of 2
    ).generate()

    # identify controlled vars
    treatment_vars = [c for c in factual_df.columns if c.startswith("t:")]

    # check that topological ordering is correct
    assert len(scm.dag.nodes) == factual_df.shape[1]
    seen_treatments = []
    seen_outputs = []
    for n in nx.topological_sort(scm.dag.graph):
        if n.startswith("t:"):
            seen_treatments.append(n)
        if n.startswith("o:"):
            seen_outputs.append(n)
        if n.startswith("c:"):
            assert not seen_treatments, f"incorrect topological ordering. The covariate {n} comes after treatments {treatment_vars}"
            assert not seen_outputs, f"incorrect topological ordering. The covariate {n} comes after outputs {seen_outputs}"

    # check that the controllable variable is indeed controllable
    assert treatment_vars == scm.controllable_vars

    # check whether all data points are contained in the original database
    num_factuals = 0
    num_counter_factuals = 0
    num_samples = 10**1
    for s in scm.generate_samples(num_samples=num_samples):
        row = np.array([s[v] for v in scm.var_names])
        is_factual = np.any([np.all(row == orig_row) for orig_row in factual_df[scm.var_names].values])
        if is_factual:
            num_factuals += 1
        else:
            num_counter_factuals += 1

    # check that we have both factual entries (from the original data) and counter-factual entries
    assert num_counter_factuals == 0, "There should be no counterfactuals in the training data, since these have been forbidden."
    assert num_factuals == num_samples, "All data points should be contained in the sampled data."

@pytest.mark.parametrize(
    "factual_df", [
        (pd.read_csv("causalitygame/data/scm/ihdp_prepared.csv"))
    ])
def test_db_scm_with_post_treatment_covariates_allowing_counterfactuals(factual_df):

    scm = DatabaseDrivenSCMGenerator(
        factual_df=factual_df,
        covariates_before_intervention=False,
        reveal_only_outcomes_of_originally_factual_covariates=False,
        random_state=np.random.RandomState(0),
        overwrite_factual_outcomes=False,
        outcome_generator=lambda x: [2] # always create a value of 2
    ).generate()

    # identify controlled vars
    treatment_vars = [c for c in factual_df.columns if c.startswith("t:")]

    # check that topological ordering is correct
    assert len(scm.dag.nodes) == factual_df.shape[1]
    seen_covariates = []
    seen_outputs = []
    for n in nx.topological_sort(scm.dag.graph):
        if n.startswith("c:"):
            seen_covariates.append(n)
        if n.startswith("o:"):
            seen_outputs.append(n)
        if n.startswith("t:"):
            assert not seen_covariates, f"incorrect topological ordering. The treatment variable {n} comes after covariates {seen_covariates}"
            assert not seen_outputs, f"incorrect topological ordering. The treatment variable {n} comes after outputs {seen_outputs}"

    # check that the controllable variable is indeed controllable
    assert treatment_vars == scm.controllable_vars

    # check whether all data points are contained in the original database
    num_factuals = 0
    num_counter_factuals = 0
    for s in scm.generate_samples(num_samples=10**1):
        row = np.array([s[v] for v in scm.var_names])
        is_factual = np.any([np.all(row == orig_row) for orig_row in factual_df[scm.var_names].values])
        if is_factual:
            num_factuals += 1
        else:
            num_counter_factuals += 1

    # check that we have both factual entries (from the original data) and counter-factual entries
    assert num_counter_factuals >= 3, "There should be at least 3 counterfactual entries even though factuals are not overwritten."
    assert num_factuals >= 3, "There should be at least 3 factual entries since original entries are preserved."

@pytest.mark.parametrize(
    "factual_df", [
        (pd.read_csv("causalitygame/data/scm/ihdp_prepared.csv"))
    ])
def test_db_scm_with_post_treatment_covariates_prohibiting_counterfactuals(factual_df):

    
    scm = DatabaseDrivenSCMGenerator(
        factual_df=factual_df,
        covariates_before_intervention=False,
        reveal_only_outcomes_of_originally_factual_covariates=True,
        random_state=np.random.RandomState(0),
        overwrite_factual_outcomes=False,
        outcome_generator=lambda x: [2] # always create a value of 2
    ).generate()

    # identify controlled vars
    treatment_vars = [c for c in factual_df.columns if c.startswith("t:")]

    # check that topological ordering is correct
    assert len(scm.dag.nodes) == factual_df.shape[1]
    seen_covariates = []
    seen_outputs = []
    for n in nx.topological_sort(scm.dag.graph):
        if n.startswith("c:"):
            seen_covariates.append(n)
        if n.startswith("o:"):
            seen_outputs.append(n)
        if n.startswith("t:"):
            assert not seen_covariates, f"incorrect topological ordering. The treatment variable {n} comes after covariates {seen_covariates}"
            assert not seen_outputs, f"incorrect topological ordering. The treatment variable {n} comes after outputs {seen_outputs}"

    # check that the controllable variable is indeed controllable
    assert treatment_vars == scm.controllable_vars

    # check whether all data points are contained in the original database
    num_factuals = 0
    num_counter_factuals = 0
    num_samples = 10**1
    for s in scm.generate_samples(num_samples=num_samples):
        row = np.array([s[v] for v in scm.var_names])
        is_factual = np.any([np.all(row == orig_row) for orig_row in factual_df[scm.var_names].values])
        if is_factual:
            num_factuals += 1
        else:
            num_counter_factuals += 1

    # check that we have both factual entries (from the original data) and counter-factual entries
    assert num_counter_factuals == 0, "There should be no counterfactuals in the training data, since these have been forbidden."
    assert num_factuals == num_samples, "All data points should be contained in the sampled data."