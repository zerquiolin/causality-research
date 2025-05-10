import pandas as pd
import numpy as np
import json
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


@pytest.mark.parametrize(
    "factual_df", [
        (pd.read_csv("causalitygame/data/scm/ihdp_prepared.csv"))
    ])
def test_that_only_instances_marked_as_revealed_are_shown_to_the_agent(factual_df):

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
    for s in scm.generate_samples(num_samples=10**1):
        row = np.array([s[v] for v in scm.var_names])
        mask = np.all(scm.df[scm.var_names].values == row, axis=1)
        assert np.count_nonzero(mask) == 1, "There should be exactly one entry for this covariate/treatment combination"
        assert scm.df[mask]["revealed"].iloc[0], f"The datapoint {scm.df[mask]["revealed"][0]} has been revealed to the user even though it is not marked for revelation."


@pytest.mark.parametrize(
    "factual_df", [
        (pd.read_csv("causalitygame/data/scm/ihdp_prepared.csv"))
    ])
def test_serializability_of_scm(factual_df):

    scm = DatabaseDrivenSCMGenerator(
        factual_df=factual_df,
        covariates_before_intervention=True,
        reveal_only_outcomes_of_originally_factual_covariates=False,
        random_state=np.random.RandomState(0),
        overwrite_factual_outcomes=False,
        outcome_generator=lambda x: [2] # always create a value of 2
    ).generate()

    def test_df_equalness(df1, df2, msg=""):
        if df1.equals(df2):
            return
        if msg != "":
            msg += "\n"
        assert df1.shape == df2.shape, f"{msg}DataFrames have different shapes. First has {df1.shape}, second has {df2.shape}"
        assert list(df1.columns) == list(df2.columns), "Column names don't match"
        assert list(df1.index) == list(df2.index), "Indices don't match"
        for i in range(len(df1)):
            row1 = df1.iloc[i]
            row2 = df2.iloc[i]
            for field, v1 in row1.items():
                v2 = row2[field]
                assert v1 == v2, f"Mismatch in field {field} in row {i} of datasets.\n\tFirst is {v1}\n\tSecond is {v2}"

        assert np.array_equal(df1.values, df2.values)

    for _ in range(2): # the 2nd loop iteration is to check whether the sample generation is also identical after there have been samples drawn before already

        if _ == 0:
            logger.info("Checking whether SCM is correctly serialized and deserialized if no samples have been generated yet.")
        if _ == 1:
            logger.info("Checking whether SCM is correctly serialized and deserialized if samples have been generated previously.")

        # recover from dict
        scm_recovered_inside_python = DatabaseSCM.from_dict(scm.to_dict())
        scm_recovered_after_json = DatabaseSCM.from_dict(json.loads(json.dumps(scm.to_dict())))

        # test that properties are equal
        logger.debug("Checking that properties are equal.")
        for other in [scm_recovered_inside_python, scm_recovered_after_json]:
            assert scm.controllable_vars == other.controllable_vars
            assert scm.latent_vars == other.latent_vars
            assert scm.observable_vars == other.observable_vars
            assert scm._output_columns == other._output_columns
            assert scm.covariates_before_intervention == other.covariates_before_intervention
            assert list(nx.topological_sort(scm.dag.graph)) == list(nx.topological_sort(other.dag.graph))

        # test that dataframes are identical
        logger.debug("Checking that dataframes are equal.")
        test_df_equalness(scm.df, scm_recovered_inside_python.df, msg="DataFrame has changed after internal Python deserialization")
        test_df_equalness(scm.df, scm_recovered_after_json.df, msg="Dataframe has changed after json.loads and json.dumps")

        # test that generated instances are the same
        logger.debug(f"Checking that sampled instances are equal.")
        num_samples = 10**1
        samples_a = scm.generate_samples(num_samples=num_samples)
        samples_b = scm_recovered_inside_python.generate_samples(num_samples=num_samples)
        samples_c = scm_recovered_after_json.generate_samples(num_samples=num_samples)
        for i, (sample_a, sample_b, sample_c) in enumerate(zip(samples_a, samples_b, samples_c)):
            assert sample_a == sample_b, f"Mismatch between {i}-th sample of original SCM and SCM after Python internal deserialization"
            assert sample_a == sample_c, f"Mismatch between {i}-th sample of original SCM and SCM after JSON deserialization"
        