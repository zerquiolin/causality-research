import pandas as pd
import numpy as np
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
    "df", [
        (pd.read_csv("causalitygame/data/scm/ihdp_prepared.csv"))
    ])
def test_db_scm_with_pre_treatment_covariates_allowing_counterfactuals(df):


    # identify controlled vars
    treatment_vars = [c for c in df.columns if c.startswith("t:")]
    df_without_factual_column = df.drop(columns="factual")

    # define SCM
    scm = DatabaseSCM(
        df=df,
        covariates_before_intervention=True,
        only_factual_outcomes=False
    )

    # check that topological ordering is correct
    assert len(scm.dag.nodes) == df_without_factual_column.shape[1]
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
        assert np.any([np.all(row == orig_row) for orig_row in df_without_factual_column[scm.var_names].values])


@pytest.mark.parametrize(
    "df", [
        (pd.read_csv("causalitygame/data/scm/ihdp_prepared.csv"))
    ])
def test_db_scm_with_pre_treatment_covariates_prohibiting_counterfactuals(df):


    # identify controlled vars
    treatment_vars = [c for c in df.columns if c.startswith("t:")]
    df["factual"] = df["factual"].astype(bool)
    df_without_factual_column = df.drop(columns="factual")

    # define SCM
    scm = DatabaseSCM(
        df=df,
        covariates_before_intervention=True,
        only_factual_outcomes=True
    )

    # check that topological ordering is correct
    assert len(scm.dag.nodes) == df_without_factual_column.shape[1]
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

    # check that all samples are really part of the original FACTUAL dataset
    for s in scm.generate_samples(num_samples=10**1):
        row = np.array([s[v] for v in scm.var_names])
        assert np.any([np.all(row == orig_row) for orig_row in df[df["factual"]].drop(columns="factual")[scm.var_names].values]), f"Record {row} was not found in original dataframe"

@pytest.mark.parametrize(
    "df", [
        (pd.read_csv("causalitygame/data/scm/ihdp_prepared.csv"))
    ])
def test_db_scm_with_post_treatment_covariates_allowing_counterfactuals(df):

    # identify controlled vars
    treatment_vars = [c for c in df.columns if c.startswith("t:")]
    df_without_factual_column = df.drop(columns="factual")

    # define SCM
    scm = DatabaseSCM(
        df=df,
        covariates_before_intervention=False,
        only_factual_outcomes=False
    )

    # check that topological ordering is correct
    assert len(scm.dag.nodes) == df_without_factual_column.shape[1]
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

    # check that all samples are really part of the original dataset
    for s in scm.generate_samples(num_samples=10**1):
        row = np.array([s[v] for v in scm.var_names])
        assert np.any([np.all(row == orig_row) for orig_row in df_without_factual_column[scm.var_names].values]), f"Record {row} was not found in original dataframe"


@pytest.mark.parametrize(
    "df", [
        (pd.read_csv("causalitygame/data/scm/ihdp_prepared.csv"))
    ])
def test_db_scm_with_post_treatment_covariates_prohibiting_counterfactuals(df):

    # identify controlled vars
    treatment_vars = [c for c in df.columns if c.startswith("t:")]
    df["factual"] = df["factual"].astype(bool)
    df_without_factual_column = df.drop(columns="factual")

    # define SCM
    scm = DatabaseSCM(
        df=df,
        covariates_before_intervention=False,
        only_factual_outcomes=True
    )

    # check that topological ordering is correct
    assert len(scm.dag.nodes) == df_without_factual_column.shape[1]
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

    # check that all samples are really part of the original FACTUAL dataset
    for s in scm.generate_samples(num_samples=10**1):
        row = np.array([s[v] for v in scm.var_names])
        assert np.any([np.all(row == orig_row) for orig_row in df[df["factual"]].drop(columns="factual")[scm.var_names].values]), f"Record {row} was not found in original dataframe"
