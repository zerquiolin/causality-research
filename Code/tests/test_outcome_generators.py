from causalitygame.generators.outcome.base import OutcomeGenerator
from causalitygame.generators.outcome._hill import SetupAOutcomeGenerator, SetupBOutcomeGenerator

import numpy as np
import json

import pytest

outcome_variable = "y"
covs = ["x1", "x2"]
treatments = ["t"]


@pytest.mark.parametrize(
        "generator", [
            SetupAOutcomeGenerator(outcome_variable=outcome_variable, required_covariates=covs, required_treatments=treatments),
            SetupBOutcomeGenerator(outcome_variable=outcome_variable, required_covariates=covs, required_treatments=treatments)
        ]
)
def test_core_functionality(generator):

    # fit model (only based on shapes actually)
    n = 42
    x = np.ones((n, len(covs)))
    t = np.ones((n, len(treatments)))
    y = np.ones(n)
    generator.fit(x, t, y)

    # generate data
    generated_data = generator.generate(x, t)
    assert generated_data.shape == (n, )
    for i, val_1 in enumerate(generated_data):
        for j, val_2 in enumerate(generated_data[:i]):
            assert val_1 != val_2, f"{i}-th and {j}-th value are identical but should be different due to noise."


@pytest.mark.parametrize(
    "generator", [
        SetupAOutcomeGenerator(outcome_variable=outcome_variable, required_covariates=covs, required_treatments=treatments, random_state=0),
        SetupBOutcomeGenerator(outcome_variable=outcome_variable, required_covariates=covs, required_treatments=treatments, random_state=0)
    ]
)
def test_serializability_before_fit(generator):

    # recovered generators
    generator_recovered_internally = OutcomeGenerator.from_dict(generator.to_dict())
    generator_recovered_json = OutcomeGenerator.from_dict(json.loads(json.dumps(generator.to_dict())))

    # fit generators
    n = 42
    x = np.ones((n, len(covs)))
    t = np.ones((n, len(treatments)))
    y = np.ones(n)
    for g in [generator, generator_recovered_internally, generator_recovered_json]:
        g.fit(x, t, y)
    
    # test that generators generate same data
    for sample_a, sample_b, sample_c in zip(generator.generate(x, t), generator_recovered_internally.generate(x, t), generator_recovered_json.generate(x, t)):
        assert sample_a == sample_b
        assert sample_a == sample_c



@pytest.mark.parametrize(
    "generator", [
        SetupAOutcomeGenerator(outcome_variable=outcome_variable, required_covariates=covs, required_treatments=treatments, random_state=0),
        SetupBOutcomeGenerator(outcome_variable=outcome_variable, required_covariates=covs, required_treatments=treatments, random_state=0)
    ]
)
def test_serializability_after_fit(generator):

    # fit generators
    n = 42
    x = np.ones((n, len(covs)))
    t = np.ones((n, len(treatments)))
    y = np.ones(n)
    generator.fit(x, t, y)

    # recovered generators
    generator_recovered_internally = OutcomeGenerator.from_dict(generator.to_dict())
    generator_recovered_json = OutcomeGenerator.from_dict(json.loads(json.dumps(generator.to_dict())))
    
    # test that generators generate same data
    for sample_a, sample_b, sample_c in zip(generator.generate(x, t), generator_recovered_internally.generate(x, t), generator_recovered_json.generate(x, t)):
        assert sample_a == sample_b
        assert sample_a == sample_c