from causalitygame.generators.outcome.base import OutcomeGenerator, ComplementaryOutcomeGenerator
from causalitygame.generators.outcome._hill import SetupAOutcomeGenerator, SetupBOutcomeGenerator

import numpy as np
import json

import pytest

outcome_variable = "y"
covs = ["x1", "x2"]
treatments = ["t"]


@pytest.mark.parametrize(
    "generator", [
        SetupAOutcomeGenerator(random_state=0),
        SetupBOutcomeGenerator(random_state=0),
        ComplementaryOutcomeGenerator(base_outcome_generator=SetupAOutcomeGenerator(random_state=0))
    ]
)
def test_core_functionality(generator):

    # fit model (only based on shapes actually)
    rs = np.random.RandomState(0)
    n = 42
    x = rs.random(size=(n, len(covs)))
    t = rs.random(size=(n, len(treatments)))
    y = rs.random(size=n)
    generator.fit(x, t, y)

    # generate data
    generated_data = generator.generate(x, t)
    assert generated_data.shape == (n, )
    for i, val_1 in enumerate(generated_data):
        for j, val_2 in enumerate(generated_data[:i]):
            assert val_1 != val_2, f"{i}-th and {j}-th value are identical but should be different due to noise."


@pytest.mark.parametrize(
    "generator", [
        SetupAOutcomeGenerator(random_state=0),
        SetupBOutcomeGenerator(random_state=0),
        ComplementaryOutcomeGenerator(base_outcome_generator=SetupAOutcomeGenerator(random_state=0))
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
        SetupAOutcomeGenerator(random_state=0),
        SetupBOutcomeGenerator(random_state=0),
        ComplementaryOutcomeGenerator(base_outcome_generator=SetupAOutcomeGenerator(random_state=0))
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


def test_complementary_outcome_generator():

    base_gen = SetupAOutcomeGenerator()
    gen = ComplementaryOutcomeGenerator(
        base_outcome_generator=base_gen
    )
    
    # fit generators
    rs = np.random.RandomState(0)
    n = 10
    x = rs.random(size=(n, len(covs)))
    t = rs.random(size=(n, len(treatments)))
    y = rs.random(size=n)
    gen.fit(x, t, y)

    # check that all values are correctly memorized
    assert np.all(gen.x == x)
    assert np.all(gen.y == y)
    assert np.all(gen.t == t)

    # check that known outcomes are not over-written
    y_gen = gen.generate(x, t)
    assert np.all(y_gen == y)

    # check that we can generate outcomes for other data
    xp = x**2
    y_gen = gen.generate(xp, t)
    assert len(y_gen) == len(xp)