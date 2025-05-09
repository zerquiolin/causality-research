from itertools import permutations
import json
import pytest
from causalitygame.evaluators.impl.BehaviorMetrics import ExperimentsBehaviorMetric
from causalitygame.evaluators.impl.DeliverableMetrics import SHDDeliverableMetric
from causalitygame.game.Environment import Environment
from causalitygame.game.GameInstance import GameInstance
from causalitygame.generators.scm_generator import EquationBasedSCMGenerator
from causalitygame.generators.dag_generator import DAGGenerator
from causalitygame.mission.impl.DAGInferenceMission import DAGInferenceMission
from causalitygame.scm.noise_distributions import (
    GaussianNoiseDistribution,
    UniformNoiseDistribution,
)
from scipy.stats import norm, uniform
import sympy as sp
import numpy as np

# Test DAGs
dag_random_state = np.random.RandomState(911)
dag = DAGGenerator(
    num_nodes=10,
    num_roots=2,
    num_leaves=2,
    edge_density=0.5,
    max_in_degree=3,
    max_out_degree=3,
    min_path_length=1,
    max_path_length=4,
    random_state=dag_random_state,
).generate()

# Test SCM
num_nodes = 10
variable_types = {f"X{i}": "numerical" for i in range(1, num_nodes)}
variable_types[f"X{num_nodes}"] = "categorical"
variable_domains = {f"X{i}": (-1, 1) for i in range(1, num_nodes)}
variable_domains[f"X{num_nodes}"] = ["A", "B", "C"]

scm_generator = EquationBasedSCMGenerator(
    dag=dag,
    variable_types=variable_types,
    variable_domains=variable_domains,
    user_constraints={"max_terms": 2},
    allowed_operations=["+", "*"],
    allowed_functions=[lambda x: x**2],
    noise_distributions=[
        GaussianNoiseDistribution(mean=0, std=1),
        UniformNoiseDistribution(low=-1, high=1),
    ],
    random_state=np.random.RandomState(911),
    num_samples_for_cdf_generation=10
)
scm = scm_generator.generate()


# Test GameInstance
# Generate a mission
behavior_metric = ExperimentsBehaviorMetric()
deliverable_metric = SHDDeliverableMetric()
mission = DAGInferenceMission(
    behavior_metric=behavior_metric, deliverable_metric=deliverable_metric
)
game_instance_random_state = np.random.RandomState(911)
game_instance = GameInstance(100, scm, mission, game_instance_random_state)


@pytest.mark.parametrize(
    "game_instance, agent, random_state_seed, interventions",
    [
        (
            game_instance,
            None,
            911,
            [
                ("X10", 5),
                ("X4", "2"),
                ("X8", 2),
                ("X9", 1),
            ],  # Interventions X4 is 0 with a log
        )
    ],  # The agent is not used in the test
)
def test_environment_is_invariant(
    game_instance, agent, random_state_seed, interventions
):
    """
    Test that the Environment is invariant to the agent.
    """
    # Generate all permutations
    all_permutations = list(permutations(interventions))

    history = {"all-in": [], "partitioned": []}

    for perm in all_permutations:
        # Create the Environment using the GameInstance and Agent
        all_in_env = Environment(
            game_instance=game_instance,
            agent=agent,
            random_state=np.random.RandomState(random_state_seed),
        )
        partitioned_env = Environment(
            game_instance=game_instance,
            agent=agent,
            random_state=np.random.RandomState(random_state_seed),
        )
        experiment = lambda t: [({var: val}, t)]
        for var, val in perm:
            # Perform the intervention for the all in environment
            all_in_env.perform_experiment(experiment(2))
            history["all-in"].append(all_in_env.get_state()["datasets"])
        for var, val in perm:
            # Perform the intervention for the partitioned environment
            partitioned_env.perform_experiment(experiment(1))
        for var, val in perm:
            # Perform the intervention for the partitioned environment
            partitioned_env.perform_experiment(experiment(1))
            history["partitioned"].append(partitioned_env.get_state()["datasets"])

    assert history["all-in"] == history["partitioned"]
