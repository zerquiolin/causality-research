from itertools import permutations
import json
import pytest
from src.game.Environment import Environment
from src.game.GameInstance import GameInstance
from src.generators.SCMGenerator import SCMGenerator
from src.generators.DagGenerator import DAGGenerator
from scipy.stats import norm, uniform
import sympy as sp
import numpy as np

# Test DAGs
dag_random_state = np.random.RandomState(911)
dag = DAGGenerator(
    num_nodes=10,
    num_roots=3,
    num_leaves=3,
    edge_density=0.3,
    max_in_degree=3,
    max_out_degree=3,
    min_path_length=2,
    max_path_length=5,
    random_state=dag_random_state,
).generate()

# Test SCM
scm_random_state = np.random.RandomState(911)
scm_constraints = {
    "variable_types": {
        f"X{i}": "numerical" if scm_random_state.rand() < 0.8 else "categorical"
        for i in range(1, 11)
    },
    "variable_domains": {},
    "user_constraints": {
        "max_terms": 3,
        "allow_non_linearity": True,
        "allow_variable_exponents": True,
    },
    "allowed_operations": ["+", "-", "*", "/"],
    "allowed_functions": [sp.sin, sp.exp, sp.log],
    "noise_distributions": {
        "gaussian": norm(loc=0, scale=0.1),
        "uniform": uniform(loc=-0.1, scale=0.2),
    },
}
# Define variable domains
for node, vtype in scm_constraints["variable_types"].items():
    if vtype == "numerical":
        # For numerical nodes, assign a random interval.
        # Here, we choose lower bound between -10 and -1, and upper bound between 1 and 10.
        lower = scm_random_state.randint(-10, -1)
        upper = scm_random_state.randint(1, 10)
        scm_constraints["variable_domains"][node] = (lower, upper)
    else:
        # For categorical nodes, assign a random list of categories.
        # For example, generate between 2 and 4 categories using letters.
        num_categories = scm_random_state.randint(2, 4)
        # This will generate categories like ['A', 'B', 'C'].
        categories = [chr(65 + i) for i in range(num_categories)]
        scm_constraints["variable_domains"][node] = categories
# Generate the SCM
scm_random_state = np.random.RandomState(911)
scm_gen = SCMGenerator(dag, **scm_constraints, random_state=np.random.RandomState(911))
scm = scm_gen.generate()

# Test GameInstance
game_instance_random_state = np.random.RandomState(911)
game_instance = GameInstance(scm, game_instance_random_state)


@pytest.mark.parametrize(
    "game_instance, agent, random_state, interventions",
    [
        (
            game_instance,
            None,
            np.random.RandomState(911),
            [("X10", 5), ("X4", "0"), ("X8", 2), ("X9", 1)],
        )
    ],  # The agent is not used in the test
)
def test_environment_is_invariant(game_instance, agent, random_state, interventions):
    """
    Test that the Environment is invariant to the agent.
    """
    # todo: this env should be inside the loop
    # Create the Environment using the GameInstance and Agent
    env = Environment(
        game_instance=game_instance,
        agent=agent,
        random_state=random_state,
    )
    # Generate all permutations
    all_permutations = list(permutations(interventions))

    history = []
    history_combined = []

    # Run once for each permutation and twice to check for the addition of new interventions
    for _ in range(2):
        for perm in all_permutations:
            for var, val in perm:
                experiment = [({var: val}, 1)]  # Only 5 samples for speed
                # Perform the intervention
                env.perform_experiment(experiment)
                history.append(env.get_state()["datasets"])
                env.perform_experiment(experiment)
                history_combined.append(env.get_state()["datasets"])

    first_round = zip(history)
    print(first_round)
    # print(len(set(frozenset(d.items()) for d in first_round.datasets)) == 1)
    second_round = zip(history_combined)

    # Check available actions
    for round1 in first_round:
        print(round1)

    # print("X9" in env.get_available_actions())

    assert False
