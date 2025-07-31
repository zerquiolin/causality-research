# Game
import numpy as np
import pandas as pd
import causalitygame as cg

# Plotting
import matplotlib.pyplot as plt

# Agents
from causalitygame.agents.random import RandomAgent
from causalitygame.agents.exhaustive import ExhaustiveAgent


# Define the agents
agents = [
    (f"Random Agent {i}", RandomAgent(seed=911 + i, samples_range=(1, 2)))
    for i in range(1, 3)
]
# Add an exhaustive agent
agents.append(("Exhaustive Agent", ExhaustiveAgent(num_obs=1)))
# Game Instance
game_instance_path = "causalitygame/data/game_instances/te/hill_instance.json"

# Create a game
game = cg.Game(
    agents=agents,
    game_spec=game_instance_path,
)


def gen_test_rs():
    return {
        "Z": np.random.RandomState(911),
        "X": np.random.RandomState(912),
        "Y": np.random.RandomState(913),
    }


# obs = game._game_instance.scm.generate_samples(
#     interventions={"Z": "0"}, num_samples=1000, random_state=gen_test_rs()
# )
# obs.to_csv("z1.csv", index=False)
# z0 = game._game_instance.scm.generate_samples(
#     interventions={"Z": 1}, num_samples=1000, random_state=gen_test_rs()
# )
obs = game._game_instance.scm.generate_samples(interventions={}, num_samples=1000)
# obs.to_csv("obs.csv", index=False)

obs_z0 = (
    obs[obs["Z"] == "0"]["X"].mean() - 200
)  # I know the base value without noise is 200
print(f"obs_z0: {obs_z0}")
samples = np.mean(np.random.normal(loc=0, scale=1, size=1000))
print(f"samples: {samples}")

obs_z1 = obs[obs["Z"] == "1"]["X"].mean() - 400
print(f"obs_z1: {obs_z1}")
