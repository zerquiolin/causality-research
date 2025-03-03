# Libs
import numpy as np
import pandas as pd

# Models
from models.Covid import Covid
from models.V4 import V4

# Modules
from src.lib.classes.environment.Environment import Environment
from src.lib.classes.player.RandomPlayer import RandomPlayer
from src.lib.classes.game.Game import CausalityGame
from src.utils.Samples import generate_binary_samples_from_joint


# Define a simple Bayesian Network for the game
# model = Covid().gen_model()
model = V4().gen_model()

# Initialize the environment and player
environment = Environment(model)
environment.bayesian_network.visualize()
random_player = RandomPlayer(environment)

# Initialize the game and play
game = CausalityGame(environment, random_player)
game.play()
game.print_state()

# TEST
inference = model.infer_probability(
    interventions={"A": 1}, output_dataframe=True
)  # Full joint probability distribution

sample = generate_binary_samples_from_joint(inference, 20000)
sample["A"] = 1.0

inference2 = model.infer_probability(
    interventions={"A": 0}, output_dataframe=True
)  # Full joint probability distribution

sample2 = generate_binary_samples_from_joint(inference, 20000)
sample2["A"] = 0.0

sample = pd.concat([sample, sample2])
print(sample)
sample_np = sample.to_numpy()
random_player.infer_model(sample, labels=sample.columns)
# random_player.causal_learning(sample_np, labels=sample.columns)
