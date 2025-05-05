import numpy as np

from causalitygame.scm.impl.basic_binary_scm import gen_binary_scm
from causalitygame.game.GameInstance import GameInstance

# Create a random state for reproducibility
random_state = np.random.RandomState(42)
# Generate a binary SCM with the specified seed
dag, scm = gen_binary_scm(random_state=random_state)
# Create a GameInstance with the generated SCM
game_instance = GameInstance(scm=scm, random_state=random_state)
# Save the game instance to a file
game_instance.save(filename="./instances/game_instance.pkl")
