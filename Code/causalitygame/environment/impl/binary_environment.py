# Math
import numpy as np

# Environment
from causalitygame.game.GameInstance import GameInstance
from causalitygame.game.Environment import Environment

# SCM Implementations
from causalitygame.scm.impl.basic_binary_scm import gen_binary_scm

# Types
from causalitygame.agents.base import BaseAgent


def gen_binary_environment(
    agent: BaseAgent,
    seed: int = 42,
):
    # Set the random seed for reproducibility
    random_state = np.random.RandomState(seed)

    # Generate a binary SCM with the specified seed
    dag, scm = gen_binary_scm(random_state=seed)

    # Create a GameInstance with the generated SCM
    game_instance = GameInstance(scm=scm, random_state=random_state)

    # Create the Environment using the GameInstance and Agent
    env = Environment(
        game_instance=game_instance,
        agent=agent,
        max_rounds=10,
        random_state=random_state,
    )
    return env
