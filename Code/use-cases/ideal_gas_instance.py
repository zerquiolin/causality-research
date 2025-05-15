# Game
import causalitygame as cg

# Agents
from causalitygame.agents.impl.RandomAgent import RandomAgent
from causalitygame.agents.impl.ExhaustiveAgent import ExhaustiveAgent

# Utils
import numpy as np

agents = []

# Define the agents
for i in range(1, 11):
    seed = 911 + i
    rs = np.random.RandomState(seed)
    agent = (
        f"Random Agent {i}",
        RandomAgent(
            stop_probability=rs.beta(0.5, 10),
            experiments_range=(1, max(rs.poisson(10), 2)),
            samples_range=(rs.randint(10, 50), rs.randint(50, 100)),
            seed=seed,
        ),
    )
    agents.append(agent)
# Add an exhaustive agent
agents.append(("Exhaustive Agent", ExhaustiveAgent()))

# Game Instance
game_instance_path = (
    "causalitygame/data/game_instances/dag_inference/ideal_gas_instance.json"
)

# Create a game
game = cg.Game(
    agents=agents,
    game_spec=game_instance_path,
)


# Run the game
runs = game.run()

# Print the results
game.plot()
