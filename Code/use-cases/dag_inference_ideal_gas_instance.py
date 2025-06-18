# Game
import causalitygame as cg

# Agents
from causalitygame.agents.RandomAgent import RandomAgent
from causalitygame.agents.ExhaustiveAgent import ExhaustiveAgent


# Define the agents
agents = [(f"Random Agent {i}", RandomAgent(seed=911 + i)) for i in range(1, 11)]
# Add an exhaustive agent
agents.append(("Exhaustive Agent", ExhaustiveAgent()))
# Game Instance
game_instance_path = (
    "causalitygame/data/game_instances/dag_inference/ideal_gas_instance.json"
)
# Create a game
game = cg.Game(agents=agents, game_spec=game_instance_path)
# Run the game
runs = game.run()
# Print the results
game.plot()
