from src.generators.DagGenerator import DAGGenerator
from src.generators.SCMGenerator import SCMGenerator
from src.game.GameInstance import GameInstance
from src.game.Environment import Environment
from src.agents.RandomAgent import RandomAgent
from src.agents.GreedyAgent import GreedyAgent
import numpy as np
import sympy as sp
import pandas as pd
from scipy.stats import norm, uniform


def main():
    seed = 911
    random_state = np.random.RandomState(seed)

    # Generate a DAG
    dag_gen = DAGGenerator(
        num_nodes=10,
        num_roots=3,
        num_leaves=3,
        edge_density=0.3,
        max_in_degree=3,
        max_out_degree=3,
        min_path_length=2,
        max_path_length=5,
        random_state=random_state,
    )
    dag_obj = dag_gen.generate()

    # Define constraints for SCM generation
    scm_constraints = {
        "variable_types": {
            f"X{i}": "numerical" if np.random.rand() < 0.8 else "categorical"
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
            lower = np.random.randint(-10, -1)
            upper = np.random.randint(1, 10)
            scm_constraints["variable_domains"][node] = (lower, upper)
        else:
            # For categorical nodes, assign a random list of categories.
            # For example, generate between 2 and 4 categories using letters.
            num_categories = np.random.randint(2, 4)
            # This will generate categories like ['A', 'B', 'C'].
            categories = [chr(65 + i) for i in range(num_categories)]
            scm_constraints["variable_domains"][node] = categories

    # Generate the SCM
    scm_gen = SCMGenerator(dag_obj, **scm_constraints, random_state=random_state)
    scm = scm_gen.generate()

    # Create a GameInstance with the generated SCM
    game_instance = GameInstance(scm, random_state)

    # Create a RandomAgent
    random_agent = RandomAgent()
    greedy_agent = GreedyAgent()

    # Create the Environment using the GameInstance and Agent
    # env = Environment(game_instance, random_agent, max_rounds=10, random_state=random_state)
    env = Environment(
        game_instance, greedy_agent, max_rounds=10, random_state=random_state
    )

    # print(f"env: {env}, game_instance: {game_instance}, agent: {agent}, scm: {scm}")

    print(game_instance.scm.nodes)

    # Run the game simulation
    final_state = env.run_game()

    # Retrieve and display the state-action history
    game_history_df = env.get_game_history()

    print("\n📊 Game History DataFrame:")
    print(game_history_df)


if __name__ == "__main__":
    main()
