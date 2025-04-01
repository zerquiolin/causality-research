from src.scm.dag import DAG
from src.generators.dag_generator import DAGGenerator
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
        num_nodes=3,
        num_roots=1,
        num_leaves=1,
        edge_density=0.3,
        max_in_degree=1,
        max_out_degree=1,
        min_path_length=1,
        max_path_length=2,
        random_state=random_state,
    )
    dag_obj = dag_gen.generate()

    # dag_obj.plot()

    # Define constraints for SCM generation
    scm_constraints = {
        "variable_types": {f"X{i}": "categorical" for i in range(1, 4)},
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
        # scm_constraints["variable_domains"][node] = [0, 1]
        scm_constraints["variable_domains"][node] = ["0", "1"]

    # Generate the SCM
    scm_gen = SCMGenerator(dag_obj, **scm_constraints, random_state=random_state)
    scm = scm_gen.generate()

    for name, node in scm.nodes.items():
        print(f"Equation for {name}")
        print(node.equation)

    # Create a GameInstance with the generated SCM
    game_instance = GameInstance(scm, random_state)

    # Create a RandomAgent
    random_agent = RandomAgent()
    greedy_agent = GreedyAgent()

    # Create the Environment using the GameInstance and Agent
    # env = Environment(
    #     game_instance, random_agent, max_rounds=10, random_state=random_state
    # )
    env = Environment(
        game_instance, greedy_agent, max_rounds=10, random_state=random_state
    )

    # print(f"env: {env}, game_instance: {game_instance}, agent: {agent}, scm: {scm}")

    # Run the game simulation
    final_state, final_history = env.run_game()
    print("\n🏁 Game simulation complete!")
    # print(f"Final State: {final_state}")
    # print(f"Final Dataset: {final_state[0]['datasets']}")

    # Retrieve and display the state-action history
    game_history_df = env.get_game_history()

    print("\n📊 Game History DataFrame:")
    print(game_history_df)

    # Show the agent's learned DAG
    print("\n🧠 Learned DAG edges:")
    print(final_state["final_answer"])

    # Create a DAG custom object
    learned_dag = DAG(final_state["final_answer"])

    dag_obj.plot()
    learned_dag.plot()


if __name__ == "__main__":
    main()
