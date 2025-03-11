import sympy as sp
import numpy as np
from src.game.GameInstance import GameInstance, GameInstanceCreator
from scipy.stats import norm, uniform
import datetime

# ===================== Example Usage =====================
if __name__ == "__main__":
    # Example parameters for DAG generation.
    dag_params = {
        "num_nodes": 10,
        "num_roots": 3,
        "num_leaves": 3,
        "edge_density": 0.3,
        "max_in_degree": 3,
        "max_out_degree": 3,
        "min_path_length": 2,
        "max_path_length": 5,
    }

    # Example parameters for SCM generation.
    scm_params = {
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
    for node, vtype in scm_params["variable_types"].items():
        if vtype == "numerical":
            # For numerical nodes, assign a random interval.
            # Here, we choose lower bound between -10 and -1, and upper bound between 1 and 10.
            lower = np.random.randint(-10, -1)
            upper = np.random.randint(1, 10)
            scm_params["variable_domains"][node] = (lower, upper)
        else:
            # For categorical nodes, assign a random list of categories.
            # For example, generate between 2 and 4 categories using letters.
            num_categories = np.random.randint(2, 4)
            # This will generate categories like ['A', 'B', 'C'].
            categories = [chr(65 + i) for i in range(num_categories)]
            scm_params["variable_domains"][node] = categories

    # Create a game instance.
    creator = GameInstanceCreator(dag_params, scm_params)
    game_instance = creator.create_instance()

    gen_random_state = np.random.RandomState(123)
    load_random_state = np.random.RandomState(123)

    # Generate samples from the SCM.
    samples = game_instance.scm.generate_samples(
        num_samples=3, random_state=gen_random_state
    )
    print("Samples:", samples)

    file_path = f"./instances/game_instance-{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"

    # Save the instance.
    game_instance.save(file_path)

    # Later, load the instance.
    loaded_instance = GameInstance.load(file_path)

    print(
        "Loaded Samples:",
        loaded_instance.scm.generate_samples(
            num_samples=3, random_state=load_random_state
        ),
    )
