from src.generators.dag_generator import DAGGenerator
from src.scm.dag import DAG
from src.generators.scm_generator import SCMGenerator

import numpy as np
import logging
import sympy as sp
from scipy.stats import norm, uniform

logging.basicConfig(level=logging.INFO)


def main():
    # Set a seed for reproducibility.
    seed = 42
    random_state = np.random.RandomState(seed)

    # Instantiate an advanced DAG generator with complex parameters.
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
    dag = dag_gen.generate()

    dag.plot()

    # Define user constraints for SCM generation.
    # Define variable types
    variable_types = {
        node: "numerical" if np.random.random() < 0.8 else "categorical"
        for node in dag.nodes
    }

    # Define variable domains
    variable_domains = {}
    for node, vtype in variable_types.items():
        if vtype == "numerical":
            # For numerical nodes, assign a random interval.
            # Here, we choose lower bound between -10 and -1, and upper bound between 1 and 10.
            lower = np.random.randint(-10, -1)
            upper = np.random.randint(1, 10)
            variable_domains[node] = (lower, upper)
        else:
            # For categorical nodes, assign a random list of categories.
            # For example, generate between 2 and 4 categories using letters.
            num_categories = np.random.randint(2, 4)
            # This will generate categories like ['A', 'B', 'C'].
            categories = [chr(65 + i) for i in range(num_categories)]
            variable_domains[node] = categories

    # Define user constraints and allowed functions/distributions
    user_constraints = {
        "max_terms": 3,
        "allow_non_linearity": True,
        "allow_variable_exponents": True,
    }
    allowed_operations = ["+", "-", "*", "/"]
    allowed_functions = [sp.sin, sp.exp, sp.log]
    noise_distributions = {
        "gaussian": norm(loc=0, scale=0.1),
        "uniform": uniform(loc=-0.1, scale=0.2),
    }

    scm_generator = SCMGenerator(
        dag,
        variable_types,
        variable_domains,
        user_constraints,
        allowed_operations,
        allowed_functions,
        noise_distributions,
        random_state=random_state,
    )
    scm = scm_generator.generate()

    print("\nObservational Samples:")
    for sample in scm.generate_samples(num_samples=5, random_state=random_state):
        print(sample)

    print("\nInterventional Samples (X1 = 2):")
    for sample in scm.generate_samples(
        interventions={"X1": 2}, num_samples=5, random_state=random_state
    ):
        print(sample)


if __name__ == "__main__":
    main()
