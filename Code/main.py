from src.generators.DagGenerator import DAGGenerator
from src.lib.models.scm.DAG import DAG
from src.generators.SCMGenerator import SCMGenerator

import random
import logging
import sympy as sp
from scipy.stats import norm, uniform

logging.basicConfig(level=logging.INFO)


def main():
    # Set a seed for reproducibility.
    seed = 42

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
    )
    dag_graph = dag_gen.generate()
    dag = DAG(dag_graph)

    dag.plot()

    # Define user constraints for SCM generation.
    # Define variable types
    variable_types = {
        node: "numerical" if random.random() < 0.8 else "categorical"
        for node in dag.nodes
    }

    # Define variable domains
    variable_domains = {}
    for node, vtype in variable_types.items():
        if vtype == "numerical":
            # For numerical nodes, assign a random interval.
            # Here, we choose lower bound between -10 and -1, and upper bound between 1 and 10.
            lower = random.randint(-10, -1)
            upper = random.randint(1, 10)
            variable_domains[node] = (lower, upper)
        else:
            # For categorical nodes, assign a random list of categories.
            # For example, generate between 2 and 4 categories using letters.
            num_categories = random.randint(2, 4)
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
        dag.graph,
        variable_types,
        variable_domains,
        user_constraints,
        allowed_operations,
        allowed_functions,
        noise_distributions,
    )
    scm = scm_generator.generate()

    print("\nObservational Samples:")
    for sample in scm.generate_samples(num_samples=5):
        print(sample)

    print("\nInterventional Samples (X1 = 2):")
    for sample in scm.generate_samples(interventions={"X1": 2}, num_samples=5):
        print(sample)


if __name__ == "__main__":
    main()
