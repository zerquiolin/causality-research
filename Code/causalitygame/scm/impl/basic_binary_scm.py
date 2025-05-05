# Math
import numpy as np
import sympy as sp
from scipy.stats import norm, uniform

# Generators
from causalitygame.generators.dag_generator import DAGGenerator
from causalitygame.generators.scm_generator import SCMGenerator


def gen_binary_scm(
    random_state: np.random.RandomState = np.random.RandomState(42),
):
    # Generate a DAG
    dag_gen = DAGGenerator(
        num_nodes=3,
        num_roots=1,
        num_leaves=1,
        edge_density=0.5,
        max_in_degree=2,
        max_out_degree=2,
        min_path_length=1,
        max_path_length=3,
        random_state=random_state,
    )
    dag = dag_gen.generate()

    # Define constraints for SCM generation
    scm_constraints = {
        "variable_types": {f"X{i}": "categorical" for i in range(1, 4)},
        "variable_domains": {},
        "user_constraints": {
            "max_terms": 3,
            "allow_non_linearity": False,
            "allow_variable_exponents": False,
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
        scm_constraints["variable_domains"][node] = [0, 1]

    # Generate the SCM
    # scm_gen = SCMGenerator(dag, **scm_constraints, random_state=scm_random_state)
    scm_gen = SCMGenerator(dag, **scm_constraints, random_state=random_state)
    scm = scm_gen.generate()

    return dag, scm
