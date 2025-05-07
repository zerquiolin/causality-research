# Math
import numpy as np
import sympy as sp

# Noise Distributions
from causalitygame.scm.noise_distributions import (
    GaussianNoiseDistribution,
    UniformNoiseDistribution,
)

# Generators
from causalitygame.generators.dag_generator import DAGGenerator
from causalitygame.generators.scm_generator import EquationBasedSCMGenerator


def gen_binary_scm(random_state: int = 42):
    rs = np.random.RandomState(random_state)
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
        random_state=rs,
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
        "noise_distributions": [
            GaussianNoiseDistribution(mean=0, std=1),
            UniformNoiseDistribution(low=-1, high=1),
        ],
    }

    # Define variable domains
    for node, vtype in scm_constraints["variable_types"].items():
        scm_constraints["variable_domains"][node] = [0, 1]

    # Generate the SCM
    # scm_gen = SCMGenerator(dag, **scm_constraints, random_state=scm_random_state)
    scm_gen = EquationBasedSCMGenerator(dag, **scm_constraints, random_state=rs)
    scm = scm_gen.generate()

    return dag, scm
