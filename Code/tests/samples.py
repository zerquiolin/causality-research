import pytest
import pandas as pd
from src.lib.models.scm.SCM import SCM
from src.generators.SCMGenerator import SCMGenerator
from src.generators.DagGenerator import DAGGenerator
import numpy as np

@pytest.mark.parametrize("num_nodes, num_samples", [(5, 10), (10, 50)])
def test_sample_generation(num_nodes, num_samples):
    """Test if SCM generates valid samples."""
    dag_generator = DAGGenerator(
        num_nodes=num_nodes,
        num_roots=1,
        num_leaves=1,
        edge_density=0.5,
        max_in_degree=3,
        max_out_degree=3,
        min_path_length=1,
        max_path_length=4,
        random_state=np.random.default_rng(42),
    )
    dag = dag_generator.generate()

    variable_types = {f"X{i}": "numerical" for i in range(1, num_nodes+1)}
    variable_domains = {f"X{i}": (-1, 1) for i in range(1, num_nodes+1)}

    scm_generator = SCMGenerator(
        dag=dag,
        variable_types=variable_types,
        variable_domains=variable_domains,
        user_constraints={"max_terms": 2},
        allowed_operations=["+", "*"],
        allowed_functions=[lambda x: x**2],
        noise_distributions={"normal": np.random.normal},
        random_state=np.random.default_rng(42),
    )

    scm = scm_generator.generate()
    samples = scm.generate_samples(num_samples=num_samples)

    assert isinstance(samples, list), "Samples should be a list of dictionaries"
    assert len(samples) == num_samples, "Incorrect number of samples"
    assert all(isinstance(sample, dict) for sample in samples), "Each sample should be a dictionary"
