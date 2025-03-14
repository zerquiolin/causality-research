import pytest
from src.generators.SCMGenerator import SCMGenerator
from src.lib.models.scm.SCM import SCM
from src.lib.models.scm.DAG import DAG
from src.generators.DagGenerator import DAGGenerator
import numpy as np


@pytest.mark.parametrize("num_nodes", [5, 10, 15])
def test_scm_generator(num_nodes):
    """Test if SCMGenerator correctly maps DAG to SCM."""
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

    variable_types = {f"X{i}": "numerical" for i in range(1, num_nodes + 1)}
    variable_domains = {f"X{i}": (-1, 1) for i in range(1, num_nodes + 1)}

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

    assert isinstance(scm, SCM), "Output should be an instance of SCM"
    assert (
        len(scm.nodes) == num_nodes
    ), "SCM should have the same number of nodes as DAG"


# 1. dos veces con la misma semilla, deberian ser iguales.
# 2. dos veces con semillas diferentes, deberian ser diferentes.
# 10 diferentes semillas, 10 grafos y siempre deberian ser identicos.
# SCM: 10 diferentes semillas, 10 SCMs y siempre deberian ser identicos.
# - scm sobre los grafos y comparar si los scm son iguales.
# - scm sobre los grafos y comparar si los samples son iguales.
# - scm sobre los grafos y comparar si los samples son diferentes.
# DAGS de verdad son DAGS.
# SCM de verdad son SCMs.
# No solo a nivel de objectos que sean iguales, pero que la serializacion sera la misma.
# Buscar que la serializacion sea la misma. -> Con la misma semilla dos veces aplicado genera las mismas instances y la serializacion de ellas es equivalente.
