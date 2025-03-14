import pytest
from src.generators.SCMGenerator import SCMGenerator
from src.lib.models.scm.SCM import SCM
from src.lib.models.scm.DAG import DAG
from src.generators.DagGenerator import DAGGenerator
from scipy.stats import norm, uniform
import numpy as np

# Test DAGs
test_dag_a = DAGGenerator(
    num_nodes=10,
    num_roots=2,
    num_leaves=2,
    edge_density=0.5,
    max_in_degree=3,
    max_out_degree=3,
    min_path_length=1,
    max_path_length=4,
    random_state=np.random.default_rng(42),
).generate()
test_dag_b = DAGGenerator(
    num_nodes=8,
    num_roots=2,
    num_leaves=2,
    edge_density=0.3,
    max_in_degree=3,
    max_out_degree=3,
    min_path_length=2,
    max_path_length=3,
    random_state=np.random.default_rng(911),
).generate()


@pytest.mark.parametrize(
    "dag, num_nodes",
    [(test_dag_a, len(test_dag_a.nodes)), (test_dag_b, len(test_dag_b.nodes))],
)
def test_scm_generator(dag, num_nodes):
    """Test if SCMGenerator correctly maps DAG to SCM."""

    variable_types = {f"X{i}": "numerical" for i in range(1, num_nodes + 1)}
    variable_domains = {f"X{i}": (-1, 1) for i in range(1, num_nodes + 1)}

    scm_generator = SCMGenerator(
        dag=dag,
        variable_types=variable_types,
        variable_domains=variable_domains,
        user_constraints={"max_terms": 2},
        allowed_operations=["+", "*"],
        allowed_functions=[lambda x: x**2],
        noise_distributions={
            "gaussian": norm(loc=0, scale=0.1),
            "uniform": uniform(loc=-0.1, scale=0.2),
        },
        random_state=np.random.default_rng(42),
    )

    scm = scm_generator.generate()

    assert isinstance(scm, SCM), "Output should be an instance of SCM."
    assert (
        len(scm.nodes) == num_nodes
    ), "SCM should have the same number of nodes as DAG."


@pytest.mark.parametrize(
    "dag, num_nodes, seed",
    [
        (test_dag_a, len(test_dag_a.nodes), 123),
        (test_dag_b, len(test_dag_b.nodes), 911),
    ],
)
def test_scm_reproducibility(dag, num_nodes, seed):
    """
    Test if SCMGenerator is reproducible with the same seed.
    """

    variable_types = {f"X{i}": "numerical" for i in range(1, num_nodes + 1)}
    variable_domains = {f"X{i}": (-1, 1) for i in range(1, num_nodes + 1)}

    scm_generator_a = SCMGenerator(
        dag=dag,
        variable_types=variable_types,
        variable_domains=variable_domains,
        user_constraints={"max_terms": 2},
        allowed_operations=["+", "*"],
        allowed_functions=[lambda x: x**2],
        noise_distributions={
            "gaussian": norm(loc=0, scale=0.1),
            "uniform": uniform(loc=-0.1, scale=0.2),
        },
        random_state=np.random.default_rng(seed),
    )
    scm_a = scm_generator_a.generate()

    scm_generator_b = SCMGenerator(
        dag=dag,
        variable_types=variable_types,
        variable_domains=variable_domains,
        user_constraints={"max_terms": 2},
        allowed_operations=["+", "*"],
        allowed_functions=[lambda x: x**2],
        noise_distributions={
            "gaussian": norm(loc=0, scale=0.1),
            "uniform": uniform(loc=-0.1, scale=0.2),
        },
        random_state=np.random.default_rng(seed),
    )
    scm_b = scm_generator_b.generate()

    # Check if the SCMs are equal
    nodes_a = [node.to_dict() for node in scm_a.nodes.values()]
    nodes_b = [node.to_dict() for node in scm_b.nodes.values()]

    assert nodes_a == nodes_b, "SCMs should be equal with the same seed."


@pytest.mark.parametrize(
    "dag, num_nodes, seed1, seed2",
    [
        (test_dag_a, len(test_dag_a.nodes), 123, 321),
        (test_dag_b, len(test_dag_b.nodes), 911, 119),
    ],
)
def test_scm_variability(dag, num_nodes, seed1, seed2):
    """
    Test if SCMGenerator is reproducible with the same seed.
    """

    variable_types = {f"X{i}": "numerical" for i in range(1, num_nodes + 1)}
    variable_domains = {f"X{i}": (-1, 1) for i in range(1, num_nodes + 1)}

    scm_generator_a = SCMGenerator(
        dag=dag,
        variable_types=variable_types,
        variable_domains=variable_domains,
        user_constraints={"max_terms": 2},
        allowed_operations=["+", "*"],
        allowed_functions=[lambda x: x**2],
        noise_distributions={
            "gaussian": norm(loc=0, scale=0.1),
            "uniform": uniform(loc=-0.1, scale=0.2),
        },
        random_state=np.random.default_rng(seed1),
    )
    scm_a = scm_generator_a.generate()

    scm_generator_b = SCMGenerator(
        dag=dag,
        variable_types=variable_types,
        variable_domains=variable_domains,
        user_constraints={"max_terms": 2},
        allowed_operations=["+", "*"],
        allowed_functions=[lambda x: x**2],
        noise_distributions={
            "gaussian": norm(loc=0, scale=0.1),
            "uniform": uniform(loc=-0.1, scale=0.2),
        },
        random_state=np.random.default_rng(seed2),
    )
    scm_b = scm_generator_b.generate()

    # Check if the SCMs are not equal
    nodes_a = [node.to_dict() for node in scm_a.nodes.values()]
    nodes_b = [node.to_dict() for node in scm_b.nodes.values()]

    assert nodes_a != nodes_b, "SCMs should not be equal with the different seeds."


@pytest.mark.parametrize(
    "dag, num_nodes, seed",
    [
        (test_dag_a, len(test_dag_a.nodes), 123),
        (test_dag_b, len(test_dag_b.nodes), 911),
    ],
)
def test_scm_serialization(dag, num_nodes, seed):
    """
    Test if SCMGenerator is reproducible with the same seed.
    """

    variable_types = {f"X{i}": "numerical" for i in range(1, num_nodes + 1)}
    variable_domains = {f"X{i}": (-1, 1) for i in range(1, num_nodes + 1)}

    scm_generator = SCMGenerator(
        dag=dag,
        variable_types=variable_types,
        variable_domains=variable_domains,
        user_constraints={"max_terms": 2},
        allowed_operations=["+", "*"],
        allowed_functions=[lambda x: x**2],
        noise_distributions={
            "gaussian": norm(loc=0, scale=0.1),
            "uniform": uniform(loc=-0.1, scale=0.2),
        },
        random_state=np.random.default_rng(seed),
    )
    scm = scm_generator.generate()

    # Serialize the SCM
    scm_data = scm.to_dict()
    # Deserialize the SCM
    scm_deserialized = SCM.from_dict(scm_data, np.random.default_rng(seed))

    # Check if the SCMs are not equal
    nodes_a = [
        node.to_dict() for node in sorted(scm.nodes.values(), key=lambda n: n.name)
    ]
    nodes_b = [
        node.to_dict()
        for node in sorted(scm_deserialized.nodes.values(), key=lambda n: n.name)
    ]

    assert nodes_a == nodes_b, "SCMs should be equal after serialization."


@pytest.mark.parametrize(
    "dag, num_nodes, num_samples, seed",
    [
        (test_dag_a, len(test_dag_a.nodes), 10, 123),
        (test_dag_b, len(test_dag_b.nodes), 21, 911),
    ],
)
def test_scm_samples_reproducibility(dag, num_nodes, num_samples, seed):
    """
    Test if SCMGenerator is reproducible with the same seed.
    """

    variable_types = {f"X{i}": "numerical" for i in range(1, num_nodes + 1)}
    variable_domains = {f"X{i}": (-1, 1) for i in range(1, num_nodes + 1)}

    scm_generator = SCMGenerator(
        dag=dag,
        variable_types=variable_types,
        variable_domains=variable_domains,
        user_constraints={"max_terms": 2},
        allowed_operations=["+", "*"],
        allowed_functions=[lambda x: x**2],
        noise_distributions={
            "gaussian": norm(loc=0, scale=0.1),
            "uniform": uniform(loc=-0.1, scale=0.2),
        },
        random_state=np.random.default_rng(seed),
    )
    scm = scm_generator.generate()

    # Serialize the SCM
    scm_data = scm.to_dict()
    # Deserialize the SCM
    scm_deserialized = SCM.from_dict(scm_data, np.random.default_rng(seed))

    # Generate samples
    samples_a = scm.generate_samples(num_samples=num_samples)
    samples_b = scm_deserialized.generate_samples(num_samples=num_samples)

    print(f"Keys A: {list(samples_a[0].keys())}")
    print(f"Keys B: {list(samples_b[0].keys())}")
    print(f"Values A: {list(samples_a[0].values())}")
    print(f"Values B: {list(samples_b[0].values())}")

    assert sorted(samples_a) == sorted(
        samples_b
    ), "Samples should be equal with the same seed."


# - scm sobre los grafos generados anteriormente y comparar si los samples son iguales.
# - scm sobre los grafos generados anteriormente y comparar si los samples son diferentes.
# - SCM de verdad son SCMs.
# - No solo a nivel de objectos que sean iguales, pero que la serializacion sera la misma.
