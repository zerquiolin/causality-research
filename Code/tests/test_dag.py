import pytest
import networkx as nx
from src.generators.DagGenerator import DAGGenerator
from src.lib.models.scm.DAG import DAG
import numpy as np
from networkx.algorithms import isomorphism


@pytest.mark.parametrize(
    "seed, num_nodes, num_roots, num_leaves, edge_density, max_in_degree, max_out_degree, min_path_length, max_path_length",
    [(42, 10, 2, 2, 0.5, 3, 3, 1, 4), (123, 8, 2, 2, 0.3, 3, 3, 2, 3)],
)
def test_dag_reproducibility(
    seed,
    num_nodes,
    num_roots,
    num_leaves,
    edge_density,
    max_in_degree,
    max_out_degree,
    min_path_length,
    max_path_length,
):
    dag_a = DAGGenerator(
        num_nodes=num_nodes,
        num_roots=num_roots,
        num_leaves=num_leaves,
        edge_density=edge_density,
        max_in_degree=max_in_degree,
        max_out_degree=max_out_degree,
        min_path_length=min_path_length,
        max_path_length=max_path_length,
        random_state=np.random.default_rng(seed),
    ).generate()
    dag_b = DAGGenerator(
        num_nodes=num_nodes,
        num_roots=num_roots,
        num_leaves=num_leaves,
        edge_density=edge_density,
        max_in_degree=max_in_degree,
        max_out_degree=max_out_degree,
        min_path_length=min_path_length,
        max_path_length=max_path_length,
        random_state=np.random.default_rng(seed),
    ).generate()

    # Check if the DAGs are isomorphic
    GM = isomorphism.GraphMatcher(dag_a.graph, dag_b.graph)
    assert GM.is_isomorphic(), "DAGs must not be isomorphic with the same seed."


@pytest.mark.parametrize(
    "seed1, seed2, num_nodes, num_roots, num_leaves, edge_density, max_in_degree, max_out_degree, min_path_length, max_path_length",
    [(42, 43, 10, 2, 2, 0.5, 3, 3, 1, 4), (123, 456, 15, 3, 3, 0.3, 3, 3, 1, 4)],
)
def test_dag_variability(
    seed1,
    seed2,
    num_nodes,
    num_roots,
    num_leaves,
    edge_density,
    max_in_degree,
    max_out_degree,
    min_path_length,
    max_path_length,
):
    dag_a = DAGGenerator(
        num_nodes=num_nodes,
        num_roots=num_roots,
        num_leaves=num_leaves,
        edge_density=edge_density,
        max_in_degree=max_in_degree,
        max_out_degree=max_out_degree,
        min_path_length=min_path_length,
        max_path_length=max_path_length,
        random_state=np.random.default_rng(seed1),
    ).generate()
    dag_b = DAGGenerator(
        num_nodes=num_nodes,
        num_roots=num_roots,
        num_leaves=num_leaves,
        edge_density=edge_density,
        max_in_degree=max_in_degree,
        max_out_degree=max_out_degree,
        min_path_length=min_path_length,
        max_path_length=max_path_length,
        random_state=np.random.default_rng(seed2),
    ).generate()

    # Check if the DAGs are isomorphic
    GM = isomorphism.GraphMatcher(dag_a.graph, dag_b.graph)
    assert not GM.is_isomorphic(), "DAGs must not be isomorphic with different seeds."


@pytest.mark.parametrize(
    "seed, num_nodes, num_roots, num_leaves, edge_density, max_in_degree, max_out_degree, min_path_length, max_path_length",
    [(123, 10, 2, 2, 0.5, 3, 3, 1, 4), (911, 9, 3, 3, 0.3, 3, 3, 1, 4)],
)
def test_dag_serialization(
    seed,
    num_nodes,
    num_roots,
    num_leaves,
    edge_density,
    max_in_degree,
    max_out_degree,
    min_path_length,
    max_path_length,
):
    dag = DAGGenerator(
        num_nodes=num_nodes,
        num_roots=num_roots,
        num_leaves=num_leaves,
        edge_density=edge_density,
        max_in_degree=max_in_degree,
        max_out_degree=max_out_degree,
        min_path_length=min_path_length,
        max_path_length=max_path_length,
        random_state=np.random.default_rng(seed),
    ).generate()
    # Serialize the DAG
    dag_data = dag.to_dict()
    # Deserialize the DAG
    dag_copy = DAG.from_dict(dag_data)
    # Check if the DAGs are isomorphic
    GM = isomorphism.GraphMatcher(dag.graph, dag_copy.graph)
    assert GM.is_isomorphic(), "DAGs must be isomorphic after serialization."


@pytest.mark.parametrize(
    "seed, num_nodes, num_roots, num_leaves, edge_density, max_in_degree, max_out_degree, min_path_length, max_path_length",
    [(123, 10, 2, 2, 0.5, 3, 3, 1, 4), (911, 9, 3, 3, 0.3, 3, 3, 1, 4)],
)
def test_dag_is_valid(
    seed,
    num_nodes,
    num_roots,
    num_leaves,
    edge_density,
    max_in_degree,
    max_out_degree,
    min_path_length,
    max_path_length,
):
    dag = DAGGenerator(
        num_nodes=num_nodes,
        num_roots=num_roots,
        num_leaves=num_leaves,
        edge_density=edge_density,
        max_in_degree=max_in_degree,
        max_out_degree=max_out_degree,
        min_path_length=min_path_length,
        max_path_length=max_path_length,
        random_state=np.random.default_rng(seed),
    ).generate()
    assert nx.is_directed_acyclic_graph(dag.graph), "Graph must be a DAG."
    assert (
        len(dag.graph.nodes()) == num_nodes
    ), "DAG should have the expected number of nodes."
