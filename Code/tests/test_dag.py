import pytest
import networkx as nx
from src.generators.DagGenerator import DAGGenerator
from src.lib.models.scm.DAG import DAG
import numpy as np


@pytest.mark.parametrize(
    "num_nodes, num_roots, num_leaves, edge_density",
    [
        (5, 1, 1, 0.5),
        (10, 2, 2, 0.7),
        (15, 3, 3, 0.3),
    ],
)
def test_dag_generator(num_nodes, num_roots, num_leaves, edge_density):
    """Test if DAGGenerator produces a valid DAG."""
    generator = DAGGenerator(
        num_nodes=num_nodes,
        num_roots=num_roots,
        num_leaves=num_leaves,
        edge_density=edge_density,
        max_in_degree=3,
        max_out_degree=3,
        min_path_length=1,
        max_path_length=4,
        random_state=np.random.default_rng(42),
    )
    dag = generator.generate()

    assert isinstance(dag, DAG), "Output should be an instance of DAG"
    assert dag.graph.number_of_nodes() == num_nodes, "Incorrect number of nodes"
    assert nx.is_directed_acyclic_graph(dag.graph), "Graph is not a DAG"
