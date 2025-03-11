import json
from networkx.readwrite import json_graph

from src.generators.DagGenerator import DAGGenerator
from src.lib.models.scm.DAG import DAG
from src.generators.SCMGenerator import SCMGenerator
from src.lib.models.scm.SCM import SCM


class GameInstance:
    def __init__(self, dag: DAG, scm: SCM, random_state=911):
        self.dag = dag
        self.scm = scm
        self.random_state = random_state

    def to_dict(self):
        # Use the 'edges' kwarg to address the FutureWarning.
        dag_data = json_graph.node_link_data(self.dag.graph, edges="edges")
        scm_data = self.scm.to_dict()
        return {"dag": dag_data, "scm": scm_data, "random_state": self.random_state}

    @classmethod
    def from_dict(cls, data):
        dag_graph = json_graph.node_link_graph(data["dag"], edges="edges")
        dag = DAG(dag_graph)
        scm = SCM.from_dict(data["scm"])
        return cls(dag, scm)

    def save(self, filename):
        """Save the game instance to a JSON file."""
        with open(filename, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load(cls, filename):
        """Load a game instance from a JSON file."""
        with open(filename, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)


class GameInstanceCreator:
    """
    Creates a game instance by first generating a DAG using the DAGGenerator
    and then generating an SCM using the SCMGenerator.
    """

    def __init__(
        self, dag_generator_params: dict, scm_generator_params: dict, random_state=911
    ):
        self.dag_generator_params = dag_generator_params
        self.scm_generator_params = scm_generator_params
        self.random_state = random_state

    def create_instance(self) -> GameInstance:
        dag_gen = DAGGenerator(**self.dag_generator_params)
        dag_graph = dag_gen.generate()
        dag = DAG(dag_graph)

        scm_gen = SCMGenerator(graph=dag_graph, **self.scm_generator_params)
        scm = scm_gen.generate()

        return GameInstance(dag, scm, random_state=self.random_state)
