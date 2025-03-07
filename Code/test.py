# import json
# import random
# import networkx as nx
# import sympy as sp
# import numpy as np
# from networkx.readwrite import json_graph


# # ===================== SCM Node =====================
# class SCMNode:
#     def __init__(
#         self,
#         name,
#         equation,
#         domain,
#         var_type,
#         cdf_mappings=None,
#         category_mappings=None,
#     ):
#         """
#         Represents a single node in the Structural Causal Model (SCM).
#         For numerical nodes, `equation` is used for generating values.
#         For categorical nodes, the equation is not used (set to 0) and values are determined via CDF mappings.
#         """
#         self.name = name
#         self.equation = equation  # A sympy expression.
#         self.domain = domain  # Tuple for numerical or list for categorical.
#         self.var_type = var_type  # "numerical" or "categorical"
#         self.cdf_mappings = cdf_mappings or {}  # (Not easily serializable.)
#         self.category_mappings = (
#             category_mappings or {}
#         )  # (Only simple numeric mappings.)
#         self.input_numeric = (
#             None  # Used for categorical nodes to supply numeric value to children.
#         )

#     def generate_value(self, parent_values):
#         """Generates a value for this node given parent values."""
#         if self.var_type == "numerical":
#             subs_dict = {}
#             for var in self.equation.free_symbols:
#                 var_str = str(var)
#                 # Use parent's numeric mapping if available.
#                 if var_str + "_num" in parent_values:
#                     subs_dict[var] = parent_values[var_str + "_num"]
#                 else:
#                     subs_dict[var] = parent_values.get(
#                         var_str, np.random.uniform(-1, 1)
#                     )
#             eval_equation = self.equation.subs(subs_dict).evalf()
#             if not eval_equation.is_number:
#                 print(f"Warning: Unresolved symbols in {self.name} ->", eval_equation)
#             # Ensure we get a real number.
#             result = (
#                 float(eval_equation.as_real_imag()[0])
#                 if not eval_equation.is_real
#                 else float(eval_equation)
#             )
#             if isinstance(self.domain, tuple):
#                 min_val, max_val = self.domain
#                 result = max(min_val, min(max_val, result))
#             return result
#         else:
#             # For categorical nodes, use the precomputed CDF mappings.
#             category_probs = {
#                 cat: self.cdf_mappings[cat](np.random.uniform(-1, 1))
#                 for cat in self.cdf_mappings
#             }
#             chosen_category = max(category_probs, key=lambda cat: category_probs[cat])
#             self.input_numeric = self.category_mappings.get(chosen_category)
#             return chosen_category

#     def to_dict(self):
#         """Serialize the node to a dict."""
#         return {
#             "name": self.name,
#             "equation": sp.srepr(self.equation),  # Save string representation.
#             "domain": self.domain,
#             "var_type": self.var_type,
#             "category_mappings": self.category_mappings,
#             "input_numeric": self.input_numeric,
#         }

#     @classmethod
#     def from_dict(cls, data):
#         """Deserialize from dict (re-generates the sympy equation from its string)."""
#         equation = sp.sympify(data["equation"])
#         node = cls(
#             name=data["name"],
#             equation=equation,
#             domain=data["domain"],
#             var_type=data["var_type"],
#             category_mappings=data.get("category_mappings", {}),
#         )
#         node.input_numeric = data.get("input_numeric")
#         # Note: cdf_mappings cannot be easily serialized; they must be re-generated if needed.
#         return node


# # ===================== SCM =====================
# class SCM:
#     def __init__(self, nodes):
#         """
#         Structural Causal Model that takes a list of nodes in topological order.
#         """
#         self.nodes = {node.name: node for node in nodes}

#     def generate_sample(self, interventions={}):
#         """Generates a single sample by evaluating each node in topological order."""
#         sample = {}
#         # Process nodes in order (assumed topological).
#         for node_name, node in self.nodes.items():
#             if node_name in interventions:
#                 sample[node_name] = interventions[node_name]
#                 if node.var_type == "categorical":
#                     sample[node_name + "_num"] = node.input_numeric
#             else:
#                 value = node.generate_value(sample)
#                 sample[node_name] = value
#                 if node.var_type == "categorical":
#                     sample[node_name + "_num"] = node.input_numeric
#         return sample

#     def generate_samples(self, interventions={}, num_samples=1, seed=None):
#         """
#         Generates multiple samples.
#         If seed is provided, seeds the random generators for reproducibility.
#         """
#         if seed is not None:
#             random.seed(seed)
#             np.random.seed(seed)
#         return [self.generate_sample(interventions) for _ in range(num_samples)]

#     def to_dict(self):
#         """Serialize the SCM as a dict."""
#         nodes_data = {name: node.to_dict() for name, node in self.nodes.items()}
#         return {"nodes": nodes_data}

#     @classmethod
#     def from_dict(cls, data):
#         nodes = [SCMNode.from_dict(nd) for nd in data["nodes"].values()]
#         # Optionally, sort nodes (if names are of the form 'X1', 'X2', …).
#         nodes.sort(key=lambda n: int(n.name[1:]))
#         return cls(nodes)


# # ===================== Game Instance =====================
# class GameInstance:
#     def __init__(self, dag: nx.DiGraph, scm: SCM):
#         self.dag = dag
#         self.scm = scm

#     def to_dict(self):
#         """Serialize the game instance (both DAG and SCM) to a dict."""
#         dag_data = json_graph.node_link_data(self.dag)
#         scm_data = self.scm.to_dict()
#         return {"dag": dag_data, "scm": scm_data}

#     @classmethod
#     def from_dict(cls, data):
#         dag = json_graph.node_link_graph(data["dag"])
#         scm = SCM.from_dict(data["scm"])
#         return cls(dag, scm)

#     def save(self, filename):
#         """Save the game instance to a JSON file."""
#         with open(filename, "w") as f:
#             json.dump(self.to_dict(), f)

#     @classmethod
#     def load(cls, filename):
#         """Load a game instance from a JSON file."""
#         with open(filename, "r") as f:
#             data = json.load(f)
#         return cls.from_dict(data)


# # ===================== Game Instance Creator =====================
# class GameInstanceCreator:
#     """
#     Creates a game instance by first generating a DAG using the DAGGenerator
#     and then generating an SCM using the SCMGenerator.
#     """

#     def __init__(self, dag_generator_params: dict, scm_generator_params: dict):
#         self.dag_generator_params = dag_generator_params
#         self.scm_generator_params = scm_generator_params

#     def create_instance(self) -> GameInstance:
#         # Create the DAG using your existing DAGGenerator.
#         from src.lib.models.abstract.BaseDagGenerator import (
#             BaseDAGGenerator,
#         )  # assuming your DAGGenerator inherits from this

#         # (Replace 'DAGGenerator' with your concrete class name if different.)
#         dag_gen = DAGGenerator(**self.dag_generator_params)
#         dag = dag_gen.generate()

#         # Create the SCM using your SCMGenerator.
#         # (Assuming SCMGenerator takes the DAG and additional parameters.)
#         from src.lib.models.some_module import (
#             SCMGenerator,
#         )  # adjust the import accordingly

#         scm_gen = SCMGenerator(graph=dag, **self.scm_generator_params)
#         scm = scm_gen.generate_scm()

#         return GameInstance(dag, scm)

import sympy as sp
import numpy as np
from src.game.GameInstance import GameInstance, GameInstanceCreator
from scipy.stats import norm, uniform

# ===================== Example Usage =====================
if __name__ == "__main__":
    # Example parameters for DAG generation.
    dag_params = {
        "num_nodes": 10,
        "num_roots": 3,
        "num_leaves": 3,
        "edge_density": 0.3,
        "max_in_degree": 3,
        "max_out_degree": 3,
        "min_path_length": 2,
        "max_path_length": 5,
    }

    # Example parameters for SCM generation.
    scm_params = {
        "variable_types": {
            f"X{i}": "numerical" if np.random.rand() < 0.8 else "categorical"
            for i in range(1, 11)
        },
        "variable_domains": {},
        "user_constraints": {
            "max_terms": 3,
            "allow_non_linearity": True,
            "allow_variable_exponents": True,
        },
        "allowed_operations": ["+", "-", "*", "/"],
        "allowed_functions": [sp.sin, sp.exp, sp.log],
        "noise_distributions": {
            "gaussian": norm(loc=0, scale=0.1),
            "uniform": uniform(loc=-0.1, scale=0.2),
        },
    }

    # Define variable domains
    for node, vtype in scm_params["variable_types"].items():
        if vtype == "numerical":
            # For numerical nodes, assign a random interval.
            # Here, we choose lower bound between -10 and -1, and upper bound between 1 and 10.
            lower = np.random.randint(-10, -1)
            upper = np.random.randint(1, 10)
            scm_params["variable_domains"][node] = (lower, upper)
        else:
            # For categorical nodes, assign a random list of categories.
            # For example, generate between 2 and 4 categories using letters.
            num_categories = np.random.randint(2, 4)
            # This will generate categories like ['A', 'B', 'C'].
            categories = [chr(65 + i) for i in range(num_categories)]
            scm_params["variable_domains"][node] = categories

    # Create a game instance.
    creator = GameInstanceCreator(dag_params, scm_params)
    game_instance = creator.create_instance()

    # Generate samples from the SCM.
    samples = game_instance.scm.generate_samples(num_samples=3, seed=42)
    print("Samples:", samples)

    # Save the instance.
    game_instance.save("game_instance.json")

    # Later, load the instance.
    loaded_instance = GameInstance.load("game_instance.json")
    print(
        "Loaded Samples:", loaded_instance.scm.generate_samples(num_samples=3, seed=42)
    )
