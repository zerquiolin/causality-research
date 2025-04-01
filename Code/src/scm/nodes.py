import numpy as np
import sympy as sp

from .dag import DAG


class SCMNode:
    """
    Represents a single node in the Structural Causal Model (SCM).

    This class handles both numerical and categorical nodes. For numerical nodes,
    the `equation` is used for generating values, while for categorical nodes,
    the value is determined via CDF mappings.
    """

    def __init__(
        self,
        name: str,
        equation: sp.Basic,
        domain: tuple,
        var_type: str,
        cdf_mappings: dict = None,
        category_mappings: dict = None,
        random_state: np.random.RandomState = np.random,
    ):
        """
        Initializes an SCMNode instance.

        Args:
            name (str): The name of the node.
            equation (sp.Basic): The symbolic equation for the node.
            domain (tuple): The range of values for numerical nodes.
            var_type (str): The type of the variable ('numerical' or 'categorical').
            cdf_mappings (dict, optional): CDF mappings for categorical nodes.
            category_mappings (dict, optional): Mappings from categories to numeric values.
            random_state (np.random.RandomState, optional): Random state for reproducibility.
        """
        self.name = name
        self.equation = equation
        self.domain = domain
        self.var_type = var_type
        self.cdf_mappings = cdf_mappings or {}
        self.category_mappings = category_mappings or {}
        self.input_numeric = None  # For categorical nodes, store a numeric mapping to be used by children.

    def generate_value(
        self, parent_values: dict, random_state: np.random.RandomState
    ) -> float:
        """
        Generates a value for the node based on its type and parent values.

        Args:
            parent_values (dict): A dictionary of parent values.
            random_state (np.random.RandomState): Random state for generating random values.

        Returns:
            float: The generated value for the node.
        """
        if self.var_type == "numerical":
            subs_dict = {}
            for var in self.equation.free_symbols:
                var_str = str(var)
                subs_dict[var] = parent_values.get(
                    var_str + "_num",
                    parent_values.get(var_str, random_state.uniform(-1, 1)),
                )

            for var in self.equation.free_symbols:
                if var not in subs_dict:
                    print(f"Warning: Unresolved symbols in {self.name} ->", var)

            eval_equation = self.equation.subs(subs_dict).evalf()
            if isinstance(eval_equation, sp.Basic) and not eval_equation.is_number:
                print(f"Warning: Unresolved symbols in {self.name} ->", eval_equation)
            result = (
                float(eval_equation.as_real_imag()[0])
                if not eval_equation.is_real
                else float(eval_equation)
            )
            if isinstance(self.domain, tuple):
                min_val, max_val = self.domain
                result = max(min_val, min(max_val, result))
            return result
        else:
            category_probs = {
                cat: self.cdf_mappings[cat](1) for cat in self.cdf_mappings
            }

            chosen_category = max(category_probs, key=lambda cat: category_probs[cat])
            self.input_numeric = self.category_mappings[chosen_category]
            print(f"Chosen category: {chosen_category}")
            return chosen_category

    def to_dict(self) -> dict:
        """
        Serializes the node to a dictionary, converting numpy arrays to lists for JSON.

        Returns:
            dict: A dictionary representation of the node.
        """
        return {
            "name": self.name,
            "equation": sp.srepr(
                self.equation
            ),  # string representation for the sympy expression
            "domain": self.domain,
            "var_type": self.var_type,
            "category_mappings": self.category_mappings,
            "input_numeric": self.input_numeric,
            "cdf_step_points": {
                cat: self._extract_step_points(self.cdf_mappings[cat])
                for cat in self.cdf_mappings
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SCMNode":
        """
        Deserializes an SCMNode instance from a dictionary.

        Args:
            data (dict): The dictionary containing node data.

        Returns:
            SCMNode: An instance of SCMNode.
        """
        safe_dict = {
            "Symbol": sp.Symbol,
            "Integer": sp.Integer,
            "Float": sp.Float,
            "Add": sp.Add,
            "Mul": sp.Mul,
            "Pow": sp.Pow,
            "Max": sp.Max,
            "Min": sp.Min,
            "re": sp.re,
            "np": np,
        }

        equation = eval(data["equation"], safe_dict)

        node = cls(
            name=data["name"],
            equation=equation,
            domain=data["domain"],
            var_type=data["var_type"],
            category_mappings=data.get("category_mappings", {}),
        )
        node.input_numeric = data.get("input_numeric")

        node.cdf_mappings = {
            category: SCMNode._create_cdf_lambda(np.array(step_points))
            for category, step_points in data.get("cdf_step_points", {}).items()
        }
        return node

    def set_cdf_function(self, category: str, cdf_lambda) -> None:
        """
        Sets a lambda function for the specified category and stores it in cdf_mappings.

        Args:
            category (str): The category for which the CDF function is set.
            cdf_lambda: The lambda function representing the CDF.
        """
        self.cdf_mappings[category] = cdf_lambda  # Store lambda function

    @staticmethod
    def _extract_step_points(cdf_lambda) -> list:
        """
        Extracts step points from a given CDF lambda function by sampling.

        Args:
            cdf_lambda: The lambda function representing the CDF.

        Returns:
            list: A list of step points extracted from the CDF.
        """
        x_values = np.linspace(0, 1, 100)  # Sample 100 points in [0,1]
        cdf_values = np.array([cdf_lambda(x) for x in x_values])
        return cdf_values.tolist()

    @staticmethod
    def _create_cdf_lambda(step_points: np.ndarray) -> callable:
        """
        Reconstructs a lambda function from step points.

        Args:
            step_points (np.ndarray): The step points used to create the CDF.

        Returns:
            callable: A lambda function representing the CDF.
        """
        return lambda x: np.searchsorted(step_points, x, side="right") / len(
            step_points
        )
