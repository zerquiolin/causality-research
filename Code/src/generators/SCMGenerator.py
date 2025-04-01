import networkx as nx
from typing import Dict, Any, List, Tuple, Callable
from src.lib.models.abstract import (
    BaseGenerator,
)
from src.lib.models.scm.SCM import SCM, SCMNode
import numpy as np
import sympy as sp
from src.lib.models.scm.DAG import DAG


class SCMGenerator:
    def __init__(
        self,
        dag: DAG,
        variable_types: Dict[str, str],
        variable_domains: Dict[str, Any],
        user_constraints: Dict[str, Any],
        allowed_operations: List[
            Any
        ],  # Currently unused; consider removal or integration.
        allowed_functions: List[Callable[[sp.Expr], sp.Expr]],
        noise_distributions: Dict[str, Any],
        random_state: np.random.RandomState,  # or np.random.Generator
    ):
        """
        Generates an SCM from a given networkx graph and constraints.
        """
        self.dag = dag
        self.variable_types = variable_types
        self.variable_domains = variable_domains
        self.user_constraints = user_constraints
        self.allowed_operations = allowed_operations
        self.allowed_functions = allowed_functions
        self.noise_distributions = noise_distributions
        self.random_state = random_state

    def _generate_function(self, input_vars: List[sp.Expr]) -> sp.Expr:
        """Creates a mathematical function from input variables."""
        function = sp.Integer(0)  # Use a Sympy integer for consistency.
        max_terms = self.user_constraints.get("max_terms", 3)
        allow_non_linear = self.user_constraints.get("allow_non_linearity", True)
        allow_variable_exponents = self.user_constraints.get(
            "allow_variable_exponents", False
        )
        term_count = 0  # Use a single counter.

        for var in input_vars:
            if term_count >= max_terms:
                break
            coeff = self.random_state.uniform(-5, 5)
            # Linear term: coeff * var
            term = coeff * var

            if (
                allow_non_linear and term_count < max_terms - 1
            ):  # Reserve one term if possible.
                # Select a non-linear function from allowed_functions.
                func = self.random_state.choice(self.allowed_functions)
                # Check if the function could potentially induce complex values.
                if func == sp.log or func == sp.sqrt or func == sp.exp:
                    # Avoid negative values in the log or square root.
                    func_expr = func(1 + sp.Abs(var))
                else:
                    func_expr = func(var)

                if self.random_state.random() > 0.7:
                    if allow_variable_exponents and input_vars:
                        exponent_var = self.random_state.choice(input_vars)
                        # Here, exponent_var remains symbolic. If you want a numeric exponent, sample a number instead.
                        # Account for the case where the exponent is negative.
                        func_expr = func_expr ** sp.Abs(exponent_var)
                    else:
                        exponent = self.random_state.uniform(0.5, 2)
                        func_expr = func_expr**exponent
                    # Add a non-linear term with its own coefficient.
                    non_linear_coeff = self.random_state.uniform(-5, 5)
                    term += non_linear_coeff * func_expr

            function += term
            term_count += 1

        return function

    def _sample_noise(self) -> sp.Expr:
        """Samples a noise value from a user-defined distribution."""
        noise_dist_name = self.random_state.choice(
            list(self.noise_distributions.keys())
        )
        noise_dist = self.noise_distributions[noise_dist_name]
        # Return a sympy Float for consistency.
        noise_value = noise_dist.rvs(random_state=self.random_state)
        return sp.Float(noise_value, precision=53)

    def generate(self) -> SCM:
        """Generates an SCM object from the given DAG."""
        topological_order = list(nx.topological_sort(self.dag.graph))
        nodes: List[SCMNode] = []

        for node_name in topological_order:
            parents = list(self.dag.graph.predecessors(node_name))
            # Create symbolic variables for the parents.
            input_vars = [sp.Symbol(var) for var in parents]

            # For categorical nodes, we do not use a numerical equation.
            if self.variable_types[node_name] == "categorical":
                eq = sp.Integer(0)
            else:
                if not parents:  # Root node
                    noise_term = self._sample_noise()
                    eq = sp.Symbol(node_name) + noise_term
                else:
                    function = self._generate_function(input_vars)
                    noise_term = self._sample_noise()
                    eq = function + noise_term
                    # Keep only the real part (if the function might produce complex expressions).
                    eq = sp.re(eq)
                    # Bound the equation if a domain (tuple) is provided.
                    if isinstance(self.variable_domains[node_name], tuple):
                        min_val, max_val = self.variable_domains[node_name]
                        eq = sp.Max(sp.Min(eq, max_val), min_val)

            cdf_mappings: Dict[Any, Callable[[float], float]] = {}
            category_mappings: Dict[Any, float] = {}

            if self.variable_types[node_name] == "categorical":
                categories = self.variable_domains[node_name]
                # For each category, compute a CDF mapping deterministically.
                for category in categories:
                    function = self._generate_function(input_vars)
                    function = sp.re(function)
                    # Use a fixed grid instead of random samples.
                    if input_vars:
                        grid = np.linspace(-1, 1, 1000)
                        samples = []
                        # Vary the first input variable over the grid and fix the rest at 0.
                        for x in grid:
                            subs_dict = {var: 0 for var in input_vars[1:]}
                            subs_dict[input_vars[0]] = x
                            try:
                                eval_value = function.subs(subs_dict).evalf()
                                samples.append(float(eval_value))
                            except Exception:
                                samples.append(0)
                    else:
                        # No input variables: evaluate the function at a fixed value.
                        try:
                            eval_value = function.subs({}).evalf()
                            samples = [float(eval_value)] * 1000
                        except Exception:
                            samples = [0] * 1000
                    sorted_samples = np.sort(samples)
                    # Create a lambda function capturing the sorted samples.
                    cdf_mappings[category] = (
                        lambda x, s=sorted_samples: np.searchsorted(s, x, side="right")
                        / len(s)
                    )
                # For categorical nodes, also create numeric mappings.
                for category in categories:
                    # Use a fixed mapping or a random number.
                    category_mappings[category] = self.random_state.uniform(-1, 1)

            node = SCMNode(
                node_name,
                eq,
                self.variable_domains[node_name],
                self.variable_types[node_name],
                cdf_mappings,
                category_mappings,
                random_state=self.random_state,
            )
            nodes.append(node)

        return SCM(self.dag, nodes, self.random_state)
