# File: causality_game/generators/scm_generator.py
import networkx as nx
from typing import Dict, Any
from src.lib.models.abstract import BaseGenerator
from src.lib.models.scm.SCM import SCM, SCMNode
import numpy as np
import sympy as sp


class SCMGenerator:
    def __init__(
        self,
        graph,
        variable_types,
        variable_domains,
        user_constraints,
        allowed_operations,
        allowed_functions,
        noise_distributions,
    ):
        """
        Generates an SCM from a given networkx graph and constraints.
        """
        self.graph = graph
        self.variable_types = variable_types
        self.variable_domains = variable_domains
        self.user_constraints = user_constraints
        self.allowed_operations = allowed_operations
        self.allowed_functions = allowed_functions
        self.noise_distributions = noise_distributions

    def _generate_function(self, input_vars):
        """Creates a mathematical function from input variables."""
        function = 0
        max_terms = self.user_constraints.get("max_terms", 3)
        allow_non_linear = self.user_constraints.get("allow_non_linearity", True)
        allow_variable_exponents = self.user_constraints.get(
            "allow_variable_exponents", False
        )
        terms = 0
        for var in input_vars:
            coeff = np.random.uniform(-5, 5)
            term = coeff * var
            if allow_non_linear and terms < max_terms:
                func = np.random.choice(self.allowed_functions)
                func_expr = func(var)
                if allow_variable_exponents and input_vars:
                    exponent_var = np.random.choice(input_vars)
                    func_expr = func_expr**exponent_var
                else:
                    exponent = np.random.uniform(0.5, 2)
                    func_expr = func_expr**exponent
                term += np.random.uniform(-5, 5) * func_expr
                terms += 1
            function += term
            terms += 1
            if terms >= max_terms:
                break
        return function

    def _sample_noise(self):
        """Samples a noise value from a user-defined distribution."""
        noise_dist_name = np.random.choice(list(self.noise_distributions.keys()))
        noise_dist = self.noise_distributions[noise_dist_name]
        return noise_dist.rvs()

    def generate(self):
        """Generates an SCM object from the given DAG."""
        topological_order = list(nx.topological_sort(self.graph))
        nodes = []
        for node_name in topological_order:
            parents = list(self.graph.predecessors(node_name))
            input_vars = [sp.Symbol(var) for var in parents]
            # For categorical nodes, we do not use a numerical equation.
            if self.variable_types[node_name] == "categorical":
                eq = sp.sympify(0)
            else:
                if not parents:  # Root node
                    noise_term = self._sample_noise()
                    eq = sp.Symbol(node_name) + noise_term
                else:
                    function = self._generate_function(input_vars)
                    noise_term = self._sample_noise()
                    eq = function + noise_term
                    eq = sp.re(eq)
                    # Bound the equation if a domain (tuple) is provided.
                    if isinstance(self.variable_domains[node_name], tuple):
                        min_val, max_val = self.variable_domains[node_name]
                        eq = sp.Max(sp.Min(eq, max_val), min_val)
            cdf_mappings = {}
            category_mappings = {}
            if self.variable_types[node_name] == "categorical":
                categories = self.variable_domains[node_name]
                # For each category, compute a CDF mapping.
                for category in categories:
                    function = self._generate_function(input_vars)
                    function = sp.re(function)
                    samples = []
                    for _ in range(1000):
                        subs_dict = {
                            var: np.random.uniform(-1, 1) for var in input_vars
                        }
                        eval_value = function.subs(subs_dict).evalf()
                        try:
                            samples.append(float(eval_value))
                        except Exception:
                            samples.append(np.random.uniform(-1, 1))
                    sorted_samples = np.sort(samples)
                    cdf_mappings[category] = (
                        lambda x, s=sorted_samples: np.searchsorted(s, x, side="right")
                        / len(s)
                    )
                # For categorical nodes, also create numeric mappings.
                for category in categories:
                    category_mappings[category] = np.random.uniform(-1, 1)
            node = SCMNode(
                node_name,
                eq,
                self.variable_domains[node_name],
                self.variable_types[node_name],
                cdf_mappings,
                category_mappings,
            )
            nodes.append(node)
        return SCM(nodes)
