# Math
import numpy as np
import sympy as sp

# Graph
import networkx as nx

# SCM
from src.scm.scm import SCM
from src.scm.nodes import SCMNode, SerializableCDF

# DAG
from src.scm.dag import DAG

# Abstract
from typing import Dict, Any, List, Tuple, Callable
from .base import AbstractSCMGenerator

# Utils
import logging


class SCMGenerator(AbstractSCMGenerator):
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
        Initializes the SCMGenerator with the DAG and configuration for generating node equations.

        Args:
            dag (DAG): The DAG structure.
            variable_types (Dict[str, str]): Mapping of node names to "numerical" or "categorical".
            variable_domains (Dict[str, Any]): Mapping of node names to domains (tuple for numerical, list for categorical).
            user_constraints (Dict[str, Any]): Additional constraints (e.g., max_terms, non-linearity).
            allowed_operations (List[Any]): Unused operations (reserved for future use).
            allowed_functions (List[Callable[[sp.Expr], sp.Expr]]): Functions allowed in equation generation.
            noise_distributions (Dict[str, Any]): Mapping of noise distribution names to distribution objects.
            random_state (np.random.RandomState): Random generator for reproducibility.
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
        """
        Generates a symbolic function using the input variables and allowed functions.

        Args:
            input_vars (List[sp.Expr]): List of symbolic parent variables.

        Returns:
            sp.Expr: A symbolic expression representing the function.
        """
        function = sp.Integer(0)
        allow_non_linear = self.user_constraints.get("allow_non_linearity", True)
        allow_variable_exponents = self.user_constraints.get(
            "allow_variable_exponents", False
        )

        for var in input_vars:
            coeff = self.random_state.uniform(-5, 5)
            term = coeff * var

            if allow_non_linear:
                func = self.random_state.choice(self.allowed_functions)
                # Avoid issues with negative values in log, sqrt, exp by applying abs.
                if func in [sp.log, sp.sqrt]:
                    func_expr = func(1 + sp.Abs(var))
                elif func == sp.exp:
                    exp = (
                        self.random_state.uniform(0.5, 5)
                        if not allow_variable_exponents
                        or self.random_state.random() < 0.7
                        else 1 + sp.Abs(var)
                    )
                    func_expr = func(exp)
                else:
                    func_expr = func(var)

                term = coeff * func_expr

            function += term

        return function

    def _sample_noise(self) -> sp.Expr:
        """
        Samples a noise value from a user-defined distribution.

        Returns:
            sp.Expr: A sympy Float representing the noise.
        """
        noise_dist_name = self.random_state.choice(
            list(self.noise_distributions.keys())
        )
        noise_dist = self.noise_distributions[noise_dist_name]
        noise_value = noise_dist.rvs(random_state=self.random_state)
        return sp.Float(noise_value, precision=53)

    def generate(self) -> SCM:
        """
        Generates an SCM object by creating SCMNodes for each node in the DAG.

        For each node, builds parent mappings (for categorical parents).
        For numerical nodes, generates a single equation.
        For categorical nodes, generates an equation per possible category and computes CDFs.

        Returns:
            SCM: An instance of the SCM with generated nodes.
        """
        topological_order = list(nx.topological_sort(self.dag.graph))
        nodes: List[SCMNode] = []

        for node_name in topological_order:
            # Get the parents of the current node.
            parents = list(self.dag.graph.predecessors(node_name))
            # Build parent mappings: for each parent that is categorical, map its domain via label encoding.
            parent_mappings = {}
            for p in parents:
                if self.variable_types.get(p) == "categorical":
                    # Simple label encoding: assign an integer based on order.
                    mapping = {
                        cat: idx for idx, cat in enumerate(self.variable_domains[p])
                    }
                    parent_mappings[p] = mapping

            if self.variable_types[node_name] == "numerical":
                if not parents:  # Root node
                    noise_term = self._sample_noise()
                    eq = sp.Symbol(node_name) + noise_term
                else:
                    input_vars = [sp.Symbol(p) for p in parents]
                    function = self._generate_function(input_vars)
                    noise_term = self._sample_noise()
                    eq = sp.re(function + noise_term)
                    if isinstance(self.variable_domains[node_name], tuple):
                        min_val, max_val = self.variable_domains[node_name]
                        eq = sp.Max(sp.Min(eq, max_val), min_val)
                cdf_mappings = {}  # Not used for numerical nodes.
            else:
                # Categorical node: generate an equation and CDF for each possible category.
                eq_dict = {}
                cdf_mappings = {}
                categories = self.variable_domains[node_name]
                for category in categories:
                    if not parents:
                        noise_term = self._sample_noise()
                        eq_cat = sp.Symbol(node_name) + noise_term
                    else:
                        input_vars = [sp.Symbol(p) for p in parents]
                        eq_cat = sp.re(
                            self._generate_function(input_vars) + self._sample_noise()
                        )
                    eq_dict[category] = eq_cat

                    # Generate CDF: sample at least 1000 datapoints.
                    if parents:
                        samples = []
                        # Generate 1000 samples
                        for _ in range(1000):
                            # Sample parent values based on their mappings.
                            parent_values = {}
                            for p in parents:
                                if self.variable_types[p] == "categorical":
                                    # Use the mapping to sample a value.
                                    parent_values[p] = self.random_state.choice(
                                        list(parent_mappings[p].keys())
                                    )
                                else:
                                    # For numerical parents, sample from the domain.
                                    min_val, max_val = self.variable_domains[p]
                                    parent_values[p] = self.random_state.uniform(
                                        min_val, max_val
                                    )

                            # Substitute parent values into the equation.
                            try:
                                val = eq_cat.subs(
                                    {sp.Symbol(p): parent_values[p] for p in parents}
                                ).evalf()
                                samples.append(float(val))
                            except Exception:
                                raise ValueError(
                                    f"Error evaluating equation for {node_name} with parents {parent_values}: {eq_cat}"
                                )
                    else:
                        try:
                            val = eq_cat.subs(
                                {
                                    sp.Symbol(
                                        node_name
                                    ): 0,  # Placeholder for self, prevent unresolved symbols.
                                }
                            ).evalf()
                            samples = [float(val)] * 1000
                        except Exception:
                            samples = [0] * 1000

                    sorted_samples = np.sort(samples)
                    # todo: check if this is correct
                    # cdf_mappings[category] = (
                    #     lambda x, s=sorted_samples: np.searchsorted(s, x, side="right")
                    #     / len(s)
                    # )
                    cdf_mappings[category] = SerializableCDF(sorted_samples)
                eq = eq_dict  # For categorical nodes, equation is a dict mapping each category to its equation.

            node = SCMNode(
                name=node_name,
                equation=eq,
                domain=self.variable_domains[node_name],
                var_type=self.variable_types[node_name],
                cdf_mappings=cdf_mappings,
                parent_mappings=parent_mappings,
                random_state=self.random_state,
            )
            nodes.append(node)

        return SCM(self.dag, nodes, self.random_state)
