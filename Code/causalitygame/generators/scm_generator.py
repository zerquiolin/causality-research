# Math
import numpy as np
import sympy as sp

# Graph
import networkx as nx

# SCM
from causalitygame.scm.base import ACCESSIBILITY_CONTROLLABLE
from causalitygame.scm.scm import SCM
from causalitygame.scm.nodes import (
    EquationBasedNumericalSCMNode,
    EquationBasedCategoricalSCMNode,
    SerializableCDF,
)

# DAG
from causalitygame.scm.dag import DAG

# Types
from causalitygame.scm.base import BaseNoiseDistribution

# Abstract
from typing import Dict, Any, List, Tuple, Callable
from .base import AbstractSCMGenerator

# Utils
import logging


class EquationBasedSCMGenerator(AbstractSCMGenerator):
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
        noise_distributions: List[BaseNoiseDistribution],
        random_state: np.random.RandomState = np.random.RandomState(911),
        num_samples_for_cdf_generation: int = 1000,
        logger: logging.Logger = None
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
        self.num_samples_for_cdf_generation = num_samples_for_cdf_generation
        self.random_state = random_state
        self.logger = logging if logger is None else logger

    def _generate_equation(self, input_vars: List[sp.Expr]) -> sp.Expr:
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
            coeff = np.round(self.random_state.uniform(-5, 5), 8)  # higher precisions can cause problems in the serialization since not supported by str
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

    def generate(self) -> SCM:
        """
        Generates an SCM object by creating SCMNodes for each node in the DAG.

        For each node, builds parent mappings (for categorical parents).
        For numerical nodes, generates a single equation.
        For categorical nodes, generates an equation per possible category and computes CDFs.

        Returns:
            SCM: An instance of the SCM with generated nodes.
        """

        self.logger.info("Generating new SCM.")

        # get topological sorting
        self.logger.debug("Generating topologically sorting of nodes")
        topological_order = list(nx.topological_sort(self.dag.graph))
        self.logger.debug(f"Done, ordering is {topological_order}")
        nodes: List[EquationBasedNumericalSCMNode | EquationBasedCategoricalSCMNode] = (
            []
        )

        for node_name in topological_order:
            # Get the parents of the current node.
            parents = list(self.dag.graph.predecessors(node_name))
            # Build parent mappings: for each parent that is categorical, map its domain via label encoding.
            parent_mappings = {
                p: {cat: idx for idx, cat in enumerate(self.variable_domains[p])}
                for p in parents
                if self.variable_types[p] == "categorical"
            }
            #  Define a mapping of variable types to generator functions
            type_generators = {
                "numerical": self._generate_numerical_node,
                "categorical": self._generate_categorical_node,
            }
            # Use the mapping to get the right generator
            node_type = self.variable_types[node_name]
            generator = type_generators.get(node_type)
            # Check for existence of the generator
            assert generator, f"Unsupported variable type: {node_type}"
            
            # Generate the node
            self.logger.debug(f"Generating node {node_name} of type {node_type} with parents {parents}")
            node = generator(
                node_name=node_name,
                parents=parents,
                parent_mappings=parent_mappings,
                nodes=nodes,
            )
            nodes.append(node)
            self.logger.debug(f"added node {node_name}")
        
        self.logger.debug("Creating SCM object")
        scm = SCM(self.dag, nodes, self.random_state)
        self.logger.info(f"SCM with {len(nodes)} variables generated.")
        return scm

    def _generate_numerical_node(
        self,
        node_name: str,
        parents: List[str],
        parent_mappings: Dict[str, int],
        nodes: List[EquationBasedNumericalSCMNode | EquationBasedCategoricalSCMNode],
    ) -> Tuple[EquationBasedNumericalSCMNode, Dict[str, Any]]:
        """
        Generates a numerical SCM node.

        Args:
            node_name (str): The name of the node.
            parents (List[str]): List of parent node names.
            parent_mappings (Dict[str, int]): Mapping of parent names to their mappings.

        Returns:
            Tuple[EquationBasedNumericalSCMNode, Dict[str, Any]]: The generated node and its parameters.
        """
        # Select a random noise distribution from the available options.
        noise_distribution = self.random_state.choice(self.noise_distributions)
        # Check for parent nodes and generate the equation accordingly
        if not parents:  # Root node
            return EquationBasedNumericalSCMNode(
                name=node_name,
                accessibility=ACCESSIBILITY_CONTROLLABLE,
                evaluation=None,
                domain=self.variable_domains[node_name],
                noise_distribution=noise_distribution,
                parents=None,
                parent_mappings=None,
                random_state=self.random_state,
            )
        # Format parent names as symbols for the equation
        input_vars = [sp.Symbol(p) for p in parents]
        # Generate the equation for the node
        equation = self._generate_equation(input_vars)
        # Create the node
        return EquationBasedNumericalSCMNode(
            name=node_name,
            accessibility=ACCESSIBILITY_CONTROLLABLE,
            evaluation=equation,
            domain=self.variable_domains[node_name],
            noise_distribution=noise_distribution,
            parents=parents,
            parent_mappings=parent_mappings,
            random_state=self.random_state,
        )

    def _generate_categorical_node(
        self,
        node_name: str,
        parents: List[str],
        parent_mappings: Dict[str, int],
        nodes: List[EquationBasedNumericalSCMNode | EquationBasedCategoricalSCMNode],
    ) -> Tuple[EquationBasedCategoricalSCMNode, Dict[str, Any]]:
        """
        Generates a categorical SCM node.

        Args:
            node_name (str): The name of the node.
            parents (List[str]): List of parent node names.
            parent_mappings (Dict[str, int]): Mapping of parent names to their mappings.

        Returns:
            Tuple[EquationBasedCategoricalSCMNode, Dict[str, Any]]: The generated node and its parameters.
        """
        # Select a random noise distribution from the available options.
        noise_distribution = self.random_state.choice(self.noise_distributions)

        # Check if the variable has parents
        if not parents:
            return EquationBasedCategoricalSCMNode(
                name=node_name,
                evaluation=None,
                domain=self.variable_domains[node_name],
                noise_distribution=noise_distribution,
                cdfs=None,
                parents=None,
                parent_mappings=None,
                random_state=self.random_state,
            )
        # Define an equation for each possible variable in the domain
        equations = {}

        # Define the CDFs for each possible variable in the domain
        cdf_mappings = {}

        # Format parent names as symbols for the equation
        input_vars = [sp.Symbol(p) for p in parents]

        # Iterate over each possible variable in the domain
        for category in self.variable_domains[node_name]:
            # Generate the equation for the node
            equation = self._generate_equation(input_vars)
            equations[category] = equation

            # Generate CDF: sample at least 1000 datapoints.
            self.logger.debug(f"Generating {self.num_samples_for_cdf_generation} samples for the case {node_name}={category}")
            samples = []
            for _ in range(self.num_samples_for_cdf_generation):

                # Values for the evaluated nodes
                node_values = {}
                # Iterate over the already generated nodes
                for n in nodes:
                    # Generate values for the current node n
                    node_values[n.name] = n.generate_value(
                        parent_values=node_values, random_state=self.random_state
                    )
                # Substitute parent values into the equation.
                try:
                    # Filter out only the parent values
                    parent_values = {
                        p: node_values[p] for p in parents if p in node_values
                    }
                    val = equation.subs(
                        {sp.Symbol(p): parent_values[p] for p in parents}
                    ).evalf()
                    samples.append(float(val))
                except Exception:
                    self.logger.error(
                        f"Error evaluating equation for {node_name} with parents {parents}: {equation}"
                    )
                    self.logger.error(f"Node values: {node_values}")
                    self.logger.error(f"Equation: {equation}")
                    raise ValueError(
                        f"Error evaluating equation for {node_name} with parents {parents}: {equation}"
                    )
            sorted_samples = np.sort(samples)
            self.logger.debug("Creating CDF from sample data.")
            cdf_mappings[category] = SerializableCDF(sorted_samples)

        # Create the node
        node = EquationBasedCategoricalSCMNode(
            name=node_name,
            accessibility=ACCESSIBILITY_CONTROLLABLE,
            evaluation=equations,
            domain=self.variable_domains[node_name],
            noise_distribution=noise_distribution,
            cdfs=cdf_mappings,
            parents=parents,
            parent_mappings=parent_mappings,
            random_state=self.random_state,
        )

        return node
