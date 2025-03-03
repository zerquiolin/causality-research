import numpy as np
import sympy as sp
from sympy import symbols
from scipy.stats import norm, uniform


class StructuralCausalModel:
    def __init__(
        self,
        dag,
        variable_types,
        variable_domains,
        user_constraints,
        allowed_operations,
        allowed_functions,
        allowed_noise_distributions,
    ):
        """
        Initialize the Structural Causal Model (SCM).

        Parameters:
        - dag: dict -> Directed Acyclic Graph (DAG) defining variable dependencies.
        - variable_types: dict -> Specifies if each variable is 'numerical' or 'categorical'.
        - variable_domains: dict -> Defines possible values for categorical variables.
        - user_constraints: dict -> Defines constraints like non-linearity, exponentiation, and max terms.
        - allowed_operations: list -> List of permitted mathematical operations (e.g., ['+', '-', '*', '/']).
        - allowed_functions: list -> List of permitted mathematical functions (e.g., [sp.sin, sp.log, sp.exp]).
        - allowed_noise_distributions: dict -> Specifies possible noise distributions (e.g., {'gaussian': norm, 'uniform': uniform}).
        """
        self.dag = dag
        self.variable_types = variable_types
        self.variable_domains = variable_domains
        self.user_constraints = user_constraints
        self.allowed_operations = allowed_operations
        self.allowed_functions = allowed_functions
        self.allowed_noise_distributions = allowed_noise_distributions

        self.structural_equations = {}
        self.cdf_mappings = {}
        self.category_mappings = {}
        self.noise_terms = {}

        self._generate_structural_equations()

    def _generate_function(self, input_vars):
        """Generates a symbolic function ensuring all parent variables are used while respecting constraints."""
        function = 0
        max_terms = self.user_constraints.get("max_terms", 3)
        allow_non_linear = self.user_constraints.get("allow_non_linearity", True)
        allow_variable_exponents = self.user_constraints.get(
            "allow_variable_exponents", False
        )

        terms = 0
        used_vars = set()

        while terms < max_terms and input_vars:
            var = input_vars.pop(0)
            used_vars.add(var)

            coeff = sp.Symbol(f"alpha_{var}")
            term = coeff * var

            if allow_non_linear and terms < max_terms:
                for func in self.allowed_functions:
                    if func == sp.log:
                        func_expr = func(abs(var) + 1)  # Ensuring valid input for log
                    elif func == sp.sqrt:
                        func_expr = func(abs(var))  # Ensuring valid input for sqrt
                    else:
                        func_expr = func(var)

                    # Apply variable exponents with some probability
                    if np.random.rand() > 0.3:
                        exponent = (
                            var
                            if allow_variable_exponents
                            else sp.Symbol(f"beta_{var}")
                        )
                        func_expr = func_expr**exponent

                    term = sp.Symbol(f"gamma_{var}") * func_expr
                    terms += 1
                    if terms >= max_terms:
                        break

            # If we still have room for more terms, apply allowed operations
            # if terms < max_terms and input_vars:
            if input_vars:
                next_var = input_vars.pop(0)
                used_vars.add(next_var)
                operation = np.random.choice(self.allowed_operations)

                if operation == "+":
                    term += sp.Symbol(f"lambda_{next_var}") * next_var
                elif operation == "-":
                    term -= sp.Symbol(f"lambda_{next_var}") * next_var
                elif operation == "*":
                    term *= sp.Symbol(f"lambda_{next_var}") * next_var
                elif operation == "/" and next_var not in {0, "0"}:
                    term /= sp.Symbol(f"lambda_{next_var}") * next_var

                # terms += 1
            function += term

        return function

    def _compute_cdf(self, function, samples=1000):
        """Computes an empirical CDF for categorical variable transformations."""
        values = [
            function.subs({v: np.random.uniform(-1, 1) for v in function.free_symbols})
            for _ in range(samples)
        ]
        values = np.sort(values)
        return lambda x: np.searchsorted(values, x, side="right") / samples

    def _one_hot_encode(self, variable):
        """Performs one-hot encoding for categorical variables."""
        return [
            f"{variable}_{category}" for category in self.variable_domains[variable]
        ]

    def _transform_categorical_parents(self, parents):
        """Transforms categorical parent variables using one-hot encoding and CDFs."""
        transformed = []
        for parent in parents:
            if self.variable_types[parent] == "categorical":
                encoded_vars = self._one_hot_encode(parent)
                for var in encoded_vars:
                    function = self._generate_function([sp.Symbol(var)])
                    cdf_function = self._compute_cdf(function)
                    self.cdf_mappings[var] = cdf_function
                    transformed.append(var)
            else:
                transformed.append(parent)
        return transformed

    def _generate_structural_equations(self):
        """Generates structural equations for all variables based on the DAG."""
        for node in self.dag:
            parents = self.dag[node]

            # Select a noise distribution for this node
            noise_dist_name = np.random.choice(
                list(self.allowed_noise_distributions.keys())
            )
            noise_dist = self.allowed_noise_distributions[noise_dist_name]
            noise_symbol = sp.Symbol(f"eta_{node}")
            self.noise_terms[node] = (
                noise_dist  # Ensure every node has an assigned noise distribution
            )

            if not parents:  # Root node (Exogenous variable)
                self.structural_equations[node] = (
                    noise_symbol  # Root nodes are purely noise-driven
                )
                continue

            # Transform categorical parents and generate function
            transformed_parents = self._transform_categorical_parents(parents)
            input_vars = [sp.Symbol(var) for var in transformed_parents]
            function = self._generate_function(input_vars)

            # Add stochastic noise
            function_with_noise = function + noise_symbol

            # Apply min/max constraints based on variable domains
            if node in self.variable_domains and isinstance(
                self.variable_domains[node], tuple
            ):
                min_val, max_val = self.variable_domains[node]
                function_with_noise = sp.Max(
                    sp.Min(function_with_noise, max_val), min_val
                )

            self.structural_equations[node] = function_with_noise

            # Handle categorical variable mappings
            if self.variable_types[node] == "categorical":
                categories = self.variable_domains[node]
                mapping = {cat: np.random.uniform() for cat in categories}
                self.category_mappings[node] = mapping

    def generate_sample(self, interventions={}):
        """
        Generates a single data sample following the structural equations.

        Parameters:
        - interventions: dict -> Specifies interventional values for variables.

        Returns:
        - sample: dict -> Generated sample with values for each variable.
        """
        sample = {}

        for node in self.dag:
            if node in interventions:
                sample[node] = interventions[node]
                continue

            function = self.structural_equations[node]
            noise_dist = self.noise_terms[node]
            noise_value = noise_dist.rvs()  # Sample noise value

            # Generate random numeric values for symbolic coefficients with a larger range
            subs_dict = {}
            for var in function.free_symbols:
                if (
                    str(var).startswith("alpha_")
                    or str(var).startswith("beta_")
                    or str(var).startswith("gamma_")
                    or str(var).startswith("lambda_")
                ):
                    subs_dict[var] = np.random.uniform(
                        -5, 5
                    )  # Increased coefficient range
                else:
                    subs_dict[var] = sample.get(str(var), np.random.uniform(-1, 1))

            # Evaluate function numerically and apply scaling
            value = function.subs(subs_dict).evalf() + noise_value

            # Adaptive scaling based on output domain range
            if node in self.variable_domains and isinstance(
                self.variable_domains[node], tuple
            ):
                min_val, max_val = self.variable_domains[node]
                range_size = max_val - min_val  # Compute variable range

                # Adjust scaling based on how large the domain range is
                scale_factor = max(
                    1, range_size / 5
                )  # Scale relative to the domain size
                value *= scale_factor

                # Clamp value within numerical domain constraints
                value = max(min(value, max_val), min_val)

            # Assign categorical variables correctly
            if self.variable_types[node] == "numerical":
                sample[node] = float(value)
            elif self.variable_types[node] == "categorical":
                # Select the closest categorical value based on stored mappings
                closest_category = min(
                    self.category_mappings[node],
                    key=lambda cat: abs(value - self.category_mappings[node][cat]),
                )
                sample[node] = closest_category

        return sample


# Define DAG structure
dag = {"X1": [], "X2": ["X1"], "X3": ["X1", "X2"], "Y": ["X2", "X3"]}

# Define variable types
variable_types = {
    "X1": "numerical",
    "X2": "numerical",
    "X3": "categorical",
    "Y": "numerical",
}

# Define variable domains
variable_domains = {
    "X1": (-5, 5),  # X1 must be between -5 and 5
    "X2": (-10, 10),  # X2 must be between -10 and 10
    "X3": ["A", "B", "C"],  # Categorical variable
    "Y": (0, 20),  # Y must be between 0 and 20
}

# Define user constraints
user_constraints = {
    "max_terms": 3,
    "allow_non_linearity": True,
    "allow_variable_exponents": True,
}

# Define allowed operations and functions
allowed_operations = ["+", "-", "*", "/"]
allowed_functions = [sp.sin, sp.exp, sp.log]

# Define allowed noise distributions
allowed_noise_distributions = {
    "gaussian": norm(loc=0, scale=0.1),
    "uniform": uniform(loc=-0.1, scale=0.2),
}

# Initialize Structural Causal Model
scm = StructuralCausalModel(
    dag,
    variable_types,
    variable_domains,
    user_constraints,
    allowed_operations,
    allowed_functions,
    allowed_noise_distributions,
)

# Generate an observational sample
observational_sample = scm.generate_sample()
print("\nObservational Sample:")
print(observational_sample)

# Generate an interventional sample with X1 = 10 (out of domain, should be clamped to 5)
intervened_sample = scm.generate_sample(interventions={"X1": 5})
print("\nInterventional Sample (X1 = 10, should be clamped to 5):")
print(intervened_sample)


print(scm.structural_equations)

for _ in range(100):
    print(scm.generate_sample())
