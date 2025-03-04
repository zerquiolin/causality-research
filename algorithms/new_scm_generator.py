import numpy as np
import sympy as sp
from scipy.stats import norm, uniform


class SCMNode:
    def __init__(
        self,
        name,
        parents,
        var_type,
        domain,
        user_constraints,
        allowed_operations,
        allowed_functions,
        noise_distributions,
    ):
        self.name = name
        self.parents = parents
        self.var_type = var_type
        self.domain = domain
        self.user_constraints = user_constraints
        self.allowed_operations = allowed_operations
        self.allowed_functions = allowed_functions
        self.noise_distributions = noise_distributions  # Store noise distributions

        self.equation = None
        self.noise_term = None
        self.category_mappings = {}
        self.cdf_mappings = {}

        self._generate_equation()
        if var_type == "categorical":
            self._generate_categorical_processing()

    def _generate_equation(self):
        """Generates the structural equation for this node based on its parents and constraints."""
        if not self.parents:  # Root variable (exogenous)
            self.noise_term = self._sample_noise()
            # For a root variable, keep a symbolic placeholder and then substitute noise
            eq = sp.Symbol(self.name)
            # Replace the symbol with the noise term and force it to be real
            eq = sp.re(self.noise_term)
            if self.var_type == "numerical" and isinstance(self.domain, tuple):
                min_val, max_val = self.domain
                self.equation = sp.Max(sp.Min(eq, max_val), min_val)
            else:
                self.equation = eq
            return

        # For non-root nodes
        input_vars = [sp.Symbol(var) for var in self.parents]
        function = self._generate_function(input_vars)
        self.noise_term = self._sample_noise()

        # Build the equation symbolically, using a placeholder for noise
        eq = function + sp.Symbol(f"eta_{self.name}")
        # Immediately substitute the noise symbol with its numerical sample
        eq = eq.subs({sp.Symbol(f"eta_{self.name}"): self.noise_term})
        # Force the entire equation to its real part
        eq = sp.re(eq)

        if self.var_type == "numerical" and isinstance(self.domain, tuple):
            min_val, max_val = self.domain
            self.equation = sp.Max(sp.Min(eq, max_val), min_val)
        else:
            self.equation = eq

    def _generate_function(self, input_vars):
        """Creates a mathematical function for the node using context-free grammar."""
        function = 0
        max_terms = self.user_constraints.get("max_terms", 3)
        allow_non_linear = self.user_constraints.get("allow_non_linearity", True)
        allow_variable_exponents = self.user_constraints.get(
            "allow_variable_exponents", False
        )

        terms = 0
        for var in input_vars:
            coeff = np.random.uniform(-5, 5)  # Precompute coefficient
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
        if not self.noise_distributions:  # Check if noise distributions exist
            raise ValueError(f"No noise distributions provided for node {self.name}")

        noise_dist_name = np.random.choice(list(self.noise_distributions.keys()))
        noise_dist = self.noise_distributions[noise_dist_name]
        return noise_dist.rvs()

    def _generate_categorical_processing(self):
        """Processes categorical inputs and outputs."""
        # Input categorical processing: Assign random values to each category
        if isinstance(self.domain, list):  # If categorical variable
            for category in self.domain:
                self.category_mappings[category] = np.random.uniform(-1, 1)

        # Output categorical processing: Generate CDF functions
        if self.parents:
            self._generate_category_cdfs()

    def _generate_category_cdfs(self):
        """Generates CDF mappings for categorical outputs."""
        for category in self.domain:
            input_vars = [sp.Symbol(var) for var in self.parents]
            function = self._generate_function(input_vars)

            # Ensure function is real-valued
            function = sp.re(function)

            # Generate 1000 numerical samples safely
            samples = []
            for _ in range(1000):
                subs_dict = {
                    var: (
                        np.random.uniform(
                            0.01, 1
                        )  # Ensure positive inputs for log/sqrt
                        if "log" in str(function) or "sqrt" in str(function)
                        else np.random.uniform(-1, 1)
                    )
                    for var in input_vars
                }
                eval_value = function.subs(subs_dict).evalf()

                # Convert complex to real if necessary
                if eval_value.is_real:
                    samples.append(float(eval_value))
                else:
                    samples.append(
                        float(eval_value.as_real_imag()[0])
                    )  # Extract only real part

            sorted_samples = np.sort(samples)  # Ensure all samples are real

            # Store CDF function
            self.cdf_mappings[category] = lambda x: np.searchsorted(
                sorted_samples, x, side="right"
            ) / len(sorted_samples)

    def generate_value(self, parent_values):
        """Generates a value for this node given parent values."""
        if self.var_type == "numerical":
            subs_dict = {}
            for var in self.equation.free_symbols:
                var_str = str(var)
                # If a numeric mapping for a categorical parent exists, use it.
                if var_str + "_num" in parent_values:
                    value = parent_values[var_str + "_num"]
                else:
                    value = parent_values.get(var_str, np.random.uniform(-1, 1))
                try:
                    value = float(value)
                except ValueError:
                    raise ValueError(
                        f"Error: Expected numerical value for {var_str}, but got {value} of type {type(value)}."
                    )
                # Ensure positive inputs for log/sqrt if needed.
                if "log" in str(self.equation) or "sqrt" in str(self.equation):
                    value = max(value, 0.01)
                subs_dict[var] = value

            # Substitute noise term explicitly.
            subs_dict[f"eta_{self.name}"] = float(self.noise_term)
            eval_equation = self.equation.subs(subs_dict).evalf()

            # Debug: Print substitutions and evaluation.
            print(f"\nEvaluating {self.name}:")
            print(f"Substitutions: {subs_dict}")
            print(f"Raw evaluation result: {eval_equation}")

            try:
                return float(eval_equation)
            except TypeError:
                return float(sp.re(eval_equation))
        else:
            # For categorical nodes: use the stored CDFs to select the most probable category.
            category_probs = {
                cat: self.cdf_mappings[cat](np.random.uniform(-1, 1))
                for cat in self.domain
            }
            chosen_category = max(category_probs, key=lambda cat: category_probs[cat])
            # Also store the numeric mapping for input purposes.
            self.input_numeric = self.category_mappings[chosen_category]
            return chosen_category


class SCM:
    def __init__(
        self,
        dag,
        variable_types,
        variable_domains,
        user_constraints,
        allowed_operations,
        allowed_functions,
        noise_distributions,
    ):
        """Initializes the entire structural causal model."""
        self.dag = dag
        self.noise_distributions = (
            noise_distributions  # Ensure noise distributions are stored
        )

        self.nodes = {
            name: SCMNode(
                name,
                parents,
                variable_types[name],
                variable_domains[name],
                user_constraints,
                allowed_operations,
                allowed_functions,
                noise_distributions,
            )
            for name, parents in dag.items()
        }

    def generate_sample(self, interventions={}):
        """Generates a single sample by evaluating each node in topological order."""
        sample = {}
        for node_name in self.dag:
            if node_name in interventions:
                sample[node_name] = interventions[node_name]
                # If the node is categorical, also store its numeric mapping.
                if self.nodes[node_name].var_type == "categorical":
                    sample[node_name + "_num"] = self.nodes[node_name].input_numeric
            else:
                value = self.nodes[node_name].generate_value(sample)
                sample[node_name] = value
                if self.nodes[node_name].var_type == "categorical":
                    sample[node_name + "_num"] = self.nodes[node_name].input_numeric
        return sample

    def generate_samples(self, interventions={}, num_samples=1):
        """
        Generates multiple samples with optional interventions.

        Parameters:
        - interventions: dict -> Fixed values for interventional variables (e.g., {'X1': 5}).
        - num_samples: int -> Number of samples to generate.

        Returns:
        - samples: list of dicts -> Each dict is a generated sample.
        """
        return [self.generate_sample(interventions) for _ in range(num_samples)]


import numpy as np
import sympy as sp
from scipy.stats import norm, uniform

# Define the DAG structure (causal relationships)
dag = {
    "X1": [],  # Root node (exogenous variable)
    "X2": ["X1"],  # X2 depends on X1
    "X3": ["X1", "X2"],  # X3 depends on X1 and X2
    "Y": ["X2", "X3"],  # Y depends on X2 and X3
}

# Define variable types (numerical or categorical)
variable_types = {
    "X1": "numerical",
    "X2": "numerical",
    "X3": "categorical",
    "Y": "numerical",
}

# Define variable domains (numerical min/max, categorical possible values)
variable_domains = {
    "X1": (-5, 5),  # X1 must be between -5 and 5
    "X2": (-10, 10),  # X2 must be between -10 and 10
    "X3": ["A", "B", "C"],  # Categorical variable with three values
    "Y": (0, 20),  # Y must be between 0 and 20
}

# Define user constraints for function generation
user_constraints = {
    "max_terms": 3,  # Limit number of terms in structural equations
    "allow_non_linearity": True,  # Allow non-linear transformations
    "allow_variable_exponents": True,  # Allow variable exponents (X1^X2)
}

# Define allowed mathematical operations
allowed_operations = ["+", "-", "*", "/"]

# Define allowed mathematical functions from SymPy
allowed_functions = [sp.sin, sp.exp, sp.log]

# Define allowed noise distributions
noise_distributions = {
    "gaussian": norm(loc=0, scale=0.1),  # Normal distribution (mean=0, std=0.1)
    "uniform": uniform(
        loc=-0.1, scale=0.2
    ),  # Uniform distribution in range [-0.1, 0.1]
}

# Initialize the Structural Causal Model
scm = SCM(
    dag,
    variable_types,
    variable_domains,
    user_constraints,
    allowed_operations,
    allowed_functions,
    noise_distributions,
)

# Generate 5 observational samples (no interventions)
observational_samples = scm.generate_samples(num_samples=5)
print("\nObservational Samples:")
for sample in observational_samples:
    print(sample)

# Generate 5 interventional samples (fixing X1 = 2)
intervened_samples = scm.generate_samples(interventions={"X1": 2}, num_samples=5)
print("\nInterventional Samples (X1 = 2):")
for sample in intervened_samples:
    print(sample)
