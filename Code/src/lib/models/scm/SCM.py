import numpy as np
import sympy as sp


class SCMNode:
    def __init__(
        self,
        name,
        equation,
        domain,
        var_type,
        cdf_mappings=None,
        category_mappings=None,
    ):
        """
        Represents a single node in the Structural Causal Model (SCM).
        For numerical nodes, `equation` is used for generating values.
        For categorical nodes, the equation is not used (set to 0) and value is determined via CDF mappings.
        """
        self.name = name
        self.equation = equation
        self.domain = domain
        self.var_type = var_type
        self.cdf_mappings = cdf_mappings or {}
        self.category_mappings = category_mappings or {}
        # For categorical nodes, store a numeric mapping to be used by children.
        self.input_numeric = None

    def generate_value(self, parent_values):
        """Generates a value for this node given parent values."""
        if self.var_type == "numerical":
            subs_dict = {}
            # For each symbol in the equation, try to substitute using the parent's numeric mapping if available.
            for var in self.equation.free_symbols:
                var_str = str(var)
                if var_str + "_num" in parent_values:
                    subs_dict[var] = parent_values[var_str + "_num"]
                else:
                    subs_dict[var] = parent_values.get(
                        var_str, np.random.uniform(-1, 1)
                    )
            eval_equation = self.equation.subs(subs_dict).evalf()
            # If still symbolic, warn
            if isinstance(eval_equation, sp.Basic) and not eval_equation.is_number:
                print(f"Warning: Unresolved symbols in {self.name} ->", eval_equation)
            # Force a float, extracting real part if needed
            if eval_equation.is_real:
                result = float(eval_equation)
            else:
                result = float(eval_equation.as_real_imag()[0])
            # Bound the result to the domain (if a tuple is provided)
            if isinstance(self.domain, tuple):
                min_val, max_val = self.domain
                result = max(min_val, min(max_val, result))
            return result
        else:
            # For categorical nodes, use the precomputed CDF mappings to pick a category.
            category_probs = {
                cat: self.cdf_mappings[cat](np.random.uniform(-1, 1))
                for cat in self.cdf_mappings
            }
            chosen_category = max(category_probs, key=lambda cat: category_probs[cat])
            # Also store the numeric mapping (to be used by children)
            self.input_numeric = self.category_mappings[chosen_category]
            return chosen_category


class SCM:
    def __init__(self, nodes):
        """
        Structural Causal Model that takes a list of nodes in topological order.
        """
        self.nodes = {node.name: node for node in nodes}

    def generate_sample(self, interventions={}):
        """Generates a single sample by evaluating each node in topological order."""
        sample = {}
        # Evaluate nodes in the given (topological) order.
        for node_name, node in self.nodes.items():
            if node_name in interventions:
                sample[node_name] = interventions[node_name]
                if node.var_type == "categorical":
                    sample[node_name + "_num"] = node.input_numeric
            else:
                value = node.generate_value(sample)
                sample[node_name] = value
                if node.var_type == "categorical":
                    sample[node_name + "_num"] = node.input_numeric
        return sample

    def generate_samples(self, interventions={}, num_samples=1):
        """Generates multiple samples."""
        return [self.generate_sample(interventions) for _ in range(num_samples)]
