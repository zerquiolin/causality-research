import random
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
        samples={},
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
        self.samples = samples

    def generate_value(self, parent_values, random_state=np.random):
        if self.var_type == "numerical":
            # (Numerical case remains the same.)
            subs_dict = {}
            for var in self.equation.free_symbols:
                var_str = str(var)
                if var_str + "_num" in parent_values:
                    subs_dict[var] = parent_values[var_str + "_num"]
                else:
                    subs_dict[var] = parent_values.get(
                        var_str,
                        random_state.uniform(
                            -1,
                            1,
                        ),
                    )
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
            # For categorical nodes, use a fixed value (e.g., 0) to evaluate each CDF mapping.
            category_probs = {
                cat: self.cdf_mappings[cat](0) for cat in self.cdf_mappings
            }
            chosen_category = max(category_probs, key=lambda cat: category_probs[cat])
            self.input_numeric = self.category_mappings[chosen_category]
            return chosen_category

    def to_dict(self):
        """Serialize the node to a dict. Convert numpy arrays to lists for JSON."""
        serializable_samples = {}
        for cat, sample in self.samples.items():
            # If sample is a NumPy array, convert it to a list.
            if isinstance(sample, np.ndarray):
                serializable_samples[cat] = sample.tolist()
            else:
                serializable_samples[cat] = sample

        return {
            "name": self.name,
            "equation": sp.srepr(
                self.equation
            ),  # string representation for the sympy expression
            "domain": self.domain,
            "var_type": self.var_type,
            "category_mappings": self.category_mappings,
            "input_numeric": self.input_numeric,
            "samples": serializable_samples,  # now JSON-serializable
        }

    @classmethod
    def from_dict(cls, data):
        """Deserialize from dict (re-generates the sympy equation from its string)."""
        equation = sp.sympify(data["equation"])
        node = cls(
            name=data["name"],
            equation=equation,
            domain=data["domain"],
            var_type=data["var_type"],
            category_mappings=data.get("category_mappings", {}),
        )
        node.input_numeric = data.get("input_numeric")
        node.cdf_mappings = {}
        for category, samples in data.get("samples", {}).items():
            s = np.array(samples)
            node.cdf_mappings[category] = lambda x, s=s: np.searchsorted(
                s, x, side="right"
            ) / len(s)
        return node


class SCM:
    def __init__(self, nodes):
        """
        Structural Causal Model that takes a list of nodes in topological order.
        """
        self.nodes = {node.name: node for node in nodes}

    def _generate_sample(self, interventions={}, random_state=np.random):
        """Generates a single sample by evaluating each node in topological order."""
        sample = {}
        # Evaluate nodes in the given (topological) order.
        for node_name, node in self.nodes.items():
            if node_name in interventions:
                sample[node_name] = interventions[node_name]
                if node.var_type == "categorical":
                    sample[node_name + "_num"] = node.input_numeric
            else:
                value = node.generate_value(sample, random_state=random_state)
                sample[node_name] = value
                if node.var_type == "categorical":
                    sample[node_name + "_num"] = node.input_numeric
        return sample

    def generate_samples(self, interventions={}, num_samples=1, random_state=None):
        """
        Generates multiple samples.

        If a seed is provided, the random generators (Python's and NumPy's) are seeded,
        ensuring reproducibility.
        """
        rd = np.random.RandomState(random_state) if random_state else np.random

        return [
            self._generate_sample(interventions, random_state=rd)
            for _ in range(num_samples)
        ]

    def to_dict(self):
        """Serialize the SCM as a dict."""
        nodes_data = {name: node.to_dict() for name, node in self.nodes.items()}
        return {"nodes": nodes_data}

    @classmethod
    def from_dict(cls, data):
        nodes = [SCMNode.from_dict(nd) for nd in data["nodes"].values()]
        # Optionally, sort nodes (if names are of the form 'X1', 'X2', …).
        nodes.sort(key=lambda n: int(n.name[1:]))
        return cls(nodes)
