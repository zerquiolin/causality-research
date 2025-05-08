# Math
import numpy as np
import sympy as sp

from causalitygame.scm.noise_distributions import (
    GaussianNoiseDistribution,
    UniformNoiseDistribution,
    DiracNoiseDistribution,
)

# Abstract Base Class
from .base import BaseNoiseDistribution, BaseSCMNode, ACCESSIBILITY_OBSERVABLE

# Types
from typing import Callable, Dict, List, Optional

# Utils
from collections import Counter
import logging


class EquationBasedNumericalSCMNode(BaseSCMNode):
    def generate_value(
        self,
        parent_values: Dict[str, float | str | int],
        random_state: Optional[np.random.RandomState] = None,
    ):
        # Define random state
        rs = random_state if random_state else self.random_state
        # Check if the node has parents
        if not self.parents:
            return self.noise_distribution.generate(rs)
        # Get the evaluation symbols for evaluation
        symbols = [
            str(symb) for symb in self.evaluation.free_symbols
        ]  # Parsed for easier comparison
        # Check if the parent values are provided
        assert set(symbols).issubset(
            parent_values.keys()
        ), "Parent values do not match the expected symbols"
        # Map categorical parent values
        substitutions = {
            symb: (
                self.parent_mappings[symb]
                if symb in self.parent_mappings
                else parent_values[symb]
            )
            for symb in symbols
        }
        # Evaluate the expression
        evaluated = self.evaluation.subs(substitutions).evalf()
        assert evaluated.is_number, "Evaluation failed"
        # Cover for "Possibly" imaginary numbers
        evaluated = (
            float(evaluated.as_real_imag()[0])
            if not evaluated.is_real
            else float(evaluated)
        )
        # Clip to the minimum and maximum values
        evaluated = min(max(evaluated, self.domain[0]), self.domain[-1])
        # Add noise to the evaluated value
        noise = self.noise_distribution.generate(rs)
        # Return the final value
        return evaluated + noise

    def to_dict(self):
        """
        Converts the node to a dictionary representation.

        Returns:
            dict: Dictionary representation of the node.
        """
        state = self.random_state.get_state()
        return {
            "class": self.__class__.__name__,
            "name": self.name,
            "accessibility": self.accessibility,
            "equation": str(self.evaluation),
            "domain": self.domain,
            "noise_distribution": self.noise_distribution.to_dict(),
            "parents": self.parents,
            "parent_mappings": self.parent_mappings,
            "random_state": {
                "state": state[0],
                "keys": state[1].tolist(),
                "pos": state[2],
                "has_gauss": state[3],
                "cached_gaussian": state[4],
            },
        }

    @classmethod
    def from_dict(cls, data: Dict):
        """
        Deserializes the node from a dictionary representation.

        Args:
            data (Dict): Dictionary containing node data.

        Returns:
            EquationBasedNumericalSCMNode: An instance of the node.
        """
        evaluation = sp.sympify(data["equation"]) if "equation" in data else None
        if evaluation is not None:
            assert (
                str(evaluation) == data["equation"]
            ), f"Evaluation structure {data['equation']} could not parsed properly. Recovered {str(evaluation)}"

        # Deserialize the noise distribution
        if "noise_distribution" in data:
            noise_distribution = data["noise_distribution"]
            if type(noise_distribution) == dict:
                if noise_distribution["class"] == GaussianNoiseDistribution.__name__:
                    noise_distribution = GaussianNoiseDistribution.from_dict(
                        noise_distribution
                    )
                elif noise_distribution["class"] == UniformNoiseDistribution.__name__:
                    noise_distribution = UniformNoiseDistribution.from_dict(
                        noise_distribution
                    )
                elif noise_distribution["class"] == DiracNoiseDistribution.__name__:
                    noise_distribution = DiracNoiseDistribution.from_dict(
                        noise_distribution
                    )
                else:
                    raise ValueError(
                        f"Unknown noise distribution class: {noise_distribution['class']}"
                    )
            elif type(noise_distribution) in [float, int, np.float64, np.int64]:
                noise_distribution = DiracNoiseDistribution(val=noise_distribution)
            else:
                raise ValueError(
                    f"Unknown noise distribution type: {type(noise_distribution)}"
                )
        else:
            noise_distribution = UniformNoiseDistribution(
                low=data["domain"][0], high=data["domain"][1]
            )

        # Deserailize the random staet
        random_state = np.random.RandomState()
        if "random_state" in data:
            random_state.set_state(
                (
                    str(data["random_state"]["state"]),
                    np.array(data["random_state"]["keys"], dtype=np.uint32),
                    int(data["random_state"]["pos"]),
                    int(data["random_state"]["has_gauss"]),
                    float(data["random_state"]["cached_gaussian"]),
                )
            )

        # Create the node
        new_class = cls(
            name=data["name"],
            accessibility=data.get("accessibility", ACCESSIBILITY_OBSERVABLE),
            evaluation=evaluation,
            domain=data["domain"],
            noise_distribution=noise_distribution,
            parents=data.get("parents"),
            parent_mappings=data.get("parent_mappings"),
        )
        # Set the random state
        new_class.random_state = random_state
        # Return the new class
        return new_class


class EquationBasedCategoricalSCMNode(BaseSCMNode):
    def __init__(
        self,
        name: str,
        accessibility: str,
        evaluation: Optional[Callable],
        domain: List[float | str],
        noise_distribution: BaseNoiseDistribution,
        cdfs: Optional[List[Callable]] = None,
        parents: Optional[List[str]] = None,
        parent_mappings: Optional[Dict[str, int | float]] = None,
        domain_distribution: Optional[Dict[str, float]] = None,
        random_state: np.random.RandomState = np.random.RandomState(911),
    ):
        # Superclass constructor
        super().__init__(
            name=name,
            accessibility=accessibility,
            evaluation=evaluation,
            domain=domain,
            noise_distribution=noise_distribution,
            parents=parents,
            parent_mappings=parent_mappings,
            random_state=random_state,
        )
        # Initialize the CDFs
        self.cdfs = cdfs
        # Initialize the noise distribution
        self.domain_noise_distribution = (
            self._noise_to_category_distribution()
            if not domain_distribution
            else domain_distribution
        )

    def _noise_to_category_distribution(
        self, n_samples: int = 10000
    ) -> Dict[str, float]:
        """
        Converts a continuous noise distribution into a discrete probability distribution over given categories.

        Args:
            n_samples (int): Number of samples to draw from the noise distribution.

        Returns:
            Dict[str, float]: A dictionary mapping each category to a probability.
        """
        # Sample from the noise distribution
        samples = [self.noise_distribution.generate() for _ in range(n_samples)]

        # Use quantiles to bin the samples into categories
        quantiles = np.percentile(samples, np.linspace(0, 100, len(self.domain) + 1))

        # Assign samples to bins
        bin_indices = np.digitize(samples, quantiles[1:-1], right=True)

        # Map bin indices to categories
        mapped = [self.domain[i] for i in bin_indices]

        # Count and normalize
        counts = Counter(mapped)
        total = sum(counts.values())
        return {cat: counts.get(cat, 0) / total for cat in self.domain}

    def generate_value(
        self,
        parent_values: Dict[str, float | str | int],
        random_state: Optional[np.random.RandomState] = np.random.RandomState(911),
    ):
        # Define random state
        rs = random_state if random_state else self.random_state
        # Check if the node has parents
        if not self.parents:
            return rs.choice(
                list(self.domain_noise_distribution.keys()),
                p=list(self.domain_noise_distribution.values()),
            )
        # Get the evaluation symbols for evaluation
        symbols = set()
        for eq in self.evaluation.values():
            symbols.update(eq.free_symbols)
        symbols = [str(symb) for symb in symbols]  # Parsed for easier comparison
        # Check if the parent values are provided
        assert set(symbols).issubset(
            parent_values.keys()
        ), "Parent values do not match the expected symbols"

        # Map categorical parent values
        substitutions = {
            symb: (
                self.parent_mappings[symb][parent_values[symb]]
                if symb in self.parent_mappings
                else parent_values[symb]
            )
            for symb in symbols
        }
        # Evaluate the expression
        evaluations = {}
        for possible_category, eq in self.evaluation.items():
            # Evaluate the expression
            evaluated = eq.subs(substitutions).evalf()
            assert evaluated.is_number, "Evaluation failed"
            # Cover for "Possibly" imaginary numbers
            evaluated = (
                float(evaluated.as_real_imag()[0])
                if not evaluated.is_real
                else float(evaluated)
            )
            # Add noise to the evaluated value
            noise = self.noise_distribution.generate(rs)
            # Calculate the CDF for the evaluated value and category
            evaluations[possible_category] = self.cdfs[possible_category](
                evaluated + noise
            )
        # Normalize the evaluations
        total = sum(evaluations.values())
        evaluations = (
            {cat: val / total for cat, val in evaluations.items()}
            if total > 0
            else {cat: 1 / len(evaluations) for cat in evaluations}
        )
        # Check if the evaluations are valid
        assert (
            all(0 <= val <= 1 for val in evaluations.values())
            and len(evaluations) == len(self.domain)
            and np.isclose(sum(evaluations.values()), 1.0)
        ), "Evaluations are not valid probabilities"
        # Sample from the categorical distribution
        return rs.choice(list(evaluations.keys()), p=list(evaluations.values()))

    def to_dict(self):
        """
        Converts the node to a dictionary representation.

        Returns:
            dict: Dictionary representation of the node.
        """
        # Serialize the random state.
        state = self.random_state.get_state()

        return {
            "class": self.__class__.__name__,
            "name": self.name,
            "accessibility": self.accessibility,
            "equation": {cat: str(eq) for cat, eq in self.evaluation.items()},
            "domain": self.domain,
            "noise_distribution": self.noise_distribution.to_dict(),
            "parents": self.parents,
            "parent_mappings": self.parent_mappings,
            "cdfs": {cat: self.cdfs[cat].to_list() for cat in self.cdfs},
            "domain_distribution": self.domain_noise_distribution,
            "random_state": {
                "state": state[0],
                "keys": state[1].tolist(),
                "pos": state[2],
                "has_gauss": state[3],
                "cached_gaussian": state[4],
            },
        }

    @classmethod
    def from_dict(cls, data: Dict):
        """
        Deserializes the node from a dictionary representation.

        Args:
            data (Dict): Dictionary containing node data.

        Returns:
            EquationBasedCategoricalSCMNode: An instance of the node.
        """
        # For categorical nodes, reconstruct the equation dictionary.
        evaluation = (
            {k: sp.sympify(v) for k, v in data["equation"].items()}
            if "equation" in data
            else None
        )

        # Reconstruct the CDF mappings from step points.
        cdfs = {
            cat: SerializableCDF.from_list(points)
            for cat, points in data["cdfs"].items()
        }
        # Deserialize the noise distribution
        noise_distribution = data["noise_distribution"]
        if noise_distribution["class"] == GaussianNoiseDistribution.__name__:
            noise_distribution = GaussianNoiseDistribution.from_dict(noise_distribution)
        elif noise_distribution["class"] == UniformNoiseDistribution.__name__:
            noise_distribution = UniformNoiseDistribution.from_dict(noise_distribution)
        else:
            raise ValueError(
                f"Unknown noise distribution class: {noise_distribution['class']}"
            )
        # Reconstruct the random state.
        random_state = np.random.RandomState()
        random_state.set_state(
            (
                str(data["random_state"]["state"]),
                np.array(data["random_state"]["keys"], dtype=np.uint32),
                int(data["random_state"]["pos"]),
                int(data["random_state"]["has_gauss"]),
                float(data["random_state"]["cached_gaussian"]),
            )
        )
        # Create the node
        new_class = cls(
            name=data["name"],
            accessibility=data.get("accessibility", ACCESSIBILITY_OBSERVABLE),
            evaluation=evaluation,
            domain=data["domain"],
            noise_distribution=noise_distribution,
            cdfs=cdfs,
            parents=data.get("parents"),
            parent_mappings=data.get("parent_mappings"),
            domain_distribution=data.get("domain_distribution"),
        )
        # Set the random state
        new_class.random_state = random_state

        # Return the new class
        return new_class


class FullyCategoricalSCMNode(BaseSCMNode):
    def generate_value(
        self,
        parent_values: Dict[str, float | str | int],
        random_state: Optional[np.random.RandomState] = np.random.RandomState(911),
    ):
        # Define random state
        rs = random_state if random_state else self.random_state
        # No need to check for parents since the self.evaluation has the probability distribution
        # Get the evaluation symbols for evaluation
        symbols = [
            str(symb) for symb in self.evaluation.free_symbols
        ]  # Parsed for easier comparison
        # Check if the parent values are provided
        assert set(symbols).issubset(
            parent_values.keys()
        ), "Parent values do not match the expected symbols"
        # Evaluate the expression
        distribution = self.evaluation(parent_values)
        # Check if the distribution is valid
        assert np.isclose(sum(distribution), 1.0), "Distribution does not sum to 1"
        # Return the final value
        return rs.choice(list(distribution.keys()), p=list(distribution.values()))

    def to_dict(self):
        """
        Converts the node to a dictionary representation.

        Returns:
            dict: Dictionary representation of the node.
        """
        return {
            "name": self.name,
            "accessibility": self.accessibility,
            "equation": self.evaluation.to_list(),
            "domain": self.domain,
            "noise": self.noise_distribution.to_dict(),
            "parents": self.parents,
            "parent_mappings": self.parent_mappings,
        }

    @classmethod
    def from_dict(cls, data: Dict):
        """
        Deserializes the node from a dictionary representation.

        Args:
            data (Dict): Dictionary containing node data.

        Returns:
            FullyCategoricalSCMNode: An instance of the node.
        """
        # Deserialize the noise distribution
        noise_distribution = BaseNoiseDistribution.from_dict(data["noise"])
        # Create the node
        return cls(
            name=data["name"],
            accessibility=data.get("accessibility", ACCESSIBILITY_OBSERVABLE),
            evaluation=SerializableCDF.from_list(data["equation"]),
            domain=data["domain"],
            noise_distribution=noise_distribution,
            parents=data.get("parents"),
            parent_mappings=data.get("parent_mappings"),
        )


class SerializableCDF:
    def __init__(self, sorted_samples):
        self.sorted_samples = np.array(sorted_samples)

    def __call__(self, x):
        return np.searchsorted(self.sorted_samples, x, side="right") / len(
            self.sorted_samples
        )

    def to_list(self):
        return self.sorted_samples.tolist()

    @classmethod
    def from_list(cls, data):
        return cls(np.array(data))


class BayesianNetworkSCMNode:
    def __init__(
        self,
        name: str,
        parents: list,
        values: list,
        probability_distribution: dict | list,
    ):
        self.name = name
        self.parents = parents
        self.domain = values
        self.probability_distribution = probability_distribution

    def generate_value(self, parent_values: dict, random_state) -> str:
        """
        Given a dictionary of parent values, returns a sampled value from the node's distribution.

        Args:
            parent_values (dict): A dictionary mapping parent names to their values.

        Returns:
            str: A sampled value from the node's distribution.
        """
        distribution = self.get_distribution(parent_values)
        return random_state.choice(self.domain, p=distribution)

    def get_distribution(self, parent_values: dict) -> list:
        if not self.parents:
            # Flatten the possibly nested list
            return [
                p
                for sub in self.probability_distribution
                for p in (sub if isinstance(sub, list) else [sub])
            ]

        key_ordered = [parent_values[parent] for parent in self.parents]
        key = ",".join(key_ordered)
        # Flatten the possibly nested list for conditional distributions
        return [
            p
            for sub in self.probability_distribution[key]
            for p in (sub if isinstance(sub, list) else [sub])
        ]

    def get_value_distribution(self, parent_values: dict) -> list:
        """
        Given a dictionary of parent values, returns the probability distribution
        over this node's values.

        Args:
            parent_values (dict): A dictionary mapping parent names to their values.

        Returns:
            list: The probability distribution over the node's values.
        """
        return self.get_distribution(parent_values)

    def to_dict(self) -> dict:
        return {
            "class": self.__class__.__name__,
            "name": self.name,
            "parents": self.parents,
            "values": self.domain,
            "probability_distribution": self.probability_distribution,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "BayesianNetworkSCMNode":
        return cls(
            name=data["name"],
            parents=data["parents"],
            values=data["values"],
            probability_distribution=data["probability_distribution"],
        )
