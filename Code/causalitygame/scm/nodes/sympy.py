from typing import Any, Callable, Dict, List, Optional, Tuple
import sympy as sp
import numpy as np
import pandas as pd


from causalitygame.scm.nodes.abstract import (
    ACCESSIBILITY_CONTROLLABLE,
    ACCESSIBILITY_LATENT,
    ACCESSIBILITY_OBSERVABLE,
    BaseNoiseDistribution,
    BaseNumericSCMNode,
    BaseCategoricSCMNode,
)

from causalitygame.scm.noises import (
    GaussianNoiseDistribution,
    NoNoiseDistribution,
    UniformNoiseDistribution,
    DiracNoiseDistribution,
)

from collections import Counter


class EquationBasedSCMNode:

    def __init__(self):

        # check that equations and parents coincide and memorize required symbols per node
        def get_symbols_for_formula_while_checking_that_those_are_declared(formula):
            symbols = set([str(s) for s in formula.free_symbols])
            assert (
                not symbols or self.parents is not None
            ), f"No parents are given (None) even though the formula has symbols: {symbols}"
            undeclared_symbols = symbols.difference(self.parents)
            assert (
                not undeclared_symbols
            ), f"Formula {formula} has undeclared variables {undeclared_symbols} that occur in the formula but not in the parents, which are specified as {self.parents}."
            return symbols

        if isinstance(self.evaluation, dict):
            self.symbols_needed_for_evaluation = {
                eq_name: get_symbols_for_formula_while_checking_that_those_are_declared(
                    eq
                )
                for eq_name, eq in self.evaluation.items()
                if eq is not None
            }
        else:
            self.symbols_needed_for_evaluation = (
                get_symbols_for_formula_while_checking_that_those_are_declared(
                    self.evaluation
                )
                if self.evaluation is not None
                else None
            )


class EquationBasedNumericalSCMNode(BaseNumericSCMNode, EquationBasedSCMNode):
    def generate_values(
        self,
        parent_values: pd.DataFrame,
        random_state: Optional[np.random.RandomState] = None,
    ):
        # Define random state
        rs = random_state if random_state else self.random_state

        # Check if the node has parents
        if not self.parents:
            # Draw random values uniformly from the domain
            values = rs.uniform(self.domain[0], self.domain[1], size=len(parent_values))
            return values

        # Check if the parent values are provided
        assert set(self.parents).issubset(
            set(parent_values.columns)
        ), "Parent values do not match the expected symbols"

        # Map all the parent values to the parent mappings
        if self.parent_mappings:
            parent_values = parent_values.copy()
            for parent, mapping in self.parent_mappings.items():
                if parent in parent_values.columns:
                    parent_values[parent] = parent_values[parent].map(mapping)

        # Evaluate the expression
        f = sp.lambdify(self.parents, self.evaluation, modules="numpy")
        evaluated = f(*tuple(parent_values[self.parents].values.T))

        assert not np.any(
            np.iscomplex(evaluated)
        ), f"Evaluation of {self.evaluation} lead to complex numbers {evaluated}"

        # Add noise to the evaluated value
        noise = self.noise_distribution.generate(size=len(evaluated), random_state=rs)

        return np.minimum(np.maximum(evaluated + noise, self.domain[0]), self.domain[1])

    def _to_dict(self):
        """
        Converts the node to a dictionary representation.

        Returns:
            dict: Dictionary representation of the node.
        """
        representation = {
            "equation": str(self.evaluation) if self.evaluation else None,
            "noise_distribution": self.noise_distribution.to_dict(),
        }

        if not self.parent_mappings:
            representation["parent_mappings"] = None

        return representation

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

        if "noise_distribution" in data:
            noise_distribution = data["noise_distribution"]
            if isinstance(noise_distribution, dict):
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
                elif noise_distribution["class"] == NoNoiseDistribution.__name__:
                    noise_distribution = NoNoiseDistribution.from_dict(
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
            noise_distribution = UniformNoiseDistribution(low=0, high=1)

        # Create the node
        new_class = cls(
            name=data["name"],
            accessibility=data.get("accessibility", ACCESSIBILITY_OBSERVABLE),
            evaluation=evaluation,
            domain=data["domain"],
            noise_distribution=noise_distribution,
            parents=data.get("parents", None),
            parent_mappings=data.get("parent_mappings", None),
            random_state=data["random_state"],
        )

        # Return the new class
        return new_class


class EquationBasedCategoricalSCMNode(BaseCategoricSCMNode, EquationBasedSCMNode):
    def __init__(
        self,
        name: str,
        evaluation: Optional[Callable],
        domain: List[float | str],
        noise_distribution: BaseNoiseDistribution,
        cdfs: Optional[List[Callable]] = None,
        accessibility: str = ACCESSIBILITY_OBSERVABLE,
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

    def prepare_new_random_state_structure(self, random_state):
        return {
            "noise": np.random.RandomState(random_state.randint(0, 10**5)),
            "choice": np.random.RandomState(random_state.randint(0, 10**5)),
        }

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
        samples = self.noise_distribution.generate(size=n_samples)

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

    def generate_values(
        self, parent_values: pd.DataFrame, random_state: Optional[dict] = None
    ):

        # Define random state
        if random_state is None:
            rs_noise, rs_choice = (self.random_state, self.random_state)
        elif isinstance(random_state, dict):
            rs_noise, rs_choice = (random_state["noise"], random_state["choice"])
        else:
            rs_noise, rs_choice = (random_state, random_state)

        # now start sampling
        self.logger.info(
            "Drawing %s values for categorical node %s with parents %s",
            len(parent_values),
            self.name,
            self.parents,
        )
        self.logger.debug(f"Parent mapping of {self.name} is %s.", self.parent_mappings)

        # Check if the node has parents
        if not self.parents:
            return rs_noise.choice(
                list(self.domain_noise_distribution.keys()),
                p=list(self.domain_noise_distribution.values()),
            )
        else:
            missing_parents = set(self.parents).difference(set(parent_values.keys()))
            assert (
                not missing_parents
            ), f"Cannot generate value for {self.name} as no values provided for some parents: {missing_parents}"

        # Check that all parent values are provided
        symbols = set()
        for eq_name, eq in self.evaluation.items():
            missing_values = self.symbols_needed_for_evaluation[eq_name].difference(
                parent_values.keys()
            )
            assert (
                not missing_values
            ), f"Cannot evaluate formula {eq} of variable {self.name} because no values are provided for parent {missing_values}"
            symbols.update(self.symbols_needed_for_evaluation[eq_name])
        symbols = list(symbols)

        # Evaluate the expression
        possible_categories = list(self.evaluation.keys())
        evaluations = []

        # determine the noise terms once (important to do this here simultaneously for all instances to not confuse the random state)
        noises = self.noise_distribution.generate(
            size=(len(parent_values), len(possible_categories)), random_state=rs_noise
        )

        for i, possible_category in enumerate(possible_categories):

            eq = self.evaluation[possible_category]

            # Evaluate the expression
            f = sp.lambdify(symbols, eq, modules="numpy")
            evaluated = f(*tuple(parent_values[symbols].values.T))

            # Calculate the CDF for the evaluated value and category to obtain values normalized between 0 and 1
            evaluations.append(self.cdfs[possible_category](evaluated + noises[:, i]))

        evaluations = np.array(evaluations).T
        expected_shape = (len(parent_values), len(self.domain))
        assert (
            expected_shape == evaluations.shape
        ), f"Shape of evaluations should be {expected_shape} but was {evaluations.shape}"

        # Normalize the evaluations
        evaluations = np.maximum(
            evaluations, 10**-20
        )  # to avoid that all entries in a row are 0
        evaluations /= evaluations.sum(axis=1)[:, np.newaxis]

        # Check if the evaluations are valid
        assert (
            np.all(0 <= evaluations)
            and np.all(evaluations <= 1)
            and np.all(np.isclose(np.sum(evaluations, axis=1), 1.0))
        ), f"Evaluations are not valid probabilities: {evaluations}"

        # Sample from the categorical distribution
        return [rs_choice.choice(self.domain, p=dist) for dist in evaluations]

    def _to_dict(self):
        """
        Converts the node to a dictionary representation.

        Returns:
            dict: Dictionary representation of the node.
        """
        representation = {
            "equation": (
                {cat: str(eq) for cat, eq in self.evaluation.items()}
                if self.evaluation
                else None
            ),
            "cdfs": (
                {cat: self.cdfs[cat].to_list() for cat in self.cdfs}
                if self.cdfs
                else None
            ),
            "parent_mappings": self.parent_mappings,
            "domain_distribution": self.domain_noise_distribution,
            "noise_distribution": self.noise_distribution.to_dict(),
        }
        return representation

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
            if "equation" in data and data["equation"] != None
            else None
        )

        # Reconstruct the CDF mappings from step points.
        cdfs = (
            {
                cat: SerializableCDF.from_list(points)
                for cat, points in data["cdfs"].items()
            }
            if data.get("cdfs", None)
            else None
        )

        # Deserialize the noise distribution
        noise_distribution = data["noise_distribution"]
        if noise_distribution["class"] == GaussianNoiseDistribution.__name__:
            noise_distribution = GaussianNoiseDistribution.from_dict(noise_distribution)
        elif noise_distribution["class"] == UniformNoiseDistribution.__name__:
            noise_distribution = UniformNoiseDistribution.from_dict(noise_distribution)
        elif noise_distribution["class"] == NoNoiseDistribution.__name__:
            noise_distribution = NoNoiseDistribution.from_dict(noise_distribution)
        else:
            raise ValueError(
                f"Unknown noise distribution class: {noise_distribution['class']}"
            )
        # Reconstruct the random state.

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
            random_state=data["random_state"],
        )

        # Return the new class
        return new_class


class SerializableCDF:
    def __init__(self, sorted_samples):
        self.sorted_samples = np.array(sorted_samples)
        assert (
            len(self.sorted_samples.shape) == 1
        ), "SerializableCDF needs a one-dimensional vector of values."

    def __call__(self, x):
        return np.searchsorted(self.sorted_samples, x, side="right") / len(
            self.sorted_samples
        )

    def to_list(self):
        return self.sorted_samples.tolist()

    @classmethod
    def from_list(cls, data):
        return cls(np.array(data))
