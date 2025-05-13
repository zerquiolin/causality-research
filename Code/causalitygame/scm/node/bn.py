# Math
import numpy as np


# Abstract Base Class
from causalitygame.scm.node.base import BaseCategoricSCMNode, ACCESSIBILITY_OBSERVABLE


class BayesianNetworkSCMNode(BaseCategoricSCMNode):
    def __init__(
        self,
        name: str,
        parents: list,
        values: list,
        probability_distribution: dict | list,
        accessibility: str = ACCESSIBILITY_OBSERVABLE,
        random_state: np.random.RandomState = np.random.RandomState(911),
    ):
        self.name = name
        self.parents = parents
        self.domain = values
        self.probability_distribution = probability_distribution
        self.accessibility = accessibility
        self.random_state = random_state

        # sanity check of given distribution
        if isinstance(probability_distribution, list):
            assert all([isinstance(v, (float, np.float64)) for v in probability_distribution]), f"invalid entries in distribution for {name}: {probability_distribution}"
            s = np.sum(probability_distribution)
            assert np.isclose(s, 1), f"Invalid distribution for leaf node {name}, which sum up to {sum(probability_distribution)}: {probability_distribution}"
            if s != 1:
                    self.probability_distribution /= s
        else:
            for parent_combo, distribution in probability_distribution.items():
                assert all([isinstance(v, (float, np.float64)) for v in distribution]), f"invalid entries in distribution for {name}: {distribution}"
                s = np.sum(distribution)
                assert np.isclose(s, 1), f"Invalid distribution for parent combination {parent_combo} node {name}, which sum up to {sum(distribution)}: {distribution}"
                if s != 1:
                    probability_distribution[parent_combo] /= s

    def generate_value(self, parent_values: dict, random_state) -> str:
        """
        Given a dictionary of parent values, returns a sampled value from the node's distribution.

        Args:
            parent_values (dict): A dictionary mapping parent names to their values.

        Returns:
            str: A sampled value from the node's distribution.
        """
        rs = random_state if random_state else self.random_state
        distribution = self.get_distribution(parent_values)
        return rs.choice(self.domain, p=distribution)

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
        data = {
            "class": self.__class__.__name__,
            "name": self.name,
            "parents": self.parents,
            "values": self.domain,
            "accessibility": self.accessibility,
            "probability_distribution": self.probability_distribution,
        }
        if self.random_state:
            state = self.random_state.get_state()
            data["random_state"] = {
                "state": state[0],
                "keys": state[1].tolist(),
                "pos": state[2],
                "has_gauss": state[3],
                "cached_gaussian": state[4],
            }
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "BayesianNetworkSCMNode":
        # Reconstruct the random state.
        random_state = None
        if "random_state" in data:
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
        return cls(
            name=data["name"],
            parents=data["parents"],
            values=data["values"],
            probability_distribution=data["probability_distribution"] if data["parents"] else [v[0] for v in data["probability_distribution"]],
            accessibility=data.get("accessibility", ACCESSIBILITY_OBSERVABLE),
            random_state=random_state,
        )
