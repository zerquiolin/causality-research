# Math
import numpy as np
import pandas as pd

# typing
from typing import Dict, Any, List

# Abstract Base Class
from causalitygame.scm.node.base import BaseCategoricSCMNode, ACCESSIBILITY_OBSERVABLE


class DatabaseDefinedSCMNode(BaseCategoricSCMNode):
    def __init__(
        self,
        name: str,
        df: pd.DataFrame,
        revealed_to_agent: list,
        accessibility: str = ACCESSIBILITY_OBSERVABLE,
        random_state: np.random.RandomState = np.random.RandomState(911),
    ):
        self.name = name
        cols = list(df.columns)
        self.df = df
        self.revealed_to_agent = revealed_to_agent
        self.parents = cols[:cols.index(name)] # all predecessor columns are parents
        self.domain = sorted(pd.unique(df[name]))
        self.accessibility = accessibility
        self.random_state = random_state
        self.eval_mode = False
    
    def enable_eval_mode(self):
        self.eval_mode = True  # enables access to the unrevealed instances
    
    def disable_eval_mode(self):
        self.eval_mode = False

    def generate_value(
        self, parent_values: Dict[str, Any], random_state: np.random.RandomState = None
    ):
        
        # take own random state if None is given
        if random_state is None:
            random_state = self.random_state
        
        # Filter the dataset for rows matching parent values
        filtered = self.df
        if not self.eval_mode:
            filtered = filtered[self.revealed_to_agent]
        for parent in self.parents:
            if parent in parent_values:
                filtered = filtered[filtered[parent] == parent_values[parent]]
        
        # If no match found, fallback to full column
        if filtered.empty:
            return random_state.choice(self.df[self.name].dropna().tolist())

        return random_state.choice(filtered[self.name].dropna().tolist())

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
            "accessibility": self.accessibility
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
    def from_dict(cls, data: dict) -> "DatabaseDefinedSCMNode":

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
            df=data["df"],
            revealed_to_agent=data["revealed_to_agent"],
            parents=data["parents"],
            values=data["values"],
            accessibility=data.get("accessibility", ACCESSIBILITY_OBSERVABLE),
            random_state=random_state,
        )
