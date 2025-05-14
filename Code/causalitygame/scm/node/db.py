# Math
import numpy as np
import pandas as pd

# typing
from typing import Dict, Any, List
from causalitygame.lib.utils.random_state_serialization import random_state_to_json, random_state_from_json

# Abstract Base Class
from causalitygame.scm.node.base import BaseCategoricSCMNode, BaseNumericSCMNode, ACCESSIBILITY_OBSERVABLE



class DatabaseDefinedSCMNode:
    
    def enable_eval_mode(self):
        self.eval_mode = True  # enables access to the unrevealed instances
    
    def disable_eval_mode(self):
        self.eval_mode = False

    def _generate_value(
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

class DatabaseDefinedNumericSCMNode(BaseNumericSCMNode, DatabaseDefinedSCMNode):
    def __init__(
        self,
        name: str,
        df: pd.DataFrame,
        revealed_to_agent: list,
        accessibility: str = ACCESSIBILITY_OBSERVABLE,
        random_state: np.random.RandomState = np.random.RandomState(911),
    ):
        self.cols = list(df.columns)
        super().__init__(
            name=name,
            accessibility=accessibility,
            evaluation=None,
            noise_distribution=None,
            parents=self.cols[:self.cols.index(name)],
            domain=sorted(pd.unique(df[name])),
            random_state=random_state
        )
        self.df = df
        self.revealed_to_agent = revealed_to_agent
        self.eval_mode = False
    
    def generate_value(
        self, parent_values: Dict[str, Any], random_state: np.random.RandomState = None
    ):
        return self._generate_value(parent_values=parent_values, random_state=random_state)

    @classmethod
    def from_dict(cls, data: dict) -> "DatabaseDefinedNumericSCMNode":
        print(data["random_state"])
        return cls(**data)



class DatabaseDefinedCategoricSCMNode(BaseCategoricSCMNode, DatabaseDefinedSCMNode):
    def __init__(
        self,
        name: str,
        df: pd.DataFrame,
        revealed_to_agent: list,
        accessibility: str = ACCESSIBILITY_OBSERVABLE,
        random_state: np.random.RandomState = np.random.RandomState(911),
    ):
        self.cols = list(df.columns)
        super().__init__(
            name=name,
            accessibility=accessibility,
            evaluation=None,
            noise_distribution=None,
            parents=self.cols[:self.cols.index(name)],
            domain=sorted(pd.unique(df[name])),
            random_state=random_state
        )
        self.df = df
        self.revealed_to_agent = revealed_to_agent
        self.eval_mode = False

    def generate_value(
        self, parent_values: Dict[str, Any], random_state: np.random.RandomState = None
    ):
        return self._generate_value(parent_values=parent_values, random_state=random_state)

    @classmethod
    def from_dict(cls, data: dict) -> "DatabaseDefinedCategoricSCMNode":
        print(data["random_state"])
        return cls(**data)
