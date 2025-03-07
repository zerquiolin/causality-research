# File: causality_game/generators/base_dag_generator.py
from abc import ABC, abstractmethod
import networkx as nx


class BaseDAGGenerator(ABC):
    @abstractmethod
    def generate(self) -> nx.DiGraph:
        """
        Generate a DAG based on provided configuration.
        """
        pass
