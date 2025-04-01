# Abstract
from abc import ABC, abstractmethod

# NetworkX
import networkx as nx


class BaseGenerator(ABC):
    @abstractmethod
    def generate(self):
        """
        Generate an object (e.g., a DAG or SCM) based on provided parameters.
        """
        pass


class BaseDAGGenerator(ABC):
    @abstractmethod
    def generate(self) -> nx.DiGraph:
        """
        Generate a DAG based on provided configuration.
        """
        pass
