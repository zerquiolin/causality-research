# File: causality_game/generators/base_generator.py
from abc import ABC, abstractmethod


class BaseGenerator(ABC):
    @abstractmethod
    def generate(self):
        """
        Generate an object (e.g., a DAG or SCM) based on provided parameters.
        """
        pass
