# Abstract
from abc import ABC, abstractmethod

# DAG
from causalitygame.scm.dag import DAG

# SCM
from causalitygame.scm.scm import SCM


class AbstractGenerator(ABC):
    @abstractmethod
    def generate(self):
        """
        Generate an object (e.g., a DAG or SCM) based on provided parameters.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")


class AbstractSCMGenerator(AbstractGenerator):
    @abstractmethod
    def generate(self) -> DAG:
        """
        Generate a DAG based on provided configuration.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")


class AbstractSCMGenerator(AbstractGenerator):
    @abstractmethod
    def generate(self) -> SCM:
        """
        Generate a SCM based on provided configuration.
        """
        raise NotImplementedError("This method should be overridden by subclasses.")
