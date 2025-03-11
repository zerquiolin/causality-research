# File: causality_game/agents/base_agent.py
from abc import ABC, abstractmethod


class BaseAgent(ABC):
    @abstractmethod
    def choose_action(self, state: dict):
        """
        Given the current state, choose an action.
        """
        pass
