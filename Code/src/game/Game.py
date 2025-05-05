# Environment
from src.game.Environment import Environment
from src.scm.impl.basic_binary_scm import gen_binary_scm

# Game Instance
from src.game.GameInstance import GameInstance

# Evaluator
from src.evaluators.Evaluator import Evaluator

# Metrics
from src.evaluators.base import WeightedScores

# Utils
from src.lib.utils.json import load_json

# Math
import numpy as np

# Figures
import matplotlib.pyplot as plt

# Utils
import joblib

# Types
from typing import List, Tuple, Union, Callable, Dict, Any


class Game:
    def __init__(
        self,
        agents: List[Tuple[str, Any]],
        game_spec: str,
        behavior_metrics: List[Any],
        deliverable_metrics: List[Any],
        max_rounds: int = 100,
        plambda: float = 0.8,
        seed: int = 911,
    ):
        """
        agents: list of (name, agent_instance)
        game_spec: route of the game instance
        behavior_metrics / deliverable_metrics: metric objects for Evaluator
        plambda: trade‐off parameter for Evaluator
        """
        self.agents = agents
        self.game_spec = game_spec
        self.behavior_metrics = behavior_metrics
        self.deliverable_metrics = deliverable_metrics
        self.max_rounds = max_rounds
        self.plambda = plambda
        self.seed = seed

        # will hold results keyed by agent name
        self.results: Dict[str, Dict[str, Any]] = {}

    def _make_env(self, agent):
        # Read the game instance from the JSON file
        game_instance = joblib.load(self.game_spec)
        # Create a game instance
        game_instance = GameInstance.from_dict(game_instance)

        # # Set the random seed for reproducibility
        # random_state = np.random.RandomState(42)
        # # Generate a binary SCM with the specified seed
        # dag, scm = gen_binary_scm(random_state=random_state)
        # # Create a GameInstance with the generated SCM
        # game_instance = GameInstance(scm=scm, random_state=random_state)

        # Create an environment
        env = Environment(
            game_instance=game_instance,
            agent=agent,
            max_rounds=self.max_rounds,
            random_state=np.random.RandomState(self.seed),
        )

        return env

    def run(self) -> Dict[str, Dict[str, Any]]:
        """
        Runs each agent through the environment, evaluates, and stores:
          - 'history' (pandas DataFrame of state‐action history)
          - 'eval'    (raw Evaluator results)
          - 'behavior_score', 'deliverable_score'
        """
        for name, agent in self.agents:
            # 1) Build a fresh environment
            env = self._make_env(agent)

            # 2) Play the game
            final_state, final_history = env.run_game()

            # 3) Evaluate
            evaluator = Evaluator(
                scm=env.game_instance.scm,
                behavior_metrics=self.behavior_metrics,
                deliverable_metrics=self.deliverable_metrics,
                plambda=self.plambda,
            )
            raw_results = evaluator.evaluate(history=final_history)

            # 4) Aggregate into two scalars
            behavior_vals = [v for _, v in raw_results["behavior"]]
            deliverable_vals = [v for _, v in raw_results["deliverable"]]

            behavior_score = WeightedScores(
                values=behavior_vals,
                weights=[1 / len(behavior_vals)] * len(behavior_vals),
            ).compute()
            deliverable_score = WeightedScores(
                values=deliverable_vals,
                weights=[1 / len(deliverable_vals)] * len(deliverable_vals),
            ).compute()

            # 5) Store
            self.results[name] = {
                "history": final_history,
                "raw": raw_results,
                "behavior_score": behavior_score,
                "deliverable_score": deliverable_score,
            }

        return self.results

    def plot(self):
        """
        Scatter each agent's behavior vs. deliverable score on one figure.
        """
        if not self.results:
            raise RuntimeError("No results to plot: run() first.")

        plt.figure(figsize=(8, 6))
        for name, res in self.results.items():
            plt.scatter(
                res["behavior_score"],
                res["deliverable_score"],
                s=100,
                edgecolor="black",
                label=name,
            )

        plt.xlabel("Behavior Score", fontsize=12)
        plt.ylabel("Deliverable Score", fontsize=12)
        plt.title("Agent Performance Comparison", fontsize=14)
        plt.legend(title="Agent", fontsize=10, title_fontsize=11)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()
