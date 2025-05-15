# Environment
import json
from causalitygame.agents.base import BaseAgent
from causalitygame.game.Environment import Environment

# Game Instance
from causalitygame.game.GameInstance import GameInstance

# Evaluator
from causalitygame.evaluators.Evaluator import Evaluator

# Metrics
from causalitygame.evaluators.base import WeightedScores

# Utils
from tqdm import tqdm

# Math
import numpy as np
import pandas as pd

# Figures
import matplotlib.pyplot as plt

# Utils
import joblib

# Types
from typing import Optional, List, Tuple, Union, Callable, Dict, Any


class Game:
    def __init__(
        self,
        agents: List[Tuple[str, BaseAgent]],
        game_spec: str,
        behavior_metrics: Optional[List[Any]] = [],
        deliverable_metrics: Optional[List[Any]] = [],
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
        self.plambda = plambda
        self.seed = seed

        # will hold results keyed by agent name
        self.results: Dict[str, Dict[str, Any]] = {}

    def _make_game_instance(self):
        # Read the game instance from the JSON file
        with open(self.game_spec, "r") as f:
            game_instance_data = f.read()
        game_instance = json.loads(game_instance_data)
        # Create a game instance
        game_instance = GameInstance.from_dict(game_instance)
        return game_instance

    def _make_env(self, agent):
        # Create a game instance
        game_instance = self._make_game_instance()

        # Create an environment
        env = Environment(
            game_instance=game_instance,
            agent=agent,
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
        for name, agent in tqdm(self.agents):
            # 1) Build a fresh environment
            env = self._make_env(agent)

            # 1.1) Inform the agent about the game instance
            agent.inform(
                goal=env.game_instance.mission.__class__.__name__,
                behavior_metric=env.game_instance.mission.behavior_metric.__class__.name,
                deliverable_metric=env.game_instance.mission.deliverable_metric.__class__.name,
            )

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

            # 4) Store
            self.results[name] = {
                "agent": agent,
                "history": final_history,
                "scores": raw_results,
                "mission": env.game_instance.mission.evaluate(
                    env.game_instance.scm, final_history
                ),
            }

        return self.results

    def plot(self):
        """
        Scatter each agent's behavior vs. deliverable score on one figure.
        """
        if not self.results:
            raise RuntimeError("No results to plot: run() first.")
        # Get the game instance from the game spec
        game_instance = self._make_game_instance()

        # Plot the scores
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for name, run in self.results.items():
            # Get the mission scores
            behavior_score, deliverable_score = run["mission"]

            # Plot behavior vs. deliverable scores
            self._plot_scores(
                name=name,
                behavior_score=behavior_score,
                deliverable_score=deliverable_score,
                title="Behavior vs. Deliverable Scores",
                ax=axes[0],
            )
            # Plot behavior score over time
            self._plot_behavior_score(
                name=name,
                mission=game_instance.mission,
                history=run["history"],
                ax=axes[1],
            )
            axes[1].scatter(
                len(run["history"]) - 1, behavior_score, s=100, edgecolor="black"
            )
            # Plot deliverable score
            self.plot_deliverable_score(
                name=name,
                scm=game_instance.scm,
                mission=game_instance.mission,
                history=run["history"],
                ax=axes[2],
            )
            axes[2].scatter(
                len(run["history"]) - 1, deliverable_score, s=100, edgecolor="black"
            )

        # Set the title and labels
        fig.suptitle("Agent Comparison Scores", fontsize=14)
        # Behavior vs. Deliverable Scores
        axes[0].grid(True, linestyle="--", alpha=0.7)
        axes[0].legend(title="Agent", fontsize=10, title_fontsize=11)
        axes[0].set_xlabel("Behavior Score", fontsize=12)
        axes[0].set_ylabel("Deliverable Score", fontsize=12)
        # Behavior Score over Time
        axes[1].grid(True, linestyle="--", alpha=0.7)
        axes[1].legend(title="Agent", fontsize=10, title_fontsize=11)
        axes[1].set_xlabel("Number of Rounds", fontsize=12)
        axes[1].set_ylabel("Behavior Score", fontsize=12)
        # Deliverable Score
        axes[2].grid(True, linestyle="--", alpha=0.7)
        axes[2].legend(title="Agent", fontsize=10, title_fontsize=11)
        axes[2].set_xlabel("Number of Rounds", fontsize=12)
        axes[2].set_ylabel("Deliverable Score", fontsize=12)
        plt.tight_layout()
        plt.show()

    def _plot_scores(self, name, behavior_score, deliverable_score, title, ax):
        """
        Plot the scores of the agents.
        """
        ax.scatter(
            behavior_score,
            deliverable_score,
            s=100,
            edgecolor="black",
            label=name,
        )

    def _plot_behavior_score(self, name, mission, history, ax):
        # history
        scores = []
        for i in range(len(history)):
            # Get the behavior score
            behavior_score = mission.behavior_metric.evaluate(history.iloc[: i + 1])
            scores.append(behavior_score)
        # Plot the behavior score
        ax.plot(
            range(len(scores)),
            scores,
            label=name,
            alpha=0.7,
        )

    def plot_deliverable_score(self, name, scm, mission, history, ax):
        # history
        scores = []
        for i in range(len(history)):
            # Create new dataframe with the action object
            current_result = history.iloc[i]["current_result"]
            new_history = pd.DataFrame(
                {
                    "action_object": [current_result],
                }
            )
            # Get the behavior score
            deliverable_score = mission.deliverable_metric.evaluate(scm, new_history)
            scores.append(deliverable_score)

        # Plot the behavior score
        ax.plot(
            range(len(scores)),
            scores,
            label=name,
            alpha=0.7,
        )

    def plot_dags(self):
        """
        Plot the DAGs of the game instance.
        """
        if not self.results:
            raise RuntimeError("No results to plot: run() first.")

        # Get the DAG from the game instance
        game_instance = joblib.load(self.game_spec)
        dag = game_instance.scm.dag

        # Plot the DAG
        plt.figure(figsize=(8, 6))
        dag.plot()
        plt.title("DAG of the Game Instance", fontsize=14)
        plt.tight_layout()
        plt.show()

        # Find the appropriate number of rows and columns for subplots
        num_agents = len(self.agents)
        num_cols = 2
        num_rows = (num_agents + num_cols - 1) // num_cols
        # Create a figure with subplots
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 6 * num_rows))
        axes = axes.flatten()
        # Plot each agent's DAG
        for i, (name, res) in enumerate(self.results.items()):
            # Get the DAG from the agent's game instance
            dag = res["agent"].game_instance.scm.dag
            # Plot the DAG
            dag.plot(ax=axes[i])
            axes[i].set_title(f"DAG of {name}", fontsize=14)
