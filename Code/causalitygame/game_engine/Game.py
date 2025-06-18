# Environment
import json
from causalitygame.agents.abstract import BaseAgent
from causalitygame.game_engine.Environment import Environment

# Game Instance
from causalitygame.game_engine.GameInstance import GameInstance

# Evaluator
from causalitygame.evaluators.Evaluator import Evaluator

# Metrics
from causalitygame.evaluators.abstract import WeightedScores

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
from typing import Optional, List, Tuple, Union, Callable, Dict, Any, TypedDict


class Hooks(TypedDict):
    """
    Hooks for the game.
    """

    on_game_start: Optional[Callable[[], None]] = None
    on_agent_game_start: Optional[Callable[[str], None]] = None
    on_round_start: Optional[Callable[[str, int, Dict, List, Dict], None]] = None
    on_action_chosen: Optional[Callable[[str, Dict, str, any], None]] = None
    on_action_evaluated: Optional[Callable[[str, Dict, str, any, Tuple], None]] = None
    on_round_end: Optional[Callable[[str, int, Dict, str, any, Dict, Tuple], None]] = (
        None
    )
    on_agent_game_end: Optional[
        Callable[
            [
                str,
            ],
            None,
        ]
    ] = None
    on_game_end: Optional[
        Callable[
            [],
            None,
        ]
    ] = None


class Game:
    def __init__(
        self,
        agents: List[Tuple[str, BaseAgent]],
        game_spec: str,
        behavior_metrics: Optional[List[Any]] = [],
        deliverable_metrics: Optional[List[Any]] = [],
        hooks: Optional[Hooks] = {},
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
        self.hooks = hooks
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

    def _make_env(self, name, agent):
        # Create a game instance
        game_instance = self._make_game_instance()

        # Create an environment
        env = Environment(
            game_instance=game_instance,
            agent=agent,
            agent_name=name,
            hooks=self.hooks,
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
        if "on_game_start" in self.hooks and callable(self.hooks["on_game_start"]):
            self.hooks["on_game_start"]()
        for name, agent in tqdm(self.agents):
            if "on_agent_game_start" in self.hooks and callable(
                self.hooks["on_agent_game_start"]
            ):
                self.hooks["on_agent_game_start"](name)
            # 1) Build a fresh environment
            env = self._make_env(name, agent)

            # 1.1) Inform the agent about the game instance
            agent.inform(
                goal={
                    "goal": env.game_instance.mission.name,
                    "description": env.game_instance.mission.description,
                },
                behavior_metric=env.game_instance.mission.behavior_metric.name,
                deliverable_metric=env.game_instance.mission.deliverable_metric.name,
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

            if "on_agent_game_end" in self.hooks and callable(
                self.hooks["on_agent_game_end"]
            ):
                self.hooks["on_agent_game_end"](name)

        if "on_game_end" in self.hooks and callable(self.hooks["on_game_end"]):
            self.hooks["on_game_end"]()

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
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        # Add a main title slightly lower (so legend can go above it)
        fig.suptitle("Agent Comparison Scores", fontsize=14)

        for name, run in self.results.items():
            # cd = DAG(run["history"].iloc[-1]["action_object"])
            # cd.plot()
            behavior_scores, deliverable_scores = [], []
            for i in range(len(run["history"])):
                current_behavior_score, current_deliverable_score = (
                    game_instance.mission.evaluate(
                        game_instance.scm, run["history"].iloc[: i + 1]
                    )
                )
                behavior_scores.append(current_behavior_score)
                deliverable_scores.append(current_deliverable_score)

            # Get the mission scores
            behavior_score, deliverable_score = run["mission"]

            # Plot behavior vs. deliverable scores
            # TODO: Fix the behavior score plot
            self._plot_scores(
                name=name,
                behavior_score=behavior_score,
                deliverable_score=deliverable_score,
                # deliverable_score=deliverable_scores[-1],
                title="Behavior vs. Deliverable Scores",
                ax=axes[0],
            )
            if deliverable_scores[-1] >= 10**4:
                axes[0].set_yscale("log")
            # Plot behavior score over time result
            self._plot_behavior_score(
                name=name,
                scores=behavior_scores,
                ax=axes[1],
            )
            # TODO: Fix the behavior score plot
            axes[1].scatter(
                # len(run["history"]) - 1, behavior_score, s=100, edgecolor="black"
                len(run["history"]) - 1,
                behavior_scores[-1],
                s=100,
                edgecolor="black",
            )
            # Plot deliverable score result
            # TODO: Fix the deliverable score plot
            self.plot_deliverable_score(
                name=name,
                scores=deliverable_scores,
                ax=axes[2],
            )
            axes[2].scatter(
                # len(run["history"]) - 1, deliverable_score, s=100, edgecolor="black"
                len(run["history"]) - 1,
                deliverable_scores[-1],
                s=100,
                edgecolor="black",
            )
            if deliverable_scores[-1] >= 10**4:
                axes[2].set_yscale("log")

        # Collect legend entries from one axis only (e.g., the first)
        handles, labels = axes[0].get_legend_handles_labels()
        unique = dict(zip(labels, handles))  # remove duplicates

        # Add a single legend above the whole figure
        fig.legend(
            handles=unique.values(),
            labels=unique.keys(),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.93),
            ncol=11,
            fontsize="small",
        )

        plt.subplots_adjust(top=0.83)  # Adjust the top to make room for the legend

        # Behavior vs. Deliverable Scores
        axes[0].grid(True, linestyle="--", alpha=0.7)
        axes[0].set_xlabel("Behavior Score", fontsize=12)
        axes[0].set_ylabel("Deliverable Score", fontsize=12)
        # Behavior Score over Time
        axes[1].grid(True, linestyle="--", alpha=0.7)
        axes[1].set_xlabel("Number of Rounds", fontsize=12)
        axes[1].set_ylabel("Behavior Score", fontsize=12)
        # Deliverable Score
        axes[2].grid(True, linestyle="--", alpha=0.7)
        axes[2].set_xlabel("Number of Rounds", fontsize=12)
        axes[2].set_ylabel("Deliverable Score", fontsize=12)

        # Show the plot
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

    def _plot_behavior_score(self, name, scores, ax):
        # Plot the behavior score
        ax.plot(
            range(len(scores)),
            scores,
            label=name,
            alpha=0.7,
        )

    def plot_deliverable_score(self, name, scores, ax):
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
