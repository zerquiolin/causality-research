# Science
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt

# Utils
import json
from tqdm import tqdm

# Types
from causalitygame.agents.abstract import BaseAgent
from causalitygame.evaluators.Evaluator import Evaluator
from causalitygame.game_engine.Environment import Environment
from causalitygame.game_engine.GameInstance import GameInstance
from typing import Optional, List, Tuple, Dict, Any, Callable, TypedDict


class Hooks(TypedDict, total=False):
    """
    Dictionary of optional lifecycle hooks for observability or instrumentation.

    All keys are optional. Hooks can be used to trace agent actions or gather logs.
    """

    on_game_start: Optional[Callable[[], None]]
    on_agent_game_start: Optional[Callable[[str], None]]
    on_round_start: Optional[Callable[[str, int, Dict, List, Dict], None]]
    on_action_chosen: Optional[Callable[[str, Dict, str, Any], None]]
    on_action_evaluated: Optional[Callable[[str, Dict, str, Any, Tuple], None]]
    on_round_end: Optional[Callable[[str, int, Dict, str, Any, Dict, Tuple], None]]
    on_agent_game_end: Optional[Callable[[str], None]]
    on_game_end: Optional[Callable[[], None]]


class Game:
    """
    Core game loop runner for evaluating causal reasoning agents.

    This class instantiates game environments, runs simulations with agents,
    applies evaluation metrics, and supports visualization and debugging via hooks.

    Attributes:
        agents (List[Tuple[str, BaseAgent]]): List of (name, agent) tuples.
        game_spec (str): Path to JSON file describing the game instance.
        behavior_metrics (List): List of callable behavior metric evaluators.
        deliverable_metrics (List): List of callable deliverable metric evaluators.
        hooks (Hooks): Optional instrumentation hooks for different lifecycle stages.
        plambda (float): Trade-off parameter between behavior and deliverable.
        seed (int): Random seed for reproducibility.
        results (Dict): Stores per-agent game results after run().
    """

    def __init__(
        self,
        agents: List[Tuple[str, BaseAgent]],
        game_spec: str,
        behavior_metrics: Optional[List[Any]] = None,
        deliverable_metrics: Optional[List[Any]] = None,
        hooks: Optional[Hooks] = None,
        plambda: float = 0.8,
        seed: int = 911,
    ):
        self.agents = agents
        self.game_spec = game_spec
        self.behavior_metrics = behavior_metrics or []
        self.deliverable_metrics = deliverable_metrics or []
        self.hooks = hooks or {}
        self.plambda = plambda
        self.seed = seed
        self.results: Dict[str, Dict[str, Any]] = {}

        # Preload a game instance template (used in score visualization)
        self._game_instance = self._make_game_instance()

    def _make_game_instance(self) -> GameInstance:
        """
        Load the game instance object from the game spec JSON.

        Returns:
            GameInstance: Initialized game instance object.
        """
        with open(self.game_spec, "r") as f:
            return GameInstance.from_dict(json.load(f))

    def _make_env(self, name: str, agent: BaseAgent) -> Environment:
        """
        Create a new environment for the given agent and name.

        Args:
            name (str): Name of the agent.
            agent (BaseAgent): The agent instance.

        Returns:
            Environment: New game environment.
        """
        return Environment(
            game_instance=self._make_game_instance(),
            agent=agent,
            agent_name=name,
            hooks=self.hooks,
            random_state=np.random.RandomState(self.seed),
        )

    def run(self) -> Dict[str, Dict[str, Any]]:
        """
        Run the simulation for all agents, evaluate their output, and record results.

        Returns:
            Dict[str, Dict[str, Any]]: Agent results including scores and histories.

        Raises:
            RuntimeError: If agents list is empty.
        """
        if not self.agents:
            raise RuntimeError("No agents provided to run.")

        # Hook: Game start
        if hook := self.hooks.get("on_game_start"):
            hook()

        for name, agent in tqdm(self.agents, desc="Running agents"):
            # Hook: Agent game start
            if hook := self.hooks.get("on_agent_game_start"):
                hook(name)

            # Setup environment and inform agent of game goal
            env = self._make_env(name, agent)
            agent.inform(
                goal={
                    "goal": env.game_instance.mission.name,
                    "description": env.game_instance.mission.description,
                },
                behavior_metric=env.game_instance.mission.behavior_metric.name,
                deliverable_metric=env.game_instance.mission.deliverable_metric.name,
            )

            # Play the game and collect history
            _, history = env.run_game()

            # Evaluate history using provided metrics
            evaluator = Evaluator(
                scm=env.game_instance.scm,
                behavior_metrics=self.behavior_metrics,
                deliverable_metrics=self.deliverable_metrics,
                plambda=self.plambda,
            )
            raw_scores = evaluator.evaluate(history=history)

            # Store evaluation and metadata
            self.results[name] = {
                "agent": agent,
                "history": history,
                "scores": raw_scores,
                "mission": env.game_instance.mission.evaluate(
                    env.game_instance.scm, history
                ),
            }

            # Hook: Round end
            if hook := self.hooks.get("on_agent_game_end"):
                hook(name)

        # Hook: Game end
        if hook := self.hooks.get("on_game_end"):
            hook()

        return self.results

    def compute_score_trajectories(
        self, history: pd.DataFrame
    ) -> Tuple[List[float], List[float]]:
        """
        Compute per-round behavior and deliverable scores.

        Args:
            history (pd.DataFrame): Agent's game history.

        Returns:
            Tuple[List[float], List[float]]: Lists of behavior and deliverable scores over time.
        """
        behavior_scores, deliverable_scores = [], []
        for i in range(len(history)):
            b, d = self._game_instance.mission.evaluate(
                self._game_instance.scm, history.iloc[: i + 1]
            )
            behavior_scores.append(b)
            deliverable_scores.append(d)
        return behavior_scores, deliverable_scores

    def _plot_scores(self, name: str, b_score: float, d_score: float, ax):
        """
        Scatter plot of final behavior vs. deliverable scores.

        Args:
            name (str): Agent name.
            b_score (float): Final behavior score.
            d_score (float): Final deliverable score.
            ax (Axes): Matplotlib axes object.
        """
        # Plot point with a specific facecolor (e.g., for auto-colors by label)
        point = ax.scatter(b_score, d_score, s=100, label=name)

        # Use the facecolor of the point for the label text
        ax.text(
            b_score,
            d_score,
            f"({b_score:.2f}, {d_score:.2f})",
            color=point.get_facecolor()[0],  # get the RGBA tuple
            fontsize=10,
            ha="left",
            va="center",
        )
        # ax.scatter(b_score, d_score, s=100, edgecolor="black", label=name)
        # # Plot final score as number with the same color as the point
        # ax.text(
        #     b_score,
        #     d_score,
        #     f"({b_score:.2f}, {d_score:.2f})",
        #     color=ax.collections[-1].get_edgecolor()[0],
        #     fontsize=10,
        #     ha="left",
        #     va="center",
        # )

    def _plot_time_series(self, name: str, scores: List[float], ylabel: str, ax):
        """
        Plot a time series of scores across rounds.

        Args:
            name (str): Agent name.
            scores (List[float]): Score trajectory.
            ylabel (str): Y-axis label.
            ax (Axes): Matplotlib axes object.
        """
        ax.plot(range(len(scores)), scores, label=name, alpha=0.7)
        ax.scatter(len(scores) - 1, scores[-1], s=100, edgecolor="black")
        # Plot final score as number with the same color as the line
        ax.text(
            len(scores) - 1,
            scores[-1],
            f"{scores[-1]:.2f}",
            color=ax.lines[-1].get_color(),
            fontsize=10,
            ha="left",
            va="center",
        )
        ax.set_ylabel(ylabel)
        if max(scores) >= 10**3 or min(scores) <= 10**-3:
            ax.set_yscale("log")

        if len(scores) >= 10**3:
            ax.set_xticks(np.arange(0, len(scores), step=1000))
            ax.set_yscale("log")
            ax.set_xscale("log")

    def plot(self):
        """
        Plot all agents' behavior vs deliverable, and their score trajectories.

        Raises:
            RuntimeError: If no results are available.
        """
        if not self.results:
            raise RuntimeError("No results to plot: run() must be called first.")

        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        fig.suptitle("Agent Comparison Scores", fontsize=14)

        for name, run in self.results.items():
            behavior_scores, deliverable_scores = self.compute_score_trajectories(
                run["history"]
            )
            final_behavior, final_deliverable = run["mission"]

            self._plot_scores(name, final_behavior, final_deliverable, axes[0])
            self._plot_time_series(name, behavior_scores, "Behavior Score", axes[1])
            self._plot_time_series(
                name, deliverable_scores, "Deliverable Score", axes[2]
            )

        # Set common axis labels and formatting
        for ax, xlabel in zip(
            axes, ["Behavior Score", "Number of Rounds", "Number of Rounds"]
        ):
            ax.set_xlabel(xlabel)
            ax.grid(True, linestyle="--", alpha=0.7)

        # Deduplicated legend
        handles, labels = axes[0].get_legend_handles_labels()
        unique = dict(zip(labels, handles))
        fig.legend(
            handles=unique.values(),
            labels=unique.keys(),
            loc="upper center",
            bbox_to_anchor=(0.5, 0.93),
            ncol=11,
            fontsize="small",
        )

        plt.subplots_adjust(top=0.83)
        plt.show()
