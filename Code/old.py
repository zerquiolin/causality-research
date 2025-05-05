import matplotlib.pyplot as plt

# Agents
from src.agents.impl.ExhaustiveAgent import ExhaustiveAgent
from src.agents.impl.RandomAgent import RandomAgent

# SCM
from src.scm.dag import DAG

# Environment
from src.environment.impl.binary_environment import gen_binary_environment


# Metrics
from src.evaluators.impl.BehaviorMetrics import (
    ExperimentsBehaviorMetric,
    TreatmentsBehaviorMetric,
    RoundsBehaviorMetric,
)
from src.evaluators.impl.DeliverableMetrics import (
    SHDDeliverableMetric,
    F1DeliverableMetric,
    EdgeAccuracyDeliverableMetric,
    PEHEDeliverableMetric,
    PolicyRiskDeliverableMetric,
)

# Evaluators
from src.evaluators.Evaluator import Evaluator
from src.evaluators.base import WeightedScores

# Utils
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


if __name__ == "__main__":
    # Create an exhaustive agent
    # agent = ExhaustiveAgent()
    agent = RandomAgent(
        stop_probability=0.01, experiments_range=(1, 20), samples_range=(500, 1000)
    )
    # Generate the binary environment with the agent
    env = gen_binary_environment(agent)

    # Save the game instance
    env.game_instance.save(filename="./instances/game_instance.pkl")

    # Run the game simulation
    final_state, final_history = env.run_game()
    print("\n🏁 Game simulation complete!")

    # Retrieve and display the state-action history
    game_history_df = env.save_game_history(path="./output")

    print("\n📊 Game History DataFrame:")
    print(final_history)

    # Show the agent's learned DAG
    print("\n🧠 Learned DAG edges:")
    print(final_state["final_answer"])
    print(final_state["final_answer"].edges)

    print(type(final_state["final_answer"]))
    print(type(final_history.iloc[-1]["action_object"]))
    # Create a DAG custom object
    learned_dag = DAG(final_history.iloc[-1]["action_object"])

    # Create the evaluator
    behavior_metrics = [
        ExperimentsBehaviorMetric(e=1.0),
        TreatmentsBehaviorMetric(t=0.5),
        RoundsBehaviorMetric(r=0.2),
    ]
    deliverable_metrics = [
        SHDDeliverableMetric(),
        F1DeliverableMetric(),
        EdgeAccuracyDeliverableMetric(),
        # PEHEDeliverableMetric(scm=env.game_instance.scm)
        # PolicyRiskDeliverableMetric(scm=env.game_instance.scm),
    ]

    evaluator = Evaluator(
        scm=env.game_instance.scm,
        behavior_metrics=behavior_metrics,
        deliverable_metrics=deliverable_metrics,
        plambda=0.8,
    )

    results = evaluator.evaluate(history=final_history)
    print("\n📊 Evaluation Results:")
    print(results)

    behavior_score = WeightedScores(
        values=[v for _, v in results["behavior"]],
        weights=[0.5, 0.3, 0.2],
    ).compute()

    deliverable_score = WeightedScores(
        values=[v for _, v in results["deliverable"]],
        weights=[0.5, 0.3, 0.2],
    ).compute()

    plt.figure(figsize=(8, 6))
    plt.scatter(
        behavior_score,
        deliverable_score,
        color="dodgerblue",
        edgecolor="black",
        s=120,
        label="Agent Performance",
    )

    # Axes labels
    plt.xlabel("Behavior Score", fontsize=12)
    plt.ylabel("Deliverable Score", fontsize=12)

    # Title
    plt.title("Behavior vs Deliverable Score", fontsize=14)

    # Legend
    plt.legend(loc="best", fontsize=10)

    # Grid and style
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    plt.show()
