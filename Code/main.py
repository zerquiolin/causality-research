import matplotlib.pyplot as plt

# Agents
from src.agents.impl.ExhaustiveAgent import ExhaustiveAgent

# SCM
from src.scm.dag import DAG

# Environment
from src.environment.impl.binary_environment import gen_binary_environment

# Metrics
from src.evaluators.impl.DagDiscoveryMetrics import gen_dag_discovery_metrics

# Utils
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


if __name__ == "__main__":
    # Create an exhaustive agent
    agent = ExhaustiveAgent()
    # Generate the binary environment with the agent
    env = gen_binary_environment(agent)
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

    # Create a DAG custom object
    print(final_history.iloc[-1].keys())
    print(final_history.iloc[-1]["action_object"])
    learned_dag = DAG(final_history.iloc[-1]["action_object"])
    # learned_dag.plot("Learned DAG")

    # Get the evaluation metrics
    metrics = gen_dag_discovery_metrics(scm=env.game_instance.scm)

    # Evaluate the performance of the agent
    performance = metrics.evaluate(
        history=final_history,
    )

    # Extract scores
    behavior_score = performance["behavior"]["score"]
    deliverable_score = performance["deliverable"]["score"]
    global_score = performance["global_score"]

    # --- First Plot: Score Summary ---
    plt.figure(figsize=(10, 4))
    x_values = [behavior_score, deliverable_score]
    labels = ["Behavior Score", "Deliverable Score"]

    plt.scatter(x_values, [1, 1], color="dodgerblue", s=100, zorder=3)
    for x, label in zip(x_values, labels):
        plt.text(x, 1.05, f"{label}: {x:.2f}", ha="center", fontsize=10)

    plt.axvline(
        global_score,
        color="red",
        linestyle="--",
        label=f"Global Score: {global_score:.2f}",
    )
    plt.text(
        global_score,
        0.95,
        f"Global Score: {global_score:.2f}",
        color="red",
        ha="center",
        fontsize=10,
    )

    plt.yticks([])
    plt.ylim(0.8, 1.2)
    plt.xlabel("Score")
    plt.title("Metric Scores Overview")
    plt.grid(axis="x", linestyle=":", linewidth=0.7)
    plt.tight_layout()
    plt.show()

    # --- Second Plot: Metric Breakdown ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Behavior metrics
    behavior_metrics = performance["behavior"]["metrics"]
    b_keys = list(behavior_metrics.keys())
    b_vals = list(behavior_metrics.values())
    axes[0].barh(b_keys, b_vals, color="skyblue")
    axes[0].set_title("Behavior Metrics")
    for i, v in enumerate(b_vals):
        axes[0].text(v, i, f"{v:.2f}", va="center", ha="left")

    # Deliverable metrics
    deliverable_metrics = performance["deliverable"]["metrics"]
    d_keys = list(deliverable_metrics.keys())
    d_vals = list(deliverable_metrics.values())
    axes[1].barh(d_keys, d_vals, color="lightgreen")
    axes[1].set_title("Deliverable Metrics")
    for i, v in enumerate(d_vals):
        axes[1].text(v, i, f"{v:.2f}", va="center", ha="left")

    plt.suptitle("Metric Breakdown by Category", fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # Print the performance metrics
    print("\n📈 Performance Metrics:")
    print(performance)

    # --- Third Plot: Score plot ---
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
