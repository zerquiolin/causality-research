# Game
import pandas as pd
import causalitygame as cg

# Plotting
import matplotlib.pyplot as plt

# Agents
from causalitygame.agents.random import RandomAgent
from causalitygame.agents.exhaustive import ExhaustiveAgent


# Define the agents
agents = [
    (f"Random Agent {i}", RandomAgent(seed=911 + i, samples_range=(1, 3)))
    for i in range(1, 3)
]
# agents = []
# Add an exhaustive agent
agents.append(("Exhaustive Agent", ExhaustiveAgent(num_obs=1)))
# Game Instance
game_instance_path = "causalitygame/data/game_instances/te/hill_instance.json"


# Data for plotting
data = {}

datasets = {}


# Hooks
def on_game_start():
    print("Game started!")


def on_agent_game_start(agent_name):
    # print(f"Game started for {agent_name}!")
    data[agent_name] = {
        "rounds": [],
        "samples": [],
        "registered_samples": [],
        "result_scores": [],
    }


def on_round_start(agent_name, round, state, actions, samples):
    # print(f"Round {round} started with state: {state}")
    # print(f"Available actions: {actions}")
    # print(f"Samples: {samples}")
    data[agent_name]["rounds"].append(round)


def on_action_chosen(agent_name, state, action, action_object):
    # print(f"Action chosen: {action} with object {action_object}")
    if isinstance(action_object, list):
        # print(f"Action object is a list with length: {len(action_object)}")
        required_samples = sum([t[1] for t in action_object])
        data[agent_name]["samples"].append(
            (
                data[agent_name]["samples"][-1]
                if len(data[agent_name]["samples"]) > 1
                else 0
            )
            + required_samples
        )
    else:
        data[agent_name]["samples"].append(0)


def on_action_evaluated(agent_name, state, action, action_object, result):
    # print(f"Action evaluated: {action} with object {action_object}, result: {result}")
    global datasets
    datasets[agent_name] = state.get("datasets", {})


def on_round_end(agent_name, round, state, action, action_object, samples, result):
    # print(f"Round {round} ended with state: {state}")
    # print(
    #     f"Action: {action}, Object: {action_object}, Samples: {samples}, Result: {result}"
    # )
    current_samples_length = sum([len(dataset) for key, dataset in samples.items()])
    data[agent_name]["registered_samples"].append(current_samples_length)


def on_agent_game_end(agent_name):
    # print(f"Game ended for {agent_name}!")
    pass


def on_game_end():
    print("Game ended!")
    for agent, agent_data in data.items():
        rounds = agent_data["rounds"]
        # samples = agent_data["samples"]
        registered_samples = agent_data["registered_samples"]

        # Plot the data for each agent
        # plt.plot(rounds, samples, marker="o", label=f"{agent} Samples per round")
        plt.plot(
            rounds,
            registered_samples,
            marker="x",
            label=f"{agent} Registered Samples",
        )

    plt.xlabel("Rounds")
    plt.ylabel("Samples")

    global datasets
    for agent in datasets.keys():
        for dataset_name, dataset in datasets[agent].items():
            # Save csv
            dataset.to_csv(f"data/tests/{agent}-{dataset_name}.csv", index=False)

    current_dataset = datasets["Exhaustive Agent"]

    union = pd.concat(current_dataset.values(), ignore_index=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(
        union["X"],
        union["Y"],
        label=f"Exhaustive Agent {dataset_name}",
        alpha=0.5,
    )


# Create a game
game = cg.Game(
    agents=agents,
    game_spec=game_instance_path,
    # hooks={
    #     "on_game_start": on_game_start,
    #     "on_agent_game_start": on_agent_game_start,
    #     "on_round_start": on_round_start,
    #     "on_action_chosen": on_action_chosen,
    #     "on_action_evaluated": on_action_evaluated,
    #     "on_round_end": on_round_end,
    #     "on_agent_game_end": on_agent_game_end,
    #     "on_game_end": on_game_end,
    # },
)
# Run the game
runs = game.run()

# For each run
fig, ax = plt.subplots(figsize=(10, 6))
for name, run in runs.items():
    behavior, result = game.compute_score_trajectories(run["history"])
    # Rolling Mean
    rolling_mean = pd.Series(result).rolling(window=100).mean()
    ax.plot(
        rolling_mean,
        label=f"{name} - Rolling Mean",
        marker="o",
        linestyle="--",
        alpha=0.7,
    )
ax.set_yscale("log")
ax.set_xlim(0, len(rolling_mean) - 1)
ax.set_xlabel("Rounds")
ax.set_ylabel("Scores")
ax.legend()
ax.set_title("Scores Trajectory")


# Print the results
game.plot()
