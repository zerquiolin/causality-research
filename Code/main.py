# Agents
from src.agents.impl.RandomAgent import RandomAgent
from src.agents.impl.ExhaustiveAgent import ExhaustiveAgent

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
)

# Game
from src.game.Game import Game

import numpy as np

# Set base seed for reproducibility
base_seed = 42
agents = []

for i in range(1, 6):
    rs = np.random.RandomState(base_seed + i)
    stop_prob = rs.beta(a=0.5, b=10)  # typically small values
    exp_upper = rs.poisson(lam=10)
    exp_upper = max(exp_upper, 2)
    experiments_range = (1, exp_upper)
    samples_lower = rs.randint(500, 800)
    samples_upper = rs.randint(samples_lower, 1000)
    samples_range = (samples_lower, samples_upper)

    agent = (
        f"random {i}",
        RandomAgent(
            stop_probability=stop_prob,
            experiments_range=experiments_range,
            samples_range=samples_range,
            seed=base_seed + i,
        ),
    )
    agents.append(agent)

agents.append(
    ("exhaustive", ExhaustiveAgent()),
)

behavior_metrics = [
    ExperimentsBehaviorMetric(),
    TreatmentsBehaviorMetric(),
    RoundsBehaviorMetric(),
]

deliverable_metrics = [
    SHDDeliverableMetric(),
    F1DeliverableMetric(),
    EdgeAccuracyDeliverableMetric(),
]

game = Game(
    agents=agents,
    game_spec="/Users/sergioamortegui/Desktop/Business/Research/Causality/Code/instances/game_instance.pkl",
    behavior_metrics=behavior_metrics,
    deliverable_metrics=deliverable_metrics,
    max_rounds=100,  # game spec
    plambda=0.8,
    seed=911,
)

results = game.run()
game.plot()
