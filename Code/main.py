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

agents = [
    (
        "random",
        RandomAgent(
            stop_probability=0.01, experiments_range=(1, 20), samples_range=(500, 1000)
        ),
    ),
    ("exhaustive", ExhaustiveAgent()),
]

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
