from causalitygame.evaluators.abstract import (
    BehaviorMetric,
    DeliverableMetric,
)
from causalitygame.missions.abstract import BaseMission

print(f"Subclasses of BaseMission: {BaseMission.__subclasses__()}")

# print(f"Subclasses of BehaviorMetric: {BehaviorMetric.__subclasses__()}")
# print(f"Subclasses of DeliverableMetric: {DeliverableMetric.__subclasses__()}")

# names = [
#     cls.__name__
#     for cls in BehaviorMetric.__subclasses__() + DeliverableMetric.__subclasses__()
# ]

# print(f"All subclasses of BehaviorMetric and DeliverableMetric: {names}")
