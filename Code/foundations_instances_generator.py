# Core
from causalitygame import GameInstance
import numpy as np

# Missions
from causalitygame.missions.AverageTreatmentEffectMission import (
    AverageTreatmentEffectMission,
)
from causalitygame.missions.ConditionalAverageTreatmentEffectMission import (
    ConditionalAverageTreatmentEffectMission,
)

# Metrics
from causalitygame.evaluators.behavior import RoundsBehaviorMetric
from causalitygame.evaluators.deliverable import AbsoluteErrorDeliverableMetric

# SCM
from causalitygame.scm.abstract import SCM

# Utils
import json


# Read scm from file
scm_path = "causalitygame/data/scms/custom/foundations.json"
with open(scm_path, "r") as f:
    scm_dict = json.load(f)

# Create SCM
scm = SCM.from_dict(scm_dict)

print(f"SCM loaded from {scm_path}")

# Create game instances
cate_instance = GameInstance(
    max_rounds=100,
    scm=scm,
    mission=ConditionalAverageTreatmentEffectMission(
        behavior_metric=RoundsBehaviorMetric(),
        deliverable_metric=AbsoluteErrorDeliverableMetric(),
    ),
    random_state=np.random.RandomState(911),
)
cate_instance.save("causalitygame/data/game_instances/cate/foundations_instance.json")
ate_instance = GameInstance(
    max_rounds=100,
    scm=scm,
    mission=AverageTreatmentEffectMission(
        behavior_metric=RoundsBehaviorMetric(),
        deliverable_metric=AbsoluteErrorDeliverableMetric(),
    ),
    random_state=np.random.RandomState(911),
)
ate_instance.save("causalitygame/data/game_instances/ate/foundations_instance.json")
