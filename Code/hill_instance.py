import numpy as np
from causalitygame.scm.base import SCM
from causalitygame.mission.impl.TreatmentEffectMission import TreatmentEffectMission
from causalitygame.game.GameInstance import GameInstance

from causalitygame.evaluators.impl.behavior import ExperimentsBehaviorMetric
from causalitygame.evaluators.impl.deliverable import SquaredErrorDeliverableMetric

# Path
hill_scm_path = "causalitygame/data/scm/papers/hill.json"

# Data
with open(hill_scm_path, "r") as f:
    hill_scm_data = f.read()
    # parse the JSON data
import json

hill_scm_data = json.loads(hill_scm_data)

# Create SCM instance from the data
scm = SCM.from_dict(hill_scm_data)

# Create the Mission
mission = TreatmentEffectMission(
    behavior_metric=ExperimentsBehaviorMetric(),
    deliverable_metric=SquaredErrorDeliverableMetric(),
)

# Create the Game Instance
game_instance = GameInstance(
    max_rounds=100,
    scm=scm,
    mission=mission,
    random_state=np.random.RandomState(911),
)

# Save the Game Instance
game_instance.save("causalitygame/data/game_instances/te/hill_instance.json")
