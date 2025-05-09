import json
import numpy as np

# Environment
from causalitygame.scm.impl.basic_binary_scm import gen_binary_scm
from causalitygame.game.GameInstance import GameInstance

# Metrics
from causalitygame.evaluators.impl.BehaviorMetrics import ExperimentsBehaviorMetric
from causalitygame.evaluators.impl.DeliverableMetrics import SHDDeliverableMetric

# Misison
from causalitygame.mission.impl.DAGInferenceMission import DAGInferenceMission

# Tests
from tests.test_scm import assert_dicts_equal


# Random State
seed = 42
# Create a random state for reproducibility
random_state = np.random.RandomState(seed)
# Generate a binary SCM with the specified seed
dag, scm = gen_binary_scm(random_state=seed)
# Generate a mission
behavior_metric = ExperimentsBehaviorMetric()
deliverable_metric = SHDDeliverableMetric()
mission = DAGInferenceMission(
    behavior_metric=behavior_metric, deliverable_metric=deliverable_metric
)
# Create a GameInstance with the generated SCM
game_instance = GameInstance(
    max_rounds=100, scm=scm, mission=mission, random_state=random_state
)
# Save the game instance to a file
game_instance.save(filename="./instances/game_instance.json")
