import numpy as np
from causalitygame.evaluators.impl.BehaviorMetrics import ExperimentsBehaviorMetric
from causalitygame.evaluators.impl.DeliverableMetrics import SHDDeliverableMetric
from causalitygame.game.GameInstance import GameInstance
from causalitygame.mission.impl.DAGInferenceMission import DAGInferenceMission
from causalitygame.repository._base import get_scm_overview
from causalitygame.scm import SCM
import logging
import json
import pytest

from causalitygame.scm.impl.basic_binary_scm import gen_binary_scm


# define stream handler
ch = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
ch.setLevel(logging.DEBUG)

# configure logger for tester
logger = logging.getLogger("tester")
logger.handlers.clear()
logger.addHandler(ch)
logger.setLevel(logging.DEBUG)


@pytest.mark.parametrize(
    "max_rounds, dag, scm, mission, seed",
    [
        (
            100,
            *gen_binary_scm(42, num_samples_for_cdf_generation=10),
            DAGInferenceMission(ExperimentsBehaviorMetric(), SHDDeliverableMetric()),
            42,
        )
    ],
)
def test_game_instance_serialization(max_rounds, dag, scm, mission, seed):
    rs = np.random.RandomState(seed)

    logger.debug(f"Creating GameInstance with max_rounds={max_rounds}, seed={seed}")
    game_instance = GameInstance(
        max_rounds=max_rounds,
        scm=scm,
        mission=mission,
        random_state=rs,
    )

    logger.debug("Generating dict from GameInstance")
    game_instance_dict = game_instance.to_dict()

    logger.debug("Serializing GameInstance to JSON")
    serialized = False
    try:
        json.dumps(game_instance_dict)
        serialized = True
    except TypeError as e:
        logger.error(f"Serialization failed: {e}")
        serialized = False
    assert serialized, "GameInstance serialization to JSON failed!"


@pytest.mark.parametrize(
    "max_rounds, dag, scm, mission, seed",
    [
        (
            100,
            *gen_binary_scm(42),
            DAGInferenceMission(ExperimentsBehaviorMetric(), SHDDeliverableMetric()),
            42,
        )
    ],
)
def test_game_instance_deserialization(max_rounds, dag, scm, mission, seed):
    rs = np.random.RandomState(seed)

    logger.debug(f"Creating GameInstance with max_rounds={max_rounds}, seed={seed}")
    game_instance = GameInstance(
        max_rounds=max_rounds,
        scm=scm,
        mission=mission,
        random_state=rs,
    )

    logger.debug("Generating dict from GameInstance")
    game_instance_dict = game_instance.to_dict()

    logger.debug("Marshalling and unmarshalling dictionary to json")
    game_instance_json = GameInstance.from_dict(
        json.loads(json.dumps(game_instance_dict))
    )

    logger.debug("Comparing original and deserialized GameInstance")
    deep_dict_equal(
        game_instance_dict,
        game_instance_json.to_dict(),
    )


def deep_dict_equal(d1, d2, path=""):
    if isinstance(d1, dict) and isinstance(d2, dict):
        if d1.keys() != d2.keys():
            print(f"Key mismatch at {path}: {d1.keys()} vs {d2.keys()}")
            return False
        for key in d1:
            new_path = f"{path}.{key}" if path else key
            if not deep_dict_equal(d1[key], d2[key], new_path):
                return False
        return True

    elif isinstance(d1, list) and isinstance(d2, list):
        if len(d1) != len(d2):
            print(f"List length mismatch at {path}: {len(d1)} vs {len(d2)}")
            return False
        for index, (item1, item2) in enumerate(zip(d1, d2)):
            new_path = f"{path}[{index}]"
            if not deep_dict_equal(item1, item2, new_path):
                return False
        return True

    else:
        if d1 != d2:
            print(f"Value mismatch at {path}: {d1} vs {d2}")
            return False
        return True
