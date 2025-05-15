import os
import json
import numpy as np
import causalitygame as cg

SOURCE_DIR = "./causalitygame/data/scm/physics"
TARGET_DIR = "./causalitygame/data/game_instances/dag_inference"

# Walk through all files and folders in SOURCE_DIR
for root, dirs, files in os.walk(SOURCE_DIR):
    for file in files:
        # Compute the name of the file without the extension
        file_name = os.path.splitext(file)[0]
        print(f"File name without extension: {file_name}")
        # Read the JSON file
        source_path = os.path.join(root, file)
        print(f"Source path: {source_path}")
        with open(source_path, "r") as json_file:
            scm_json = json.load(json_file)
        newton_scm = cg.SCM.from_dict(scm_json)

        # Create the corresponding target path
        target_path = os.path.join(TARGET_DIR, file_name + "_instance.json")

        # Mission
        mission = cg.DAGInferenceMission(
            behavior_metric=cg.ExperimentsBehaviorMetric(),
            deliverable_metric=cg.EdgeAccuracyDeliverableMetric(),
        )

        # Create a Game Instance
        game_instance = cg.GameInstance(
            max_rounds=100,
            scm=newton_scm,
            mission=mission,
            random_state=np.random.RandomState(911),
        )

        game_instance.save(target_path)
