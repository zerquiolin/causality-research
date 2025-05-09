import json
import numpy as np
from causalitygame.agents.impl.RandomAgent import RandomAgent
from causalitygame.agents.impl.ExhaustiveAgent import ExhaustiveAgent
from causalitygame.translators.impl.bif_translator import BifTranslator
from causalitygame.generators.bayesian_network_scm_generator import (
    BayesianNetworkBasedSCMGenerator,
)
from causalitygame.game.GameInstance import GameInstance

import os
import shutil

# Define your source and target folders
SOURCE_DIR = "./bif_files"
TARGET_DIR = "./causalitygame/data/scm/literature_cases"

# Walk through all files and folders in SOURCE_DIR
for root, dirs, files in os.walk(SOURCE_DIR):
    for file in files:
        # Full path to the source file
        source_path = os.path.join(root, file)
        print(f"Source path: {source_path}")

        # Compute the relative path from the source root
        relative_path = os.path.relpath(source_path, SOURCE_DIR)
        print(f"Relative path: {relative_path}")

        # Compute the folder structure
        folder_structure = os.path.dirname(relative_path)
        print(f"Folder structure: {folder_structure}")

        # Compute the name of the file without the extension
        file_name = os.path.splitext(file)[0]
        print(f"File name without extension: {file_name}")

        # Compute the new file name
        new_file_name = f"{file_name}.json"
        print(f"New file name: {new_file_name}")

        # Create the corresponding target path
        target_path = os.path.join(TARGET_DIR, folder_structure, new_file_name)
        print(f"Target path: {target_path}")

        # Compute the scm object
        translator = BifTranslator()
        by_gen = BayesianNetworkBasedSCMGenerator(translator, source_path)
        scm = by_gen.generate()
        scm_dict = scm.to_dict()
        del scm_dict["random_state"]  # remove the random state from the dictionary

        # Save the JSON file to the target path
        os.makedirs(os.path.dirname(target_path), exist_ok=True)
        with open(target_path, "w") as json_file:
            json.dump(scm_dict, json_file, indent=2)

# agents = []

# base_seed = 42
# for i in range(1, 2):
#     rs = np.random.RandomState(base_seed + i)
#     stop_prob = rs.beta(a=0.5, b=10)  # typically small values
#     exp_upper = rs.poisson(lam=10)
#     exp_upper = max(exp_upper, 2)
#     experiments_range = (1, exp_upper)
#     samples_lower = rs.randint(500, 800)
#     samples_upper = rs.randint(samples_lower, 1000)
#     samples_range = (samples_lower, samples_upper)

#     agent = (
#         f"random {i}",
#         RandomAgent(
#             stop_probability=stop_prob,
#             experiments_range=experiments_range,
#             samples_range=samples_range,
#             seed=base_seed + i,
#         ),
#     )
#     agents.append(agent)

# agents.append(
#     ("exhaustive", ExhaustiveAgent()),
# )

# behavior_metrics = [
#     ExperimentsBehaviorMetric(),
#     TreatmentsBehaviorMetric(),
#     RoundsBehaviorMetric(),
# ]

# deliverable_metrics = [
#     SHDDeliverableMetric(),
#     F1DeliverableMetric(),
#     EdgeAccuracyDeliverableMetric(),
# ]


# # parent_values = {}
# # for node in result:
# #     print("=" * 20)
# #     print(f"Node Name: {node.name}")
# #     print(f"Parents: {node.parents}")
# #     print(f"Parent Values: {parent_values}")
# #     print(f"Domain: {node.values}")
# #     print(
# #         f"Probability Distribution: {json.dumps(node.probability_distribution, indent=2)}"
# #     )
# #     distribution = node.get_distribution(parent_values)
# #     print(f"Distribution: {distribution}")
# #     value = np.random.choice(
# #         node.values, p=distribution
# #     )  # Sample a value based on the distribution
# #     parent_values[node.name] = value
# #     print(f"Sampled Value: {value}")
# #     print("=" * 20)
