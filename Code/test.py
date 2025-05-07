import json
import numpy as np
from causalitygame.agents.impl.RandomAgent import RandomAgent
from causalitygame.agents.impl.ExhaustiveAgent import ExhaustiveAgent
from causalitygame.translators.impl.bif_translator import BifTranslator
from causalitygame.generators.bayesian_network_scm_generator import (
    BayesianNetworkBasedSCMGenerator,
)
from causalitygame.game.GameInstance import GameInstance


path = "survey.bif"
translator = BifTranslator()
by_gen = BayesianNetworkBasedSCMGenerator(translator, path)
scm = by_gen.generate()
gameInstance = GameInstance(scm=scm, random_state=np.random.RandomState(42))
gameInstance.save(filename="./instances/bn_game_instance.json")

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
