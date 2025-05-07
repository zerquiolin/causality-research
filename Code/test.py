import json
import numpy as np
from causalitygame.translators.impl.bif_translator import BifTranslator
from causalitygame.generators.bayesian_network_scm_generator import (
    BayesianNetworkBasedSCMGenerator,
)


path = "survey.bif"
translator = BifTranslator()
by_gen = BayesianNetworkBasedSCMGenerator(translator, path)
result = by_gen.generate_nodes()

parent_values = {}
for node in result:
    print("=" * 20)
    print(f"Node Name: {node.name}")
    print(f"Parents: {node.parents}")
    print(f"Parent Values: {parent_values}")
    print(f"Domain: {node.values}")
    print(
        f"Probability Distribution: {json.dumps(node.probability_distribution, indent=2)}"
    )
    distribution = node.get_distribution(parent_values)
    print(f"Distribution: {distribution}")
    value = np.random.choice(
        node.values, p=distribution
    )  # Sample a value based on the distribution
    parent_values[node.name] = value
    print(f"Sampled Value: {value}")
    print("=" * 20)
