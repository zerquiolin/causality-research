# Abstract
from .abstract import BaseMission

# Science
import numpy as np

# Utils
from causalitygame.lib.utils.imports import find_importable_classes

# Types
from causalitygame.evaluators.abstract import BehaviorMetric, DeliverableMetric
from causalitygame.scm.abstract import SCM

# Constants
from causalitygame.lib.constants.routes import METRICS_FOLDER_PATH

# Identify specific metric classes
metric_classes = find_importable_classes(METRICS_FOLDER_PATH, base_class=BehaviorMetric)
metric_classes.update(
    find_importable_classes(METRICS_FOLDER_PATH, base_class=DeliverableMetric)
)


class TreatmentEffectMission(BaseMission):
    """
    A mission that focuses on inferring the structure of a Directed Acyclic Graph (DAG).
    This mission is designed to evaluate the performance of agents in inferring the
    underlying causal structure from observational data.
    """

    name = "Treatment Effect Mission"
    description = "This mission evaluates the ability to infer the treatment effects in a causal graph given a intervention Z, covariates X, and outcome Y."

    def evaluate(self, scm: SCM, history):
        # Define a random state for reproducibility
        rs = np.random.RandomState(911)
        # Get the agents' function
        empirical_te_function = history.iloc[-1]["current_result"]
        # Evaluate the behavior and deliverable metrics
        te_measurable_nodes = [
            node
            for node in scm.nodes.values()
            if node.name in scm.observable_vars
            and node.parents
            and all(parent in scm.observable_vars for parent in node.parents)
        ]

        assert len(te_measurable_nodes) > 0, "No measurable nodes found for CATE"

        node = te_measurable_nodes[-1]

        Y = node.name
        Z = {
            node.parents[0]: current_node.domain
            for current_node in scm.nodes.values()
            if current_node.name == node.parents[0]
        }
        X = {
            current_node.name: (
                rs.choice(current_node.domain)
                if type(current_node.domain[0]) == str
                else rs.uniform(current_node.domain[0], current_node.domain[1], size=1)
            )
            for current_node in scm.nodes.values()
            if current_node.name != Y and current_node.name != node.parents[0]
        }

        Z_name, Z_values = list(Z.items())[0]

        Y0, Y1 = [
            scm.generate_samples(
                interventions={
                    Z_name: value,
                    **X,
                },
                num_samples=1,
                random_state=rs,
            )[Y][0]
            for value in Z_values
        ]

        te = Y1 - Y0
        # Comput the empirical treatment effect
        empirical_te = empirical_te_function(Y, Z, X)
        behavior_score = self.behavior_metric.evaluate(history)
        deliverable_score = self.deliverable_metric.evaluate(scm, (te, empirical_te))

        return behavior_score, deliverable_score

    def to_dict(self):
        return {
            "class": TreatmentEffectMission.__name__,
            "behavior_metric": self.behavior_metric.__class__.__name__,
            "deliverable_metric": self.deliverable_metric.__class__.__name__,
        }

    @classmethod
    def from_dict(cls, dict):
        global metric_classes
        # Check if the behavior metric class is known
        behavior_cls = metric_classes.get(dict["behavior_metric"])
        if behavior_cls is None:
            raise ValueError(f"Unknown mission class: {dict['behavior_metric']}")
        # Instantiate the mission from the data
        behavior_metric = behavior_cls()
        # Check if the deliverable metric class is known
        deliverable_cls = metric_classes.get(dict["deliverable_metric"])
        if deliverable_cls is None:
            raise ValueError(f"Unknown mission class: {dict['deliverable_metric']}")
        # Instantiate the mission from the data
        deliverable_metric = deliverable_cls()

        return cls(
            behavior_metric=behavior_metric,
            deliverable_metric=deliverable_metric,
        )
