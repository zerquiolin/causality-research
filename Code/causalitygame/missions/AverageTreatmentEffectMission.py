# Abstract
from causalitygame.lib.constants.nodes import (
    ACCESSIBILITY_CONTROLLABLE,
    ACCESSIBILITY_OBSERVABLE,
)
from .abstract import BaseMission

# Science
import numpy as np
import pandas as pd

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


class AverageTreatmentEffectMission(BaseMission):
    """
    A mission that focuses on inferring the structure of a Directed Acyclic Graph (DAG).
    This mission is designed to evaluate the performance of agents in inferring the
    underlying causal structure from observational data.
    """

    name = "Average Treatment Effect Mission"
    description = "This mission evaluates the ability to infer the treatment effects in a causal graph given a intervention Z, covariates X, and outcome Y."

    def evaluate(self, scm: SCM, history):
        # Check if the mission is mounted
        if not self.is_mounted:
            raise ValueError("Mission is not mounted")

        # Define a random state for reproducibility
        rs = np.random.RandomState(911)
        # Get the agents' function
        empirical_ate_function = history.iloc[-1]["current_result"]
        # Select the predictive node
        possible_outcomes = [
            var for var in scm.leaf_vars if type(scm.nodes[var].domain[0]) is not str
        ]
        assert len(possible_outcomes) > 0, "No measurable nodes found for TE"
        te_node = rs.choice(possible_outcomes)
        # Select the treatment node
        possible_treatments = [
            node.name
            for node in scm.nodes.values()
            if node.accessibility == ACCESSIBILITY_CONTROLLABLE
        ]
        assert len(possible_treatments) > 0, "No controllable nodes found for TE"
        treatment_node = rs.choice(possible_treatments)
        # Generate samples for each treatment value
        Z_0, Z_1 = [
            scm.generate_samples(
                interventions={treatment_node: value},
                num_samples=10,
                cancel_noise=True,
                random_state=rs,
            )
            for value in scm.nodes[treatment_node].domain
        ]
        # Extract the outcome variable Y
        ate = Z_1[te_node] - Z_0[te_node]
        # Compute the empirical treatment effect
        empirical_ate = empirical_ate_function(
            Y=str(te_node),
            Z=str(treatment_node),
            samples=pd.concat([Z_0[treatment_node], Z_1[treatment_node]]),
        )
        behavior_score = self.behavior_metric.evaluate(history)
        deliverable_score = self.deliverable_metric.evaluate(
            scm, (ate.mean(), empirical_ate)
        )

        return behavior_score, deliverable_score

    def to_dict(self):
        return {
            "class": AverageTreatmentEffectMission.__name__,
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
