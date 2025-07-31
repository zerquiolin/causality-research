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


class ConditionalAverageTreatmentEffectMission(BaseMission):
    """
    A mission that focuses on inferring the structure of a Directed Acyclic Graph (DAG).
    This mission is designed to evaluate the performance of agents in inferring the
    underlying causal structure from observational data.
    """

    name = "Conditional Average Treatment Effect Mission"
    description = "This mission evaluates the ability to infer the treatment effects in a causal graph given a intervention Z, covariates X, and outcome Y."

    def mount(self, scm: SCM):
        """
        Mount the mission to the given SCM.

        Args:
            scm (SCM): Structural Causal Model to mount the mission to.
        """
        # Mount Behavior Metric
        self.behavior_metric.mount(scm)
        # Mount Deliverable Metric
        self.deliverable_metric.mount(scm)
        # Define a random state for reproducibility
        rs = np.random.RandomState(911)
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
        # Generate values for the covariant nodes
        cov_sample = scm.generate_samples(
            num_samples=100,
            cancel_noise=True,
            random_state=rs,
        )
        # Drop columns that are either the treatment node or the outcome node
        cov_sample = cov_sample.drop(columns=[te_node, treatment_node])
        # Convert the result to a Dict
        cov_sample = cov_sample.to_dict(orient="records")[rs.choice(range(100))]
        # Generate samples for each treatment value
        treatment_samples = [
            scm.generate_samples(
                # interventions={treatment_node: value, "X": 1.5},
                interventions={treatment_node: value, **cov_sample},
                num_samples=1000,
                cancel_noise=True,
                random_state=rs,
            )
            for value in scm.nodes[treatment_node].domain
        ]

        # Extract the outcome variable Y
        T0, T1 = treatment_samples
        Y0, Y1 = T0[te_node].values, T1[te_node].values
        true_cate = Y1 - Y0
        # Save the treatment node
        self.treatment_node = treatment_node
        # Save the treatment effect node
        self.te_node = te_node
        # Save the treatment samples
        self.treatment_samples = treatment_samples
        # Save the true CATE
        self.true_cate = true_cate[0]
        # Update the is_mounted flag
        self.is_mounted = True

    def evaluate(self, scm: SCM, history):
        # Check if the mission is mounted
        if not self.is_mounted:
            raise ValueError("Mission is not mounted")

        # Get the agents' function
        empirical_te_function = history.iloc[-1]["current_result"]
        # Compute the empirical treatment effect
        estimated_cate = empirical_te_function(
            Y=self.te_node,
            Z=self.treatment_node,
            X=[
                var
                for var in scm.vars
                if var != self.te_node
                and var != self.treatment_node
                and (
                    scm.nodes[var].accessibility == ACCESSIBILITY_CONTROLLABLE
                    or scm.nodes[var].accessibility == ACCESSIBILITY_OBSERVABLE
                )
            ],
            covariate_values=(
                self.treatment_samples[0].drop(columns=[self.te_node]),
                self.treatment_samples[1].drop(columns=[self.te_node]),
            ),
        )
        behavior_score = self.behavior_metric.evaluate(history)
        deliverable_score = self.deliverable_metric.evaluate(
            scm, (self.true_cate, estimated_cate)
        )

        return behavior_score, deliverable_score

    def to_dict(self):
        return {
            "class": ConditionalAverageTreatmentEffectMission.__name__,
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
