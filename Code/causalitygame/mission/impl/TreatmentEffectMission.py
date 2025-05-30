import numpy as np
from causalitygame.evaluators.base import BaseMetric
from causalitygame.scm.base import SCM
from ..base import BaseMission
from causalitygame.evaluators.impl.behavior import (
    ExperimentsBehaviorMetric,
    TreatmentsBehaviorMetric,
    RoundsBehaviorMetric,
)

from causalitygame.evaluators.impl.deliverable import (
    SHDDeliverableMetric,
    F1DeliverableMetric,
    EdgeAccuracyDeliverableMetric,
    AbsoluteErrorDeliverableMetric,
    SquaredErrorDeliverableMetric,
    MeanSquaredErrorDeliverableMetric,
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
            )[
                Y
            ][0]
            for value in Z_values
        ]

        te = Y1 - Y0
        print(f"Estimated Treatment Effect (TE): {te}")
        # Comput the empirical treatment effect
        empirical_te = empirical_te_function(Y, Z, X)
        print(f"Empirical Treatment Effect (TE): {empirical_te}")
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
        mapping = {
            c.__name__: c
            for c in [
                ExperimentsBehaviorMetric,
                TreatmentsBehaviorMetric,
                RoundsBehaviorMetric,
                SHDDeliverableMetric,
                F1DeliverableMetric,
                EdgeAccuracyDeliverableMetric,
                AbsoluteErrorDeliverableMetric,
                SquaredErrorDeliverableMetric,
                MeanSquaredErrorDeliverableMetric,
            ]
        }
        behavior_metric = mapping[dict["behavior_metric"]]()
        deliverable_metric = mapping[dict["deliverable_metric"]]()

        # Ensure the behavior and deliverable metrics are initialized correctly
        assert isinstance(behavior_metric, BaseMetric), "Invalid behavior metric type"

        assert isinstance(
            deliverable_metric,
            BaseMetric,
        ), "Invalid deliverable metric type"

        return cls(
            behavior_metric=behavior_metric,
            deliverable_metric=deliverable_metric,
        )
