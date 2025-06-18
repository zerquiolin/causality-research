from collections import Counter

import numpy as np
from causalitygame.evaluators.abstract import BaseMetric
from causalitygame.scm.abstract import SCM
from .abstract import BaseMission
from causalitygame.evaluators.behavior import (
    ExperimentsBehaviorMetric,
    TreatmentsBehaviorMetric,
    RoundsBehaviorMetric,
)

from causalitygame.evaluators.deliverable import (
    SHDDeliverableMetric,
    F1DeliverableMetric,
    EdgeAccuracyDeliverableMetric,
    AbsoluteErrorDeliverableMetric,
    SquaredErrorDeliverableMetric,
)


class CATEMission(BaseMission):
    """
    A mission that focuses on inferring the structure of a Directed Acyclic Graph (DAG).
    This mission is designed to evaluate the performance of agents in inferring the
    conditional average treatment effect (CATE) from observational data.
    """

    name = "Conditional Average Treatment Effect (CATE) Mission"
    description = "This mission evaluates the ability to infer the conditional average treatment effect from observational data. The user is expected to provide a function f(Y, T, X) that computes the CATE given the outcome Y, treatment T, and covariates X."

    def __init__(self, behavior_metric, deliverable_metric):
        super().__init__(behavior_metric, deliverable_metric)

    def evaluate(self, scm: SCM, history):
        # Define a random state for reproducibility
        rs = np.random.RandomState(911)
        # Get the agents' cate function
        empirical_cate_function = history.iloc[-1]["current_result"]
        # Evaluate the behavior and deliverable metrics
        cate_measurable_nodes = [
            node
            for node in scm.nodes.values()
            if node.name in scm.observable_vars
            and node.parents
            and all(parent in scm.observable_vars for parent in node.parents)
        ]

        assert len(cate_measurable_nodes) > 0, "No measurable nodes found for CATE"

        cate_experiments = []

        for node in cate_measurable_nodes[:3]:
            Y = node.name
            T = {
                node.parents[0]: current_node.domain[:2]
                for current_node in scm.nodes.values()
                if current_node.name == node.parents[0]
            }
            X = {
                current_node.name: (
                    rs.choice(current_node.domain)
                    if type(current_node.domain[0]) is str
                    else rs.uniform(current_node.domain[0], current_node.domain[1])
                )
                for current_node in scm.nodes.values()
                if current_node.name != Y and current_node.name != node.parents[0]
            }
            cate_experiments.append((Y, T, X))

        CATEs = []
        # Measure the CATE
        for experiment in cate_experiments:
            Y, T, X = experiment
            Ys = []
            T, dos = list(T.items())[0]
            for do in dos:
                interventions = {T: do, **X}
                samples = scm.generate_samples(
                    interventions={T: do, **X},
                    num_samples=100,
                )
                doi = []
                for sample in samples:
                    doi.append(sample[Y])
                Ys.append(doi)

            # Compute the CATE
            if type(Ys[0][0]) is str:
                # Handle categorical variables
                classes = set(Ys[0]) | set(Ys[1])
                do1 = {k: v / len(Ys[0]) for k, v in Counter(Ys[0]).items()}
                do2 = {k: v / len(Ys[1]) for k, v in Counter(Ys[1]).items()}
                cate = np.mean([abs(do1.get(k, 0) - do2.get(k, 0)) for k in classes])
                CATEs.append(cate)

            else:
                # Handle continuous variables
                cate = np.mean(Ys[0]) - np.mean(Ys[1])
                CATEs.append(cate)

        # Compute the average CATE
        CATE = np.mean(CATEs)
        # Comput the empirical average CATE
        empirical_CATE = np.mean(
            [empirical_cate_function(Y, T, X) for Y, T, X in cate_experiments]
        )
        behavior_score = self.behavior_metric.evaluate(history)
        deliverable_score = self.deliverable_metric.evaluate(
            scm, (CATE, empirical_CATE)
        )

        return behavior_score, np.log10(deliverable_score) / 100

    def to_dict(self):
        return {
            "class": CATEMission.__name__,
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
