from ..Metrics import BehaviorMetrics, DeliverableMetrics, Metrics


def gen_dag_discovery_metrics(scm):
    deliverable_metrics = DeliverableMetrics(scm)
    behavior_metrics = BehaviorMetrics(
        e=1.0, t=0.5, r=0.2, alpha=0.5, beta=0.3, gamma=0.2
    )

    return Metrics(
        deliverable_metric=deliverable_metrics,
        behavior_metric=behavior_metrics,
        goal_type="dag_discovery",
    )
