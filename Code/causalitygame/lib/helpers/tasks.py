# Math
import pandas as pd

# Scripts
from causalitygame.lib.scripts.pc import learn as learn_dag
from causalitygame.lib.scripts.empiricalCATE import compute_empirical_cate_fuzzy
from causalitygame.lib.scripts.xgboostTE import te_estimation

# Types
from typing import Callable, List


class TaskFactory:
    @staticmethod
    def create_dag_task(data: pd.DataFrame, is_numeric: bool) -> Callable:
        # Clousure
        dataset = data.copy()
        return learn_dag(dataset, is_numeric)

    @staticmethod
    def create_cate_task(data: pd.DataFrame) -> Callable:
        # Clousure
        dataset = data.copy()

        def compute_cate(Y: str, T: str, Z: list):
            return compute_empirical_cate_fuzzy(
                query={"Y": Y, "T": T, "Z": Z},
                data=(dataset,),
                distance_threshold=1e2,
            )

        return compute_cate

    @staticmethod
    def create_te_task(data: pd.DataFrame) -> Callable:
        # Clousure
        dataset = data.copy()

        def compute_te(Y: str, Z: str, X: List[str], samples: pd.DataFrame):
            return te_estimation(Y=Y, Z=Z, X=X, samples=samples, data=dataset)

        return compute_te
