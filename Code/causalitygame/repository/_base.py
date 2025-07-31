from pathlib import Path
from causalitygame.scm import SCM
from causalitygame.scm.nodes.abstract import BaseNumericSCMNode, BaseCategoricSCMNode

import numpy as np
import pandas as pd
import json


def get_scm_stats(scm):
    dag = scm.dag
    nodes = dag.nodes
    edges = dag.edges
    num_nodes = len(nodes)
    num_edges = len(edges)

    # count number of nodes per data type
    num_numerical_nodes = len(
        [1 for v in scm.nodes.values() if isinstance(v, BaseNumericSCMNode)]
    )
    num_categorical_nodes = len(
        [1 for v in scm.nodes.values() if isinstance(v, BaseCategoricSCMNode)]
    )

    # check in-degrees and out-degrees of the variables
    out_degrees = {v: len([1 for e in edges if e[0] == v]) for v in nodes}
    in_degrees = {v: len([1 for e in edges if e[1] == v]) for v in nodes}

    # check domain sizes
    domains = {c: scm.nodes[c].domain for c in nodes}
    domain_sizes = {c: len(d) for c, d in domains.items()}

    # check domain sizes
    return {
        "num_nodes": num_nodes,
        "num_edges": num_edges,
        "num_roots": (
            len([v for v, d in in_degrees.items() if d == 0])
            if in_degrees is not None
            else None
        ),
        "num_leaves": (
            len([v for v, d in out_degrees.items() if d == 0])
            if out_degrees is not None
            else None
        ),
        "max_parents": max(in_degrees.values()) if in_degrees is not None else None,
        "max_children": max(out_degrees.values()) if out_degrees is not None else None,
        "num_categorical_vars": num_categorical_nodes,
        "num_numerical_vars": num_numerical_nodes,
        "min/avg/max domain size": f"{min(domain_sizes.values())}/{int(np.round(np.mean(list(domain_sizes.values()))))}/{max(domain_sizes.values())}",
    }


def get_scm_overview(folder=None):

    # define folder
    if folder is None:
        folder = f"{Path(__file__).parent}/../data/scm"

    # recursively read all JSON files in folder
    path = Path(folder)
    if not path.exists():
        raise ValueError(
            f"The folder {folder} does not exist, so we cannot retrieve SCMs"
        )

    rows = []
    for f in path.rglob("*"):
        if f.is_file() and str(f).endswith(".json"):
            name = f.stem
            with open(f) as h:

                try:
                    print(f)
                    scm_json = json.load(h)
                    scm = SCM.from_dict(scm_json)
                    descriptor = {"name": name, "filename": f}
                    descriptor.update(get_scm_stats(scm))
                    rows.append(descriptor)

                except json.decoder.JSONDecodeError:
                    print(f"JSONDecodeError")

                except KeyError:
                    print(f"KeyError")
                    raise
    return pd.DataFrame(rows).astype(
        {
            "num_nodes": "Int64",
            "num_edges": "Int64",
            "num_roots": "Int64",
            "num_leaves": "Int64",
            "max_parents": "Int64",
            "max_children": "Int64",
            "num_categorical_vars": "Int64",
            "num_numerical_vars": "Int64",
        }
    )
