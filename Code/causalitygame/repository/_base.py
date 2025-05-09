from pathlib import Path
from causalitygame.scm import SCM
from causalitygame.scm.node.base import BaseNumericSCMNode, BaseCategoricSCMNode

import numpy as np
import pandas as pd
import json

def get_scm_overview(folder=None):
    
    # define folder
    if folder is None:
        folder = f"{Path(__file__).parent}/../data/scm"
    
    # recursively read all JSON files in folder
    path = Path(folder)
    if not path.exists():
        raise ValueError(f"The folder {folder} does not exist, so we cannot retrieve SCMs")

    rows = []
    for f in path.rglob("*"):
        if f.is_file() and str(f).endswith(".json"):
            name = f.stem
            with open(f) as h:
                
                # initialize variables
                num_nodes = None
                num_edges = None
                num_numerical_nodes = None
                num_categorical_nodes = None
                out_degrees = None
                in_degrees = None

                try:
                    scm_json = json.load(h)
                    scm = SCM.from_dict(scm_json)

                    dag = scm.dag
                    nodes = dag.nodes
                    edges = dag.edges
                    num_nodes = len(nodes)
                    num_edges = len(edges)
                    
                    # count number of nodes per data type
                    num_numerical_nodes = len([1 for v in scm.nodes.values() if isinstance(v, BaseNumericSCMNode)])
                    num_categorical_nodes = len([1 for v in scm.nodes.values() if isinstance(v, BaseCategoricSCMNode)])
                    
                    # check in-degrees and out-degrees of the variables
                    out_degrees = {v: len([1 for e in edges if e[0] == v]) for v in nodes}
                    in_degrees = {v: len([1 for e in edges if e[1] == v]) for v in nodes}

                except json.decoder.JSONDecodeError:
                    print(f"JSONDecodeError")
                
                except KeyError:
                    print(f"KeyError")
                
                rows.append([
                    name,
                    num_nodes,
                    num_edges,
                    len([v for v, d in in_degrees.items() if d == 0]) if in_degrees is not None else None,
                    len([v for v, d in out_degrees.items() if d == 0]) if out_degrees is not None else None,
                    max(in_degrees.values()) if in_degrees is not None else None,
                    max(out_degrees.values()) if out_degrees is not None else None,
                    num_categorical_nodes,
                    num_numerical_nodes,
                    f
                ])
    return pd.DataFrame(
        rows,
        columns=["name", "num_nodes", "num_edges", "num_roots", "num_leaves", "max_parents", "max_children", "num_categorical_vars", "num_numerical_vars", "filename"]
    ).astype({
        "num_nodes": "Int64",
        "num_edges": "Int64",
        "num_roots": "Int64",
        "num_leaves": "Int64",
        "max_parents": "Int64",
        "max_children": "Int64",
        "num_categorical_vars": "Int64",
        "num_numerical_vars": "Int64"
    })
