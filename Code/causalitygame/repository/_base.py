from pathlib import Path
from causalitygame.scm import SCM

import numpy as np
import pandas as pd
import json

def get_scm_overview(folder=None):
    
    # define folder
    if folder is None:
        folder = "data/scm"
    
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
                types_per_counts = None
                out_degrees = None
                in_degrees = None

                try:
                    scm_json = json.load(h)

                    dag = scm_json["dag"]
                    nodes = dag["nodes"]
                    edges = dag["edges"]
                    defs = scm_json["scm"]["vars"]
                    num_nodes = len(nodes)
                    num_edges = len(edges)
                    
                    # count number of nodes per data type
                    types_per_counts = {}
                    for dtype, cnt in zip(*np.unique([v["var_type"] for v in defs.values()], return_counts=True)):
                        types_per_counts[str(dtype)] = int(cnt)
                    
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
                    types_per_counts.get("categorical", 0) if types_per_counts is not None else None,
                    types_per_counts.get("numerical", 0) if types_per_counts is not None else None,
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
