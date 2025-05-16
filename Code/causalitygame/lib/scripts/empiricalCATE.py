from collections import Counter
import numpy as np


def compute_empirical_cate_fuzzy(query: dict, data: dict, distance_threshold=0.1):
    """
    Compute empirical CATE from approximate matches in continuous space.

    Args:
        query (dict): { "Y": str, "T": {T_name: [t1, t0]}, "Z": {X_name: value, ...} }
        data (dict): agent-collected data in the form: {Y: {treatment_val: [records]}, "empty": [...]}
        distance_threshold (float): maximum distance allowed for a match (L2 norm)

    Returns:
        float: Estimated CATE (or np.nan if insufficient data)
    """
    Y = query["Y"]
    T_var, (t1, t0) = list(query["T"].items())[0]
    context = query["Z"]

    def get_close_matches(records, treatment_value):
        close = []
        for record in records:
            # Must have outcome and treatment
            if Y not in record or T_var not in record:
                continue
            if not isinstance(record[T_var], (int, float)):
                continue  # only support continuous for this case

            distances = []

            # Distance in T
            t_dist = abs(record[T_var] - treatment_value)
            distances.append(t_dist)

            # Distance in Z
            x_dists = []
            for key, val in context.items():
                distances.append(abs(record.get(key, 0) - val))
            if all(d <= distance_threshold for d in distances):
                close.append(record[Y])
        return close

    Y_records = []
    # Iterate over all treatments and observational data
    for row in data.get("empty", []):
        Y_records.append(row)
    for var, treatments in data.items():
        if var == "empty":
            continue
        for treatment_val, records in treatments.items():
            Y_records.extend(records)

    Y_t1 = get_close_matches(Y_records, t1)
    Y_t0 = get_close_matches(Y_records, t0)

    if not Y_t1 or not Y_t0:
        return 0

    # Handle categorical or continuous Y
    if isinstance(Y_t1[0], str):
        classes = set(Y_t1) | set(Y_t0)
        p1 = Counter(Y_t1)
        p0 = Counter(Y_t0)
        n1, n0 = len(Y_t1), len(Y_t0)
        return np.mean([abs(p1.get(c, 0) / n1 - p0.get(c, 0) / n0) for c in classes])
    else:
        return np.mean(Y_t1) - np.mean(Y_t0)
