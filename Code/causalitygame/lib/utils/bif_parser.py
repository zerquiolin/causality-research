from pgmpy.readwrite import BIFReader
import json


def bif_to_dict(file_path: str) -> dict:
    """
    Parses a .bif file and returns its contents in a structured format suitable for JSON export.

    Args:
        file_path (str): Path to the .bif file.

    Returns:
        dict: Structured dictionary with variables, values, parents, CPDs, and edges.
    """
    reader = BIFReader(file_path)
    model = reader.get_model()

    variables = {}

    for node in model.nodes():
        cpd = model.get_cpds(node)
        states = reader.get_states()[node]
        parents = list(cpd.get_evidence()) if cpd.get_evidence() else []
        cpd_values = cpd.get_values()

        # Handle CPDs based on whether there are parents
        if not parents:
            prob = cpd_values.tolist()
        else:
            prob = {}
            evidence_card = cpd.cardinality[1:]  # Cardinalities of the parents
            parent_states = [reader.get_states()[parent] for parent in parents]

            # Generate all possible combinations of parent states
            from itertools import product

            for index, parent_values in enumerate(product(*parent_states)):
                flat_index = index
                key = ",".join(parent_values)  # or str(parent_values) if you prefer
                prob[key] = cpd_values[:, flat_index].tolist()

        variables[node] = {
            "values": states,
            "parents": parents,
            "probability_distribution": prob,
        }

    return {"variables": variables, "edges": [list(edge) for edge in model.edges()]}


def bif_to_json(file_path: str, output_path: str) -> None:
    """
    Converts a .bif file to JSON format and saves it.

    Args:
        file_path (str): Path to the .bif file.
        output_path (str): Path to save the output JSON file.
    """
    data = bif_to_dict(file_path)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=4)


def get_probability_distribution(
    json_data: dict, variable: str, parent_values: dict
) -> list:
    """
    Returns the probability distribution for a variable given its parent values.

    Args:
        json_data (dict): The parsed JSON structure containing the Bayesian network.
        variable (str): The name of the variable to retrieve probabilities for.
        parent_values (dict): A dictionary mapping parent variable names to their current values.

    Returns:
        list: The probability distribution for the specified variable and parent combination.
    """
    var_data = json_data["variables"][variable]
    parents = var_data["parents"]
    dist = var_data["probability_distribution"]

    # No parents: return the flat distribution
    if not parents:
        return dist

    # Ensure all expected parents are provided
    key_ordered = [parent_values[parent] for parent in parents]
    key = ",".join(key_ordered)

    return dist[key]
