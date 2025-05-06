from pgmpy.readwrite import BIFReader
import json


def bif_to_dict(file_path: str) -> dict:
    """
    Parses a .bif file and returns its contents as a Python dictionary.

    Args:
        file_path (str): Path to the .bif file.

    Returns:
        dict: Dictionary containing variables, states, and CPDs.
    """
    reader = BIFReader(file_path)
    model = reader.get_model()

    bif_dict = {
        "variables": model.nodes(),
        "edges": list(model.edges()),
        "cpds": {},
    }

    for cpd in model.get_cpds():
        bif_dict["cpds"][cpd.variable] = {
            "variables": cpd.variables,
            "values": cpd.values.tolist(),
            "cardinality": cpd.cardinality,
        }

    return bif_dict


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
