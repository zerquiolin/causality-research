import pandas as pd
from src.lib.classes.network.SCM import StructuralCausalModel


class Covid:
    def __init__(self):
        # Create a Bayesian Network
        self.bn = StructuralCausalModel()

    def gen_model(self):

        # Add nodes
        self.bn.add_nodes(["Fever", "Covid"])

        # Add edges
        self.bn.add_edges([("Covid", "Fever")])

        # Add Conditional Probability Distribution
        cpds = {
            "Covid": pd.DataFrame({"Covid": [0, 1], "P": [0.75, 0.25]}),
            "Fever": pd.DataFrame(
                {"Fever": [0, 0, 1, 1], "Covid": [0, 1, 0, 1], "P": [1, 0, 0, 1]}
            ),
        }

        self.bn.add_cpds(cpds)

        return self.bn

    def test_model(self):
        # Check if the model is valid
        try:
            bn.is_valid
            print("Model is valid")
        except Exception as e:
            print(e)


# Example usage
if __name__ == "__main__":
    model = Covid()
    bn = model.gen_model()
    model.test_model()
