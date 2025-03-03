import pandas as pd
from src.lib.classes.network.SCM import StructuralCausalModel


class V4:
    def __init__(self):
        # Create a Bayesian Network
        self.bn = StructuralCausalModel()

    def gen_model(self):
        # Add nodes
        self.bn.add_nodes(["A", "B", "C", "D"])

        # Add edges: A -> B, B -> C, C -> D
        self.bn.add_edges([("A", "B"), ("B", "C"), ("C", "D")])

        # Add Conditional Probability Distribution
        cpds = {
            "A": pd.DataFrame({"A": [0, 1], "P": [0.6, 0.4]}),  # Prior for A
            "B": pd.DataFrame(
                {
                    "A": [0, 1, 0, 1],  # Values of A (parent of B)
                    "B": [0, 0, 1, 1],  # Values of B
                    "P": [0.8, 0.2, 0.2, 0.8],  # P(B|A)
                }
            ),
            "C": pd.DataFrame(
                {
                    "B": [0, 1, 0, 1],  # Values of B (parent of C)
                    "C": [0, 0, 1, 1],  # Values of C
                    "P": [0.7, 0.3, 0.3, 0.7],  # P(C|B)
                }
            ),
            "D": pd.DataFrame(
                {
                    "C": [0, 1, 0, 1],  # Values of C (parent of D)
                    "D": [0, 0, 1, 1],  # Values of D
                    "P": [0.9, 0.4, 0.1, 0.6],  # P(D|C)
                }
            ),
        }

        self.bn.add_cpds(cpds)

        return self.bn

    def test_model(self):
        # Check if the model is valid
        try:
            self.bn.is_valid
            print("Model is valid")
        except Exception as e:
            print(e)


# Example usage
if __name__ == "__main__":
    model = V4()
    bn = model.gen_model()
    model.test_model()
