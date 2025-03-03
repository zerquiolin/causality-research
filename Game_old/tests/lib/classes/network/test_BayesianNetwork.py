from src.lib.classes.network.SCM import StructuralCausalModel
from pgmpy.factors.discrete import TabularCPD
import pandas as pd


def test_create_dag_with_tabular_data():
    # Create a Bayesian Network object
    bn = StructuralCausalModel()

    # Add nodes and edges
    bn.add_nodes(["A", "B", "C"])
    bn.add_edges([("A", "B"), ("A", "C"), ("B", "C")])

    # Create TabularCPD objects from the DataFrames
    cpd_A = TabularCPD(
        variable="A",
        variable_card=2,
        values=[[0.8], [0.2]],  # Probabilities for A=0 and A=1
    )

    cpd_B = TabularCPD(
        variable="B",
        variable_card=2,
        values=[[0.7, 0.4], [0.3, 0.6]],  # Conditional probabilities P(B|A)
        evidence=["A"],
        evidence_card=[2],
    )

    cpd_C = TabularCPD(
        variable="C",
        variable_card=2,
        values=[
            [0.9, 0.8, 0.9, 0.1],
            [0.1, 0.2, 0.1, 0.9],
        ],  # Conditional probabilities P(C|A,B)
        evidence=["A", "B"],
        evidence_card=[2, 2],
    )

    CPDs = {"A": cpd_A, "B": cpd_B, "C": cpd_C}

    # Add the CPDs to the Bayesian Network
    bn.add_cpds(CPDs)

    # Check if the model is valid
    assert bn.is_valid == True


def test_create_dag_with_pandas_data():
    # Create a Bayesian Network object
    bn = StructuralCausalModel()

    # Add nodes and edges
    bn.add_nodes(["A", "B", "C"])
    bn.add_edges([("A", "B"), ("A", "C"), ("B", "C")])

    CPDs = {
        "A": pd.DataFrame(
            {
                "A": [0, 1],
                "P": [0.8, 0.2],
            }
        ),
        "B": pd.DataFrame(
            {
                "A": [0, 0, 1, 1],
                "B": [0, 1, 0, 1],
                "P": [0.7, 0.3, 0.4, 0.6],
            }
        ),
        "C": pd.DataFrame(
            {
                "A": [0, 0, 1, 1, 0, 0, 1, 1],
                "B": [0, 1, 0, 1, 0, 1, 0, 1],
                "C": [0, 0, 0, 0, 1, 1, 1, 1],
                "P": [0.9, 0.8, 0.9, 0.1, 0.1, 0.2, 0.1, 0.9],
            }
        ),
    }

    # Add the CPDs to the Bayesian Network
    bn.add_cpds(CPDs)

    # Check if the model is valid
    assert bn.is_valid == True


def test_covid_dag():
    # Create a Bayesian Network
    bn = StructuralCausalModel()

    # Add nodes
    bn.add_nodes(["Fever", "Covid"])

    # Add edges
    bn.add_edges([("Covid", "Fever")])

    # Add Conditional Probability Distribution
    cpds = {
        "Covid": pd.DataFrame({"Covid": [0, 1], "P": [0.75, 0.25]}),
        "Fever": pd.DataFrame(
            {"Fever": [0, 0, 1, 1], "Covid": [0, 1, 0, 1], "P": [1, 0, 0, 1]}
        ),
    }

    bn.add_cpds(cpds)

    # Check if the model is valid
    try:
        bn.is_valid
        print("Model is valid")
    except Exception as e:
        print(e)

    # Full joint probability distribution
    print("Full joint probability distribution:")
    print(bn.infer_probability(output_dataframe=True))
    print("Intervention: P(Fever=1):")
    print(bn.infer_probability(interventions={"Fever": 0}, output_dataframe=True))
    print("Evidences: P(Fever=1):")
    print(bn.infer_probability(evidences={"Fever": 0}, output_dataframe=True))
    print("Intervention: P(Covid=1):")
    print(bn.infer_probability(interventions={"Covid": 1}, output_dataframe=True))
    print("Evidences: P(Covid=1):")
    print(bn.infer_probability(evidences={"Covid": 1}, output_dataframe=True))


def test():
    # Create a Bayesian Network
    bn = StructuralCausalModel()

    # Add nodes (variables)
    bn.add_nodes(["Fever", "Covid", "Cough", "TestResult"])

    # Add edges (relationships)
    bn.add_edges([("Covid", "Fever"), ("Covid", "Cough"), ("Covid", "TestResult")])

    # Create CPDs using pandas DataFrames

    # CPD for Covid (prior probability)
    cpds = {
        "Covid": pd.DataFrame({"Covid": [0, 1], "P": [0.75, 0.25]}),
        # CPD for Fever given Covid
        "Fever": pd.DataFrame(
            {"Fever": [0, 0, 1, 1], "Covid": [0, 1, 0, 1], "P": [1, 0, 0, 1]}
        ),
        # CPD for Cough given Covid
        "Cough": pd.DataFrame(
            {"Cough": [0, 0, 1, 1], "Covid": [0, 1, 0, 1], "P": [0.8, 0.2, 0.2, 0.8]}
        ),
        # CPD for TestResult given Covid
        "TestResult": pd.DataFrame(
            {
                "TestResult": [0, 0, 1, 1],
                "Covid": [0, 1, 0, 1],
                "P": [0.9, 0.1, 0.1, 0.9],
            }
        ),
    }

    # Add CPDs to the network (directly using the DataFrame method as in the example)
    bn.add_cpds(cpds)

    # Check if the model is valid
    try:
        bn.is_valid
        print("Model is valid")
    except Exception as e:
        print(e)

    in_1 = bn.infer_probability(evidences={"Covid": 1}, interventions=dict())
    print(type(in_1))
    print(in_1)


# test()
test_covid_dag()
