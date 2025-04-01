import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from causallearn.search.ConstraintBased.PC import pc
from causallearn.utils.GraphUtils import GraphUtils
from causallearn.utils.cit import chisq


# Function to generate binary data with the defined causal structure
def generate_data(n_samples=1000, intervention=None):
    if intervention is None:
        intervention = {}

    A = (
        np.random.binomial(1, 0.5, n_samples)
        if "A" not in intervention
        else np.full(n_samples, intervention["A"], dtype=int)
    )
    B = np.zeros(n_samples, dtype=int)
    C = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        if "B" in intervention:
            B[i] = intervention["B"]
        else:
            B[i] = A[i] ^ np.random.binomial(1, 0.1)

        if "C" in intervention:
            C[i] = intervention["C"]
        else:
            C[i] = (A[i] & B[i]) ^ np.random.binomial(1, 0.1)

    return pd.DataFrame({"A": A, "B": B, "C": C})


# Generate datasets
datasets = {
    "observational": generate_data(5000),
    "intervention_A": generate_data(100, intervention={"A": 1}),
    "intervention_B": generate_data(100, intervention={"B": 0}),
    "intervention_C": generate_data(100, intervention={"C": 1}),
}


# Run PC algorithm on each dataset and extract directed edges
def extract_directed_edges(data, labels=["A", "B", "C"]):
    result = pc(data.to_numpy(), alpha=0.05, indep_test_method=chisq, labels=labels)
    edges = set()
    for i, src in enumerate(labels):
        for j, dst in enumerate(labels):
            edge_type = result.G.graph[i][j]
            if edge_type == 1:  # 1 means i --> j
                edges.add((src, dst))
    return edges


# Extract DAGs
edge_sets = []
for name, df in datasets.items():
    print(f"Processing {name} data...")
    edges = extract_directed_edges(df)
    edge_sets.append(edges)

# Intersect all edge sets
common_edges = set.union(*edge_sets)

# Build consensus DAG
consensus_dag = nx.DiGraph()
consensus_dag.add_edges_from(common_edges)


# Plotting
def plot_graph(G):
    pos = nx.spring_layout(G)
    nx.draw(
        G,
        pos,
        with_labels=True,
        arrows=True,
        node_size=2000,
        node_color="lightgreen",
        font_size=12,
    )
    plt.title("Consensus DAG from Common Edges Across All Datasets")
    plt.show()


plot_graph(consensus_dag)
