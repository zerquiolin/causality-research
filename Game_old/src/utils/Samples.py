import numpy as np
import pandas as pd


def generate_binary_samples_from_joint(df, n) -> pd.DataFrame:
    """
    Generate binary samples from a joint probability distribution.

    Parameters:
    - df (pandas.DataFrame): The DataFrame representing the joint probability distribution.
                            The last column should contain the probabilities, which should sum to 1.
                            The other columns should represent the binary variables.
    - n (int): The number of samples to generate.

    Returns:
    - pandas.DataFrame: A DataFrame containing the sampled binary variable rows.

    """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("The input data must be a pandas DataFrame.")

    if "P" not in df.columns:
        raise ValueError(
            "DataFrame must contain a 'P' column representing probabilities."
        )

    # Ensure the joint probability column is normalized (sums to 1)
    if not np.isclose(df.iloc[:, -1].sum(), 1.0):
        df.iloc[:, -1] = df.iloc[:, -1] / df.iloc[:, -1].sum()

    # Extract the binary variable columns (without the probability column)
    binary_variables = df.iloc[:, :-1]

    # Sample `n` rows from the DataFrame based on the joint probability
    sampled_indices = np.random.choice(df.index, size=n, p=df["P"])

    # Return the sampled binary variable rows
    sampled_data = binary_variables.loc[sampled_indices].reset_index(drop=True)

    return sampled_data
