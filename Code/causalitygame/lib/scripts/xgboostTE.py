from typing import List
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor


def te_estimation(
    Y: str, Z: str, X: List[str], samples: pd.DataFrame, data: pd.DataFrame
):
    """
    qstimate ehe treatment effect given Y, Z, and X.

    - Y:      name of outcome column
    - Z:      name of treatment column
    - X:      list of names of covariate columns
    - samples: DataFrame containing the samples to estimate the treatment effect (two rows)
    - data:   DataFrame containing those columns
    """
    # Check if there is any data
    if len(data) == 0:
        return 0.0  # No data to estimate treatment effect

    # Define features and target variable
    features = [Z] + X
    target = Y

    # Define the training and prediction dataframes
    X_train = data[features]
    y_train = data[target]
    X_pred = samples[features]

    # Check both training and prediction dataframes have the same columns order
    assert list(X_train.columns) == list(X_pred.columns), "Column order mismatch!"

    # Generate model
    model = RandomForestRegressor(n_estimators=100, random_state=911)

    # Train the model
    model.fit(X_train, y_train)

    # Predict on the last two appended rows
    Y0, Y1 = model.predict(X_pred)

    return Y1 - Y0
