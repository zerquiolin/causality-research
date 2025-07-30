from typing import List
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
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


def cate_estimation(
    Y: str, Z: str, X: List[str], samples: pd.DataFrame, data: pd.DataFrame
):
    """
    Estimate the treatment effect given Y, Z, and X.

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


# def ate_estimation(Y: str, Z: str, samples: pd.DataFrame, data: pd.DataFrame):
#     """
#     estimate the treatment effect given Y, Z, and X.

#     - Y:      name of outcome column
#     - Z:      name of treatment column
#     - Samples: DataFrame containing the intervention values for Z.
#     - data:   DataFrame containing those columns
#     """
#     # Check if there is any data
#     if len(data) == 0:
#         return 0.0
#     # Filter the data given the first possible value of Z
#     Z_0, Z_1 = samples.unique()
#     data_0 = data[data[Z] == Z_0]
#     # Mean of Y given Z=0
#     Y_0_mean = data_0[Y].mean()
#     # Filter the data given the second possible value of Z
#     data_1 = data[data[Z] == Z_1]
#     # Mean of Y given Z=1
#     Y_1_mean = data_1[Y].mean()
#     # Compute the Average Treatment Effect (ATE)
#     return Y_1_mean - Y_0_mean


def ate_estimation(Y: str, Z: str, samples: pd.DataFrame, data: pd.DataFrame) -> float:
    """
    Estimate the Average Treatment Effect (ATE) using regression adjustment.
    Automatically uses all other columns as covariates.

    - Y: name of outcome column
    - Z: name of treatment column
    - data: DataFrame containing all variables
    """
    if len(data) == 0:
        return 0.0

    # Automatically determine covariates (all columns except Y and Z)
    X = [col for col in data.columns if col not in [Y, Z]]

    # If no covariates, fall back to simple difference
    if not X:
        unique_treatments = sorted(samples.unique())
        if len(unique_treatments) < 2:
            return 0.0

        Z_0, Z_1 = unique_treatments[0], unique_treatments[1]
        Y_0_mean = data[data[Z] == Z_0][Y].mean()
        Y_1_mean = data[data[Z] == Z_1][Y].mean()
        return Y_1_mean - Y_0_mean

    # Prepare features
    features = [Z] + X

    # Train regression model
    model = LinearRegression()
    X_train = data[features]
    y_train = data[Y]

    # Handle case of insufficient data
    if len(X_train) == 0 or len(y_train) == 0:
        return 0.0

    model.fit(X_train, y_train)

    # Get unique treatment values
    unique_treatments = samples.unique()
    if len(unique_treatments) < 2:
        return 0.0

    Z_0, Z_1 = sorted(unique_treatments)[:2]

    # Create counterfactual datasets
    data_control = data.copy()
    data_treatment = data.copy()

    data_control[Z] = Z_0
    data_treatment[Z] = Z_1

    # Predict outcomes
    Y_control_pred = model.predict(data_control[features])
    Y_treatment_pred = model.predict(data_treatment[features])

    # Calculate ATE
    ate = np.mean(Y_treatment_pred - Y_control_pred)
    return ate
