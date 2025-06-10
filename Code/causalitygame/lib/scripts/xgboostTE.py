import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor


def te_estimation(Y: str, Z: dict, X: dict, data: pd.DataFrame):
    """
    Estimate the treatment effect given Y, Z, and X.

    - Y:      name of outcome column
    - Z:      {treatment_name: [z0, z1]}
    - X:      {covariate_name: cov_value, ...}
    - data:   DataFrame containing those columns
    """
    if len(data) == 0:
        return 0.0  # No data to estimate treatment effect
    # Update the dataframe to include the treatment and covariates
    feat_Z, Z_values = next(iter(Z.items()))  # e.g. "Z", [0,1]
    feature_names = [feat_Z] + list(X.keys())

    # Note: Ensure the order of feature_names matches the data columns

    # 2) Fit the model on exactly those columns, in that order
    model = GradientBoostingRegressor(
        n_estimators=10, max_depth=1, learning_rate=1.0, subsample=1.0, random_state=911
    )
    # model = GradientBoostingRegressor(
    #     n_estimators=200,
    #     max_depth=3,
    #     learning_rate=0.03,
    #     subsample=0.8,
    #     random_state=np.random.RandomState(911),
    # )
    X_train = data[feature_names]
    y_train = data[Y].values
    model.fit(X_train, y_train)

    # Note: X_train, y_train -> Check
    # Note: model.predict(X_train) -> Error 0

    # ATE: Error 0 sin ruido

    # Note: No distinction between random states (unit test)

    # 3) Append two rows for the counterfactual treatments, then train/predict by slicing
    rows_to_add = []
    for z_val in Z_values:
        row = {feat_Z: z_val}
        for cov_name, cov_val in X.items():
            row[cov_name] = cov_val
        row[Y] = np.nan
        rows_to_add.append(row)

    data_extended = pd.concat([data, pd.DataFrame(rows_to_add)], ignore_index=True)

    # Train on all but the last two rows
    train_df = data_extended.iloc[:-2]
    X_train = train_df[feature_names]
    y_train = train_df[Y].values
    model.fit(X_train, y_train)

    # Predict on the last two appended rows
    pred_df = data_extended.iloc[-2:]
    X_pred = pred_df[feature_names]
    Y0 = model.predict(X_pred.iloc[[0]])[0]
    Y1 = model.predict(X_pred.iloc[[1]])[0]

    return Y1 - Y0
