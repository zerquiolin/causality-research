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
    # 1) Build a stable feature‐name list in the order we’ll train & predict
    feat_Z, Z_values = next(iter(Z.items()))  # e.g. "Z", [0,1]
    feature_names = [feat_Z] + list(X.keys())

    # 2) Fit the model on exactly those columns, in that order
    model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.8,
        random_state=np.random.RandomState(911),
    )
    X_train = data[feature_names]
    y_train = data[Y].values
    model.fit(X_train, y_train)

    # 3) Helper to build a single‐row DataFrame and predict
    def predict_for(z_val):
        row_dict = {feat_Z: z_val, **X}
        # the columns= feature_names ensures the right order & fills missing
        X_new = pd.DataFrame([row_dict], columns=feature_names)
        return model.predict(X_new)[0]

    # 4) Compute potential outcomes
    Y0 = predict_for(Z_values[0])
    Y1 = predict_for(Z_values[1])
    return Y1 - Y0
