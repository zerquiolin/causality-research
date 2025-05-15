# import joblib


# path = "runs.pkl"
# # Read pickle file
# with open(path, "rb") as f:
#     data = joblib.load(f)
# # Print the data
# print(len(data["Random Agent 1"]["history"]))

import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 100, 500)


def log_penalty(x, alpha=1.0, floor=0.01):
    return 1 / (np.log(alpha * x + 2)) + floor


def inverse_root_penalty(x, scale=1.0, floor=0.01):
    return 1 / (scale * np.sqrt(x + 1)) + floor


def rational_penalty(x, alpha=1.0, floor=0.01):
    return 1 / (alpha * x + 1) + floor


plt.plot(x, log_penalty(x, alpha=0.025), label="Log Penalty")
plt.plot(x, inverse_root_penalty(x, scale=1.0), label="Inverse Root")
plt.plot(x, rational_penalty(x, alpha=0.1), label="Rational (alpha=0.1)")
plt.plot(x, rational_penalty(x, alpha=0.01), label="Rational (alpha=0.05)")

plt.ylim(0, 1.2)
plt.title("Decay Penalty Functions")
plt.xlabel("x (e.g. number of experiments)")
plt.ylabel("Penalty")
plt.legend()
plt.grid(True)
plt.show()
