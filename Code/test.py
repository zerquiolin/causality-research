import pandas as pd


# Read the CSV file
# df = pd.read_csv("data/tests/Exhaustive Agent-(('Z', '0'),).csv")
df = pd.read_csv("data/tests/Random Agent 2-(('Z', np.str_('0')),).csv")
# df = pd.read_csv("data/tests/Exhaustive Agent-observable.csv")

print("DataFrame shape:", df.shape)

print("DataFrame columns:", df.columns.tolist())

print("First few rows of the DataFrame:")
print(df.head())

# Check if X == Y
print("Checking if X == Y:")
print((df["Y"] == df["Y"]).all())

# Get values that are not equal
unequal_values = df[df["X"] != df["Y"]]
print("Values where X != Y:")
print(unequal_values)

# Check mean of X
mean_x = df["X"].mean()
print(f"Mean of X: {mean_x}")
