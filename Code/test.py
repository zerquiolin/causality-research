import matplotlib.pyplot as plt
import numpy as np

# Performance metrics
performance = {
    "behavior": {
        "score": 1053.54,
        "metrics": {"PE": 7.0, "PT": 3500.0, "PR": 0.2, "total": 0.0},
    },
    "deliverable": {
        "score": np.float64(0.5),
        "metrics": {"SHD": 0, "F1": np.float64(1.0), "EdgeOrientationAccuracy": 1.0},
    },
    "global_score": np.float64(211.108),
}

# Extract scores
behavior_score = performance["behavior"]["score"]
deliverable_score = performance["deliverable"]["score"]
global_score = performance["global_score"]

# --- First Plot: Score Summary ---
plt.figure(figsize=(10, 4))
x_values = [behavior_score, deliverable_score]
labels = ["Behavior Score", "Deliverable Score"]

plt.scatter(x_values, [1, 1], color="dodgerblue", s=100, zorder=3)
for x, label in zip(x_values, labels):
    plt.text(x, 1.05, f"{label}: {x:.2f}", ha="center", fontsize=10)

plt.axvline(
    global_score, color="red", linestyle="--", label=f"Global Score: {global_score:.2f}"
)
plt.text(
    global_score,
    0.95,
    f"Global Score: {global_score:.2f}",
    color="red",
    ha="center",
    fontsize=10,
)

plt.yticks([])
plt.ylim(0.8, 1.2)
plt.xlabel("Score")
plt.title("Metric Scores Overview")
plt.grid(axis="x", linestyle=":", linewidth=0.7)
plt.tight_layout()
plt.show()

# --- Second Plot: Metric Breakdown ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Behavior metrics
behavior_metrics = performance["behavior"]["metrics"]
b_keys = list(behavior_metrics.keys())
b_vals = list(behavior_metrics.values())
axes[0].barh(b_keys, b_vals, color="skyblue")
axes[0].set_title("Behavior Metrics")
for i, v in enumerate(b_vals):
    axes[0].text(v, i, f"{v:.2f}", va="center", ha="left")

# Deliverable metrics
deliverable_metrics = performance["deliverable"]["metrics"]
d_keys = list(deliverable_metrics.keys())
d_vals = list(deliverable_metrics.values())
axes[1].barh(d_keys, d_vals, color="lightgreen")
axes[1].set_title("Deliverable Metrics")
for i, v in enumerate(d_vals):
    axes[1].text(v, i, f"{v:.2f}", va="center", ha="left")

plt.suptitle("Metric Breakdown by Category", fontsize=14)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
