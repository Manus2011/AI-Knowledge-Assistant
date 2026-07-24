# quick histograms comparing feature distributions for
# purchased vs not purchased

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from customer_prediction import generate_data

df = generate_data()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

features = ["time_on_site", "pages_viewed", "past_purchases"]
for ax, feature in zip(axes, features):
    df[df["will_purchase"] == 1][feature].hist(ax=ax, alpha=0.6, label="purchased", bins=15)
    df[df["will_purchase"] == 0][feature].hist(ax=ax, alpha=0.6, label="did not purchase", bins=15)
    ax.set_title(feature)
    ax.legend()

plt.tight_layout()
plt.savefig("visualization_output.png", dpi=100)
print("Saved chart to visualization_output.png")
