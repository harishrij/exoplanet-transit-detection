import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("Loading training data...")
train = pd.read_csv("exoTrain.csv")
test  = pd.read_csv("exoTest.csv")

# See the shape of the data ──
# .shape returns (rows, columns)
print(f"Training set: {train.shape[0]} stars, {train.shape[1]} columns")
print(f"Test set:     {test.shape[0]} stars, {test.shape[1]} columns")

# Check the labels ──
# value_counts() counts how many of each label exists
print("\nTraining labels:")
print(train["LABEL"].value_counts())
print("(2 = has planet, 1 = no planet)")

# Separate labels from brightness readings ──
# The first column is the label, everything else is flux readings over time
X_train = train.drop("LABEL", axis=1).values  # brightness data only
y_train = train["LABEL"].values                # labels only

X_test  = test.drop("LABEL", axis=1).values
y_test  = test["LABEL"].values

# Convert labels: make 2 → 1 (planet) and 1 → 0 (no planet)
# ML models prefer 0 and 1 over 1 and 2
y_train = (y_train == 2).astype(int)
y_test  = (y_test  == 2).astype(int)

print(f"\nPlanets in training set: {y_train.sum()}")
print(f"No planet in training set: {(y_train==0).sum()}")

# Plot one star WITH a planet vs one WITHOUT ──
planet_idx    = np.where(y_train == 1)[0][0]  # first star with planet
no_planet_idx = np.where(y_train == 0)[0][0]  # first star without

fig, axes = plt.subplots(2, 1, figsize=(14, 7))

axes[0].plot(X_train[planet_idx],    color="royalblue", linewidth=0.8)
axes[0].set_title("⭐ Star WITH a confirmed exoplanet")
axes[0].set_xlabel("Time step")
axes[0].set_ylabel("Flux")

axes[1].plot(X_train[no_planet_idx], color="tomato", linewidth=0.8)
axes[1].set_title("Star with NO planet")
axes[1].set_xlabel("Time step")
axes[1].set_ylabel("Flux")

plt.tight_layout()
plt.show()

print("\nData is ready for ML!")
print(f"Each star has {X_train.shape[1]} brightness measurements")

# Save for next stage
np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
np.save("X_test.npy",  X_test)
np.save("y_test.npy",  y_test)
print("Saved processed data as .npy files ✅")