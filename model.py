import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.metrics import (classification_report,
                             confusion_matrix,
                             ConfusionMatrixDisplay)
import matplotlib.pyplot as plt

# ── Load data ──
print("Loading data...")
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_test  = np.load("X_test.npy")
y_test  = np.load("y_test.npy")

# ── Balance: oversample planet stars ──
print("Balancing dataset...")
X_planet    = X_train[y_train == 1]
y_planet    = y_train[y_train == 1]
X_no_planet = X_train[y_train == 0]
y_no_planet = y_train[y_train == 0]

X_planet_up, y_planet_up = resample(
    X_planet, y_planet,
    replace=True,
    n_samples=len(X_no_planet),
    random_state=42
)

X_balanced = np.vstack([X_no_planet, X_planet_up])
y_balanced = np.hstack([y_no_planet, y_planet_up])

# ── Scale ──
print("Scaling...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_balanced)
X_test_scaled  = scaler.transform(X_test)

# ── Build model with class_weight to penalize missing planets ──
# class_weight="balanced" tells the model: 
# "missing a planet is a much bigger mistake than a false alarm"
print("\nTraining improved model... (1-2 mins)")
model = RandomForestClassifier(
    n_estimators=200,        # more trees = more robust voting
    max_depth=15,            # slightly deeper trees
    class_weight="balanced", # KEY FIX: penalize planet misses heavily
    random_state=42,
    n_jobs=-1
)
model.fit(X_train_scaled, y_balanced)
print("Training complete! ✅")

# ── Lower the decision threshold ──
# Instead of "if confidence > 50% → planet"
# We use "if confidence > 30% → planet"
# This makes the model more willing to flag potential planets
print("\nPredicting with lowered threshold (0.30)...")
y_proba = model.predict_proba(X_test_scaled)[:, 1]
# predict_proba gives a probability for each star
# [:, 1] = take the "has planet" probability column

threshold = 0.30
y_pred = (y_proba >= threshold).astype(int)
# Any star where confidence >= 30% gets called "planet"

# ── Results ──
print("\n── Model Performance (threshold=0.30) ──")
print(classification_report(y_test, y_pred,
      target_names=["No Planet", "Has Planet"]))

# ── Confusion matrix ──
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=["No Planet", "Has Planet"])
disp.plot(cmap="Blues")
plt.title("Confusion Matrix — Improved Model")
plt.show()

# ── Bonus: show confidence scores for the 5 real planet stars ──
print("\n── Confidence scores for the 5 actual planet stars ──")
planet_indices = np.where(y_test == 1)[0]
for i, idx in enumerate(planet_indices):
    confidence = y_proba[idx] * 100
    called = "PLANET ✅" if y_pred[idx] == 1 else "missed ❌"
    print(f"  Planet star {i+1}: {confidence:.1f}% confident → {called}")