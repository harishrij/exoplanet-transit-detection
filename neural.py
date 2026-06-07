import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# ── Load data ──
print("Loading data...")
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_test  = np.load("X_test.npy")
y_test  = np.load("y_test.npy")

# ── Balance ──
print("Balancing...")
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

X_bal = np.vstack([X_no_planet, X_planet_up])
y_bal = np.hstack([y_no_planet, y_planet_up])

# ── Scale ──
scaler = StandardScaler()
X_bal_sc   = scaler.fit_transform(X_bal)
X_test_sc  = scaler.transform(X_test)

# ── Build neural network using MLPClassifier ──
# MLP = Multi-Layer Perceptron — a classic neural network
# No need to install anything new, it's inside sklearn
from sklearn.neural_network import MLPClassifier

print("\nTraining neural network... (2-3 mins)")
model = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    # 3 layers: first sees 256 neurons, then 128, then 64
    # Neurons = tiny units that learn patterns, like brain cells
    activation="relu",
    # relu = the neuron only "fires" if input is positive
    # Most common activation function in modern ML
    solver="adam",
    # adam = smart optimizer that adjusts learning speed automatically
    max_iter=100,
    # maximum 100 passes through the full dataset
    early_stopping=True,
    # stops automatically if it stops improving (saves time)
    validation_fraction=0.1,
    # uses 10% of training data to check if it's improving
    random_state=42,
    verbose=True
    # prints progress every 10 iterations so you see it working
)

model.fit(X_bal_sc, y_bal)
print("\nTraining complete! ✅")

# ── Predict with lowered threshold ──
y_proba = model.predict_proba(X_test_sc)[:, 1]
threshold = 0.30
y_pred = (y_proba >= threshold).astype(int)

print("\n── Neural Network Performance ──")
print(classification_report(y_test, y_pred,
      target_names=["No Planet", "Has Planet"]))

# ── Confusion matrix ──
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["No Planet", "Has Planet"])
disp.plot(cmap="Greens")
plt.title("Neural Network — Confusion Matrix")
plt.show()

# ── Confidence for real planet stars ──
print("\n── Confidence for the 5 real planet stars ──")
planet_idx = np.where(y_test == 1)[0]
for i, idx in enumerate(planet_idx):
    conf = y_proba[idx] * 100
    result = "PLANET ✅" if y_pred[idx] == 1 else "missed ❌"
    print(f"  Planet star {i+1}: {conf:.1f}% confident → {result}")

# ── Plot learning curve ──
plt.figure(figsize=(10, 4))
plt.plot(model.loss_curve_, label="Training loss", color="royalblue")
if model.validation_scores_ is not None:
    plt.plot(model.validation_scores_, label="Validation score", color="tomato")
plt.xlabel("Iteration")
plt.ylabel("Score")
plt.title("Neural Network Learning Curve")
plt.legend()
plt.tight_layout()
plt.show()