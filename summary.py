import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import lightkurve as lk
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

print("Building portfolio summary... (takes a few minutes)")

# ── Reload and retrain everything ──
X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_test  = np.load("X_test.npy")
y_test  = np.load("y_test.npy")

X_planet    = X_train[y_train == 1]
y_planet    = y_train[y_train == 1]
X_no_planet = X_train[y_train == 0]
y_no_planet = y_train[y_train == 0]
X_planet_up, y_planet_up = resample(X_planet, y_planet,
    replace=True, n_samples=len(X_no_planet), random_state=42)
X_bal = np.vstack([X_no_planet, X_planet_up])
y_bal = np.hstack([y_no_planet, y_planet_up])

scaler = StandardScaler()
X_bal_sc  = scaler.fit_transform(X_bal)
X_test_sc = scaler.transform(X_test)

model = MLPClassifier(hidden_layer_sizes=(256,128,64),
    activation="relu", solver="adam", max_iter=100,
    early_stopping=True, validation_fraction=0.1,
    random_state=42, verbose=False)
model.fit(X_bal_sc, y_bal)

y_proba = model.predict_proba(X_test_sc)[:, 1]
y_pred  = (y_proba >= 0.30).astype(int)

# ── Download a real light curve for display ──
print("Downloading Kepler-22 for display...")
lc_raw  = lk.search_lightcurve("Kepler-22", mission="Kepler", quarter=1).download()
lc_flat = lc_raw.remove_outliers(sigma=3).flatten(window_length=51)

# ── Build the summary figure ──
fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor("#0d1117")  # dark background like GitHub

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

title_color   = "white"
accent_color  = "#58a6ff"   # GitHub blue
planet_color  = "#3fb950"   # GitHub green
no_planet_col = "#f85149"   # GitHub red

# ── Panel 1: Raw light curve ──
ax1 = fig.add_subplot(gs[0, :2])
ax1.plot(lc_raw.time.value, lc_raw.flux.value,
         color=accent_color, linewidth=0.6, alpha=0.8)
ax1.set_facecolor("#161b22")
ax1.set_title("Kepler-22 — Raw Light Curve (NASA Data)",
              color=title_color, fontsize=11, pad=8)
ax1.set_xlabel("Time (BKJD days)", color="gray", fontsize=9)
ax1.set_ylabel("Flux (e⁻/s)", color="gray", fontsize=9)
ax1.tick_params(colors="gray")
for spine in ax1.spines.values():
    spine.set_edgecolor("#30363d")

# ── Panel 2: Flattened ──
ax2 = fig.add_subplot(gs[0, 2])
ax2.plot(lc_flat.time.value, lc_flat.flux.value,
         color=planet_color, linewidth=0.6, alpha=0.9)
ax2.axhline(0, color="white", linewidth=0.5, linestyle="--", alpha=0.4)
ax2.set_facecolor("#161b22")
ax2.set_title("After Detrending\n(Transit Dips Visible)",
              color=title_color, fontsize=11, pad=8)
ax2.set_xlabel("Time (BKJD days)", color="gray", fontsize=9)
ax2.set_ylabel("Normalized Flux", color="gray", fontsize=9)
ax2.tick_params(colors="gray")
for spine in ax2.spines.values():
    spine.set_edgecolor("#30363d")

# ── Panel 3: Planet vs No Planet comparison ──
ax3 = fig.add_subplot(gs[1, :2])
planet_idx    = np.where(y_test == 1)[0]
no_planet_idx = np.where(y_test == 0)[0][0]
ax3.plot(X_test[planet_idx[4]], color=planet_color,
         linewidth=0.7, label="✅ Has Planet (72% confidence)", alpha=0.9)
ax3.plot(X_test[no_planet_idx], color=no_planet_col,
         linewidth=0.7, label="❌ No Planet", alpha=0.7)
ax3.set_facecolor("#161b22")
ax3.set_title("Planet Star vs Non-Planet Star — Model Input",
              color=title_color, fontsize=11, pad=8)
ax3.set_xlabel("Time step", color="gray", fontsize=9)
ax3.set_ylabel("Flux", color="gray", fontsize=9)
ax3.tick_params(colors="gray")
ax3.legend(facecolor="#21262d", labelcolor="white", fontsize=8)
for spine in ax3.spines.values():
    spine.set_edgecolor("#30363d")

# ── Panel 4: Confusion matrix ──
ax4 = fig.add_subplot(gs[1, 2])
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["No Planet", "Planet"])
disp.plot(ax=ax4, cmap="Greens", colorbar=False)
ax4.set_facecolor("#161b22")
ax4.set_title("Model Results", color=title_color, fontsize=11, pad=8)
ax4.tick_params(colors="gray")
ax4.xaxis.label.set_color("gray")
ax4.yaxis.label.set_color("gray")
for spine in ax4.spines.values():
    spine.set_edgecolor("#30363d")

# ── Main title ──
fig.suptitle(
    "Exoplanet Transit Detection — Neural Network on NASA Kepler Data",
    color="white", fontsize=14, fontweight="bold", y=0.98
)

# ── Stats annotation ──
fig.text(0.5, 0.01,
    "Dataset: 5,087 Kepler stars  |  Model: MLP Neural Network  "
    "|  Training: Oversampled & Scaled  |  Threshold: 0.30",
    ha="center", color="gray", fontsize=8)

plt.savefig("portfolio_summary.png", dpi=150,
            bbox_inches="tight", facecolor="#0d1117")
print("Saved as portfolio_summary.png ✅")
plt.show()