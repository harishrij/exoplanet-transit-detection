import lightkurve as lk
import matplotlib.pyplot as plt

print("Downloading Kepler-22 data...")
lc = lk.search_lightcurve("Kepler-22", mission="Kepler", quarter=1).download()

#Remove extreme outliers (cosmic rays, bad readings) ──
# remove_outliers() drops any data point more than 3 "sigma" away from normal
# Sigma = standard deviation, a measure of how far from average something is
lc_clean = lc.remove_outliers(sigma=3)

print(f"Original data points: {len(lc)}")
print(f"After removing outliers: {len(lc_clean)}")

#Flatten the light curve (remove the slow wavy trend) ──
# flatten() fits a smooth curve to the data and divides it out
# window_length controls how smooth — must be an odd number
lc_flat = lc_clean.flatten(window_length=51)

#Plot original vs cleaned side by side ──
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
# subplots(2, 1) = 2 rows, 1 column of charts
# figsize = width and height in inches

lc_clean.plot(ax=axes[0], title="BEFORE cleaning — raw light curve")
lc_flat.plot(ax=axes[1], title="AFTER cleaning — flattened light curve")

plt.tight_layout()  # prevents charts from overlapping
plt.show()
# .truncate() cuts the data to only a time range we specify
transit_zoom = lc_flat.truncate(before=133, after=136)

transit_zoom.plot(title="Zoomed in — One Planet Transit of Kepler-22")
plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7)
# axhline draws a horizontal reference line at y=1.0 (baseline brightness)
# alpha=0.7 means 70% opaque (slightly transparent)

plt.show()