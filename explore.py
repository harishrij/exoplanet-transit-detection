import lightkurve as lk
import matplotlib.pyplot as plt

print("Searching NASA for Kepler-22 data...")

# Search NASA's archive for Kepler-22, observed in quarter 1
search_result = lk.search_lightcurve("Kepler-22", mission="Kepler", quarter=1)

print("Found this data:")
print(search_result)
print("Downloading the light curve...")

# Download the first result (index 0 means "first item")
lc = search_result.download()

print("Download complete! Here is what the data looks like:")
print(lc)

# Plot it - this opens a window with the graph
print("Opening chart...")
lc.plot(title="Kepler-22 — Real NASA Light Curve")
plt.show()