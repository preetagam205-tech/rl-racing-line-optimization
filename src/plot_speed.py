import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# -------------------------
# FIND THE SAVED LAP DATA
# -------------------------

project_dir = Path(__file__).resolve().parent.parent

results_dir = project_dir / "results"

csv_files = list(results_dir.glob("lap_*.csv"))

if not csv_files:
    raise FileNotFoundError(
        "No lap CSV files found. "
        "Complete a valid lap first."
    )

# Select the most recently modified CSV
latest_file = max(
    csv_files,
    key=lambda file: file.stat().st_mtime_ns
)

print("Loading:", latest_file.name)


# -------------------------
# LOAD SPEED AND PROGRESS
# -------------------------

speed = []
progress = []
time = []

with open(
    latest_file,
    "r",
    newline="",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        speed.append(float(row["speed"]))

        progress.append(float(row["progress"]))

        time.append(float(row["time"]))


# Convert lists into NumPy arrays
speed = np.array(speed)
progress = np.array(progress)
time = np.array(time)

if len(speed) == 0:
    raise ValueError("The CSV file contains no driving data.")

# -------------------------
# DEBUG SPEED DATA
# -------------------------

print("\n--- SPEED DATA CHECK ---")

print("Minimum speed:", np.min(speed))

print("Maximum speed:", np.max(speed))

print("First 10 speeds:", speed[:10])

print("First 10 times:", time[:10])

print("First 10 progress values:", progress[:10])

print("Number of unique speeds:", len(np.unique(speed)))

print(f"Loaded {len(speed)} data points")

print(f"Maximum speed: {np.max(speed):.2f} units/s")

print(f"Average speed: {np.mean(speed):.2f} units/s")


# -------------------------
# PLOT SPEED PROFILE
# -------------------------

plt.figure(figsize=(12, 6))

# Scatter avoids drawing a false connecting line
# when progress wraps from 100% back to 0%.
plt.scatter(
    progress,
    speed,
    s=8,
    label="Vehicle speed"
)

# Mark all seven checkpoints
num_checkpoints = 7

for i in range(1, num_checkpoints + 1):

    checkpoint_progress = (
        i / (num_checkpoints + 1)
    ) * 100

    plt.axvline(
        checkpoint_progress,
        linestyle="--",
        alpha=0.3
    )

# Titles and labels
plt.title("Vehicle Speed Along the Racing Track")

plt.xlabel("Track Progress (%)")
plt.ylabel("Speed (units/s)")

plt.xlim(0, 100)
plt.ylim(bottom=0)

plt.grid(True)
plt.legend()

plt.tight_layout()

plt.show()