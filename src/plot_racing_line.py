import csv
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# -------------------------
# LOCATE SAVED RACING DATA
# -------------------------

project_dir = Path(__file__).resolve().parent.parent
results_dir = project_dir / "results"

# Find all saved lap CSV files
csv_files = list(results_dir.glob("lap_*.csv"))

if not csv_files:
    raise FileNotFoundError(
        "No lap CSV files found in results/. "
        "Complete a valid lap in the simulator first."
    )

# Select the most recently modified lap
latest_file = max(
    csv_files,
    key=lambda file: file.stat().st_mtime_ns
)

print("Loading:", latest_file.name)


# -------------------------
# LOAD CAR TRAJECTORY
# -------------------------

car_x = []
car_y = []

with open(
    latest_file,
    "r",
    newline="",
    encoding="utf-8"
) as file:

    reader = csv.DictReader(file)

    for row in reader:
        car_x.append(float(row["x"]))
        car_y.append(float(row["y"]))

car_x = np.array(car_x)
car_y = np.array(car_y)

if len(car_x) == 0:
    raise ValueError("The selected CSV has no trajectory data.")

print(f"Loaded {len(car_x)} trajectory points")


# -------------------------
# RECREATE TRACK GEOMETRY
# -------------------------

a = 40
b = 20
track_width = 12

theta = np.linspace(
    0,
    2 * np.pi,
    720,
    endpoint=False
)

x = a * np.cos(theta)
y = b * np.sin(theta)

centerline = np.column_stack((x, y))

# Calculate normal vectors
nx = b * np.cos(theta)
ny = a * np.sin(theta)

length = np.sqrt(nx**2 + ny**2)

nx = nx / length
ny = ny / length

normals = np.column_stack((nx, ny))

# Generate boundaries
outer = centerline + track_width / 2 * normals
inner = centerline - track_width / 2 * normals


# -------------------------
# PLOT RACING LINE
# -------------------------

plt.figure(figsize=(12, 7))

# Track boundaries
plt.plot(
    outer[:, 0],
    outer[:, 1],
    color="black",
    label="Outer boundary"
)

plt.plot(
    inner[:, 0],
    inner[:, 1],
    color="black",
    label="Inner boundary"
)

# Track centerline
plt.plot(
    centerline[:, 0],
    centerline[:, 1],
    color="orange",
    linestyle="--",
    label="Track centerline"
)

# Your recorded trajectory
plt.plot(
    car_x,
    car_y,
    color="blue",
    linewidth=2,
    label="Recorded trajectory"
)

# Start position
plt.scatter(
    car_x[0],
    car_y[0],
    color="green",
    s=80,
    label="Trajectory start"
)

# Plot settings
plt.title("Racing-Line Visualization")

plt.xlabel("X Position")
plt.ylabel("Y Position")

plt.axis("equal")
plt.grid(True)
plt.legend()

plt.show()