from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np

from src.track_geometry import TrackGeometry


# -------------------------
# PROJECT PATHS
# -------------------------

project_dir = Path(__file__).resolve().parent.parent

trajectory_path = (
    project_dir
    / "results"
    / "ppo_stage3_trajectory.csv"
)


# -------------------------
# LOAD TRAJECTORY
# -------------------------

x_values = []
y_values = []
speed_values = []
progress_values = []


with open(trajectory_path, "r") as file:

    reader = csv.DictReader(file)

    for row in reader:

        x_values.append(
            float(row["x"])
        )

        y_values.append(
            float(row["y"])
        )

        speed_values.append(
            float(row["speed"])
        )

        progress_values.append(
            float(row["progress"])
        )


x_values = np.array(x_values)
y_values = np.array(y_values)
speed_values = np.array(speed_values)


print(
    f"Loaded {len(x_values)} PPO trajectory points."
)


# -------------------------
# LOAD TRACK
# -------------------------

track = TrackGeometry()


# -------------------------
# CREATE PLOT
# -------------------------

plt.figure(
    figsize=(12, 7)
)


# Outer track boundary
plt.plot(
    track.outer_boundary[:, 0],
    track.outer_boundary[:, 1],
    label="Outer Boundary"
)


# Inner track boundary
plt.plot(
    track.inner_boundary[:, 0],
    track.inner_boundary[:, 1],
    label="Inner Boundary"
)


# Centerline
plt.plot(
    track.centerline[:, 0],
    track.centerline[:, 1],
    "--",
    label="Centerline"
)


# PPO trajectory
plt.plot(
    x_values,
    y_values,
    linewidth=2.5,
    label="PPO Stage 3"
)


# Starting point
plt.scatter(
    x_values[0],
    y_values[0],
    s=100,
    label="Start"
)


# -------------------------
# PLOT SETTINGS
# -------------------------

plt.title(
    "PPO Stage 3 Learned Trajectory"
)

plt.xlabel("X Position")
plt.ylabel("Y Position")

plt.axis("equal")

plt.grid(True)

plt.legend()

plt.tight_layout()

plt.show()