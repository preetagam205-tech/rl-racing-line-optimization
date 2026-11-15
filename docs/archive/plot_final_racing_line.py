from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO

from src.racing_env import RacingEnv
from src.track_geometry import TrackGeometry
from src.track_factory import make_wavy_track


project_dir = Path(__file__).resolve().parent


# =========================================================
# LOAD FINAL MODEL
# =========================================================

model = PPO.load(
    project_dir
    / "models"
    / "ppo_hardfivefamily_stage1"
)

print("Final model loaded!")


# =========================================================
# CREATE REPRESENTATIVE WAVY TRACK
# =========================================================

centerline = make_wavy_track(
    num_points=720
)

track = TrackGeometry(
    centerline_points=centerline,
    track_width=12.0,
    num_checkpoints=7
)


# =========================================================
# COMPUTE TRACK BOUNDARIES FOR PLOTTING
# =========================================================

previous_points = np.roll(
    centerline,
    1,
    axis=0
)

next_points = np.roll(
    centerline,
    -1,
    axis=0
)

tangent = (
    next_points
    - previous_points
)

tangent_length = np.linalg.norm(
    tangent,
    axis=1,
    keepdims=True
)

tangent = (
    tangent
    / np.maximum(
        tangent_length,
        1e-12
    )
)

normal = np.column_stack(
    (
        -tangent[:, 1],
        tangent[:, 0]
    )
)

half_width = (
    track.track_width
    / 2.0
)

left_boundary = (
    centerline
    + normal * half_width
)

right_boundary = (
    centerline
    - normal * half_width
)


# =========================================================
# RUN PPO LAP
# =========================================================

env = RacingEnv(
    track=track,
    random_start=False,
    curvature_observation=False
)

observation, info = env.reset(
    options={
        "start_index": 0
    }
)


trajectory_x = [
    env.sim.car.x
]

trajectory_y = [
    env.sim.car.y
]

speeds = [
    env.sim.car.speed
]


terminated = False
truncated = False


while not (
    terminated
    or truncated
):

    action, _ = model.predict(
        observation,
        deterministic=True
    )

    (
        observation,
        reward,
        terminated,
        truncated,
        info
    ) = env.step(action)

    trajectory_x.append(
        env.sim.car.x
    )

    trajectory_y.append(
        env.sim.car.y
    )

    speeds.append(
        env.sim.car.speed
    )


trajectory_x = np.asarray(
    trajectory_x
)

trajectory_y = np.asarray(
    trajectory_y
)


print(
    "Lap completed:",
    info["lap_completed"]
)

print(
    "Lap time:",
    f"{info['time']:.2f}s"
)

print(
    "Final progress:",
    f"{info['progress']:.2f}%"
)


# =========================================================
# PLOT
# =========================================================

plt.figure(
    figsize=(10, 8)
)


# Track boundaries
plt.plot(
    left_boundary[:, 0],
    left_boundary[:, 1],
    linewidth=1.5,
    label="Track boundary"
)

plt.plot(
    right_boundary[:, 0],
    right_boundary[:, 1],
    linewidth=1.5
)


# Centerline
plt.plot(
    centerline[:, 0],
    centerline[:, 1],
    linestyle="--",
    linewidth=1.2,
    label="Centerline"
)


# PPO racing line
plt.plot(
    trajectory_x,
    trajectory_y,
    linewidth=2.5,
    label="PPO racing line"
)


# Start position
plt.scatter(
    trajectory_x[0],
    trajectory_y[0],
    s=70,
    label="Start"
)


plt.axis(
    "equal"
)

plt.xlabel(
    "X position (m)"
)

plt.ylabel(
    "Y position (m)"
)

plt.title(
    "Learned PPO Racing Line on Wavy Track"
)

plt.legend()

plt.grid(
    alpha=0.25
)

plt.tight_layout()


output_file = (
    project_dir
    / "results"
    / "final_racing_line.png"
)


plt.savefig(
    output_file,
    dpi=300
)

plt.show()


print(
    "\nSaved figure to:"
)

print(
    output_file
)


env.close()