from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO

from src.racing_env import RacingEnv
from src.track_geometry import TrackGeometry
from src.track_factory import make_wavy_track


project_dir = Path(__file__).resolve().parent


# =========================================================
# TRACK
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
# TRACK BOUNDARIES
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
# PPO TRAJECTORY
# =========================================================

model = PPO.load(
    project_dir
    / "models"
    / "ppo_hardfivefamily_stage1"
)

ppo_env = RacingEnv(
    track=track,
    random_start=False,
    curvature_observation=False
)

observation, info = ppo_env.reset(
    options={
        "start_index": 0
    }
)

ppo_x = [
    ppo_env.sim.car.x
]

ppo_y = [
    ppo_env.sim.car.y
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
    ) = ppo_env.step(action)

    ppo_x.append(
        ppo_env.sim.car.x
    )

    ppo_y.append(
        ppo_env.sim.car.y
    )


ppo_time = info["time"]

ppo_env.close()


# =========================================================
# BASELINE HELPERS
# =========================================================

def wrap_angle(angle):

    return (
        angle + np.pi
    ) % (
        2.0 * np.pi
    ) - np.pi


def nearest_centerline_index(
    x,
    y
):

    differences = (
        centerline
        - np.array([x, y])
    )

    distances_squared = np.sum(
        differences ** 2,
        axis=1
    )

    return int(
        np.argmin(
            distances_squared
        )
    )


def lookahead_target(
    start_index,
    distance_ahead=12.0
):

    total_distance = 0.0

    index = start_index

    while total_distance < distance_ahead:

        next_index = (
            index + 1
        ) % len(centerline)

        segment = (
            centerline[next_index]
            - centerline[index]
        )

        total_distance += (
            np.linalg.norm(segment)
        )

        index = next_index

    return centerline[index]


# =========================================================
# BASELINE TRAJECTORY
# =========================================================

baseline_env = RacingEnv(
    track=track,
    random_start=False,
    curvature_observation=False
)

observation, info = baseline_env.reset(
    options={
        "start_index": 0
    }
)


baseline_x = [
    baseline_env.sim.car.x
]

baseline_y = [
    baseline_env.sim.car.y
]

terminated = False
truncated = False


while not (
    terminated
    or truncated
):

    car = baseline_env.sim.car

    nearest_index = (
        nearest_centerline_index(
            car.x,
            car.y
        )
    )

    target = lookahead_target(
        nearest_index,
        distance_ahead=12.0
    )


    desired_heading = np.arctan2(
        target[1] - car.y,
        target[0] - car.x
    )

    heading_error = wrap_angle(
        desired_heading
        - car.heading
    )


    steering_input = np.clip(
        1.8
        * heading_error
        / car.max_steering,
        -1.0,
        1.0
    )


    lookahead_heading = (
        track.get_lookahead_heading_changes(
            car.x,
            car.y,
            distances=(
                20.0,
                40.0
            )
        )
    )


    corner_severity = max(
        abs(lookahead_heading[0]),
        abs(lookahead_heading[1])
    )


    if corner_severity > 1.0:

        target_speed = 10.0

    elif corner_severity > 0.65:

        target_speed = 13.0

    elif corner_severity > 0.35:

        target_speed = 16.0

    else:

        target_speed = 20.0


    speed_error = (
        target_speed
        - car.speed
    )


    if speed_error >= 0.0:

        throttle_brake = np.clip(
            speed_error / 5.0,
            0.0,
            1.0
        )

    else:

        throttle_brake = np.clip(
            speed_error / 4.0,
            -1.0,
            0.0
        )


    action = np.array(
        [
            steering_input,
            throttle_brake
        ],
        dtype=np.float32
    )


    (
        observation,
        reward,
        terminated,
        truncated,
        info
    ) = baseline_env.step(
        action
    )


    baseline_x.append(
        baseline_env.sim.car.x
    )

    baseline_y.append(
        baseline_env.sim.car.y
    )


baseline_time = info["time"]

baseline_env.close()


# =========================================================
# RESULTS
# =========================================================

print("\n")
print("=" * 60)
print(" PPO VS CENTERLINE BASELINE")
print("=" * 60)

print(
    f"PPO lap time: "
    f"{ppo_time:.2f}s"
)

print(
    f"Baseline lap time: "
    f"{baseline_time:.2f}s"
)

difference = (
    baseline_time
    - ppo_time
)

improvement = (
    100.0
    * difference
    / baseline_time
)

print(
    f"Time saved by PPO: "
    f"{difference:.2f}s"
)

print(
    f"Lap-time reduction: "
    f"{improvement:.2f}%"
)


# =========================================================
# PLOT
# =========================================================

plt.figure(
    figsize=(10, 8)
)


plt.plot(
    left_boundary[:, 0],
    left_boundary[:, 1],
    linewidth=1.2,
    label="Track boundary"
)

plt.plot(
    right_boundary[:, 0],
    right_boundary[:, 1],
    linewidth=1.2
)


plt.plot(
    centerline[:, 0],
    centerline[:, 1],
    linestyle="--",
    linewidth=1.1,
    label="Centerline"
)


plt.plot(
    baseline_x,
    baseline_y,
    linewidth=2.0,
    label=(
        f"Centerline baseline "
        f"({baseline_time:.2f}s)"
    )
)


plt.plot(
    ppo_x,
    ppo_y,
    linewidth=2.5,
    label=(
        f"PPO policy "
        f"({ppo_time:.2f}s)"
    )
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
    "PPO Racing Line vs Centerline-Following Baseline"
)

plt.legend()

plt.grid(
    alpha=0.25
)

plt.tight_layout()


output_file = (
    project_dir
    / "results"
    / "ppo_vs_centerline_baseline.png"
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