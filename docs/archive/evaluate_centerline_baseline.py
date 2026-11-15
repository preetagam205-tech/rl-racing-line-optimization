from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

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


env = RacingEnv(
    track=track,
    random_start=False,
    curvature_observation=False
)


# =========================================================
# HELPERS
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
# RESET
# =========================================================

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

speed_values = [
    env.sim.car.speed
]

terminated = False
truncated = False


# =========================================================
# CENTERLINE CONTROLLER
# =========================================================

while not (
    terminated
    or truncated
):

    car = env.sim.car

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


    # ---------------------------------
    # STEERING
    # ---------------------------------

    steering_input = np.clip(
        1.8
        * heading_error
        / car.max_steering,
        -1.0,
        1.0
    )


    # ---------------------------------
    # SPEED CONTROL
    # ---------------------------------

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


    # Simple curvature-aware speed target
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
    ) = env.step(action)


    trajectory_x.append(
        env.sim.car.x
    )

    trajectory_y.append(
        env.sim.car.y
    )

    speed_values.append(
        env.sim.car.speed
    )


# =========================================================
# RESULTS
# =========================================================

print("\n")
print("=" * 60)
print(" CENTERLINE BASELINE")
print("=" * 60)

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


print("\nPPO reference:")
print(
    "Lap completed: True"
)

print(
    "Lap time: 16.45s"
)


if info["lap_completed"]:

    difference = (
        info["time"]
        - 16.45
    )

    print(
        "\nBaseline - PPO difference:",
        f"{difference:+.2f}s"
    )


# =========================================================
# TRAJECTORY PLOT
# =========================================================

trajectory_x = np.asarray(
    trajectory_x
)

trajectory_y = np.asarray(
    trajectory_y
)


plt.figure(
    figsize=(10, 8)
)


plt.plot(
    centerline[:, 0],
    centerline[:, 1],
    linestyle="--",
    label="Track centerline"
)


plt.plot(
    trajectory_x,
    trajectory_y,
    linewidth=2.0,
    label="Centerline controller"
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
    "Non-RL Centerline-Following Baseline"
)

plt.legend()

plt.grid(
    alpha=0.25
)

plt.tight_layout()


output_file = (
    project_dir
    / "results"
    / "centerline_baseline.png"
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