from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from src.racing_env import RacingEnv
from src.track_geometry import TrackGeometry


project_dir = Path(__file__).resolve().parent


# --------------------------------
# LOAD BROAD-DOMAIN MODEL
# --------------------------------

model = PPO.load(
    project_dir
    / "models"
    / "ppo_curvature_stage1"
)

print("Model loaded!")


# --------------------------------
# CATMULL-ROM CLOSED SPLINE
# --------------------------------

def catmull_rom_closed(
    control_points,
    points_per_segment=60
):

    control_points = np.asarray(
        control_points,
        dtype=np.float64
    )

    generated = []

    n = len(control_points)

    for i in range(n):

        p0 = control_points[
            (i - 1) % n
        ]

        p1 = control_points[
            i % n
        ]

        p2 = control_points[
            (i + 1) % n
        ]

        p3 = control_points[
            (i + 2) % n
        ]

        for j in range(
            points_per_segment
        ):

            t = (
                j
                / points_per_segment
            )

            t2 = t * t
            t3 = t2 * t

            point = 0.5 * (
                (2.0 * p1)
                + (-p0 + p2) * t
                + (
                    2.0 * p0
                    - 5.0 * p1
                    + 4.0 * p2
                    - p3
                ) * t2
                + (
                    -p0
                    + 3.0 * p1
                    - 3.0 * p2
                    + p3
                ) * t3
            )

            generated.append(
                point
            )

    return np.asarray(
        generated
    )


# --------------------------------
# COMPLETELY DIFFERENT TRACK SHAPE
# --------------------------------

control_points = np.array(
    [
        [48.00,   0.00],
        [38.11,  17.60],
        [26.00,  36.03],
        [0.00,   32.00],
        [-23.50, 32.56],
        [-37.24, 17.20],
        [-50.00,  0.00],
        [-35.51, -16.40],
        [-24.50, -33.95],
        [0.00,   -36.00],
        [26.50,  -36.72],
        [36.37,  -16.80],
    ],
    dtype=np.float64
)


points = catmull_rom_closed(
    control_points,
    points_per_segment=60
)


track = TrackGeometry(
    centerline_points=points,
    track_width=12.0,
    num_checkpoints=7
)


env = RacingEnv(
    track=track,
    random_start=False,
    curvature_observation=True
)


print(
    "Control-point OOD track length:",
    track.total_track_length
)


# --------------------------------
# TEST 36 STARTS
# --------------------------------

start_indices = np.linspace(
    0,
    track.num_points - 1,
    36,
    dtype=int
)


successful_laps = 0
lap_times = []


for start_index in start_indices:

    observation, info = env.reset(
        options={
            "start_index": int(
                start_index
            )
        }
    )

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


    if info["lap_completed"]:

        successful_laps += 1

        lap_times.append(
            info["time"]
        )


    print(
        f"Start {start_index:3d} | "
        f"EndProg "
        f"{info['progress']:6.2f}% | "
        f"Time "
        f"{info['time']:6.2f}s | "
        f"Lap "
        f"{info['lap_completed']}"
    )


print("\n")
print("====================================")
print(" CONTROL-POINT OOD EVALUATION")
print("====================================")

print(
    f"Laps completed: "
    f"{successful_laps}/"
    f"{len(start_indices)}"
)

success_rate = (
    100.0
    * successful_laps
    / len(start_indices)
)

print(
    f"Success rate: "
    f"{success_rate:.2f}%"
)


if lap_times:

    print(
        f"Average successful lap time: "
        f"{np.mean(lap_times):.2f}s"
    )

    print(
        f"Best successful lap time: "
        f"{min(lap_times):.2f}s"
    )

    print(
        f"Worst successful lap time: "
        f"{max(lap_times):.2f}s"
    )


env.close()