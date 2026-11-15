from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from src.racing_env import RacingEnv
from src.track_geometry import TrackGeometry


project_dir = Path(__file__).resolve().parent


# =========================================================
# MODEL
# =========================================================

model = PPO.load(
    project_dir
    / "models"
    / "ppo_hardfivefamily_stage1"
)

print("Model loaded!")


# =========================================================
# HELD-OUT TRACK
# =========================================================

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

        p1 = control_points[i]

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
                2.0 * p1
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


# =========================================================
# CENTERLINE CURVATURE
# =========================================================

previous_points = np.roll(
    points,
    1,
    axis=0
)

next_points = np.roll(
    points,
    -1,
    axis=0
)

incoming = (
    points
    - previous_points
)

outgoing = (
    next_points
    - points
)

heading_in = np.arctan2(
    incoming[:, 1],
    incoming[:, 0]
)

heading_out = np.arctan2(
    outgoing[:, 1],
    outgoing[:, 0]
)

heading_change = (
    heading_out
    - heading_in
)

heading_change = (
    heading_change
    + np.pi
) % (
    2.0 * np.pi
) - np.pi

segment_length = np.linalg.norm(
    outgoing,
    axis=1
)

curvature = (
    heading_change
    / np.maximum(
        segment_length,
        1e-12
    )
)


# Arc-length progress for each track point.
cumulative_distance = np.concatenate(
    (
        [0.0],
        np.cumsum(
            segment_length[:-1]
        )
    )
)

curve_progress = (
    100.0
    * cumulative_distance
    / np.sum(segment_length)
)


# =========================================================
# THEORETICAL GRIP SPEED
# =========================================================

max_lateral_acceleration = 8.0
max_vehicle_speed = 20.0


def safe_speed_at_progress(
    target_progress
):

    index = np.argmin(
        np.abs(
            curve_progress
            - target_progress
        )
    )

    kappa = abs(
        curvature[index]
    )

    if kappa < 1e-8:

        safe_speed = (
            max_vehicle_speed
        )

    else:

        safe_speed = np.sqrt(
            max_lateral_acceleration
            / kappa
        )

        safe_speed = min(
            safe_speed,
            max_vehicle_speed
        )

    return (
        curvature[index],
        safe_speed
    )


# =========================================================
# COMPARE TWO TRAJECTORIES
# =========================================================

def evaluate_approach(
    start_index,
    name
):

    env = RacingEnv(
        track=track,
        random_start=False
    )

    observation, info = env.reset(
        options={
            "start_index":
                start_index
        }
    )

    targets = list(
        range(
            35,
            51
        )
    )

    target_position = 0

    terminated = False
    truncated = False

    previous_progress = info.get(
    "progress",
    info["start_progress"]
)


    print("\n")
    print("=" * 120)
    print(name)
    print("=" * 120)

    print(
        print(
    "Prog | Curv    | SafeV | Speed | "
    "LatErr | H10   | H20   | H30   | "
    "H40   | H50   | H60   | Throttle"
)
    )


    while not (
        terminated
        or truncated
    ):

        action, _ = model.predict(
            observation,
            deterministic=True
        )

        (
            next_observation,
            reward,
            terminated,
            truncated,
            info
        ) = env.step(action)

        current_progress = (
            info["progress"]
        )


        while (
            target_position
            < len(targets)
        ):

            target = targets[
                target_position
            ]

            crossed = (
                previous_progress
                < target
                <= current_progress
            )

            if not crossed:
                break

            (
                kappa,
                safe_speed
            ) = safe_speed_at_progress(
                target
            )

            print(
                print(
                f"{target:4d}% | "
                f"{kappa:+.4f} | "
                f"{safe_speed:5.2f} | "
                f"{info['speed']:5.2f} | "
                f"{next_observation[1]:6.3f} | "
                f"{next_observation[3]:5.3f} | "
                f"{next_observation[4]:5.3f} | "
                f"{next_observation[5]:5.3f} | "
                f"{next_observation[6]:5.3f} | "
                f"{next_observation[7]:5.3f} | "
                f"{next_observation[8]:5.3f} | "
                f"{float(action[1]):8.3f}"
            )
            )

            target_position += 1


        previous_progress = (
            current_progress
        )

        observation = (
            next_observation
        )


    print(
        "\nLap completed:",
        info["lap_completed"]
    )

    print(
        "Final progress:",
        f"{info['progress']:.2f}%"
    )

    env.close()



evaluate_approach(
    start_index=0,
    name="FAILED TRAJECTORY — CORNER VISIBILITY"
)