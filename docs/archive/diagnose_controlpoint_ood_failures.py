from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from src.racing_env import RacingEnv
from src.track_geometry import TrackGeometry


project_dir = Path(__file__).resolve().parent


# --------------------------------
# MODEL
# --------------------------------

model = PPO.load(
    project_dir
    / "models"
    / "ppo_hardfivefamily_stage1"
)

print("Model loaded!")


# --------------------------------
# CATMULL-ROM SPLINE
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


env = RacingEnv(
    track=track,
    random_start=False
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


print("\n")
print("=" * 150)
print(" CONTROL-POINT OOD FAILURE DIAGNOSIS")
print("=" * 150)

print(
    "Start | EndProg | Speed | LatErr | HeadErr | "
    "H10 | H20 | H30 | H40 | H50 | H60 | "
    "Authority | Steer | Throttle | Lap"
)


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

    final_action = np.array(
        [0.0, 0.0],
        dtype=np.float32
    )


    while not (
        terminated
        or truncated
    ):

        action, _ = model.predict(
            observation,
            deterministic=True
        )

        final_action = action.copy()

        (
            next_observation,
            reward,
            terminated,
            truncated,
            info
        ) = env.step(action)

        observation = next_observation


    print(
        f"{start_index:5d} | "
        f"{info['progress']:7.2f}% | "
        f"{info['speed']:5.2f} | "
        f"{observation[1]:6.3f} | "
        f"{observation[2]:7.3f} | "
        f"{observation[3]:5.3f} | "
        f"{observation[4]:5.3f} | "
        f"{observation[5]:5.3f} | "
        f"{observation[6]:5.3f} | "
        f"{observation[7]:5.3f} | "
        f"{observation[8]:5.3f} | "
        f"{observation[9]:9.3f} | "
        f"{float(final_action[0]):6.3f} | "
        f"{float(final_action[1]):8.3f} | "
        f"{info['lap_completed']}"
    )


env.close()