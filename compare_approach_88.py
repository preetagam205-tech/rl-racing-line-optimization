from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from src.racing_env import RacingEnv
from src.track_geometry import TrackGeometry
from src.track_factory import make_wavy_track


# --------------------------------
# MODEL
# --------------------------------

project_dir = Path(__file__).resolve().parent

model = PPO.load(
    project_dir
    / "models"
    / "ppo_mixedtracks_stage1"
)


# --------------------------------
# TRACK
# --------------------------------

track = TrackGeometry(
    centerline_points=make_wavy_track(
        num_points=720
    ),
    track_width=12.0,
    num_checkpoints=7
)


# --------------------------------
# TWO APPROACHES
# --------------------------------

test_starts = [
    ("FAIL", 0),
    ("SUCCESS", 320),
]


for label, start_index in test_starts:

    env = RacingEnv(
        track=track,
        random_start=False
    )

    observation, info = env.reset(
        options={
            "start_index": start_index
        }
    )

    terminated = False
    truncated = False

    printed_counter = 0

    print("\n")
    print("=" * 90)
    print(
        f"{label} APPROACH — "
        f"START INDEX {start_index}"
    )
    print("=" * 90)

    print(
        "Prog | Speed | LatErr | HeadErr | "
        "L10 | L20 | L30 | Authority | "
        "Steer | Throttle"
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

        progress = info["progress"]

        # Only inspect the region leading into
        # the problematic ~88% section.
        if 80.0 <= progress <= 89.0:

            printed_counter += 1

            # Print every 4th step to keep
            # the output readable.
            if printed_counter % 4 == 0:

                print(
                    f"{progress:5.2f} | "
                    f"{info['speed']:5.2f} | "
                    f"{observation[1]:6.3f} | "
                    f"{observation[2]:7.3f} | "
                    f"{observation[3]:5.3f} | "
                    f"{observation[4]:5.3f} | "
                    f"{observation[5]:5.3f} | "
                    f"{observation[6]:9.3f} | "
                    f"{float(action[0]):6.3f} | "
                    f"{float(action[1]):8.3f}"
                )

        observation = next_observation

    print("\nFINAL")

    print(
        f"End progress: "
        f"{info['progress']:.2f}%"
    )

    print(
        f"On track: "
        f"{info['on_track']}"
    )

    print(
        f"Lap: "
        f"{info['lap_completed']}"
    )

    env.close()