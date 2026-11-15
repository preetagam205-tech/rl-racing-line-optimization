from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from src.racing_env import RacingEnv
from src.track_geometry import TrackGeometry


project_dir = Path(__file__).resolve().parent


# --------------------------------
# LOAD CURRENT BEST MODEL
# --------------------------------

model = PPO.load(
    project_dir
    / "models"
    / "ppo_nonlinear_laptime_stage1"
)

print("Model loaded!")


# --------------------------------
# OOD TRACK GENERATOR
# --------------------------------

def make_ood_track(
    num_points=720
):

    theta = np.linspace(
        0.0,
        2.0 * np.pi,
        num_points,
        endpoint=False
    )

    # Different geometry from training.
    #
    # Training wavy tracks mainly used
    # 2nd and 3rd harmonics.
    #
    # This track introduces 4th and
    # 5th harmonic structure.

    radius = (
        39.0
        + 4.0 * np.sin(
            4.0 * theta + 0.4
        )
        + 2.5 * np.cos(
            5.0 * theta - 0.7
        )
    )

    x = (
        radius
        * np.cos(theta)
    )

    y = (
        0.78
        * radius
        * np.sin(theta)
    )

    return np.column_stack(
        (x, y)
    )


points = make_ood_track()

track = TrackGeometry(
    centerline_points=points,
    track_width=12.0,
    num_checkpoints=7
)


env = RacingEnv(
    track=track,
    random_start=False
)


print(
    "OOD track length:",
    track.total_track_length
)


# --------------------------------
# TEST 36 STARTS
# --------------------------------

start_indices = list(
    range(0, 720, 20)
)

successful_laps = 0
lap_times = []


for start_index in start_indices:

    observation, info = env.reset(
        options={
            "start_index": start_index
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
        f"EndProg {info['progress']:6.2f}% | "
        f"Time {info['time']:6.2f}s | "
        f"Lap {info['lap_completed']}"
    )


print("\n")
print("================================")
print(" OOD TRACK EVALUATION")
print("================================")

print(
    f"Laps completed: "
    f"{successful_laps}/"
    f"{len(start_indices)}"
)

print(
    f"Success rate: "
    f"{100.0 * successful_laps / len(start_indices):.2f}%"
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

env.close()