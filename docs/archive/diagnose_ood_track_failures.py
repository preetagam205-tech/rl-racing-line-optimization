from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from src.racing_env import RacingEnv
from src.track_geometry import TrackGeometry


# --------------------------------
# PROJECT
# --------------------------------

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
# SAME OOD TRACK FROM STAGE 10.1
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


points = make_ood_track(
    num_points=720
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


print(
    "OOD track length:",
    track.total_track_length
)


# --------------------------------
# TEST SAME 36 START POSITIONS
# --------------------------------

start_indices = list(
    range(0, 720, 20)
)


print("\n")
print("=" * 120)
print(" OOD TRACK FAILURE LOCATIONS")
print("=" * 120)


for start_index in start_indices:

    observation, reset_info = env.reset(
        options={
            "start_index": start_index
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
            observation,
            reward,
            terminated,
            truncated,
            info
        ) = env.step(action)


    checkpoints = (
        env.sim.checkpoints_passed_count
    )


    print(
        f"Start {start_index:3d} | "
        f"StartProg "
        f"{reset_info['start_progress']:6.2f}% | "
        f"EndProg "
        f"{info['progress']:6.2f}% | "
        f"Speed "
        f"{info['speed']:5.2f} | "
        f"Steer "
        f"{float(final_action[0]):6.3f} | "
        f"Throttle "
        f"{float(final_action[1]):6.3f} | "
        f"CP "
        f"{checkpoints}/7 | "
        f"OnTrack "
        f"{info['on_track']} | "
        f"Lap "
        f"{info['lap_completed']}"
    )


env.close()