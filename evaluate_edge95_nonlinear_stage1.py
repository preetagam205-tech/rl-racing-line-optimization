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

model_path = (
    project_dir
    / "models"
    / "ppo_edge95_nonlinear_stage1"
)

model = PPO.load(model_path)

print("Model loaded!")


# --------------------------------
# WAVY TRACK
# --------------------------------

points = make_wavy_track(
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
    "Wavy track length:",
    track.total_track_length
)


# --------------------------------
# TEST 36 START POSITIONS
# --------------------------------

start_indices = list(
    range(0, 720, 20)
)

results = []
successful_lap_times = []

for start_index in start_indices:

    observation, reset_info = env.reset(
        options={
            "start_index": start_index
        }
    )

    terminated = False
    truncated = False

    previous_progress = (
        reset_info["start_progress"]
    )

    cumulative_progress = 0.0
    max_relative_progress = 0.0

    lap_completed = False

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

        current_progress = (
            info["progress"]
        )

        progress_delta = (
            current_progress
            - previous_progress
        )

        if progress_delta < -50:
            progress_delta += 100

        elif progress_delta > 50:
            progress_delta -= 100

        cumulative_progress += (
            progress_delta
        )

        max_relative_progress = max(
            max_relative_progress,
            cumulative_progress
        )

        previous_progress = (
            current_progress
        )

        if info["lap_completed"]:
            lap_completed = True

    if info["lap_completed"]:
        successful_lap_times.append(
        info["time"]
    )


    checkpoints = (
        env.sim.checkpoints_passed_count
    )


    results.append(
        {
            "start_index": start_index,
            "progress": max_relative_progress,
            "checkpoints": checkpoints,
            "lap": lap_completed
        }
    )


    print(
        f"Start {start_index:3d} | "
        f"Progress {max_relative_progress:6.2f}% | "
        f"CP {checkpoints}/7 | "
        f"Time {info['time']:6.2f}s | "
        f"Lap {lap_completed}"
    )


# --------------------------------
# SUMMARY
# --------------------------------

successful_laps = sum(
    result["lap"]
    for result in results
)

average_progress = np.mean(
    [
        result["progress"]
        for result in results
    ]
)

average_checkpoints = np.mean(
    [
        result["checkpoints"]
        for result in results
    ]
)


print("\n")
print("================================")
print(" WAVY TRACK EVALUATION")
print("================================")

print(
    f"Starts tested: "
    f"{len(start_indices)}"
)

print(
    f"Average progress: "
    f"{average_progress:.2f}%"
)

print(
    f"Average checkpoints: "
    f"{average_checkpoints:.2f}/7"
)

print(
    f"Laps completed: "
    f"{successful_laps}/"
    f"{len(start_indices)}"
)

print(
    f"Success rate: "
    f"{100.0 * successful_laps / len(start_indices):.2f}%"
)

if successful_lap_times:

    average_lap_time = np.mean(
        successful_lap_times
    )

    best_lap_time = min(
        successful_lap_times
    )

    worst_lap_time = max(
        successful_lap_times
    )

    print(
        f"Average successful lap time: "
        f"{average_lap_time:.2f}s"
    )

    print(
        f"Best successful lap time: "
        f"{best_lap_time:.2f}s"
    )

    print(
        f"Worst successful lap time: "
        f"{worst_lap_time:.2f}s"
    )
env.close()