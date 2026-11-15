from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from src.racing_env import RacingEnv
from src.track_geometry import TrackGeometry


# --------------------------------
# MODEL
# --------------------------------

project_dir = Path(__file__).resolve().parent

model_path = (
    project_dir
    / "models"
    / "ppo_nonlinear_laptime_stage1"
)

model = PPO.load(model_path)

print("Model loaded!")


# --------------------------------
# UNSEEN TRACKS
# --------------------------------

test_tracks = [
    ("Wide Oval", 45.0, 18.0),
    ("Flat Oval", 50.0, 15.0),
    ("Round Oval", 35.0, 25.0),
    ("Long Oval", 55.0, 18.0),
    ("Very Flat Oval", 60.0, 15.0),
]


# 36 starting positions around each track
start_indices = list(
    range(0, 720, 20)
)


all_track_results = []


# --------------------------------
# TEST EACH TRACK
# --------------------------------

for track_name, a, b in test_tracks:

    print("\n")
    print("================================")
    print(f" TESTING: {track_name}")
    print("================================")

    track = TrackGeometry(
        a=a,
        b=b,
        track_width=12.0,
        num_points=720,
        num_checkpoints=7
    )

    env = RacingEnv(
        track=track,
        random_start=False
    )

    results = []


    # --------------------------------
    # TEST ALL START POSITIONS
    # --------------------------------

    for start_index in start_indices:

        observation, reset_info = env.reset(
            options={
                "start_index": start_index
            }
        )

        terminated = False
        truncated = False

        total_reward = 0.0

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

            total_reward += reward


            # -------------------------
            # RELATIVE PROGRESS
            # -------------------------

            current_progress = (
                info["progress"]
            )

            progress_delta = (
                current_progress
                - previous_progress
            )

            # Handle 100 -> 0 wrap
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


        checkpoints_passed = (
            env.sim.checkpoints_passed_count
        )


        results.append(
            {
                "start_index": start_index,
                "progress": max_relative_progress,
                "checkpoints": checkpoints_passed,
                "lap": lap_completed,
                "reward": total_reward
            }
        )


        print(
            f"Start {start_index:3d} | "
            f"Progress "
            f"{max_relative_progress:6.2f}% | "
            f"CP {checkpoints_passed}/7 | "
            f"Lap {lap_completed}"
        )


    # --------------------------------
    # TRACK SUMMARY
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


    print("\n------------------------------")
    print(f"RESULT: {track_name}")
    print("------------------------------")

    print(
        f"a={a}, b={b}"
    )

    print(
        f"Track length: "
        f"{track.total_track_length:.2f}"
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


    all_track_results.append(
        {
            "name": track_name,
            "a": a,
            "b": b,
            "progress": average_progress,
            "checkpoints": average_checkpoints,
            "laps": successful_laps
        }
    )

    env.close()


# --------------------------------
# FINAL SUMMARY
# --------------------------------

print("\n")
print("========================================")
print(" MULTI-TRACK GENERALIZATION SUMMARY")
print("========================================")

for result in all_track_results:

    print(
        f"{result['name']:16s} | "
        f"Progress "
        f"{result['progress']:6.2f}% | "
        f"CP "
        f"{result['checkpoints']:.2f}/7 | "
        f"Laps "
        f"{result['laps']}/36"
    )


total_laps = sum(
    result["laps"]
    for result in all_track_results
)

total_tests = (
    len(test_tracks)
    * len(start_indices)
)

print("\n----------------------------------------")

print(
    f"Total successful laps: "
    f"{total_laps}/{total_tests}"
)

print(
    f"Overall success rate: "
    f"{100.0 * total_laps / total_tests:.2f}%"
)