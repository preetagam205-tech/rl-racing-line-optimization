from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from src.racing_env import RacingEnv
from src.track_geometry import TrackGeometry
from src.track_factory import make_random_wavy_track


project_dir = Path(__file__).resolve().parent


# --------------------------------
# MODEL
# --------------------------------

model = PPO.load(
    project_dir
    / "models"
    / "ppo_nonlinear_laptime_stage1"
)

print("Model loaded!")


# --------------------------------
# UNSEEN TRACK SEEDS
# --------------------------------

track_seeds = [
    1001,
    1002,
    1003,
    1004,
    1005,
]


all_success_rates = []
all_average_times = []


for seed in track_seeds:

    rng = np.random.default_rng(seed)

    points = make_random_wavy_track(
        rng=rng,
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

    # 12 evenly spaced starts
    start_indices = list(
        range(0, 720, 60)
    )

    lap_times = []
    successful_laps = 0


    print("\n")
    print("=" * 70)
    print(
        f"UNSEEN WAVY TRACK — SEED {seed}"
    )
    print("=" * 70)

    print(
        f"Track length: "
        f"{track.total_track_length:.2f} m"
    )


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


    success_rate = (
        100.0
        * successful_laps
        / len(start_indices)
    )

    all_success_rates.append(
        success_rate
    )


    print("\nTRACK SUMMARY")

    print(
        f"Laps: "
        f"{successful_laps}/"
        f"{len(start_indices)}"
    )

    print(
        f"Success rate: "
        f"{success_rate:.2f}%"
    )


    if lap_times:

        average_time = np.mean(
            lap_times
        )

        all_average_times.append(
            average_time
        )

        print(
            f"Average successful lap: "
            f"{average_time:.2f}s"
        )

        print(
            f"Best successful lap: "
            f"{min(lap_times):.2f}s"
        )

    env.close()


print("\n")
print("=" * 70)
print("OVERALL UNSEEN WAVY GENERALIZATION")
print("=" * 70)

print(
    f"Tracks tested: "
    f"{len(track_seeds)}"
)

print(
    f"Average track success rate: "
    f"{np.mean(all_success_rates):.2f}%"
)

if all_average_times:

    print(
        f"Average successful lap time "
        f"across tracks: "
        f"{np.mean(all_average_times):.2f}s"
    )