from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from src.racing_env import RacingEnv
from src.track_geometry import TrackGeometry
from src.track_factory import make_random_spline_track


project_dir = Path(__file__).resolve().parent


# =========================================================
# LOAD MODEL
# =========================================================

model = PPO.load(
    project_dir
    / "models"
    / "ppo_fourfamily_stage1"
)

print("Model loaded!")


# =========================================================
# TEST SETTINGS
# =========================================================

test_seeds = [
    3001,
    3002,
    3003,
    3004,
    3005,
]

starts_per_track = 12


all_success_rates = []
all_lap_times = []


# =========================================================
# EVALUATE
# =========================================================

for seed in test_seeds:

    rng = np.random.default_rng(
        seed
    )

    points = make_random_spline_track(
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

    start_indices = np.linspace(
        0,
        track.num_points - 1,
        starts_per_track,
        dtype=int
    )

    successful_laps = 0
    lap_times = []


    print("\n")
    print(
        "=" * 80
    )

    print(
        f"SPLINE TRACK SEED {seed}"
    )

    print(
        f"Track length: "
        f"{track.total_track_length:.2f} m"
    )

    print(
        "=" * 80
    )


    for start_index in start_indices:

        observation, info = env.reset(
            options={
                "start_index":
                    int(start_index)
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

            all_lap_times.append(
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


    success_rate = (
        100.0
        * successful_laps
        / starts_per_track
    )

    all_success_rates.append(
        success_rate
    )


    print("\n")
    print(
        f"Track success: "
        f"{successful_laps}/"
        f"{starts_per_track} "
        f"({success_rate:.2f}%)"
    )


    if lap_times:

        print(
            f"Average successful lap: "
            f"{np.mean(lap_times):.2f}s"
        )


    env.close()


# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n")
print("=" * 80)
print(" UNSEEN SPLINE FAMILY SUMMARY")
print("=" * 80)

print(
    f"Average track success rate: "
    f"{np.mean(all_success_rates):.2f}%"
)

print(
    f"Minimum track success rate: "
    f"{np.min(all_success_rates):.2f}%"
)

print(
    f"Maximum track success rate: "
    f"{np.max(all_success_rates):.2f}%"
)


if all_lap_times:

    print(
        f"Average successful lap time: "
        f"{np.mean(all_lap_times):.2f}s"
    )

    print(
        f"Best successful lap time: "
        f"{np.min(all_lap_times):.2f}s"
    )