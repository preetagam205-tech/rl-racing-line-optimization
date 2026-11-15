from pathlib import Path

from stable_baselines3 import PPO

from src.racing_env import RacingEnv
from src.track_geometry import TrackGeometry


project_dir = Path(__file__).resolve().parent

model = PPO.load(
    project_dir
    / "models"
    / "ppo_randomstart_stage1"
)


test_tracks = [
    ("Flat Oval - SUCCESS", 50.0, 15.0),
    ("Round Oval - FAILURE", 35.0, 25.0),
    ("Very Flat - FAILURE", 60.0, 15.0),
]


# Use the normal starting position first
start_index = 0


for name, a, b in test_tracks:

    print("\n")
    print("=" * 50)
    print(name)
    print("=" * 50)

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

    observation, info = env.reset(
        options={
            "start_index": start_index
        }
    )

    terminated = False
    truncated = False

    step_number = 0

    max_progress = 0.0

    while not (
        terminated
        or truncated
    ):

        action, _ = model.predict(
            observation,
            deterministic=True
        )

        observation, reward, terminated, truncated, info = (
            env.step(action)
        )

        step_number += 1

        max_progress = max(
            max_progress,
            info["progress"]
        )

        # Print every 20 simulation steps
        if step_number % 20 == 0:

            print(
                f"Step {step_number:4d} | "
                f"Progress {info['progress']:6.2f}% | "
                f"Speed {info['speed']:5.2f} | "
                f"Steer {float(action[0]):6.3f} | "
                f"Throttle {float(action[1]):6.3f} | "
                f"CP {env.sim.checkpoints_passed_count}/7 | "
                f"On track {info['on_track']}"
            )


    print("\nFINAL")

    print(
        f"Progress: "
        f"{info['progress']:.2f}%"
    )

    print(
        f"Max progress: "
        f"{max_progress:.2f}%"
    )

    print(
        f"Speed: "
        f"{info['speed']:.2f}"
    )

    print(
        f"Checkpoints: "
        f"{env.sim.checkpoints_passed_count}/7"
    )

    print(
        f"Lap completed: "
        f"{info['lap_completed']}"
    )

    print(
        f"On track: "
        f"{info['on_track']}"
    )

    print(
        f"Episode time: "
        f"{info['time']:.2f} s"
    )

    env.close()