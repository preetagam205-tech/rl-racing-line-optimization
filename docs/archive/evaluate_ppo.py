from pathlib import Path

import numpy as np
import csv

from stable_baselines3 import PPO

from src.racing_env import RacingEnv


# -------------------------
# MODEL PATH
# -------------------------

project_dir = Path(__file__).resolve().parent

model_path = (
    project_dir
    / "models"
    / "ppo_physics_stage2"
)


# -------------------------
# LOAD ENVIRONMENT + MODEL
# -------------------------

env = RacingEnv()

model = PPO.load(
    model_path
)

print("PPO model loaded successfully!")


# -------------------------
# EVALUATION SETTINGS
# -------------------------

num_episodes = 1

episode_rewards = []
episode_lengths = []
episode_progress = []
episode_checkpoints = []

laps_completed = 0


# -------------------------
# RUN EVALUATION
# -------------------------

for episode in range(1, num_episodes + 1):

    observation, info = env.reset()

    trajectory = []

    debug_data = []

    terminated = False
    truncated = False

    total_reward = 0.0
    steps = 0

    max_progress = 0.0
    max_checkpoint_passed = 0

    completed_lap = False

    while not (terminated or truncated):

        # PPO chooses the action
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
        steps += 1

        trajectory.append({
            "step": steps,
            "x": env.sim.car.x,
            "y": env.sim.car.y,
            "speed": info["speed"],
            "progress": info["progress"],
            "steering": float(action[0]),
            "throttle": float(action[1])
        })

        debug_data.append({
        "step": steps,
        "progress": info["progress"],
        "speed": info["speed"],
        "steering": float(action[0]),
        "throttle": float(action[1]),
        "on_track": info["on_track"]
    })


        # Furthest progress reached
        max_progress = max(
            max_progress,
            info["progress"]
        )

        # next_checkpoint = 1 means
        # zero checkpoints completed
        checkpoint_passed = (
            info["next_checkpoint"] - 1
        )

        max_checkpoint_passed = max(
            max_checkpoint_passed,
            checkpoint_passed
        )

        if info["lap_completed"]:
            completed_lap = True
            laps_completed += 1

    if completed_lap:

        results_dir = (
            project_dir / "results"
        )

        results_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        trajectory_path = (
            results_dir
            / "ppo_physics_stage2_trajectory.csv"
        )

        with open(
            trajectory_path,
            "w",
            newline=""
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=[
                    "step",
                    "x",
                    "y",
                    "speed",
                    "progress",
                    "steering",
                    "throttle"
                ]
            )

            writer.writeheader()

            writer.writerows(
                trajectory
            )

        print(
            f"\nTrajectory saved to:"
            f" {trajectory_path}"
        )

    # Save episode results
    episode_rewards.append(total_reward)
    episode_lengths.append(steps)
    episode_progress.append(max_progress)
    episode_checkpoints.append(
        max_checkpoint_passed
    )

    print("\nLast 20 steps before episode ended:")

    for row in debug_data[-20:]:
        print(
            f"Step {row['step']:3d} | "
            f"Progress {row['progress']:6.2f}% | "
            f"Speed {row['speed']:5.2f} | "
            f"Steer {row['steering']:6.3f} | "
            f"Throttle {row['throttle']:6.3f} | "
            f"On track {row['on_track']}"
        )

    print(
        f"\nEpisode {episode}"
    )

    print(
        f"Reward: {total_reward:.2f}"
    )

    print(
        f"Steps survived: {steps}"
    )

    print(
        f"Maximum progress: {max_progress:.2f}%"
    )

    print(
        f"Checkpoints passed: "
        f"{max_checkpoint_passed}/7"
    )

    print(
        f"Lap completed: {completed_lap}"
    )
    print(
    f"Final speed: {info['speed']:.2f}"
    )

    print(
    f"Final progress: {info['progress']:.2f}%"
    )

    print(
    f"On track at end: {info['on_track']}"
    )

    print(
    f"Episode time: {info['time']:.2f} s"
    )


# -------------------------
# SUMMARY
# -------------------------

print("\n==============================")
print("      PPO EVALUATION")
print("==============================")

print(
    f"Episodes: {num_episodes}"
)

print(
    f"Average reward: "
    f"{np.mean(episode_rewards):.2f}"
)

print(
    f"Average episode length: "
    f"{np.mean(episode_lengths):.1f} steps"
)

print(
    f"Average max progress: "
    f"{np.mean(episode_progress):.2f}%"
)

print(
    f"Best progress: "
    f"{np.max(episode_progress):.2f}%"
)

print(
    f"Best checkpoint reached: "
    f"{np.max(episode_checkpoints):.0f}/7"
)

print(
    f"Laps completed: "
    f"{laps_completed}/{num_episodes}"
)

env.close()