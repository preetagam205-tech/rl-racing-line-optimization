from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from src.racing_env import RacingEnv


# -------------------------
# PATHS
# -------------------------

project_dir = Path(__file__).resolve().parent

model_path = (
    project_dir
    / "models"
    / "ppo_randomstart_stage1"
)


# -------------------------
# ENVIRONMENT + MODEL
# -------------------------

env = RacingEnv(
    random_start=False
)

model = PPO.load(
    model_path
)

print("Random-start PPO model loaded!")


# -------------------------
# FIXED TEST STARTS
# -------------------------

start_indices = list(
    range(
        0,
        720,
        20
    )
)

results = []


# -------------------------
# EVALUATE
# -------------------------

for start_index in start_indices:

    observation, reset_info = env.reset(
        options={
            "start_index": start_index
        }
    )

    terminated = False
    truncated = False

    steps = 0
    total_reward = 0.0

    start_progress = (
        reset_info["start_progress"]
    )

    previous_progress = start_progress

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

        steps += 1
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

        # Handle 100% -> 0% wrap
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


    # Number of checkpoints actually passed
    checkpoints_passed = (
        env.sim.checkpoints_passed_count
    )


    result = {
        "start_index": start_index,
        "start_progress": start_progress,
        "relative_progress":
            max_relative_progress,
        "checkpoints":
            checkpoints_passed,
        "lap":
            lap_completed,
        "steps":
            steps,
        "time":
            info["time"],
        "reward":
            total_reward
    }

    results.append(result)


    print("\n------------------------------")

    print(
        f"Start index: {start_index}"
    )

    print(
        f"Start progress: "
        f"{start_progress:.2f}%"
    )

    print(
        f"Progress travelled: "
        f"{max_relative_progress:.2f}%"
    )

    print(
        f"Checkpoints passed: "
        f"{checkpoints_passed}/7"
    )

    print(
        f"Lap completed: "
        f"{lap_completed}"
    )

    print(
        f"Episode time: "
        f"{info['time']:.2f} s"
    )

    print(
        f"Reward: "
        f"{total_reward:.2f}"
    )


# -------------------------
# SUMMARY
# -------------------------

successful_laps = sum(
    result["lap"]
    for result in results
)

average_progress = np.mean(
    [
        result["relative_progress"]
        for result in results
    ]
)

average_checkpoints = np.mean(
    [
        result["checkpoints"]
        for result in results
    ]
)


print("\n==============================")
print(" RANDOM-START EVALUATION")
print("==============================")

print(
    f"Starts tested: "
    f"{len(start_indices)}"
)

print(
    f"Average progress travelled: "
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

env.close()