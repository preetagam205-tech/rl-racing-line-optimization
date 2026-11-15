import numpy as np

from src.racing_env import RacingEnv


env = RacingEnv()

observation, info = env.reset()

# Test settings
total_steps = 5000

episode_count = 0
terminated_count = 0
truncated_count = 0

total_reward = 0.0


for step_number in range(total_steps):

    # Random steering + throttle/brake
    action = env.action_space.sample()

    (
        observation,
        reward,
        terminated,
        truncated,
        info
    ) = env.step(action)

    # -------------------------
    # SAFETY CHECKS
    # -------------------------

    if not np.all(np.isfinite(observation)):
        raise ValueError(
            f"Invalid observation at step {step_number}: "
            f"{observation}"
        )

    if not np.isfinite(reward):
        raise ValueError(
            f"Invalid reward at step {step_number}: "
            f"{reward}"
        )

    if not env.observation_space.contains(
        observation
    ):
        raise ValueError(
            f"Observation outside space at step "
            f"{step_number}: {observation}"
        )

    total_reward += reward

    # -------------------------
    # EPISODE FINISHED
    # -------------------------

    if terminated or truncated:

        episode_count += 1

        if terminated:
            terminated_count += 1

        if truncated:
            truncated_count += 1

        observation, info = env.reset()


print("\n--- RANDOM ENVIRONMENT TEST ---")

print(
    f"Total simulation steps: {total_steps}"
)

print(
    f"Episodes completed: {episode_count}"
)

print(
    f"Terminated episodes: {terminated_count}"
)

print(
    f"Truncated episodes: {truncated_count}"
)

print(
    f"Total reward: {total_reward:.2f}"
)

print("\nRandom-action stress test passed!")