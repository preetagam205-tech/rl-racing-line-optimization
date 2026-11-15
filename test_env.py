from stable_baselines3.common.env_checker import check_env

from src.racing_env import RacingEnv


env = RacingEnv()

print("Checking RacingEnv...")

check_env(
    env,
    warn=True
)

print("Environment check passed!")