from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from src.racing_env import RacingEnv


# -------------------------
# PATHS
# -------------------------

project_dir = Path(__file__).resolve().parent

models_dir = project_dir / "models"

old_model_path = (
    models_dir / "ppo_physics_stage1"
)

new_model_path = (
    models_dir / "ppo_physics_stage2"
)


# -------------------------
# ENVIRONMENT
# -------------------------

env = RacingEnv()
env = Monitor(env)


# -------------------------
# LOAD STAGE 2
# -------------------------

model = PPO.load(
    old_model_path,
    env=env
)

print("\nLoaded ppo_physics_stage2.")
print("Continuing training...\n")


# -------------------------
# CONTINUE TRAINING
# -------------------------

model.learn(
    total_timesteps=700_000,
    reset_num_timesteps=False
)


# -------------------------
# SAVE STAGE 3
# -------------------------

model.save(new_model_path)

print("\nTraining complete!")

print(
    f"Model saved to: {new_model_path}"
)

env.close()