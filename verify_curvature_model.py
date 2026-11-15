from pathlib import Path

from stable_baselines3 import PPO


project_dir = Path(__file__).resolve().parent


model = PPO.load(
    project_dir
    / "models"
    / "ppo_curvature_stage1"
)


print(
    "Model observation space:",
    model.observation_space
)

print(
    "Observation shape:",
    model.observation_space.shape
)

print(
    "Policy input dimension:",
    model.policy.features_extractor.features_dim
)