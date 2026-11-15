from src.racing_env import RacingEnv


# -------------------------
# OLD OBSERVATION
# -------------------------

old_env = RacingEnv(
    random_start=True,
    hard_broad_tracks=True,
    curvature_observation=False
)

old_obs, old_info = old_env.reset(
    seed=42
)

print(
    "Old observation shape:",
    old_obs.shape
)

print(
    "Old observation valid:",
    old_env.observation_space.contains(
        old_obs
    )
)

old_env.close()


# -------------------------
# NEW OBSERVATION
# -------------------------

new_env = RacingEnv(
    random_start=True,
    hard_broad_tracks=True,
    curvature_observation=True
)

new_obs, new_info = new_env.reset(
    seed=42
)

print(
    "New observation shape:",
    new_obs.shape
)

print(
    "New observation valid:",
    new_env.observation_space.contains(
        new_obs
    )
)

print(
    "\nNew observation:"
)

print(new_obs)

new_env.close()