from collections import Counter

from src.racing_env import RacingEnv


env = RacingEnv(
    random_start=True,
    hard_broad_tracks=True
)

family_counts = Counter()


for i in range(50):

    observation, info = env.reset(
        seed=4000 + i
    )

    family = info[
        "track_family"
    ]

    family_counts[
        family
    ] += 1

    print(
        f"Episode {i + 1:2d} | "
        f"Family: {family:11s} | "
        f"Start: {info['start_index']:3d} | "
        f"Obs: {observation.shape}"
    )


print("\n")
print("============================")
print(" FIVE-FAMILY COUNTS")
print("============================")

for family, count in sorted(
    family_counts.items()
):

    print(
        f"{family:11s}: "
        f"{count}"
    )


print(
    "\nObservation valid:",
    env.observation_space.contains(
        observation
    )
)

env.close()