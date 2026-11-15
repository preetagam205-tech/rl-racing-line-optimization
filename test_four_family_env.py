from collections import Counter

from src.racing_env import RacingEnv


env = RacingEnv(
    random_start=True,
    broad_tracks=True
)

family_counts = Counter()


for i in range(40):

    observation, info = env.reset(
        seed=2000 + i
    )

    family = info[
        "track_family"
    ]

    family_counts[
        family
    ] += 1

    print(
        f"Episode {i + 1:2d} | "
        f"Family: {family:8s} | "
        f"Start: {info['start_index']:3d} | "
        f"Obs: {observation.shape}"
    )


print("\n")
print("============================")
print(" FOUR-FAMILY COUNTS")
print("============================")

for family, count in sorted(
    family_counts.items()
):

    print(
        f"{family:8s}: "
        f"{count}"
    )


print(
    "\nObservation valid:",
    env.observation_space.contains(
        observation
    )
)

env.close()