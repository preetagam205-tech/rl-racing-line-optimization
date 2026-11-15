from pathlib import Path

import torch
from stable_baselines3 import PPO


project_dir = Path(__file__).resolve().parent


model_old = PPO.load(
    project_dir
    / "models"
    / "ppo_fourfamily_stage1"
)

model_new = PPO.load(
    project_dir
    / "models"
    / "ppo_hardfivefamily_stage1"
)


old_state = (
    model_old.policy.state_dict()
)

new_state = (
    model_new.policy.state_dict()
)


print("\n")
print("=" * 70)
print(" POLICY PARAMETER COMPARISON")
print("=" * 70)


different_tensors = 0

max_difference = 0.0


for name in old_state:

    old_tensor = old_state[name]
    new_tensor = new_state[name]

    difference = torch.max(
        torch.abs(
            old_tensor
            - new_tensor
        )
    ).item()

    if difference > 0.0:
        different_tensors += 1

    max_difference = max(
        max_difference,
        difference
    )


print(
    "Number of parameter tensors:",
    len(old_state)
)

print(
    "Different parameter tensors:",
    different_tensors
)

print(
    "Maximum parameter difference:",
    max_difference
)


if different_tensors == 0:

    print(
        "\nWARNING: Policies are identical."
    )

else:

    print(
        "\nPolicies are different."
    )