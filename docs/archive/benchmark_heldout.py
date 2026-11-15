
"""Evaluate the strongest PPO models on new test circuits."""

import argparse
import csv
from pathlib import Path
from statistics import mean

from stable_baselines3 import PPO
from benchmark_ppo import FAMILIES, evaluate

ROOT = Path(__file__).resolve().parent

MODELS = {
    "broaddomain": "models/ppo_broaddomain_stage1.zip",
    "hardfivefamily": "models/ppo_hardfivefamily_stage1.zip",
    "fourfamily": "models/ppo_fourfamily_stage1.zip",
}

# Previous evaluation used trial IDs 0-4.
# These IDs generate different, reproducible test circuits.
TRIAL_OFFSET = 10000


def write_csv(filename, rows):
    if not rows:
        return

    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0])
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=20)
    args = parser.parse_args()

    if args.trials < 1:
        parser.error("Trials must be positive")

    results = []

    for name, filename in MODELS.items():
        path = ROOT / filename
        model = PPO.load(str(path), device="cpu")

        print(f"\nEvaluating {name}")

        for family in FAMILIES:
            for trial in range(args.trials):
                trial_id = TRIAL_OFFSET + trial

                result = evaluate(
                    model, family, trial_id
                )

                results.append({
                    "model": name,
                    "family": family,
                    "trial": trial_id,
                    **result,
                })

                print(
                    f"{name} | {family} | "
                    f"Trial {trial+1}/{args.trials} | "
                    f"Progress: {result['progress_percent']:.1f}% | "
                    f"Lap: {bool(result['completed'])}",
                    flush=True,
                )

    write_csv(
        ROOT / "ppo_heldout_runs.csv",
        results,
    )

    summary = []

    for name in MODELS:
        for family in FAMILIES:
            group = [
                row for row in results
                if row["model"] == name
                and row["family"] == family
            ]

            successful = [
                float(row["lap_seconds"])
                for row in group
                if row["completed"]
            ]

            summary.append({
                "model": name,
                "family": family,
                "trials": len(group),

                "completion_rate_percent": round(
                    100 * mean(
                        row["completed"] for row in group
                    ), 2
                ),

                "mean_progress_percent": round(
                    mean(
                        row["progress_percent"]
                        for row in group
                    ), 2
                ),

                "mean_lap_seconds": (
                    round(mean(successful), 3)
                    if successful else ""
                ),

                "mean_steering_change": round(
                    mean(
                        row["mean_steering_change"]
                        for row in group
                    ), 5
                ),

                "mean_abs_lateral_error": round(
                    mean(
                        row["mean_abs_lateral_error"]
                        for row in group
                    ), 3
                ),
            })

    write_csv(
        ROOT / "ppo_heldout_summary.csv",
        summary,
    )

    print("\nOVERALL RANKING")

    ranking = []

    for name in MODELS:
        group = [
            row for row in results
            if row["model"] == name
        ]

        completion = 100 * mean(
            row["completed"] for row in group
        )

        progress = mean(
            row["progress_percent"] for row in group
        )

        ranking.append((name, completion, progress))

    for name, completion, progress in sorted(
        ranking, key=lambda x: (x[1], x[2]),
        reverse=True
    ):
        print(
            f"{name}: "
            f"Completion={completion:.1f}% "
            f"Progress={progress:.1f}%"
        )

    print("\nResults saved successfully.")


if __name__ == "__main__":
    main()
