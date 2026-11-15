
"""Compare PPO models on the same five track families."""

import argparse
import csv
from pathlib import Path
from statistics import mean

from stable_baselines3 import PPO
from benchmark_ppo import FAMILIES, evaluate

MODELS = {
    "visual_100k":
        "models/visual_training/ppo_visual_100000.zip",
    "visual_250k":
        "models/visual_training/ppo_visual_250000.zip",
    "visual_500k":
        "models/visual_training/ppo_visual_500000.zip",
    "multitrack":
        "models/ppo_multitrack_stage1.zip",
    "mixedtracks":
        "models/ppo_mixedtracks_stage1.zip",
    "fourfamily":
        "models/ppo_fourfamily_stage1.zip",
    "broaddomain":
        "models/ppo_broaddomain_stage1.zip",
    "hardfivefamily":
        "models/ppo_hardfivefamily_stage1.zip",
}


def save_csv(filename, records):
    if not records:
        return

    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(records[0])
        )
        writer.writeheader()
        writer.writerows(records)

    print(f"Saved {filename}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    trials = 1 if args.quick else args.trials

    if trials < 1:
        parser.error("Trials must be positive")

    root = Path(__file__).resolve().parent
    results = []

    for name, relative_path in MODELS.items():
        path = root / relative_path

        if not path.is_file():
            print(f"SKIP {name}: missing {path}")
            continue

        try:
            model = PPO.load(str(path), device="cpu")
        except Exception as exc:
            print(f"SKIP {name}: {exc}")
            continue

        obs_shape = tuple(model.observation_space.shape)
        action_shape = tuple(model.action_space.shape)

        print(
            f"{name}: observations={obs_shape}, "
            f"actions={action_shape}",
            flush=True,
        )

        if (
            obs_shape not in ((10,), (16,))
            or action_shape != (2,)
        ):
            print("SKIP: incompatible model spaces")
            continue

        for family in FAMILIES:
            for trial in range(trials):
                result = evaluate(
                    model, family, trial
                )

                row = {
                    "model": name,
                    "family": family,
                    "trial": trial + 1,
                    **result,
                }

                results.append(row)

                print(
                    f"{family:<12} "
                    f"test {trial + 1}/{trials} "
                    f"progress="
                    f"{result['progress_percent']:6.1f}% "
                    f"lap="
                    f"{'YES' if result['completed'] else 'NO'}",
                    flush=True,
                )

    if not results:
        raise SystemExit(
            "No compatible models were evaluated."
        )

    save_csv(
        root / "ppo_multitrack_model_runs.csv",
        results,
    )

    summary = []

    for name in MODELS:
        for family in FAMILIES:
            group = [
                r for r in results
                if r["model"] == name
                and r["family"] == family
            ]

            if not group:
                continue

            lap_times = [
                float(r["lap_seconds"])
                for r in group
                if r["completed"]
            ]

            summary.append({
                "model": name,
                "family": family,
                "trials": len(group),
                "completion_rate_percent": round(
                    100 * mean(
                        r["completed"]
                        for r in group
                    ), 1
                ),
                "mean_progress_percent": round(
                    mean(
                        r["progress_percent"]
                        for r in group
                    ), 2
                ),
                "mean_successful_lap_seconds": (
                    round(mean(lap_times), 3)
                    if lap_times else ""
                ),
                "mean_steering_change": round(
                    mean(
                        r["mean_steering_change"]
                        for r in group
                    ), 5
                ),
                "mean_abs_lateral_error": round(
                    mean(
                        r["mean_abs_lateral_error"]
                        for r in group
                    ), 3
                ),
            })

    save_csv(
        root / "ppo_multitrack_model_summary.csv",
        summary,
    )

    print(
        "\nNON-ELLIPSE PERFORMANCE:"
    )

    rankings = []

    for name in MODELS:
        group = [
            r for r in results
            if r["model"] == name
            and r["family"] != "ellipse"
        ]

        if group:
            rankings.append((
                name,
                100 * mean(
                    r["completed"]
                    for r in group
                ),
                mean(
                    r["progress_percent"]
                    for r in group
                ),
                len(group),
            ))

    rankings.sort(
        key=lambda r: (r[1], r[2]),
        reverse=True,
    )

    for name, success, progress, count in rankings:
        print(
            f"{name:<18} "
            f"completion={success:5.1f}% "
            f"mean_progress={progress:6.2f}% "
            f"runs={count}"
        )


if __name__ == "__main__":
    main()
