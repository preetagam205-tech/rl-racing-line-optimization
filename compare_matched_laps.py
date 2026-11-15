
import csv
from pathlib import Path
from statistics import mean

path = Path(__file__).resolve().parent / (
    "baseline_comparison_tuned/baseline_comparison_runs.csv"
)

with path.open(newline="") as file:
    rows = list(csv.DictReader(file))

families = [
    "ellipse", "wavy", "complex",
    "spline", "hard_spline"
]

baseline_name = "Centerline Pure Pursuit"

ppo_models = [
    "Broad Domain PPO",
    "Hard Five Family PPO",
]

lookup = {
    (r["family"], r["trial"], r["model"]): r
    for r in rows
}

print("\nMATCHED SUCCESSFUL LAP COMPARISON\n")

for family in families:
    trials = sorted({
        r["trial"]
        for r in rows
        if r["family"] == family
    })

    print(f"\n{family.upper()}")

    for model in ppo_models:
        pairs = []

        for trial in trials:
            baseline = lookup.get(
                (family, trial, baseline_name)
            )

            ppo = lookup.get(
                (family, trial, model)
            )

            if baseline is None or ppo is None:
                continue

            if (
                int(baseline["completed"]) == 1
                and int(ppo["completed"]) == 1
            ):
                pairs.append((baseline, ppo))

        if not pairs:
            print(f"  {model}: No matched successful laps")
            continue

        baseline_times = [
            float(b["lap_seconds"])
            for b, p in pairs
        ]

        ppo_times = [
            float(p["lap_seconds"])
            for b, p in pairs
        ]

        time_improvements = [
            100 * (
                float(b["lap_seconds"])
                - float(p["lap_seconds"])
            ) / float(b["lap_seconds"])
            for b, p in pairs
        ]

        path_differences = [
            100 * (
                float(p["distance_driven"])
                - float(b["distance_driven"])
            ) / float(b["distance_driven"])
            for b, p in pairs
        ]

        print(f"  {model}")
        print(f"    Matched laps: {len(pairs)}")
        print(
            f"    Baseline lap: "
            f"{mean(baseline_times):.3f}s"
        )
        print(
            f"    PPO lap: "
            f"{mean(ppo_times):.3f}s"
        )
        print(
            f"    Mean time improvement: "
            f"{mean(time_improvements):.2f}%"
        )
        print(
            f"    Mean path length difference: "
            f"{mean(path_differences):+.2f}%"
        )
