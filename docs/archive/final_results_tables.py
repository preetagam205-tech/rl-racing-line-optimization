from pathlib import Path
import csv


project_dir = Path(__file__).resolve().parent

results_dir = (
    project_dir
    / "results"
)

results_dir.mkdir(
    exist_ok=True
)


# =========================================================
# FINAL MODEL BENCHMARK
# =========================================================

final_benchmark = [
    [
        "Ellipse",
        60,
        60,
        100.00,
        17.84,
        65.23,
    ],
    [
        "Wavy",
        60,
        60,
        100.00,
        17.66,
        69.03,
    ],
    [
        "Complex",
        48,
        60,
        80.00,
        17.83,
        69.90,
    ],
    [
        "Spline",
        46,
        60,
        76.67,
        18.33,
        71.47,
    ],
    [
        "Hard spline",
        17,
        60,
        28.33,
        18.66,
        70.76,
    ],
    [
        "Held-out spline",
        9,
        36,
        25.00,
        18.31,
        74.95,
    ],
]


with open(
    results_dir
    / "final_benchmark.csv",
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow(
        [
            "Track family",
            "Successful laps",
            "Attempts",
            "Success rate (%)",
            "Average successful lap time (s)",
            "Average efficiency (%)",
        ]
    )

    writer.writerows(
        final_benchmark
    )


# =========================================================
# FIXED WAVY ABLATIONS
# =========================================================

wavy_ablations = [
    [
        "Long lookahead",
        "H10-H60 observation",
        61.11,
        15.98,
    ],
    [
        "Linear lap-time reward",
        "Edge penalty + linear terminal reward",
        100.00,
        16.83,
    ],
    [
        "Fast progress reward",
        "Added local speed-progress bonus",
        36.11,
        16.89,
    ],
    [
        "Nonlinear lap-time reward",
        "Edge penalty + 300*efficiency^4",
        100.00,
        16.43,
    ],
    [
        "Edge penalty only",
        "No lap-time efficiency bonus",
        100.00,
        17.09,
    ],
    [
        "95% edge threshold",
        "Weaker edge constraint + nonlinear reward",
        100.00,
        16.93,
    ],
]


with open(
    results_dir
    / "fixed_wavy_ablation.csv",
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow(
        [
            "Experiment",
            "Main change",
            "Success rate (%)",
            "Average successful lap time (s)",
        ]
    )

    writer.writerows(
        wavy_ablations
    )


# =========================================================
# HELD-OUT SPLINE GENERALIZATION
# =========================================================

generalization = [
    [
        "Broad-domain",
        "Ellipse + wavy + complex",
        10,
        2.78,
        16.60,
    ],
    [
        "Four-family",
        "+ regular spline",
        10,
        0.00,
        "",
    ],
    [
        "Hard five-family",
        "+ targeted hard spline",
        10,
        25.00,
        18.31,
    ],
    [
        "Curvature observation",
        "Hard five-family + six curvature inputs",
        16,
        5.56,
        18.28,
    ],
]


with open(
    results_dir
    / "heldout_generalization.csv",
    "w",
    newline=""
) as file:

    writer = csv.writer(file)

    writer.writerow(
        [
            "Model",
            "Training / observation change",
            "Observation dimensions",
            "Held-out success rate (%)",
            "Average successful lap time (s)",
        ]
    )

    writer.writerows(
        generalization
    )


print(
    "Saved:"
)

print(
    results_dir
    / "final_benchmark.csv"
)

print(
    results_dir
    / "fixed_wavy_ablation.csv"
)

print(
    results_dir
    / "heldout_generalization.csv"
)