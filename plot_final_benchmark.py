from pathlib import Path
import csv

import matplotlib.pyplot as plt


project_dir = Path(__file__).resolve().parent

results_file = (
    project_dir
    / "results"
    / "final_benchmark.csv"
)

output_file = (
    project_dir
    / "results"
    / "final_success_rates.png"
)


# =========================================================
# LOAD RESULTS
# =========================================================

families = []
success_rates = []


with open(
    results_file,
    "r"
) as file:

    reader = csv.DictReader(
        file
    )

    for row in reader:

        families.append(
            row["Track family"]
        )

        success_rates.append(
            float(
                row["Success rate (%)"]
            )
        )


print("Loaded results:")

for family, rate in zip(
    families,
    success_rates
):

    print(
        f"{family:16s} | "
        f"{rate:6.2f}%"
    )


# =========================================================
# SUCCESS-RATE FIGURE
# =========================================================

plt.figure(
    figsize=(10, 6)
)


bars = plt.bar(
    families,
    success_rates
)


plt.ylabel(
    "Lap completion rate (%)"
)

plt.xlabel(
    "Track family"
)

plt.title(
    "Final PPO Policy Generalization Across Track Families"
)


plt.ylim(
    0,
    110
)


plt.xticks(
    rotation=20
)


# =========================================================
# ADD VALUES ABOVE BARS
# =========================================================

for bar, value in zip(
    bars,
    success_rates
):

    plt.text(
        bar.get_x()
        + bar.get_width() / 2,
        value + 2,
        f"{value:.1f}%",
        ha="center",
        va="bottom"
    )


# =========================================================
# SAVE FIGURE
# =========================================================

plt.tight_layout()


plt.savefig(
    output_file,
    dpi=300
)


plt.show()


print(
    "\nSaved figure to:"
)

print(
    output_file
)