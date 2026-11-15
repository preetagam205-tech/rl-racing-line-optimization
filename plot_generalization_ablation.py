from pathlib import Path
import csv

import matplotlib.pyplot as plt


project_dir = Path(__file__).resolve().parent

results_file = (
    project_dir
    / "results"
    / "heldout_generalization.csv"
)

output_file = (
    project_dir
    / "results"
    / "heldout_generalization_ablation.png"
)


# =========================================================
# LOAD RESULTS
# =========================================================

models = []
success_rates = []


with open(
    results_file,
    "r"
) as file:

    reader = csv.DictReader(file)

    for row in reader:

        models.append(
            row["Model"]
        )

        success_rates.append(
            float(
                row[
                    "Held-out success rate (%)"
                ]
            )
        )


print("Loaded results:")

for model, rate in zip(
    models,
    success_rates
):

    print(
        f"{model:24s} | "
        f"{rate:6.2f}%"
    )


# =========================================================
# PLOT
# =========================================================

plt.figure(
    figsize=(10, 6)
)


bars = plt.bar(
    models,
    success_rates
)


plt.ylabel(
    "Held-out spline lap completion rate (%)"
)

plt.xlabel(
    "Model / training configuration"
)

plt.title(
    "Effect of Training Diversity and Observation Design "
    "on OOD Generalization"
)


plt.ylim(
    0,
    32
)


plt.xticks(
    rotation=20
)


# Add values above bars
for bar, value in zip(
    bars,
    success_rates
):

    plt.text(
        bar.get_x()
        + bar.get_width() / 2,
        value + 0.8,
        f"{value:.1f}%",
        ha="center",
        va="bottom"
    )


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