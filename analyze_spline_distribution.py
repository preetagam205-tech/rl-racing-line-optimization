import numpy as np

from src.track_factory import (
    make_random_spline_track,
)


# =========================================================
# GEOMETRY STATISTICS
# =========================================================

def geometry_statistics(points):

    previous_points = np.roll(
        points,
        1,
        axis=0
    )

    next_points = np.roll(
        points,
        -1,
        axis=0
    )

    incoming = (
        points
        - previous_points
    )

    outgoing = (
        next_points
        - points
    )

    heading_in = np.arctan2(
        incoming[:, 1],
        incoming[:, 0]
    )

    heading_out = np.arctan2(
        outgoing[:, 1],
        outgoing[:, 0]
    )

    delta = (
        heading_out
        - heading_in
    )

    delta = (
        delta
        + np.pi
    ) % (
        2.0 * np.pi
    ) - np.pi

    segment_length = np.linalg.norm(
        outgoing,
        axis=1
    )

    curvature = (
        delta
        / np.maximum(
            segment_length,
            1e-12
        )
    )

    abs_curvature = np.abs(
        curvature
    )

    signs = np.sign(
        curvature
    )

    sign_changes = np.sum(
        signs[:-1]
        != signs[1:]
    )

    return (
        np.mean(abs_curvature),
        np.max(abs_curvature),
        np.std(curvature),
        sign_changes
    )


# =========================================================
# SAMPLE MANY TRAINING-FAMILY TRACKS
# =========================================================

rng = np.random.default_rng(
    12345
)

num_tracks = 500


mean_values = []
max_values = []
std_values = []
sign_values = []


for _ in range(num_tracks):

    points = make_random_spline_track(
        rng=rng,
        num_points=720
    )

    (
        mean_k,
        max_k,
        std_k,
        signs
    ) = geometry_statistics(
        points
    )

    mean_values.append(mean_k)
    max_values.append(max_k)
    std_values.append(std_k)
    sign_values.append(signs)


mean_values = np.asarray(
    mean_values
)

max_values = np.asarray(
    max_values
)

std_values = np.asarray(
    std_values
)

sign_values = np.asarray(
    sign_values
)


# =========================================================
# HELD-OUT VALUES
# =========================================================

heldout = {
    "MeanK": 0.0403,
    "MaxK": 0.1981,
    "StdK": 0.0488,
    "Signs": 12,
}


def percentile_position(
    values,
    target
):

    return (
        100.0
        * np.mean(
            values <= target
        )
    )


# =========================================================
# OUTPUT
# =========================================================

print("\n")
print("=" * 90)
print(" RANDOM SPLINE TRAINING DISTRIBUTION")
print("=" * 90)


for name, values, target in [
    (
        "MeanK",
        mean_values,
        heldout["MeanK"]
    ),
    (
        "MaxK",
        max_values,
        heldout["MaxK"]
    ),
    (
        "StdK",
        std_values,
        heldout["StdK"]
    ),
    (
        "Signs",
        sign_values,
        heldout["Signs"]
    ),
]:

    print(f"\n{name}")

    print(
        f"Training mean: "
        f"{np.mean(values):.4f}"
    )

    print(
        f"Training median: "
        f"{np.median(values):.4f}"
    )

    print(
        f"90th percentile: "
        f"{np.percentile(values, 90):.4f}"
    )

    print(
        f"95th percentile: "
        f"{np.percentile(values, 95):.4f}"
    )

    print(
        f"99th percentile: "
        f"{np.percentile(values, 99):.4f}"
    )

    print(
        f"Held-out value: "
        f"{target:.4f}"
    )

    print(
        f"Held-out percentile position: "
        f"{percentile_position(values, target):.2f}%"
    )


# =========================================================
# JOINT DIFFICULTY
# =========================================================

hard_tracks = (
    (max_values >= heldout["MaxK"])
    & (std_values >= heldout["StdK"])
    & (sign_values >= heldout["Signs"])
)


print("\n")
print("=" * 90)
print(" JOINT DIFFICULTY")
print("=" * 90)

print(
    "Training tracks as difficult or "
    "harder on all three metrics:",
    int(np.sum(hard_tracks)),
    "/",
    num_tracks
)

print(
    "Percentage:",
    f"{100.0 * np.mean(hard_tracks):.2f}%"
)