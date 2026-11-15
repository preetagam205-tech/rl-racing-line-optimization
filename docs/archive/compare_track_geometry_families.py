import numpy as np

from src.track_factory import (
    make_random_wavy_track,
    make_random_complex_track,
)

from src.track_geometry import TrackGeometry


# =========================================================
# CURVATURE CALCULATION
# =========================================================

def calculate_signed_curvature(
    points
):

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

    incoming_heading = np.arctan2(
        incoming[:, 1],
        incoming[:, 0]
    )

    outgoing_heading = np.arctan2(
        outgoing[:, 1],
        outgoing[:, 0]
    )

    heading_change = (
        outgoing_heading
        - incoming_heading
    )

    heading_change = (
        heading_change
        + np.pi
    ) % (
        2.0 * np.pi
    ) - np.pi

    segment_length = np.linalg.norm(
        outgoing,
        axis=1
    )

    curvature = (
        heading_change
        / np.maximum(
            segment_length,
            1e-12
        )
    )

    return curvature


# =========================================================
# CONTROL-POINT TRACK
# =========================================================

def catmull_rom_closed(
    control_points,
    points_per_segment=60
):

    control_points = np.asarray(
        control_points,
        dtype=np.float64
    )

    generated = []

    n = len(
        control_points
    )

    for i in range(n):

        p0 = control_points[
            (i - 1) % n
        ]

        p1 = control_points[
            i
        ]

        p2 = control_points[
            (i + 1) % n
        ]

        p3 = control_points[
            (i + 2) % n
        ]

        for j in range(
            points_per_segment
        ):

            t = (
                j
                / points_per_segment
            )

            t2 = t * t
            t3 = t2 * t

            point = 0.5 * (
                2.0 * p1
                + (-p0 + p2) * t
                + (
                    2.0 * p0
                    - 5.0 * p1
                    + 4.0 * p2
                    - p3
                ) * t2
                + (
                    -p0
                    + 3.0 * p1
                    - 3.0 * p2
                    + p3
                ) * t3
            )

            generated.append(
                point
            )

    return np.asarray(
        generated
    )


control_points = np.array(
    [
        [48.00,   0.00],
        [38.11,  17.60],
        [26.00,  36.03],
        [0.00,   32.00],
        [-23.50, 32.56],
        [-37.24, 17.20],
        [-50.00,  0.00],
        [-35.51, -16.40],
        [-24.50, -33.95],
        [0.00,   -36.00],
        [26.50,  -36.72],
        [36.37,  -16.80],
    ],
    dtype=np.float64
)


control_track_points = (
    catmull_rom_closed(
        control_points,
        points_per_segment=60
    )
)


# =========================================================
# GENERATE TRAINING-LIKE TRACKS
# =========================================================

rng = np.random.default_rng(
    42
)


families = {
    "wavy": [],
    "complex": [],
}


for _ in range(20):

    families["wavy"].append(
        make_random_wavy_track(
            rng=rng,
            num_points=720
        )
    )

    families["complex"].append(
        make_random_complex_track(
            rng=rng,
            num_points=720
        )
    )


# =========================================================
# STATISTICS
# =========================================================

def geometry_statistics(
    points
):

    curvature = (
        calculate_signed_curvature(
            points
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

    return {
        "mean_abs": np.mean(
            abs_curvature
        ),

        "max_abs": np.max(
            abs_curvature
        ),

        "std": np.std(
            curvature
        ),

        "sign_changes": sign_changes,
    }


print("\n")
print("=" * 90)
print(" TRAINING FAMILY GEOMETRY")
print("=" * 90)


for family_name, tracks in (
    families.items()
):

    mean_values = []
    max_values = []
    std_values = []
    sign_values = []

    for points in tracks:

        stats = (
            geometry_statistics(
                points
            )
        )

        mean_values.append(
            stats["mean_abs"]
        )

        max_values.append(
            stats["max_abs"]
        )

        std_values.append(
            stats["std"]
        )

        sign_values.append(
            stats["sign_changes"]
        )

    print(
        f"\n{family_name.upper()}"
    )

    print(
        f"Mean absolute curvature: "
        f"{np.mean(mean_values):.4f}"
    )

    print(
        f"Average maximum curvature: "
        f"{np.mean(max_values):.4f}"
    )

    print(
        f"Largest maximum curvature: "
        f"{np.max(max_values):.4f}"
    )

    print(
        f"Curvature std: "
        f"{np.mean(std_values):.4f}"
    )

    print(
        f"Average sign changes: "
        f"{np.mean(sign_values):.2f}"
    )


# =========================================================
# CONTROL-POINT OOD STATISTICS
# =========================================================

ood_stats = (
    geometry_statistics(
        control_track_points
    )
)


print("\n")
print("=" * 90)
print(" CONTROL-POINT OOD GEOMETRY")
print("=" * 90)

print(
    f"Mean absolute curvature: "
    f"{ood_stats['mean_abs']:.4f}"
)

print(
    f"Maximum absolute curvature: "
    f"{ood_stats['max_abs']:.4f}"
)

print(
    f"Curvature std: "
    f"{ood_stats['std']:.4f}"
)

print(
    f"Curvature sign changes: "
    f"{ood_stats['sign_changes']}"
)