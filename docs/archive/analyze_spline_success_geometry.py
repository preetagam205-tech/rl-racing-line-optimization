import numpy as np

from src.track_factory import (
    make_random_spline_track,
)


# =========================================================
# CURVATURE STATISTICS
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

    total_length = np.sum(
        segment_length
    )

    return {
        "length":
            total_length,

        "mean_k":
            np.mean(abs_curvature),

        "max_k":
            np.max(abs_curvature),

        "std_k":
            np.std(curvature),

        "signs":
            sign_changes,
    }


# =========================================================
# HELD-OUT SPLINE
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

    n = len(control_points)

    for i in range(n):

        p0 = control_points[
            (i - 1) % n
        ]

        p1 = control_points[i]

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


heldout_points = catmull_rom_closed(
    control_points,
    points_per_segment=60
)


# =========================================================
# RESULTS FROM STAGE 10.15
# =========================================================

results = {
    3001: 50.00,
    3002: 0.00,
    3003: 16.67,
    3004: 75.00,
    3005: 8.33,
}


print("\n")
print("=" * 105)
print(" SPLINE SUCCESS VS GEOMETRY")
print("=" * 105)

print(
    "Track       | Success | Length  | "
    "MeanK  | MaxK   | StdK   | Signs"
)

print("-" * 105)


for seed, success in results.items():

    rng = np.random.default_rng(
        seed
    )

    points = make_random_spline_track(
        rng=rng,
        num_points=720
    )

    stats = geometry_statistics(
        points
    )

    print(
        f"Seed {seed}   | "
        f"{success:6.2f}% | "
        f"{stats['length']:7.2f} | "
        f"{stats['mean_k']:.4f} | "
        f"{stats['max_k']:.4f} | "
        f"{stats['std_k']:.4f} | "
        f"{stats['signs']:5d}"
    )


heldout = geometry_statistics(
    heldout_points
)


print("-" * 105)

print(
    f"HELD-OUT    | "
    f"{0.00:6.2f}% | "
    f"{heldout['length']:7.2f} | "
    f"{heldout['mean_k']:.4f} | "
    f"{heldout['max_k']:.4f} | "
    f"{heldout['std_k']:.4f} | "
    f"{heldout['signs']:5d}"
)