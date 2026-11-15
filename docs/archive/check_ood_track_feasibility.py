import numpy as np


# --------------------------------
# VEHICLE PARAMETERS
# --------------------------------

wheelbase = 2.5

max_steering = np.radians(
    30.0
)

mechanical_max_curvature = (
    np.tan(max_steering)
    / wheelbase
)


# --------------------------------
# CONTROL-POINT TRACK
# --------------------------------

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


points = catmull_rom_closed(
    control_points,
    points_per_segment=60
)


# --------------------------------
# CURVATURE
# --------------------------------

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

abs_curvature = np.abs(
    curvature
)


# --------------------------------
# FEASIBILITY
# --------------------------------

too_sharp = (
    abs_curvature
    > mechanical_max_curvature
)

num_too_sharp = np.sum(
    too_sharp
)

percentage_too_sharp = (
    100.0
    * num_too_sharp
    / len(points)
)


print("\n")
print("=" * 80)
print(" OOD TRACK PHYSICAL FEASIBILITY")
print("=" * 80)

print(
    f"Mechanical max curvature: "
    f"{mechanical_max_curvature:.4f} 1/m"
)

print(
    f"Track maximum curvature: "
    f"{np.max(abs_curvature):.4f} 1/m"
)

print(
    f"Minimum track radius: "
    f"{1.0 / np.max(abs_curvature):.2f} m"
)

print(
    f"Minimum vehicle turning radius: "
    f"{1.0 / mechanical_max_curvature:.2f} m"
)

print(
    f"Points exceeding steering limit: "
    f"{num_too_sharp}/{len(points)}"
)

print(
    f"Percentage physically too sharp: "
    f"{percentage_too_sharp:.2f}%"
)


print("\n")
print("=" * 80)
print(" WORST CURVATURE LOCATIONS")
print("=" * 80)


worst_indices = np.argsort(
    abs_curvature
)[-15:][::-1]


for index in worst_indices:

    progress = (
        100.0
        * index
        / len(points)
    )

    required_steering = np.degrees(
        np.arctan(
            wheelbase
            * abs_curvature[index]
        )
    )

    print(
        f"Progress {progress:6.2f}% | "
        f"Curvature "
        f"{abs_curvature[index]:.4f} | "
        f"Required steer "
        f"{required_steering:6.2f}°"
    )