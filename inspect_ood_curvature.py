import numpy as np

from src.track_geometry import TrackGeometry


def make_ood_track(
    num_points=720
):

    theta = np.linspace(
        0.0,
        2.0 * np.pi,
        num_points,
        endpoint=False
    )

    radius = (
        39.0
        + 4.0 * np.sin(
            4.0 * theta + 0.4
        )
        + 2.5 * np.cos(
            5.0 * theta - 0.7
        )
    )

    x = (
        radius
        * np.cos(theta)
    )

    y = (
        0.78
        * radius
        * np.sin(theta)
    )

    return np.column_stack(
        (x, y)
    )


track = TrackGeometry(
    centerline_points=make_ood_track(),
    track_width=12.0,
    num_checkpoints=7
)


points = track.centerline


# --------------------------------
# TANGENTS
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


# --------------------------------
# SIGNED HEADING CHANGE
# --------------------------------

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


signed_curvature = (
    heading_change
    / np.maximum(
        segment_length,
        1e-12
    )
)


# --------------------------------
# FAILURE REGIONS
# --------------------------------

regions = [
    (6.0, 10.0),
    (63.0, 67.0),
    (86.0, 92.0),
]


print("\n")
print("=" * 80)
print(" OOD SIGNED CURVATURE ANALYSIS")
print("=" * 80)


for start_percent, end_percent in regions:

    start_index = int(
        start_percent
        / 100.0
        * track.num_points
    )

    end_index = int(
        end_percent
        / 100.0
        * track.num_points
    )

    values = signed_curvature[
        start_index:end_index + 1
    ]

    sign_changes = np.sum(
        np.sign(values[:-1])
        != np.sign(values[1:])
    )

    print(
        f"\nREGION "
        f"{start_percent:.1f}%"
        f"–{end_percent:.1f}%"
    )

    print(
        f"Minimum signed curvature: "
        f"{np.min(values):.4f} rad/m"
    )

    print(
        f"Maximum signed curvature: "
        f"{np.max(values):.4f} rad/m"
    )

    print(
        f"Maximum absolute curvature: "
        f"{np.max(np.abs(values)):.4f} rad/m"
    )

    print(
        f"Curvature sign changes: "
        f"{sign_changes}"
    )


# --------------------------------
# SAMPLE MAIN FAILURE REGION
# --------------------------------

print("\n")
print("=" * 80)
print(" MAIN FAILURE REGION — 84% TO 94%")
print("=" * 80)

for progress in np.arange(
    84.0,
    94.1,
    1.0
):

    index = int(
        progress
        / 100.0
        * track.num_points
    )

    print(
        f"Progress {progress:5.1f}% | "
        f"Signed curvature "
        f"{signed_curvature[index]: .4f} rad/m"
    )