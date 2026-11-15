import numpy as np
import matplotlib.pyplot as plt

from src.track_factory import make_wavy_track
from src.track_geometry import TrackGeometry


# --------------------------------
# CREATE TRACK
# --------------------------------

track = TrackGeometry(
    centerline_points=make_wavy_track(
        num_points=720
    ),
    track_width=10.0,
    num_checkpoints=7
)


# --------------------------------
# HEADING OF EVERY SEGMENT
# --------------------------------

vectors = track.segment_vectors

headings = np.arctan2(
    vectors[:, 1],
    vectors[:, 0]
)


# --------------------------------
# CHANGE IN HEADING
# --------------------------------

heading_change = (
    np.roll(headings, -1)
    - headings
)

heading_change = (
    heading_change
    + np.pi
) % (2 * np.pi) - np.pi


# --------------------------------
# APPROXIMATE CURVATURE
# --------------------------------

segment_length = (
    track.segment_lengths
)

curvature = (
    heading_change
    / np.maximum(
        segment_length,
        1e-9
    )
)


# --------------------------------
# PROGRESS %
# --------------------------------

progress = (
    track.cumulative_distances[:-1]
    / track.total_track_length
    * 100.0
)


# --------------------------------
# PRINT STRONGEST CURVES
# --------------------------------

indices = np.argsort(
    np.abs(curvature)
)[-20:]

indices = indices[
    np.argsort(
        progress[indices]
    )
]


print("\nStrong-curvature locations:")

for i in indices:

    print(
        f"Progress {progress[i]:6.2f}% | "
        f"Curvature {curvature[i]: .4f}"
    )


# --------------------------------
# PLOT
# --------------------------------

plt.figure(
    figsize=(11, 6)
)

plt.plot(
    progress,
    curvature,
    label="Track curvature"
)


# Known PPO failure regions
failure_regions = [
    3.5,
    29.5,
    49.0,
    71.0,
    81.0
]

for failure_progress in failure_regions:

    plt.axvline(
        failure_progress,
        linestyle="--"
    )


plt.xlabel(
    "Track progress (%)"
)

plt.ylabel(
    "Approximate curvature (rad/m)"
)

plt.title(
    "Wavy Track Curvature and PPO Failure Regions"
)

plt.grid()

plt.legend()

plt.show()