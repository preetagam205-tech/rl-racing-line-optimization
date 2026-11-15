import matplotlib.pyplot as plt

from src.track_geometry import TrackGeometry
from src.track_factory import make_wavy_track


# -------------------------
# CREATE CUSTOM TRACK
# -------------------------

points = make_wavy_track(
    num_points=720
)

track = TrackGeometry(
    centerline_points=points,
    track_width=10.0,
    num_checkpoints=7
)


# -------------------------
# PRINT INFORMATION
# -------------------------

print(
    "Points:",
    track.num_points
)

print(
    "Track length:",
    track.total_track_length
)

print(
    "Checkpoints:",
    len(track.checkpoint_gates)
)


# -------------------------
# PLOT
# -------------------------

plt.figure(
    figsize=(10, 7)
)

plt.plot(
    track.outer_boundary[:, 0],
    track.outer_boundary[:, 1],
    label="Outer boundary"
)

plt.plot(
    track.inner_boundary[:, 0],
    track.inner_boundary[:, 1],
    label="Inner boundary"
)

plt.plot(
    track.centerline[:, 0],
    track.centerline[:, 1],
    "--",
    label="Centerline"
)


# Checkpoint gates
for inner, outer in track.checkpoint_gates:

    plt.plot(
        [inner[0], outer[0]],
        [inner[1], outer[1]]
    )


plt.axis("equal")

plt.xlabel("X")
plt.ylabel("Y")

plt.title(
    "First Non-Elliptical Track"
)

plt.legend()

plt.grid()

plt.show()