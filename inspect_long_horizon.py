import numpy as np

from src.track_factory import make_wavy_track
from src.track_geometry import TrackGeometry


track = TrackGeometry(
    centerline_points=make_wavy_track(
        num_points=720
    ),
    track_width=12.0,
    num_checkpoints=7
)


progress_targets = [
    60.0,
    65.0,
    70.0,
    72.0,
    74.0,
    76.0,
    78.0,
    80.0,
]


print("\n")
print("=" * 100)
print(" LONG-HORIZON TRACK INFORMATION")
print("=" * 100)

print(
    "Prog | "
    "H30 | H40 | H50 | H60 | "
    "O30 | O40 | O50 | O60"
)


for target_progress in progress_targets:

    target_distance = (
        target_progress
        / 100.0
        * track.total_track_length
    )

    index = (
        np.searchsorted(
            track.cumulative_distances,
            target_distance,
            side="right"
        )
        - 1
    )

    index = int(
        np.clip(
            index,
            0,
            track.num_points - 1
        )
    )

    x = track.centerline[index, 0]
    y = track.centerline[index, 1]


    headings = (
        track.get_lookahead_heading_changes(
            x,
            y,
            distances=(
                30.0,
                40.0,
                50.0,
                60.0
            )
        )
    )


    offsets = (
        track.get_lookahead_lateral_offsets(
            x,
            y,
            distances=(
                30.0,
                40.0,
                50.0,
                60.0
            )
        )
    )


    print(
        f"{target_progress:5.1f} | "
        f"{headings[0]:5.2f} | "
        f"{headings[1]:5.2f} | "
        f"{headings[2]:5.2f} | "
        f"{headings[3]:5.2f} | "
        f"{offsets[0]:6.2f} | "
        f"{offsets[1]:6.2f} | "
        f"{offsets[2]:6.2f} | "
        f"{offsets[3]:6.2f}"
    )