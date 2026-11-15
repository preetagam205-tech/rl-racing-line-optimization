import numpy as np

from src.track_factory import (
    make_random_complex_track
)

from src.track_geometry import TrackGeometry


rng = np.random.default_rng(
    42
)


for i in range(5):

    points = (
        make_random_complex_track(
            rng=rng,
            num_points=720
        )
    )

    track = TrackGeometry(
        centerline_points=points,
        track_width=12.0,
        num_checkpoints=7
    )

    print(
        f"Track {i + 1} | "
        f"Points {len(points)} | "
        f"Length "
        f"{track.total_track_length:.2f} m"
    )