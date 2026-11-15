import numpy as np

from src.track_factory import (
    make_random_hard_spline_track,
    _track_curvature_statistics,
)

from src.track_geometry import (
    TrackGeometry,
)


rng = np.random.default_rng(
    777
)


for i in range(10):

    points = (
        make_random_hard_spline_track(
            rng=rng,
            num_points=720
        )
    )

    track = TrackGeometry(
        centerline_points=points,
        track_width=12.0,
        num_checkpoints=7
    )

    (
        mean_k,
        max_k,
        std_k,
        signs
    ) = (
        _track_curvature_statistics(
            points
        )
    )

    print(
        f"Track {i + 1:2d} | "
        f"Length {track.total_track_length:7.2f} m | "
        f"MeanK {mean_k:.4f} | "
        f"MaxK {max_k:.4f} | "
        f"StdK {std_k:.4f} | "
        f"Signs {signs}"
    )