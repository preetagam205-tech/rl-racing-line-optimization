import numpy as np

from src.track_factory import (
    make_random_spline_track,
)

from src.track_geometry import (
    TrackGeometry,
)


def curvature_statistics(
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


rng = np.random.default_rng(
    42
)


for i in range(10):

    points = (
        make_random_spline_track(
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
        mean_abs,
        max_abs,
        curvature_std,
        sign_changes
    ) = curvature_statistics(
        points
    )

    print(
        f"Track {i + 1:2d} | "
        f"Length {track.total_track_length:7.2f} m | "
        f"MeanK {mean_abs:.4f} | "
        f"MaxK {max_abs:.4f} | "
        f"StdK {curvature_std:.4f} | "
        f"Signs {sign_changes}"
    )