import numpy as np


def make_wavy_track(
    num_points=720
):

    theta = np.linspace(
        0.0,
        2.0 * np.pi,
        num_points,
        endpoint=False
    )

    # Radius changes around the lap.
    # This creates an asymmetric,
    # non-elliptical closed circuit.
    radius = (
        38.0
        + 6.0 * np.sin(3.0 * theta)
        + 3.0 * np.cos(2.0 * theta)
    )

    x = (
        radius
        * np.cos(theta)
    )

    y = (
        0.70
        * radius
        * np.sin(theta)
    )

    centerline = np.column_stack(
        (x, y)
    )

    return centerline

def make_random_wavy_track(
    rng,
    num_points=720
):

    theta = np.linspace(
        0.0,
        2.0 * np.pi,
        num_points,
        endpoint=False
    )

    # Overall track size
    base_radius = rng.uniform(
        34.0,
        44.0
    )

    # Different amounts of waviness
    amplitude_2 = rng.uniform(
        1.0,
        4.0
    )

    amplitude_3 = rng.uniform(
        2.0,
        7.0
    )

    # Move the waves around the circuit
    phase_2 = rng.uniform(
        0.0,
        2.0 * np.pi
    )

    phase_3 = rng.uniform(
        0.0,
        2.0 * np.pi
    )

    # Change how stretched the track is
    y_scale = rng.uniform(
        0.65,
        0.90
    )

    radius = (
        base_radius
        + amplitude_2
        * np.cos(
            2.0 * theta
            + phase_2
        )
        + amplitude_3
        * np.sin(
            3.0 * theta
            + phase_3
        )
    )

    x = (
        radius
        * np.cos(theta)
    )

    y = (
        y_scale
        * radius
        * np.sin(theta)
    )

    return np.column_stack(
        (x, y)
    )

def make_random_complex_track(
    rng,
    num_points=720
):

    theta = np.linspace(
        0.0,
        2.0 * np.pi,
        num_points,
        endpoint=False
    )

    base_radius = rng.uniform(
        36.0,
        44.0
    )

    amplitude_2 = rng.uniform(
        1.0,
        3.0
    )

    amplitude_3 = rng.uniform(
        1.0,
        4.0
    )

    amplitude_4 = rng.uniform(
        1.0,
        4.0
    )

    amplitude_5 = rng.uniform(
        0.5,
        3.0
    )

    phase_2 = rng.uniform(
        0.0,
        2.0 * np.pi
    )

    phase_3 = rng.uniform(
        0.0,
        2.0 * np.pi
    )

    phase_4 = rng.uniform(
        0.0,
        2.0 * np.pi
    )

    phase_5 = rng.uniform(
        0.0,
        2.0 * np.pi
    )

    y_scale = rng.uniform(
        0.70,
        0.90
    )

    radius = (
        base_radius
        + amplitude_2
        * np.cos(
            2.0 * theta
            + phase_2
        )
        + amplitude_3
        * np.sin(
            3.0 * theta
            + phase_3
        )
        + amplitude_4
        * np.sin(
            4.0 * theta
            + phase_4
        )
        + amplitude_5
        * np.cos(
            5.0 * theta
            + phase_5
        )
    )

    x = (
        radius
        * np.cos(theta)
    )

    y = (
        y_scale
        * radius
        * np.sin(theta)
    )

    return np.column_stack(
        (x, y)
    )

def _catmull_rom_closed(
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


def _maximum_track_curvature(
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

    return float(
        np.max(
            np.abs(curvature)
        )
    )


def make_random_spline_track(
    rng,
    num_points=720,
    max_curvature=0.20
):
    num_control_points = 12

    points_per_segment = (
        num_points
        // num_control_points
    )

    # Try several random tracks until
    # we obtain one that is physically
    # reasonable for the vehicle.
    for _ in range(100):

        theta = np.linspace(
            0.0,
            2.0 * np.pi,
            num_control_points,
            endpoint=False
        )

        # Random global orientation.
        theta = (
            theta
            + rng.uniform(
                0.0,
                2.0 * np.pi
            )
        )

        x_radius = rng.uniform(
            38.0,
            52.0
        )

        y_radius = rng.uniform(
            26.0,
            40.0
        )

        radial_variation = rng.uniform(
            -7.0,
            7.0,
            size=num_control_points
        )

        # Alternating variation deliberately
        # introduces richer left-right
        # curvature sequences.
        alternating = (
            rng.uniform(
                1.0,
                4.0
            )
            * (
                (-1.0)
                ** np.arange(
                    num_control_points
                )
            )
        )

        radial_variation += (
            alternating
        )

        x = (
            (
                x_radius
                + radial_variation
            )
            * np.cos(theta)
        )

        y = (
            (
                y_radius
                + 0.75
                * radial_variation
            )
            * np.sin(theta)
        )

        control_points = (
            np.column_stack(
                (x, y)
            )
        )

        points = (
            _catmull_rom_closed(
                control_points,
                points_per_segment=
                    points_per_segment
            )
        )

        curvature = (
            _maximum_track_curvature(
                points
            )
        )

        if curvature <= max_curvature:
            return points

    raise RuntimeError(
        "Could not generate a feasible "
        "random spline track."
    )

def _track_curvature_statistics(
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

    heading_change = (
        heading_out
        - heading_in
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

    signs = np.sign(
        curvature
    )

    sign_changes = int(
        np.sum(
            signs[:-1]
            != signs[1:]
        )
    )

    return (
        float(
            np.mean(
                abs_curvature
            )
        ),
        float(
            np.max(
                abs_curvature
            )
        ),
        float(
            np.std(
                curvature
            )
        ),
        sign_changes
    )

def make_random_hard_spline_track(
    rng,
    num_points=720,
    max_attempts=250
):
    """
    Generate a physically feasible spline
    from the difficult tail of the spline
    geometry distribution.

    The held-out benchmark itself is NOT
    used here.
    """

    for _ in range(
        max_attempts
    ):

        points = (
            make_random_spline_track(
                rng=rng,
                num_points=num_points,
                max_curvature=0.20
            )
        )

        (
            mean_k,
            max_k,
            std_k,
            sign_changes
        ) = (
            _track_curvature_statistics(
                points
            )
        )

        # Difficult but still physically
        # feasible geometry.
        if (
            max_k >= 0.17
            and std_k >= 0.040
            and sign_changes >= 10
        ):
            return points

    raise RuntimeError(
        "Could not generate a hard "
        "spline track within "
        f"{max_attempts} attempts."
    )