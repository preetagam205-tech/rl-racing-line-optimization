import numpy as np


class TrackGeometry:

    def __init__(
        self,
        a=40.0,
        b=20.0,
        track_width=12.0,
        num_points=720,
        num_checkpoints=7,
        centerline_points=None
    ):

        self.a = float(a)
        self.b = float(b)

        self.track_width = float(
            track_width
        )

        self.num_checkpoints = int(
            num_checkpoints
        )
        # -------------------------
        # GENERATE / LOAD CENTERLINE
        # -------------------------

        if centerline_points is None:

            self.num_points = int(
                num_points
            )

            theta = np.linspace(
                0,
                2 * np.pi,
                self.num_points,
                endpoint=False
            )

            x = self.a * np.cos(theta)
            y = self.b * np.sin(theta)

            self.centerline = np.column_stack(
                (x, y)
            )

        else:

            self.centerline = np.asarray(
                centerline_points,
                dtype=float
            )

            # Must look like:
            # [[x1, y1],
            #  [x2, y2],
            #  ...]
            if (
                self.centerline.ndim != 2
                or self.centerline.shape[1] != 2
            ):
                raise ValueError(
                    "centerline_points must have shape (N, 2)"
                )

            if len(self.centerline) < 4:
                raise ValueError(
                    "centerline_points must contain at least 4 points"
                )

            # If first point was repeated at the end,
            # remove the duplicate.
            if np.allclose(
                self.centerline[0],
                self.centerline[-1]
            ):
                self.centerline = (
                    self.centerline[:-1]
                )

            self.num_points = len(
                self.centerline
            )


        # -------------------------
        # ENSURE COUNTER-CLOCKWISE
        # -------------------------

        x = self.centerline[:, 0]
        y = self.centerline[:, 1]

        signed_area = 0.5 * np.sum(
            x * np.roll(y, -1)
            - np.roll(x, -1) * y
        )

        if signed_area < 0:
            self.centerline = (
                self.centerline[::-1].copy()
            )

        # -------------------------
        # CALCULATE NORMAL VECTORS
        # -------------------------

        previous_points = np.roll(
            self.centerline,
            1,
            axis=0
        )

        next_points = np.roll(
            self.centerline,
            -1,
            axis=0
        )

        # Approximate tangent at every point
        tangents = (
            next_points
            - previous_points
        )

        tangent_lengths = np.linalg.norm(
            tangents,
            axis=1,
            keepdims=True
        )

        tangents = (
            tangents
            / np.maximum(
                tangent_lengths,
                1e-12
            )
        )

        # Right-hand normal.
        # For our counter-clockwise centerline,
        # this points outward.
        normals = np.column_stack(
            (
                tangents[:, 1],
                -tangents[:, 0]
            )
        )

        # -------------------------
        # TRACK BOUNDARIES
        # -------------------------

        self.outer_boundary = (
            self.centerline
            + self.track_width / 2 * normals
        )

        self.inner_boundary = (
            self.centerline
            - self.track_width / 2 * normals
        )

        # -------------------------
        # TRACK LENGTH
        # -------------------------

        self.segment_starts = self.centerline

        self.segment_ends = np.roll(
            self.centerline,
            -1,
            axis=0
        )

        self.segment_vectors = (
            self.segment_ends - self.segment_starts
        )

        self.segment_lengths = np.linalg.norm(
            self.segment_vectors,
            axis=1
        )

        self.cumulative_distances = np.concatenate((
            [0.0],
            np.cumsum(self.segment_lengths)
        ))

        self.total_track_length = (
            self.cumulative_distances[-1]
        )

        # -------------------------
        # CHECKPOINT GENERATION
        # -------------------------

        self.checkpoint_gates = []

        for number in range(
            1,
            self.num_checkpoints + 1
        ):

            target_distance = (
                number / (self.num_checkpoints + 1)
            ) * self.total_track_length

            index = np.searchsorted(
                self.cumulative_distances,
                target_distance,
                side="right"
            ) - 1

            next_index = (
                index + 1
            ) % self.num_points

            fraction = (
                target_distance
                - self.cumulative_distances[index]
            ) / self.segment_lengths[index]

            inner_point = (
                self.inner_boundary[index]
                + fraction * (
                    self.inner_boundary[next_index]
                    - self.inner_boundary[index]
                )
            )

            outer_point = (
                self.outer_boundary[index]
                + fraction * (
                    self.outer_boundary[next_index]
                    - self.outer_boundary[index]
                )
            )

            self.checkpoint_gates.append(
                (inner_point, outer_point)
            )

    # -------------------------
    # DISTANCE TO CENTERLINE
    # -------------------------

    def distance_to_centerline(self, x, y):

        car_pos = np.array([x, y])

        relative = car_pos - self.segment_starts

        segment_length_squared = np.sum(
            self.segment_vectors ** 2,
            axis=1
        )

        t = np.sum(
            relative * self.segment_vectors,
            axis=1
        ) / segment_length_squared

        t = np.clip(t, 0.0, 1.0)

        projected_points = (
            self.segment_starts
            + t[:, None] * self.segment_vectors
        )

        distances = np.linalg.norm(
            projected_points - car_pos,
            axis=1
        )

        return float(np.min(distances))


    # -------------------------
    # ON-TRACK DETECTION
    # -------------------------

    def is_on_track(self, x, y):

        distance = self.distance_to_centerline(
            x,
            y
        )

        return distance <= self.track_width / 2


    # -------------------------
    # TRACK PROGRESS
    # -------------------------

    def calculate_progress(self, x, y):

        car_pos = np.array([x, y])

        relative = car_pos - self.segment_starts

        segment_length_squared = np.sum(
            self.segment_vectors ** 2,
            axis=1
        )

        t = np.sum(
            relative * self.segment_vectors,
            axis=1
        ) / segment_length_squared

        t = np.clip(t, 0.0, 1.0)

        projected_points = (
            self.segment_starts
            + t[:, None] * self.segment_vectors
        )

        distances_squared = np.sum(
            (projected_points - car_pos) ** 2,
            axis=1
        )

        nearest_index = np.argmin(
            distances_squared
        )

        distance_along_track = (
            self.cumulative_distances[nearest_index]
            + t[nearest_index]
            * self.segment_lengths[nearest_index]
        )

        progress = (
            distance_along_track
            / self.total_track_length
        ) * 100.0

        return float(progress)

    # -------------------------
    # TRACK-RELATIVE STATE
    # -------------------------

    def get_track_relative_state(self,x,y,heading):

        car_pos = np.array([x, y])

        # Position relative to every segment start
        relative = (car_pos - self.segment_starts)

        segment_length_squared = np.sum(self.segment_vectors ** 2,axis=1)

        # Projection along every segment
        t = np.sum(relative * self.segment_vectors,axis=1) / segment_length_squared

        t = np.clip(t,0.0,1.0)

        # Nearest points on all track segments
        projected_points = (
            self.segment_starts
            + t[:, None] * self.segment_vectors
        )

        difference = (
            car_pos - projected_points
        )

        distances_squared = np.sum(
            difference ** 2,
            axis=1
        )

        # Closest track segment
        nearest_index = int(
            np.argmin(distances_squared)
        )

        # Track direction
        tangent = (
            self.segment_vectors[nearest_index]
            / self.segment_lengths[nearest_index]
        )

        # Perpendicular direction
        normal = np.array([
            -tangent[1],
            tangent[0]
        ])

        # Signed distance from centerline
        lateral_error = np.dot(
            difference[nearest_index],
            normal
        )

        # Track heading
        track_heading = np.arctan2(
            tangent[1],
            tangent[0]
        )

        # Difference between car direction
        # and track direction
        heading_error = (
            heading
            - track_heading
            + np.pi
        ) % (2 * np.pi) - np.pi

        # Track progress
        distance_along_track = (
            self.cumulative_distances[nearest_index]
            + t[nearest_index]
            * self.segment_lengths[nearest_index]
        )

        progress = (
            distance_along_track
            / self.total_track_length
        ) * 100.0

        return (
            float(lateral_error),
            float(heading_error),
            float(progress)
        )

    def get_lookahead_heading_changes(self,x,y,distances=(10.0, 20.0, 30.0)):
        """
        Return the change in track heading at several
        distances ahead of the car.

        Positive = track turns left
        Negative = track turns right

        Returned values are in radians.
        """

        position = np.array(
            [x, y],
            dtype=float
        )

        # ---------------------------------
        # FIND NEAREST CENTERLINE SEGMENT
        # ---------------------------------

        relative = (
            position
            - self.segment_starts
        )

        segment_length_squared = np.sum(
            self.segment_vectors ** 2,
            axis=1
        )

        projection_fraction = (
            np.sum(
                relative
                * self.segment_vectors,
                axis=1
            )
            / segment_length_squared
        )

        projection_fraction = np.clip(
            projection_fraction,
            0.0,
            1.0
        )

        projections = (
            self.segment_starts
            + projection_fraction[:, None]
            * self.segment_vectors
        )

        distances_squared = np.sum(
            (
                position
                - projections
            ) ** 2,
            axis=1
        )

        nearest_index = int(
            np.argmin(distances_squared)
        )


        # ---------------------------------
        # CURRENT POSITION ALONG TRACK
        # ---------------------------------

        current_distance = (
            self.cumulative_distances[
                nearest_index
            ]
            + projection_fraction[
                nearest_index
            ]
            * self.segment_lengths[
                nearest_index
            ]
        )


        # ---------------------------------
        # CURRENT TRACK HEADING
        # ---------------------------------

        current_vector = (
            self.segment_vectors[
                nearest_index
            ]
        )

        current_heading = np.arctan2(
            current_vector[1],
            current_vector[0]
        )


        # ---------------------------------
        # LOOK AHEAD
        # ---------------------------------

        heading_changes = []

        for lookahead_distance in distances:

            target_distance = (
                current_distance
                + lookahead_distance
            ) % self.total_track_length

            future_index = (
                np.searchsorted(
                    self.cumulative_distances,
                    target_distance,
                    side="right"
                )
                - 1
            )

            future_index = int(
                np.clip(
                    future_index,
                    0,
                    len(self.segment_vectors) - 1
                )
            )

            future_vector = (
                self.segment_vectors[
                    future_index
                ]
            )

            future_heading = np.arctan2(
                future_vector[1],
                future_vector[0]
            )


            # Heading difference
            heading_change = (
                future_heading
                - current_heading
            )

            # Wrap to [-pi, pi]
            heading_change = (
                heading_change
                + np.pi
            ) % (2 * np.pi) - np.pi

            heading_changes.append(
                float(heading_change)
            )

        return heading_changes

    def get_lookahead_lateral_offsets(
    self,
    x,
    y,
    distances=(10.0, 20.0, 30.0)
):
        """
        Return the lateral position of future
        centerline points relative to the current
        track direction.

        Positive = left
        Negative = right
        """

        position = np.array(
            [x, y],
            dtype=float
        )

        # ---------------------------------
        # FIND CURRENT TRACK POSITION
        # ---------------------------------

        relative = (
            position
            - self.segment_starts
        )

        segment_length_squared = np.sum(
            self.segment_vectors ** 2,
            axis=1
        )

        projection_fraction = (
            np.sum(
                relative
                * self.segment_vectors,
                axis=1
            )
            / segment_length_squared
        )

        projection_fraction = np.clip(
            projection_fraction,
            0.0,
            1.0
        )

        projections = (
            self.segment_starts
            + projection_fraction[:, None]
            * self.segment_vectors
        )

        distances_squared = np.sum(
            (position - projections) ** 2,
            axis=1
        )

        nearest_index = int(
            np.argmin(distances_squared)
        )

        current_point = (
            projections[nearest_index]
        )

        current_distance = (
            self.cumulative_distances[
                nearest_index
            ]
            + projection_fraction[
                nearest_index
            ]
            * self.segment_lengths[
                nearest_index
            ]
        )


        # ---------------------------------
        # CURRENT LOCAL COORDINATE FRAME
        # ---------------------------------

        tangent = (
            self.segment_vectors[
                nearest_index
            ]
            / self.segment_lengths[
                nearest_index
            ]
        )

        left_normal = np.array(
            [
                -tangent[1],
                tangent[0]
            ]
        )


        # ---------------------------------
        # FUTURE CENTERLINE OFFSETS
        # ---------------------------------

        offsets = []

        for lookahead_distance in distances:

            target_distance = (
                current_distance
                + lookahead_distance
            ) % self.total_track_length

            future_index = (
                np.searchsorted(
                    self.cumulative_distances,
                    target_distance,
                    side="right"
                )
                - 1
            )

            future_index = int(
                np.clip(
                    future_index,
                    0,
                    self.num_points - 1
                )
            )

            distance_into_segment = (
                target_distance
                - self.cumulative_distances[
                    future_index
                ]
            )

            fraction = (
                distance_into_segment
                / self.segment_lengths[
                    future_index
                ]
            )

            future_point = (
                self.segment_starts[
                    future_index
                ]
                + fraction
                * self.segment_vectors[
                    future_index
                ]
            )

            vector_to_future = (
                future_point
                - current_point
            )

            lateral_offset = np.dot(
                vector_to_future,
                left_normal
            )

            offsets.append(
                float(lateral_offset)
            )

        return offsets