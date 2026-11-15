import numpy as np

from src.track_geometry import TrackGeometry
from src.vehicle import Vehicle


class RacingSimulator:

    def __init__(self,track=None):

        # Use supplied track, otherwise use default
        if track is None:
            self.track = TrackGeometry()
        else:
            self.track = track

        # Starting position
        self.start_position = self.track.centerline[0]

        # Track direction at the start
        next_position = self.track.centerline[1]

        direction = (
            next_position
            - self.start_position
        )

        self.start_heading = np.arctan2(
            direction[1],
            direction[0]
        )

        # Create vehicle
        self.car = Vehicle(
            x=self.start_position[0],
            y=self.start_position[1],
            heading=self.start_heading
        )

        # Simulation state
        self.time = 0.0

        self.reset()

    def reset(self, start_index=0):

        # Keep index inside valid centerline range
        start_index = int(
            start_index % self.track.num_points
        )
        self.start_index = start_index

        # -------------------------
        # START POSITION
        # -------------------------

        start_position = (
            self.track.centerline[start_index]
        )

        # Next point gives the direction of the track
        next_index = (
            start_index + 1
        ) % self.track.num_points

        next_position = (
            self.track.centerline[next_index]
        )

        # Calculate starting heading
        start_heading = np.arctan2(
            next_position[1] - start_position[1],
            next_position[0] - start_position[0]
        )

        # Actually move/reset the car
        self.car.reset(
            start_position[0],
            start_position[1],
            start_heading
        )


        # -------------------------
        # RESET SIMULATION STATE
        # -------------------------

        self.time = 0.0


        # -------------------------
        # CHECKPOINT TRACKING
        # -------------------------


        self.run_valid = True


        # -------------------------
        # LAP TRACKING
        # -------------------------

        self.lap_count = 0

        self.lap_elapsed = 0.0

        self.last_lap_time = None

        self.lap_started = False


        # -------------------------
        # PREVIOUS TRACK STATE
        # -------------------------

        self.previous_progress = (
            self.track.calculate_progress(
                self.car.x,
                self.car.y
            )
        )

        # -------------------------
        # CHECKPOINT ORDER
        # -------------------------

        checkpoint_progresses = [
            (
                checkpoint_number
                / (self.track.num_checkpoints + 1)
            ) * 100.0
            for checkpoint_number
            in range(
                1,
                self.track.num_checkpoints + 1
            )
        ]

        # Checkpoints still ahead of our start
        checkpoints_ahead = [
            checkpoint_number
            for checkpoint_number, checkpoint_progress
            in enumerate(
                checkpoint_progresses,
                start=1
            )
            if checkpoint_progress
            > self.previous_progress + 1e-9
        ]

        # Checkpoints behind us will be visited
        # after wrapping around the track
        checkpoints_after_wrap = [
            checkpoint_number
            for checkpoint_number, checkpoint_progress
            in enumerate(
                checkpoint_progresses,
                start=1
            )
            if checkpoint_progress
            <= self.previous_progress + 1e-9
        ]

        self.checkpoint_order = (
            checkpoints_ahead
            + checkpoints_after_wrap
        )

        self.checkpoint_order_position = 0

        self.checkpoints_passed_count = 0

        self.next_checkpoint = (
            self.checkpoint_order[0]
        )

        self.previous_on_track = (
            self.track.is_on_track(
                self.car.x,
                self.car.y
            )
        )


        return self.get_state()

    def crossed_start_gate_forward(
    self,
    previous_pos,
    current_pos
):

        # Gate at this episode's starting position
        gate_start = (
            self.track.inner_boundary[
                self.start_index
            ]
        )

        gate_end = (
            self.track.outer_boundary[
                self.start_index
            ]
        )

        movement = (
            current_pos
            - previous_pos
        )

        gate = (
            gate_end
            - gate_start
        )

        def cross2d(a, b):
            return (
                a[0] * b[1]
                - a[1] * b[0]
            )

        denominator = cross2d(
            movement,
            gate
        )

        if abs(denominator) < 1e-9:
            return False

        relative = (
            gate_start
            - previous_pos
        )

        t = (
            cross2d(
                relative,
                gate
            )
            / denominator
        )

        u = (
            cross2d(
                relative,
                movement
            )
            / denominator
        )

        intersects = (
            0 < t <= 1
            and 0 <= u <= 1
        )

        # Correct forward direction at
        # this episode's starting point
        track_direction = (
            self.track.segment_vectors[
                self.start_index
            ]
            / self.track.segment_lengths[
                self.start_index
            ]
        )

        moving_forward = (
            np.dot(
                movement,
                track_direction
            ) > 0
        )

        return (
            intersects
            and moving_forward
        )
    
    def step(
        self,
        acceleration,
        steering,
        dt
    ):
        previous_car_pos = np.array([self.car.x,self.car.y])

        # Update vehicle
        self.car.update(
            acceleration,
            steering,
            dt
        )

        self.time += dt

        # Current track information
        current_progress = (
            self.track.calculate_progress(
                self.car.x,
                self.car.y
            )
        )

        current_on_track = (
            self.track.is_on_track(
                self.car.x,
                self.car.y
            )
        )

        # Leaving the road invalidates the run
        if not current_on_track:
            self.run_valid = False
        # -------------------------
        # LAP TIMER
        # -------------------------

        # Start timing when the car first moves
        if (
            self.run_valid
            and not self.lap_started
            and self.car.speed > 0
        ):
            self.lap_started = True

        # Update current lap time
        if (
            self.run_valid
            and self.lap_started
        ):
            self.lap_elapsed += dt

        checkpoint_passed = False

        # Check next required checkpoint
        if (
            self.run_valid
            and self.previous_on_track
            and current_on_track
            and self.next_checkpoint
            <= self.track.num_checkpoints
        ):

            checkpoint_progress = (
                self.next_checkpoint
                / (self.track.num_checkpoints + 1)
            ) * 100.0

            progress_change = (
                current_progress
                - self.previous_progress
            )

            crossed = (
                0 < progress_change < 5
                and self.previous_progress
                < checkpoint_progress
                <= current_progress
            )

            if crossed:

                checkpoint_passed = True

                self.checkpoints_passed_count += 1

                self.checkpoint_order_position += 1

                if (
                    self.checkpoint_order_position
                    < self.track.num_checkpoints
                ):

                    self.next_checkpoint = (
                        self.checkpoint_order[
                            self.checkpoint_order_position
                        ]
                    )

                else:

                    # All checkpoints passed
                    self.next_checkpoint = (
                        self.track.num_checkpoints + 1
                    )

        # -------------------------
        # LAP COMPLETION
        # -------------------------

        lap_completed = False

        current_car_pos = np.array([
            self.car.x,
            self.car.y
        ])

        if (
            self.run_valid
            and self.previous_on_track
            and current_on_track
            and self.next_checkpoint
            > self.track.num_checkpoints
            and self.crossed_start_gate_forward(
                previous_car_pos,
                current_car_pos
            )
        ):

            self.lap_count += 1

            self.last_lap_time = (
                self.lap_elapsed
            )

            # Reset timer for next lap
            self.lap_elapsed = 0.0

            # Require checkpoints again
            self.next_checkpoint = 1

            lap_completed = True

        # Save state for next simulation step
        self.previous_progress = current_progress
        self.previous_on_track = current_on_track

        state = self.get_state()

        state["checkpoint_passed"] = (
            checkpoint_passed
        )

        state["lap_completed"] = (
            lap_completed
        )

        return state

    def get_state(self):

        progress = self.track.calculate_progress(
            self.car.x,
            self.car.y
        )

        on_track = self.track.is_on_track(
            self.car.x,
            self.car.y
        )

        state = {
            "x": float(self.car.x),
            "y": float(self.car.y),
            "speed": float(self.car.speed),
            "heading": float(self.car.heading),
            "progress": float(progress),
            "on_track": bool(on_track),
            "time": float(self.time),

            "next_checkpoint": int(
                self.next_checkpoint
            ),

                        "run_valid": bool(
                self.run_valid
            ),

            "lap_count": int(
                self.lap_count
            ),

            "lap_elapsed": float(
                self.lap_elapsed
            ),

            "last_lap_time": (
                None
                if self.last_lap_time is None
                else float(self.last_lap_time)
            ),

            "lap_started": bool(
                self.lap_started
            )
        }

        return state