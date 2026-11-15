import gymnasium as gym
import numpy as np

from gymnasium import spaces

from src.simulator import RacingSimulator

from src.track_geometry import TrackGeometry

from src.track_factory import (
    make_random_wavy_track,
    make_random_complex_track,
    make_random_spline_track,
    make_random_hard_spline_track,
)


class RacingEnv(gym.Env):

    def __init__(
        self,
        track=None,
        random_start=False,
        random_track=False,
        mixed_tracks=False,
        broad_tracks=False,
        hard_broad_tracks=False,
        curvature_observation=False
    ):

        super().__init__()

        # Simulator uses whichever track we provide
        self.sim = RacingSimulator(
            track=track
        )

        self.random_start = random_start

        self.random_track = random_track

        self.mixed_tracks = mixed_tracks

        self.broad_tracks = broad_tracks

        self.hard_broad_tracks = hard_broad_tracks

        self.curvature_observation = curvature_observation

        self.current_track_family = "fixed"

        # Fixed simulation timestep
        self.dt = 0.05

        # Maximum simulated time for one episode
        self.max_episode_time = 60.0

        # -------------------------
        # OBSERVATION SPACE
        # -------------------------

        if self.curvature_observation:

            # 16 values:
            #
            # 0  speed
            # 1  lateral error
            # 2  heading error
            #
            # 3-8:
            # H10 H20 H30 H40 H50 H60
            #
            # 9-14:
            # curvature for each successive
            # 10 m road segment
            #
            # 15 steering authority

            self.observation_space = spaces.Box(
                low=np.array(
                    [
                        0.0,      # speed
                        -2.0,     # lateral error
                        -1.0,     # heading error

                        -1.0,     # H10
                        -1.0,     # H20
                        -1.0,     # H30
                        -1.0,     # H40
                        -1.0,     # H50
                        -1.0,     # H60

                        -1.0,     # K 0-10 m
                        -1.0,     # K 10-20 m
                        -1.0,     # K 20-30 m
                        -1.0,     # K 30-40 m
                        -1.0,     # K 40-50 m
                        -1.0,     # K 50-60 m

                        0.0       # steering authority
                    ],
                    dtype=np.float32
                ),

                high=np.array(
                    [
                        1.0,
                        2.0,
                        1.0,

                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,

                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,

                        1.0
                    ],
                    dtype=np.float32
                ),

                dtype=np.float32
            )

        else:

            # Original 10-value observation.
            # Keep this unchanged so all old
            # models remain compatible.

            self.observation_space = spaces.Box(
                low=np.array(
                    [
                        0.0,
                        -2.0,
                        -1.0,

                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,
                        -1.0,

                        0.0
                    ],
                    dtype=np.float32
                ),

                high=np.array(
                    [
                        1.0,
                        2.0,
                        1.0,

                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,
                        1.0,

                        1.0
                    ],
                    dtype=np.float32
                ),

                dtype=np.float32
            )


        # -------------------------
        # ACTION SPACE
        # -------------------------

        self.action_space = spaces.Box(
            low=np.array(
                [-1.0, -1.0],
                dtype=np.float32
            ),

            high=np.array(
                [1.0, 1.0],
                dtype=np.float32
            ),

            dtype=np.float32
        )

    def _get_observation(self):

        car = self.sim.car
        track = self.sim.track

        lateral_error, heading_error, progress = (
            track.get_track_relative_state(
                car.x,
                car.y,
                car.heading
            )
        )

        # -------------------------
        # BASIC STATE
        # -------------------------

        normalized_speed = (
            car.speed
            / car.max_speed
        )

        normalized_lateral = (
            lateral_error
            / (track.track_width / 2.0)
        )

        normalized_heading = (
            heading_error
            / np.pi
        )


        # -------------------------
        # HEADING LOOKAHEAD
        # -------------------------

        lookahead = (
            track.get_lookahead_heading_changes(
                car.x,
                car.y,
                distances=(
                    10.0,
                    20.0,
                    30.0,
                    40.0,
                    50.0,
                    60.0
                )
            )
        )

        normalized_lookahead = np.clip(
            np.asarray(
                lookahead,
                dtype=np.float64
            ) / np.pi,
            -1.0,
            1.0
        )


        # -------------------------
        # STEERING AUTHORITY
        # -------------------------

        steering_authority = (
            car.get_steering_authority()
        )


        # =====================================================
        # NEW 16-VALUE OBSERVATION
        # =====================================================

        if self.curvature_observation:

            # Convert cumulative future heading
            # information into six local 10 m
            # curvature estimates.
            #
            # Example:
            #
            # first segment:
            #   H10 / 10
            #
            # second segment:
            #   (H20 - H10) / 10
            #
            # etc.

            segment_curvatures = []

            previous_heading_change = 0.0

            for heading_change in lookahead:

                local_heading_change = (
                    heading_change
                    - previous_heading_change
                )

                # Wrap angle to [-pi, pi].
                local_heading_change = (
                    local_heading_change
                    + np.pi
                ) % (
                    2.0 * np.pi
                ) - np.pi

                local_curvature = (
                    local_heading_change
                    / 10.0
                )

                segment_curvatures.append(
                    local_curvature
                )

                previous_heading_change = (
                    heading_change
                )


            segment_curvatures = np.asarray(
                segment_curvatures,
                dtype=np.float64
            )


            # 0.20 1/m is approximately the
            # upper curvature used by our
            # physically-feasible training tracks.
            curvature_scale = 0.20

            normalized_curvature = np.clip(
                segment_curvatures
                / curvature_scale,
                -1.0,
                1.0
            )


            observation = np.array(
                [
                    np.clip(
                        normalized_speed,
                        0.0,
                        1.0
                    ),

                    np.clip(
                        normalized_lateral,
                        -2.0,
                        2.0
                    ),

                    np.clip(
                        normalized_heading,
                        -1.0,
                        1.0
                    ),

                    # Heading lookahead
                    normalized_lookahead[0],
                    normalized_lookahead[1],
                    normalized_lookahead[2],
                    normalized_lookahead[3],
                    normalized_lookahead[4],
                    normalized_lookahead[5],

                    # Local curvature lookahead
                    normalized_curvature[0],
                    normalized_curvature[1],
                    normalized_curvature[2],
                    normalized_curvature[3],
                    normalized_curvature[4],
                    normalized_curvature[5],

                    steering_authority
                ],
                dtype=np.float32
            )


        # =====================================================
        # ORIGINAL 10-VALUE OBSERVATION
        # =====================================================

        else:

            observation = np.array(
                [
                    np.clip(
                        normalized_speed,
                        0.0,
                        1.0
                    ),

                    np.clip(
                        normalized_lateral,
                        -2.0,
                        2.0
                    ),

                    np.clip(
                        normalized_heading,
                        -1.0,
                        1.0
                    ),

                    normalized_lookahead[0],
                    normalized_lookahead[1],
                    normalized_lookahead[2],
                    normalized_lookahead[3],
                    normalized_lookahead[4],
                    normalized_lookahead[5],

                    steering_authority
                ],
                dtype=np.float32
            )


        return observation
               

    def reset(self, seed=None, options=None):

        # Required Gymnasium behaviour
        super().reset(seed=seed)

        if self.hard_broad_tracks:

            track_choice = int(
                self.np_random.integers(
                    0,
                    5
                )
            )

            # -------------------------
            # ELLIPSE
            # -------------------------

            if track_choice == 0:

                a = float(
                    self.np_random.uniform(
                        35.0,
                        60.0
                    )
                )

                b = float(
                    self.np_random.uniform(
                        15.0,
                        25.0
                    )
                )

                new_track = TrackGeometry(
                    a=a,
                    b=b,
                    track_width=12.0,
                    num_points=720,
                    num_checkpoints=7
                )

                self.current_track_family = (
                    "ellipse"
                )

            # -------------------------
            # WAVY
            # -------------------------

            elif track_choice == 1:

                points = make_random_wavy_track(
                    rng=self.np_random,
                    num_points=720
                )

                new_track = TrackGeometry(
                    centerline_points=points,
                    track_width=12.0,
                    num_checkpoints=7
                )

                self.current_track_family = (
                    "wavy"
                )

            # -------------------------
            # COMPLEX
            # -------------------------

            elif track_choice == 2:

                points = make_random_complex_track(
                    rng=self.np_random,
                    num_points=720
                )

                new_track = TrackGeometry(
                    centerline_points=points,
                    track_width=12.0,
                    num_checkpoints=7
                )

                self.current_track_family = (
                    "complex"
                )

            # -------------------------
            # REGULAR SPLINE
            # -------------------------

            elif track_choice == 3:

                points = make_random_spline_track(
                    rng=self.np_random,
                    num_points=720
                )

                new_track = TrackGeometry(
                    centerline_points=points,
                    track_width=12.0,
                    num_checkpoints=7
                )

                self.current_track_family = (
                    "spline"
                )

            # -------------------------
            # HARD SPLINE
            # -------------------------

            else:

                points = (
                    make_random_hard_spline_track(
                        rng=self.np_random,
                        num_points=720
                    )
                )

                new_track = TrackGeometry(
                    centerline_points=points,
                    track_width=12.0,
                    num_checkpoints=7
                )

                self.current_track_family = (
                    "hard_spline"
                )

            self.sim = RacingSimulator(
                track=new_track
            )

        elif self.broad_tracks:

            track_choice = int(
                self.np_random.integers(
                    0,
                    4
                )
            )

            # -------------------------
            # ELLIPSE
            # -------------------------

            if track_choice == 0:

                a = float(
                    self.np_random.uniform(
                        35.0,
                        60.0
                    )
                )

                b = float(
                    self.np_random.uniform(
                        15.0,
                        25.0
                    )
                )

                new_track = TrackGeometry(
                    a=a,
                    b=b,
                    track_width=12.0,
                    num_points=720,
                    num_checkpoints=7
                )

                self.current_track_family = (
                    "ellipse"
                )

            # -------------------------
            # ORIGINAL WAVY
            # -------------------------

            elif track_choice == 1:

                points = make_random_wavy_track(
                    rng=self.np_random,
                    num_points=720
                )

                new_track = TrackGeometry(
                    centerline_points=points,
                    track_width=12.0,
                    num_checkpoints=7
                )

                self.current_track_family = (
                    "wavy"
                )

            # -------------------------
            # COMPLEX HARMONIC
            # -------------------------

            elif track_choice == 2:

                points = make_random_complex_track(
                    rng=self.np_random,
                    num_points=720
                )

                new_track = TrackGeometry(
                    centerline_points=points,
                    track_width=12.0,
                    num_checkpoints=7
                )

                self.current_track_family = (
                    "complex"
                )

            # -------------------------
            # RANDOM SPLINE
            # -------------------------

            else:

                points = make_random_spline_track(
                    rng=self.np_random,
                    num_points=720
                )

                new_track = TrackGeometry(
                    centerline_points=points,
                    track_width=12.0,
                    num_checkpoints=7
                )

                self.current_track_family = (
                    "spline"
                )

            self.sim = RacingSimulator(
                track=new_track
            )

       

       # ---------------------------------
        # RANDOMIZE TRACK GEOMETRY
        # ---------------------------------

        elif self.mixed_tracks:

            track_choice = int(
                self.np_random.integers(
                    0,
                    3
                )
            )

            # -------------------------
            # ELLIPSE
            # -------------------------

            if track_choice == 0:

                a = float(
                    self.np_random.uniform(
                        35.0,
                        60.0
                    )
                )

                b = float(
                    self.np_random.uniform(
                        15.0,
                        25.0
                    )
                )

                new_track = TrackGeometry(
                    a=a,
                    b=b,
                    track_width=12.0,
                    num_points=720,
                    num_checkpoints=7
                )

                self.current_track_family = (
                    "ellipse"
                )

            # -------------------------
            # ORIGINAL WAVY
            # -------------------------

            elif track_choice == 1:

                points = make_random_wavy_track(
                    rng=self.np_random,
                    num_points=720
                )

                new_track = TrackGeometry(
                    centerline_points=points,
                    track_width=12.0,
                    num_checkpoints=7
                )

                self.current_track_family = (
                    "wavy"
                )

            # -------------------------
            # COMPLEX WAVY
            # -------------------------

            else:

                points = (
                    make_random_complex_track(
                        rng=self.np_random,
                        num_points=720
                    )
                )

                new_track = TrackGeometry(
                    centerline_points=points,
                    track_width=12.0,
                    num_checkpoints=7
                )

                self.current_track_family = (
                    "complex"
                )

            self.sim = RacingSimulator(
                track=new_track
            )

              

        elif self.random_track:

            a = float(
                self.np_random.uniform(
                    35.0,
                    60.0
                )
            )

            b = float(
                self.np_random.uniform(
                    15.0,
                    25.0
                )
            )

            new_track = TrackGeometry(
                a=a,
                b=b,
                track_width=12.0,
                num_points=720,
                num_checkpoints=7
            )

            self.sim = RacingSimulator(
                track=new_track
            )

            self.current_track_family = "ellipse"

        # -------------------------
        # CHOOSE START POSITION
        # -------------------------

        # Allows us to manually choose a start
        # during testing/evaluation
        if (
            options is not None
            and "start_index" in options
        ):

            start_index = int(
                options["start_index"]
            )

        # Random start during training
        elif self.random_start:

            start_index = int(
                self.np_random.integers(
                    0,
                    self.sim.track.num_points
                )
            )

        # Normal start line
        else:

            start_index = 0


        # Reset simulator at chosen location
        state = self.sim.reset(
            start_index=start_index
        )

        # Required for reward calculation
        self.previous_progress = (
            state["progress"]
        )

        observation = (
            self._get_observation()
        )

        info = {
            "start_index": start_index,
            "start_progress": state["progress"],
            "track_family": self.current_track_family
        }

        return observation, info

    def step(self, action):

        # Make sure action is a NumPy array
        action = np.asarray(
            action,
            dtype=np.float32
        )

        # Keep actions inside [-1, 1]
        action = np.clip(
            action,
            -1.0,
            1.0
        )

        # -------------------------
        # STEERING
        # -------------------------

        steering_input = float(action[0])

        steering = (
            steering_input
            * self.sim.car.max_steering
        )

        # -------------------------
        # THROTTLE / BRAKE
        # -------------------------

        throttle_brake = float(action[1])

        if throttle_brake >= 0:

            # Full throttle = +6
            acceleration = (
                throttle_brake * 6.0
            )

        else:

            # Full braking = -10
            acceleration = (
                throttle_brake * 10.0
            )

        # -------------------------
        # UPDATE SIMULATOR
        # -------------------------

        state = self.sim.step(
            acceleration=acceleration,
            steering=steering,
            dt=self.dt
        )

        # Get RL observation
        observation = (
            self._get_observation()
        )

        
        
        # -------------------------
        # REWARD FUNCTION V2
        # -------------------------

        current_progress = state["progress"]

        # Change in progress since previous step
        progress_delta = (
            current_progress
            - self.previous_progress
        )

        # Handle progress wrapping around finish line
        if progress_delta < -50:
            progress_delta += 100

        elif progress_delta > 50:
            progress_delta -= 100

        # Prevent abnormal progress jumps
        progress_delta = np.clip(
            progress_delta,
            -1.0,
            1.0
        )

        # -------------------------
        # REWARD COMPONENTS v3
        # -------------------------

        # Reward forward movement
        progress_reward = (
            2.0 * progress_delta
        )

        

        # Small penalty for taking time
        time_penalty = -0.01

        # Penalize staying almost stationary
        low_speed_penalty = 0.0

        if state["speed"] < 1.0:
            low_speed_penalty = -0.10

        # -------------------------
        # SOFT EDGE PENALTY
        # -------------------------

        lateral_error, _, _ = (
            self.sim.track.get_track_relative_state(
                self.sim.car.x,
                self.sim.car.y,
                self.sim.car.heading
            )
        )

        normalized_lateral_error = (
            abs(lateral_error)
            / (self.sim.track.track_width / 2.0)
        )

        edge_penalty = 0.0

        # Only penalize the outermost 10%
        # of the available track width.
        if normalized_lateral_error > 0.90:

            edge_amount = np.clip(
                normalized_lateral_error,
                0.90,
                1.00
            )

            edge_penalty = (
                -0.05
                * (
                    edge_amount
                    - 0.90
                )
                / 0.10
            )

        # Reward checkpoints
        checkpoint_bonus = 0.0

        if state["checkpoint_passed"]:
            checkpoint_bonus = 5.0

        # Penalize leaving the track
        off_track_penalty = 0.0

        if not state["on_track"]:
            off_track_penalty = -100.0

        # Large reward for completing a lap
        lap_bonus = 0.0

        if state["lap_completed"]:
            lap_bonus = 200.0

        # -------------------------
        # LAP-TIME BONUS
        # -------------------------

        lap_time_bonus = 0.0
        lap_efficiency = 0.0

        theoretical_min_time = (
            self.sim.track.total_track_length
            / self.sim.car.max_speed
        )

        if state["lap_completed"]:

            lap_efficiency = (
                theoretical_min_time
                / max(
                    state["time"],
                    1e-6
                )
            )

            lap_efficiency = np.clip(
                lap_efficiency,
                0.0,
                1.0
            )

            lap_time_bonus = (
                300.0
                * (
                    lap_efficiency ** 4
                )
            )

        # Total reward
        reward = (
            progress_reward
            + time_penalty
            + low_speed_penalty
            + edge_penalty
            + checkpoint_bonus
            + off_track_penalty
            + lap_bonus
            + lap_time_bonus
        )

        # Save progress for next step
        self.previous_progress = (
            current_progress
        )
        
        
        # -------------------------
        # EPISODE TERMINATION
        # -------------------------

        # End episode if car leaves track
        # or successfully completes a lap
        terminated = (
            not state["on_track"]
            or state["lap_completed"]
        )

        # Stop excessively long episodes
        truncated = (
            not terminated
            and state["time"] >= self.max_episode_time
        )

        # Extra information useful for debugging
        info = {
            "speed": state["speed"],
            "progress": state["progress"],
            "on_track": state["on_track"],
            "reward_edge": float(edge_penalty),

            "next_checkpoint": state[
                "next_checkpoint"
            ],

            "lap_completed": state[
                "lap_completed"
            ],

            "run_valid": state["run_valid"],
            "time": state["time"],

            # Reward debugging
            "progress_delta": float(
                progress_delta
            ),

            "reward_progress": float(
                progress_reward
            ),

            "reward_time": float(
                time_penalty
            ),

            "reward_low_speed": float(
                low_speed_penalty
            ),

            "reward_checkpoint": float(
                checkpoint_bonus
            ),

            "reward_off_track": float(
                off_track_penalty
            ),

            "reward_lap": float(
                lap_bonus
            ),

            "reward_lap_time": float(
                lap_time_bonus
            ),

            "lap_efficiency": float(
                lap_efficiency
            ),

            "theoretical_min_time": float(
                theoretical_min_time
            ),


            "reward_total": float(
                reward
            )
        }

        reward = float(reward)
        terminated = bool(terminated)
        truncated = bool(truncated)

        return (
            observation,
            reward,
            terminated,
            truncated,
            info
        )