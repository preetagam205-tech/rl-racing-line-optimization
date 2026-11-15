import numpy as np


class Vehicle:

    def __init__(self, x, y, heading):

        # Vehicle state
        self.x = x
        self.y = y
        self.heading = heading
        self.speed = 0.0

        # Vehicle parameters
        self.wheelbase = 2.5
        self.max_speed = 20.0
        self.max_steering = np.radians(30)
        self.coast_deceleration = 1.5
        self.max_lateral_acceleration = 8.0

    def reset(self, x, y, heading):

        # Restore starting position
        self.x = float(x)
        self.y = float(y)

        # Restore starting direction
        self.heading = float(heading)

        # Stop the vehicle
        self.speed = 0.0

    def update(self, acceleration, steering, dt):

        # -------------------------
        # MECHANICAL STEERING LIMIT
        # -------------------------

        steering = np.clip(
            steering,
            -self.max_steering,
            self.max_steering
        )


        # -------------------------
        # SPEED UPDATE
        # -------------------------

        net_acceleration = acceleration

        # Rolling/coasting resistance acts whenever
        # the vehicle is moving
        if self.speed > 0.0:

            net_acceleration -= (
                self.coast_deceleration
            )

        # Do not let resistance move the car backwards
        if self.speed <= 0.0 and net_acceleration < 0.0:

            net_acceleration = 0.0

        self.speed += (
            net_acceleration * dt
        )

        self.speed = np.clip(
            self.speed,
            0.0,
            self.max_speed
        )


        # -------------------------
        # TIRE GRIP LIMIT
        # -------------------------

        if self.speed > 0.1:

            grip_steering_limit = np.arctan(
                (
                    self.max_lateral_acceleration
                    * self.wheelbase
                )
                / (self.speed ** 2)
            )

            allowed_steering = min(
                self.max_steering,
                grip_steering_limit
            )

            steering = np.clip(
                steering,
                -allowed_steering,
                allowed_steering
            )


        # -------------------------
        # HEADING UPDATE
        # -------------------------

        self.heading += (
            self.speed
            / self.wheelbase
            * np.tan(steering)
            * dt
        )


        # -------------------------
        # POSITION UPDATE
        # -------------------------

        self.x += (
            self.speed
            * np.cos(self.heading)
            * dt
        )

        self.y += (
            self.speed
            * np.sin(self.heading)
            * dt
        )

    def get_steering_authority(self):
        """
        Returns how much of the car's mechanical steering
        is physically available at the current speed.

        1.0 = full steering authority
        0.0 = almost no steering authority
        """

        if self.speed <= 0.1:
            return 1.0

        grip_steering_limit = np.arctan(
            (
                self.max_lateral_acceleration
                * self.wheelbase
            )
            / (self.speed ** 2)
        )

        allowed_steering = min(
            self.max_steering,
            grip_steering_limit
        )

        steering_authority = (
            allowed_steering
            / self.max_steering
        )

        return float(
            np.clip(
                steering_authority,
                0.0,
                1.0
            )
        )