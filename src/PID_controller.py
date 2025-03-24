import numpy as np


class PIDController:

    def __init__(self, Kp, Ki, Kd, Ki_sat):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.Ki_sat = Ki_sat

        self.previous_error = [0.0, 0.0, 0.0]

        # Integration total
        self.int = [0.0, 0.0, 0.0]

    def reset(self):
        self.int = [0.0, 0.0, 0.0]
        self.previous_error = [0.0, 0.0, 0.0]

    def control_update(self, error, timestep):

        # Update integral controller
        self.int += error * timestep

        # Prevent windup
        over_mag = np.argwhere(np.array(self.int) > np.array(self.Ki_sat))
        if over_mag.size != 0:
            for i in range(over_mag.size):
                mag = abs(self.int[over_mag[i][0]])  # get magnitude to find sign
                self.int[over_mag[i][0]] = (
                    self.int[over_mag[i][0]] / mag
                ) * self.Ki_sat[
                    over_mag[i][0]
                ]  # maintain sign but limit to saturation
        # Prevent windup using clipping
        self.int = np.clip(self.int, -np.array(self.Ki_sat), np.array(self.Ki_sat))
        derivative = (error - self.previous_error) / timestep
        self.previous_error = error

        # Calculate controller output
        output = self.Kp * error + self.Ki * self.int + self.Kd * derivative
        return output
