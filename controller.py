import numpy as np
from src.PID_controller import PIDController

# Ziegler–Nichols tuner with "max velocity limit" scaling
class ZNTuner:
    def __init__(self, name="unnamed", kp_start=0.1, kp_step=0.05, kp_max=5.0):
        """
        :param name: Tuner name (e.g. "X", "Y"), used in logs
        :param kp_start: Initial Kp
        :param kp_step:  Kp increment per step
        :param kp_max:   Maximum Kp to search
        """
        self.name = name
        self.state = "OFF"
        self.Kp_current = kp_start
        self.Kp_step = kp_step
        self.Kp_max = kp_max

        self.last_error_sign = 0
        self.zero_cross_times = []
        self.K_u = None
        self.T_u = None
        self.best_pid = (0.0, 0.0, 0.0)  # (kp, ki, kd)

    def start_tuning(self):
        """Enter FIND_KU state, start searching for critical gain K_u"""
        self.state = "FIND_KU"
        self.Kp_current = 0.1
        self.zero_cross_times = []
        self.last_error_sign = 0
        print(f"[ZNTuner-{self.name}] Start searching K_u ...")

    def update_tuner(self, error, t):
        """
        :param error: scalar (single axis)
        :param t: current time
        """
        if self.state not in ["FIND_KU", "OBSERVE_OSC"]:
            return

        sign = np.sign(error)

        if self.state == "FIND_KU":
            # Detect continuous zero-crossings (very simplified)
            if sign != 0 and sign != self.last_error_sign:
                self.last_error_sign = sign
                self.zero_cross_times.append(t)
                if len(self.zero_cross_times) >= 3:
                    print(f"[ZNTuner-{self.name}] Potential oscillation at Kp={self.Kp_current:.3f}")
                    self.K_u = self.Kp_current
                    self.state = "OBSERVE_OSC"
                    self.zero_cross_times = [t]

        elif self.state == "OBSERVE_OSC":
            # Record zero-crossing intervals => estimate oscillation period
            if sign != 0 and sign != self.last_error_sign:
                self.last_error_sign = sign
                self.zero_cross_times.append(t)
                if len(self.zero_cross_times) >= 2:
                    period_est = self.zero_cross_times[-1] - self.zero_cross_times[-2]
                    print(f"[ZNTuner-{self.name}] Observed period ~ {period_est:.3f}s => finish!")
                    self.T_u = period_est
                    # Apply velocity limit scaling in finish_tuning()
                    self.finish_tuning(max_vel=1.0, max_err=4.0)

    def finish_tuning(self, max_vel=1.0, max_err=4.0):
        """Use Z-N formula to calculate PID gains based on K_u, T_u, considering velocity limit"""
        if (self.K_u is not None) and (self.T_u is not None):
            Ku = self.K_u
            Tu = self.T_u
            # Classic Z-N
            kp0 = 0.6 * Ku
            # Extra attenuation
            ki0 = 1.2 * Ku / Tu * 0.7
            kd0 = 0.075 * Ku * Tu

            # If kp0 * max_err > max_vel => scale down
            if kp0 * max_err > max_vel:
                alpha = max_vel / (kp0 * max_err)
                kp = alpha * kp0
                ki = alpha * ki0
                kd = alpha * kd0
                print(f"[ZNTuner-{self.name}] Gains scaled by alpha={alpha:.3f}")
            else:
                kp, ki, kd = kp0, ki0, kd0

            self.best_pid = (kp, ki, kd)
            print(f"[ZNTuner-{self.name}] => Kp={kp:.3f}, Ki={ki:.3f}, Kd={kd:.3f}")
        else:
            print(f"[ZNTuner-{self.name}] No Ku or Tu??")
        self.state = "DONE"

    def increase_Kp(self):
        """In FIND_KU stage, increase kp if oscillation not detected"""
        if self.state == "FIND_KU":
            self.Kp_current += self.Kp_step
            if self.Kp_current > self.Kp_max:
                print(f"[ZNTuner-{self.name}] Reached maxKp={self.Kp_current:.2f}, no oscillation => stop.")
                self.state = "DONE"

    def is_tuning_done(self):
        return (self.state == "DONE")

    def get_current_Kp_for_control(self):
        if self.state in ["FIND_KU","OBSERVE_OSC"]:
            return self.Kp_current
        return None

    def get_final_pid(self):
        """Get (kp, ki, kd) after DONE"""
        return self.best_pid

import numpy as np

# Initialize PID controller
pid_pos = PIDController([0.4, 0.4, 0.6], [0.05, 0.05, 0.1], [0.2, 0.2, 0.3], [1, 1, 1])
pid_vel = PIDController([0.3, 0.3, 0.5], [0.02, 0.02, 0.05], [0.15, 0.15, 0.2], [0.5, 0.5, 0.5])
pid_att = PIDController([0.8, 0.8, 0.5], [0, 0, 0.01], [0.1, 0.1, 0.05], [0.1, 0.1, 0.1])

last_target_pos = None
last_position = None
sim_time = 0
NotReached = True

def reset_all_pid():
    pid_pos.reset()
    pid_vel.reset()
    pid_att.reset()

def rotation_matrix(roll, pitch, yaw):
    c, s = np.cos, np.sin
    Rz = np.array([
        [c(yaw), -s(yaw), 0],
        [s(yaw), c(yaw), 0],
        [0, 0, 1]
    ])
    Ry = np.array([
        [c(pitch), 0, s(pitch)],
        [0, 1, 0],
        [-s(pitch), 0, c(pitch)]
    ])
    Rx = np.array([
        [1, 0, 0],
        [0, c(roll), -s(roll)],
        [0, s(roll), c(roll)]
    ])
    return Rz @ Ry @ Rx

def controller(state, target_pos, dt):
    global sim_time, last_target_pos, last_position, NotReached
    sim_time += dt

    x, y, z, roll, pitch, yaw = state
    tx, ty, tz, tyaw = target_pos

    error_yaw = yaw-tyaw
    while(error_yaw < 0):
        error_yaw += 2*np.pi
    # If reached target
    if np.abs(x-tx) + np.abs(y-ty) + np.abs(z-tz) + error_yaw < 0.1 and NotReached:
        NotReached = False
        print(f"[INFO] Reached target!")
    
    if np.abs(x-tx) + np.abs(y-ty) + np.abs(z-tz) + error_yaw > 0.1 and NotReached == False:
        NotReached = True
        print(f"[INFO] Off target! Adjusting...")

    # If the target changes, reset the PID and initialize the position
    if target_pos != last_target_pos or last_position is None:
        print(f"[DEBUG] Target changed from {last_target_pos} to {target_pos}, resetting all PIDs.")
        reset_all_pid()
        last_target_pos = target_pos
        last_position = np.array([x, y, z])
        NotReached = True
        return (0, 0, 0, 0)

    # Calculate the current speed of the drone (by position difference)
    current_position = np.array([x, y, z])
    current_velocity_global = (current_position - last_position) / dt
    last_position = current_position

    # Outer PID: Position → Expected speed (Global coordinate system)
    pos_error = np.array([tx - x, ty - y, tz - z])
    vel_desired_global = pid_pos.control_update(pos_error, dt)
    vel_desired_global = np.clip(vel_desired_global, -1.0, 1.0)

    # Middle level PID: Velocity → Expected acceleration
    vel_error = vel_desired_global - current_velocity_global
    accel_desired_global = pid_vel.control_update(vel_error, dt)

    # Convert expected acceleration to attitude angle
    g = 9.81
    desired_roll = np.clip(np.arctan2(-accel_desired_global[1], g), -np.pi/6, np.pi/6)
    desired_pitch = np.clip(np.arctan2(accel_desired_global[0], g), -np.pi/6, np.pi/6)

    # Inner loop PID: attitude control (including yaw)
    att_error = np.array([desired_roll - roll, desired_pitch - pitch, tyaw - yaw])
    att_error[2] = np.arctan2(np.sin(att_error[2]), np.cos(att_error[2]))
    angular_rates_cmd = pid_att.control_update(att_error, dt)

    yaw_rate = np.clip(angular_rates_cmd[2], -0.5, 0.5)
    vz = np.clip(vel_desired_global[2], -1.0, 1.0)

    # Convert the speed from the global coordinate system to the drone's own coordinate system
    R = rotation_matrix(roll, pitch, yaw).T
    vel_body = R @ vel_desired_global

    vx = np.clip(vel_body[0], -1.0, 1.0)
    vy = np.clip(vel_body[1], -1.0, 1.0)

    # print(f"vx: {vx}, vy: {vy}, vz: {vz}, vyaw: {yaw_rate}")

    return (vx, vy, vz, yaw_rate)
