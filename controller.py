# -*- coding: utf-8 -*-
"""
Three-level cascade PID controller with Ziegler–Nichols autotuning.
"""

import numpy as np
from src.PID_controller import PIDController


class ZNTuner:
    """Ziegler–Nichols tuner for PID autotuning with optional velocity limiting."""
    def __init__(
        self,
        name: str = "unnamed",
        kp_start: float = 0.1,
        kp_step: float = 0.05,
        kp_max: float = 5.0
    ):
        self.name = name
        self.state = "OFF"
        self.Kp_current = kp_start
        self.Kp_step = kp_step
        self.Kp_max = kp_max
        self.last_error_sign = 0
        self.zero_cross_times: list[float] = []
        self.K_u: float | None = None
        self.T_u: float | None = None
        self.best_pid: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def start_tuning(self) -> None:
        """Begin searching for critical gain K_u."""
        self.state = "FIND_KU"
        self.Kp_current = 0.1
        self.zero_cross_times.clear()
        self.last_error_sign = 0
        print(f"[ZNTuner-{self.name}] Start searching K_u …")

    def update_tuner(self, error: float, t: float) -> None:
        """
        Feed the scalar error into the tuner to detect zero crossings.
        When in FIND_KU, look for first sustained oscillation to set K_u;
        in OBSERVE_OSC, measure T_u from consecutive zero crossings.
        """
        if self.state not in ("FIND_KU", "OBSERVE_OSC"):
            return

        sign = np.sign(error)
        if sign != 0 and sign != self.last_error_sign:
            self.last_error_sign = sign
            self.zero_cross_times.append(t)
            print(f"[ZNTuner-{self.name}] Zero-cross at t={t:.3f}s, sign={sign}")

            if self.state == "FIND_KU" and len(self.zero_cross_times) >= 2:
                # Detected critical gain
                self.K_u = self.Kp_current
                self.state = "OBSERVE_OSC"
                self.zero_cross_times = [t]
                print(f"[ZNTuner-{self.name}] Detected Ku ≈ {self.K_u:.3f}")

            elif self.state == "OBSERVE_OSC" and len(self.zero_cross_times) >= 2:
                # Estimate oscillation period
                self.T_u = self.zero_cross_times[-1] - self.zero_cross_times[-2]
                print(f"[ZNTuner-{self.name}] Observed Tu ≈ {self.T_u:.3f}s")
                self.finish_tuning(max_vel=1.0, max_err=4.0)

    def finish_tuning(self, max_vel: float = 1.0, max_err: float = 4.0) -> None:
        """
        Compute PID gains using Z-N formulas and apply velocity-limit scaling.
        If tuning failed, fall back to safe defaults.
        """
        if self.K_u is not None and self.T_u is not None:
            Ku, Tu = self.K_u, self.T_u
            kp0 = 0.6 * Ku
            ki0 = 1.2 * Ku / Tu * 0.7
            kd0 = 0.075 * Ku * Tu
            if kp0 * max_err > max_vel:
                alpha = max_vel / (kp0 * max_err)
                kp, ki, kd = alpha * kp0, alpha * ki0, alpha * kd0
                print(f"[ZNTuner-{self.name}] Gains scaled by α={alpha:.3f}")
            else:
                kp, ki, kd = kp0, ki0, kd0
            self.best_pid = (kp, ki, kd)
            print(f"[ZNTuner-{self.name}] → Kp={kp:.3f}, Ki={ki:.3f}, Kd={kd:.3f}")
        else:
            print(f"[ZNTuner-{self.name}] No Ku or Tu—using safe defaults")
            # Safe default gains if tuning fails
            self.best_pid = (0.4, 0.05, 0.2)
        self.state = "DONE"

    def increase_kp(self) -> None:
        """Increment Kp in FIND_KU mode; stop if beyond max."""
        if self.state == "FIND_KU":
            self.Kp_current += self.Kp_step
            if self.Kp_current > self.Kp_max:
                self.state = "DONE"
                print(f"[ZNTuner-{self.name}] Max Kp reached—abort tuning")

    def is_tuning_done(self) -> bool:
        return self.state == "DONE"

    def get_current_kp_for_control(self) -> float | None:
        return self.Kp_current if self.state in ("FIND_KU", "OBSERVE_OSC") else None

    def get_final_pid(self) -> tuple[float, float, float]:
        return self.best_pid


# --- Configuration ---
AUTO_TUNE = False

# --- Initialize PID Controllers ---
pid_pos = PIDController(
    Kp=[0.4, 0.4, 0.6], Ki=[0.05, 0.05, 0.1], Kd=[0.2, 0.2, 0.3], K_i_limit=[1, 1, 1]
)
pid_vel = PIDController(
    Kp=[0.3, 0.3, 0.5], Ki=[0.02, 0.02, 0.05], Kd=[0.15, 0.15, 0.2], K_i_limit=[0.5, 0.5, 0.5]
)
pid_att = PIDController(
    Kp=[0.8, 0.8, 0.5], Ki=[0.0, 0.0, 0.01], Kd=[0.1, 0.1, 0.05], K_i_limit=[0.1, 0.1, 0.1]
)

# Autotuner instance for position loop
pos_tuner = ZNTuner(name="POS")
if AUTO_TUNE:
    pos_tuner.start_tuning()

# --- Internal State ---
last_target_pos: tuple[float, float, float, float] | None = None
last_position: np.ndarray | None = None
sim_time: float = 0.0
not_reached: bool = True


def reset_all_pid() -> None:
    """Reset all PID controllers' internal state."""
    pid_pos.reset()
    pid_vel.reset()
    pid_att.reset()


def rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Compute rotation matrix from roll, pitch, yaw angles."""
    c, s = np.cos, np.sin
    Rz = np.array([[c(yaw), -s(yaw), 0], [s(yaw), c(yaw), 0], [0, 0, 1]])
    Ry = np.array([[c(pitch), 0, s(pitch)], [0, 1, 0], [-s(pitch), 0, c(pitch)]])
    Rx = np.array([[1, 0, 0], [0, c(roll), -s(roll)], [0, s(roll), c(roll)]])
    return Rz @ Ry @ Rx


def controller(
    state: tuple[float, float, float, float, float, float],
    target_pos: tuple[float, float, float, float],
    dt: float
) -> tuple[float, float, float, float]:
    """
    Main controller with optional Z-N autotuning for the position loop.

    Implements three-level cascade PID:
      1. Outer loop: Position PID → desired global velocity
      2. Middle loop: Velocity PID → desired global acceleration
      3. Inner loop: Attitude PID → angular rate commands (roll, pitch, yaw)
    """
    global sim_time, last_target_pos, last_position, not_reached

    sim_time += dt
    x, y, z, roll, pitch, yaw = state
    tx, ty, tz, tyaw = target_pos

    # --- Autotune position loop using Ziegler–Nichols ---
    if AUTO_TUNE and not pos_tuner.is_tuning_done():
        # Use x-axis error to drive tuner
        error_x = tx - x
        pos_tuner.update_tuner(error_x, sim_time)

        # P-only control while tuning
        kp_now = pos_tuner.get_current_kp_for_control()
        if kp_now is not None:
            pid_pos.Kp = [kp_now] * 3
            pid_pos.Ki = [0.0] * 3
            pid_pos.Kd = [0.0] * 3
        pos_tuner.increase_kp()

    if AUTO_TUNE and pos_tuner.is_tuning_done() and not hasattr(controller, "_tuned"):
        kp, ki, kd = pos_tuner.get_final_pid()
        pid_pos.Kp = [kp] * 3
        pid_pos.Ki = [ki] * 3
        pid_pos.Kd = [kd] * 3
        controller._tuned = True
        print(f"[PID_POS] Auto-tuned gains Kp={kp:.3f}, Ki={ki:.3f}, Kd={kd:.3f}")

    # --- Reset PIDs when target changes ---
    if target_pos != last_target_pos or last_position is None:
        reset_all_pid()
        last_target_pos = target_pos
        last_position = np.array([x, y, z])
        not_reached = True
        return (0.0, 0.0, 0.0, 0.0)

    # --- Compute current velocity global frame ---
    prev_pos = last_position
    current_pos = np.array([x, y, z])
    current_vel_global = (current_pos - prev_pos) / dt
    last_position = current_pos

    # --- Outer loop: Position Cascade PID ---
    # Calculate position error and desired velocity
    pos_error = np.array([tx - x, ty - y, tz - z])
    desired_vel_global = pid_pos.control_update(pos_error, dt)
    desired_vel_global = np.clip(desired_vel_global, -1.0, 1.0)

    # --- Middle loop: Velocity Cascade PID ---
    # Calculate velocity error and desired acceleration
    vel_error = desired_vel_global - current_vel_global
    desired_accel_global = pid_vel.control_update(vel_error, dt)

    # --- Inner loop: Attitude PID including yaw control ---
    g = 9.81
    # Map acceleration to desired roll/pitch angles
    desired_roll = np.clip(np.arctan2(-desired_accel_global[1], g), -np.pi/6, np.pi/6)
    desired_pitch = np.clip(np.arctan2(desired_accel_global[0], g), -np.pi/6, np.pi/6)
    # Compute yaw error
    yaw_error = np.arctan2(np.sin(tyaw - yaw), np.cos(tyaw - yaw))

    att_error = np.array([desired_roll - roll,
                          desired_pitch - pitch,
                          yaw_error])
    angular_rates_cmd = pid_att.control_update(att_error, dt)
    vz = np.clip(desired_vel_global[2], -1.0, 1.0)
    yaw_rate = np.clip(angular_rates_cmd[2], -0.5, 0.5)

    # --- Convert to body-frame velocities ---
    R = rotation_matrix(roll, pitch, yaw).T
    vel_body = R @ desired_vel_global
    vx, vy = np.clip(vel_body[:2], -1.0, 1.0)

    return (vx, vy, vz, yaw_rate)
