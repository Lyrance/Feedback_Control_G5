import numpy as np

###############################################################################
# 1) Your PIDController (supports 3D vectors, integral saturation)
###############################################################################
class PIDController:
    def __init__(self, Kp, Ki, Kd, Ki_sat):
        """
        :param Kp, Ki, Kd: list/array, length 3, for x/y/z directions
        :param Ki_sat: list/array, length 3, integral saturation for x/y/z
        """
        self.Kp = np.array(Kp)
        self.Ki = np.array(Ki)
        self.Kd = np.array(Kd)
        self.Ki_sat = np.array(Ki_sat)

        self.previous_error = np.zeros(3)  # Previous error
        self.int = np.zeros(3)            # Integral term

    def reset(self):
        """Reset integral and derivative terms"""
        self.int = np.zeros(3)
        self.previous_error = np.zeros(3)

    def control_update(self, error, timestep):
        """
        :param error: np.array(3) current error [ex, ey, ez]
        :param timestep: time step
        :return: np.array(3) output [out_x, out_y, out_z]
        """
        self.int += error * timestep
        # Anti-windup
        self.int = np.clip(self.int, -self.Ki_sat, self.Ki_sat)

        derivative = (error - self.previous_error) / timestep
        self.previous_error = error

        output = self.Kp*error + self.Ki*self.int + self.Kd*derivative
        return output


###############################################################################
# 2) Ziegler–Nichols tuner with "max velocity limit" scaling
###############################################################################
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


###############################################################################
# 3) Global: PID + ZN tuner for X/Y/Z/Yaw
###############################################################################
# Initialize with small gains (or Ki=0 to prevent windup)
pid_x = PIDController(Kp=[0.1,0,0], Ki=[0,0,0], Kd=[0.05,0,0], Ki_sat=[0.5,0,0])
pid_y = PIDController(Kp=[0.1,0,0], Ki=[0,0,0], Kd=[0.05,0,0], Ki_sat=[0.5,0,0])
pid_z = PIDController(Kp=[0.1,0,0], Ki=[0,0,0], Kd=[0.03,0,0], Ki_sat=[0.5,0,0])
pid_yaw = PIDController(Kp=[0.1,0,0], Ki=[0,0,0], Kd=[0.01,0,0], Ki_sat=[0.5,0,0])

zntuner_x = ZNTuner("X")
zntuner_y = ZNTuner("Y")
zntuner_z = ZNTuner("Z")
zntuner_yaw = ZNTuner("Yaw")

# Whether to enable auto Z-N tuning for X/Y/Z/Yaw
TUNE_X = False
TUNE_Y = False
TUNE_Z = False
TUNE_YAW = False

if TUNE_X:
    zntuner_x.start_tuning()
if TUNE_Y:
    zntuner_y.start_tuning()
if TUNE_Z:
    zntuner_z.start_tuning()
if TUNE_YAW:
    zntuner_yaw.start_tuning()

# If known good gains are available, set them here:
pid_x.Kp = [0.250,0,0]
pid_x.Ki = [0.324,0,0]
pid_x.Kd = [0.034,0,0]
pid_y.Kp = [0.250,0,0]
pid_y.Ki = [0.324,0,0]
pid_y.Kd = [0.034,0,0]
pid_z.Kp = [0.250,0,0]
pid_z.Ki = [0.155,0,0]
pid_z.Kd = [0.071,0,0]
pid_yaw.Kp = [0.210,0,0]
pid_yaw.Ki = [0.160,0,0]
pid_yaw.Kd = [0.048,0,0]

low_gain_mode = False
time_of_switch = 0.0

# Save original gains for restoration
orig_gains = {
    "x": {
        "Kp": pid_x.Kp.copy(),
        "Ki": pid_x.Ki.copy(),
        "Kd": pid_x.Kd.copy(),
    },
    "y": {
        "Kp": pid_y.Kp.copy(),
        "Ki": pid_y.Ki.copy(),
        "Kd": pid_y.Kd.copy(),
    },
    "z": {
        "Kp": pid_z.Kp.copy(),
        "Ki": pid_z.Ki.copy(),
        "Kd": pid_z.Kd.copy(),
    },
    "yaw": {
        "Kp": pid_yaw.Kp.copy(),
        "Ki": pid_yaw.Ki.copy(),
        "Kd": pid_yaw.Kd.copy(),
    },
}

# Define a low-gain configuration
low_gains = {
    "x": {"Kp": [0.05,0,0], "Ki":[0,0,0], "Kd":[0.01,0,0]},
    "y": {"Kp": [0.05,0,0], "Ki":[0,0,0], "Kd":[0.01,0,0]},
    "z": {"Kp": [0.05,0,0], "Ki":[0,0,0], "Kd":[0.01,0,0]},
    "yaw": {"Kp": [0.05,0,0], "Ki":[0,0,0], "Kd":[0.01,0,0]},
}

# Global timer
sim_time = 0.0

# Detect target changes
last_target_pos = (None, None, None, None)

debug_counter = 0 

# Internal yaw target for smooth transition
internal_target_yaw = None

# Key: track control phase (first xyz, then yaw)
phase = 1  # 1: move xyz, 2: then control yaw

def reset_all_pid():
    pid_x.reset()
    pid_y.reset()
    pid_z.reset()
    pid_yaw.reset()

###############################################################################
# 4) controller() function: reset PID on target change + multi-axis
###############################################################################
def controller(state, target_pos, dt):
    """
    :param state: [x, y, z, roll, pitch, yaw]
    :param target_pos: (tx, ty, tz, tyaw)
    :param dt: time step
    :return: (vx, vy, vz, yaw_rate)
    """
    global sim_time, last_target_pos, debug_counter, phase, internal_target_yaw
    global low_gain_mode, time_of_switch
    sim_time += dt

    x, y, z, roll, pitch, yaw = state
    tx, ty, tz, tyaw = target_pos

    # 1) If current target_pos is different from last one => reset all 4 PIDs
    if target_pos != last_target_pos:
        print(f"[DEBUG] Target changed from {last_target_pos} to {target_pos}, resetting all PIDs.")
        reset_all_pid()
        # Start with phase=1 (control xyz), keep yaw fixed
        phase = 1
        # When target changes, set internal yaw target to current yaw to avoid sudden error
        internal_target_yaw = yaw
        last_target_pos = target_pos
        return (0, 0, 0, 0)
    

    # 2) Compute error
    ex = tx - x
    ey = ty - y
    ez = tz - z
    e_yaw = tyaw - yaw
    
    # Print if target is reached (xyz only)
    if np.abs(ex) < 0.05 and np.abs(ey) < 0.05 and np.abs(ez) < 0.05 and np.abs(e_yaw) < 0.05:
        print(f"[INFO] Reached target!")

    # Step 1: If still in phase=1, only control xyz, yaw_rate = 0
    # if phase == 1:
    # compute xyz pid
    out_x = pid_x.control_update(np.array([ex,0,0]), dt)[0]
    out_y = pid_y.control_update(np.array([ey,0,0]), dt)[0]
    out_z = pid_z.control_update(np.array([ez,0,0]), dt)[0]

    # yaw = 0 => yaw_rate = 0
    out_yaw = 0

    # Velocity saturation
    vx_old = np.clip(out_x, -0.8, 0.8)
    vy_old = np.clip(out_y, -0.8, 0.8)
    vz = np.clip(out_z, -0.8, 0.8)
    
    # Translate from global vx,vy to local
    vx = vx_old*np.cos(yaw)+vy_old*np.sin(yaw)
    vy = -vx_old*np.sin(yaw)+vy_old*np.cos(yaw)
    
    # Smoothly update internal_target_yaw to gradually approach external tyaw
    alpha = 0.1  # Update rate, can be tuned
    internal_target_yaw = internal_target_yaw + alpha * (tyaw - internal_target_yaw)
    
    # Calculate yaw error and wrap to (-pi, pi)
    e_yaw = internal_target_yaw - yaw
    e_yaw = np.arctan2(np.sin(e_yaw), np.cos(e_yaw))

    # Use yaw PID controller to compute output
    out_yaw = pid_yaw.control_update(np.array([e_yaw,0,0]), dt)[0]
    yaw_rate = np.clip(out_yaw, -0.5, 0.5)

    return (vx, vy, vz, yaw_rate)

