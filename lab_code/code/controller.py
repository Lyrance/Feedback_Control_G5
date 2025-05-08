# Implement a controller
# def controller(state, target_pos, dt):
#     # state format: [position_x (m), position_y (m), position_z (m),
#     #                roll (radians), pitch (radians), yaw (radians)]
#     # target_pos format: (x (m), y (m), z (m), yaw (radians))
#     # timestamp (ms)
#     # return velocity command format: (velocity_x_setpoint (m/s), velocity_y_setpoint (m/s), velocity_z_setpoint (m/s), yaw_rate_setpoint (degrees/s))


import numpy as np
import os
from src.PID_controller import PIDController

# Initialize PID controller
pid_pos = PIDController([0.4, 0.4, 0.6], [0.05, 0.05, 0.1], [0.2, 0.2, 0.3], [1, 1, 1])
pid_vel = PIDController([0.3, 0.3, 0.5], [0.02, 0.02, 0.05], [0.15, 0.15, 0.2], [0.5, 0.5, 0.5])
pid_att = PIDController([0.9, 0.8, 0.5], [0, 0, 0.01], [0.1, 0.1, 0.05], [0.1, 0.1, 0.1])

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


DEBUG_MODE = False

TELLO_VICON_NAME = "tello_marker3" # Vicon object name
TELLO_ID = "111" # Tello ID

POSITION_ERROR = 0.2
YAW_ERROR = 0.2
MAX_SPEED = 50
last_timestamp = None

LOG_FILE = "output-lab3.csv"

def log_data(timestamp_ms, state, target_pos):
    """
    	timestamp_ms, x, y, z, roll, pitch, yaw, tx, ty, tz, tyaw
    """
    row = np.hstack((timestamp_ms, state, target_pos))   # shape = (11,)

    need_header = (not os.path.exists(LOG_FILE)) or os.path.getsize(LOG_FILE) == 0
    with open(LOG_FILE, "a") as f:
        if need_header:
            header = ("timestamp_ms,x,y,z,roll,pitch,yaw,"
                      "target_x,target_y,target_z,target_yaw\n")
            f.write(header)
        np.savetxt(f, [row], delimiter=",", fmt="%.6f")

def controller(state, target_pos, timestamp):

    x, y, z = state[0], state[1], state[2]
    yaw = state[5]
    
    log_data(timestamp, state, target_pos)
    
    print(f"x: {x}, y: {y}, z: {z}, yaw: {yaw}")
    print(f"Target: {target_pos}")
    
    global sim_time, last_target_pos, last_position, NotReached, last_timestamp
    sim_time = timestamp
    if last_timestamp is None:
        dt = 1e-3               
    else:
        dt = (timestamp - last_timestamp) / 1000.0
    last_timestamp = timestamp

    x, y, z, roll, pitch, yaw = state
    tx, ty, tz, tyaw = target_pos

    error_yaw = tyaw-yaw
    # while(error_yaw < 0):
    #     error_yaw += 2*np.pi

    print(f"error_yaw: {error_yaw}")
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
    yaw_rate_deg = yaw_rate * 180.0 / np.pi

    return (vx, vy, vz, yaw_rate)


