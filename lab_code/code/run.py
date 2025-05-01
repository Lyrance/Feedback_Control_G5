# Script to run controller for hardware lab

import controller
from pyvicon_datastream import tools # need to be installed
import time
from djitellopy import tello # need to be installed
import numpy as np
import csv
import cv2
import math

VICON_ON = True

#Set vicon server IP and the object name to be tracked
VICON_TRACKER_IP = "192.168.10.1"
#OBJECT_NAME = controller.TELLO_VICON_NAME
OBJECT_NAME = controller.TELLO_VICON_NAME
FD = None
if VICON_ON:
	mytracker = tools.ObjectTracker(VICON_TRACKER_IP)
     
TELLO_IP = f"192.168.10.{controller.TELLO_ID}"
FREQUENCY = 10 # Hz

POSITION_ERROR = controller.POSITION_ERROR
YAW_ERROR = controller.YAW_ERROR

def load_targets():
        targets = []
        with open("targets.csv", "r") as file:
            csvreader = csv.reader(file)
            header = next(csvreader)
            for row in csvreader:
                if float(row[2]) < 0:
                    print("WARNING: Target z below the ground, not loading target")
                else:
                    targets.append((float(row[0]), float(row[1]), float(row[2]), float(row[3])))
        return targets
def clamp(n, max_speed):
    if (n > max_speed or n < -max_speed):
        print("Control value is out of bounds...")
    clamped = max(-max_speed, min(n, max_speed))
    return clamped

def calculate_error(current_pose, target_pose):
    current_position = np.array([current_pose[0], current_pose[1], current_pose[2]])
    target_position = np.array([target_pose[0], target_pose[1], target_pose[2]])
    current_yaw = current_pose[5]
    target_yaw = target_pose[3]

    position_error = np.linalg.norm(target_position - current_position)

    def normalise_angle(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi
    
    yaw_error = normalise_angle(target_yaw - current_yaw)

    return position_error, yaw_error

def transform_pose(position):
    pose_data = position[0][2:8] # x=right, y=forward, z=up
    pos_x = pose_data[0]/1000
    pos_y = pose_data[1]/1000
    pos_z = pose_data[2]/1000
    ang_x = pose_data[3]
    ang_y = pose_data[4]
    ang_z = pose_data[5]
    # Transform to correct coordinate frame (x: forward, y: left, z: up)
    # Needs this transform unless it is possible to set the Vicon up like this
    new_x = pos_y
    new_y = -pos_x
    new_z = pos_z
    new_ang_x = ang_y
    new_ang_y = -ang_x
    new_ang_z = ang_z

    state = [new_x,new_y,new_z,new_ang_x, new_ang_y, new_ang_z]

    return state
     

def main():
    drone = tello.Tello(host=TELLO_IP) # Instantiate a Tello Object and should connect to the right IP address

    drone.connect(False) # Connect to the Tello
    print("Tello taking off in 3 seconds...")
     #time.sleep(3)
    drone.takeoff()
    previous_state = [0,0,0,0,0,0]
    # current_target = [0,0,0,0] # target_pos format: (x (m), y (m), z (m), yaw (degrees))
    target_poses = load_targets()
    pose_index = 0
    current_target = target_poses[pose_index]
    print(current_target)
    max_speed = controller.MAX_SPEED
    if (max_speed < 0):
         print("Speed has to be positive value...")
         return -1
    elif (max_speed > 100):
         print("Max speed cannot be more than 100 cm/s")
         print("Setting speed to be 100...")
         max_speed = 100
    else:
         print(f"Max speed: {max_speed}")

    while(True):
        start_time = time.time()
        # Get current pose from Vicon
        _, _, position = mytracker.get_position(OBJECT_NAME)
        if len(position)>0:
            state = transform_pose(position)
        else:
            state = previous_state

        # Call controller function
        timestamp = round(time.time() * 1000)
        vx, vy, vz, v_yaw = controller.controller(state, current_target, timestamp)
        a = int(-vy * 100) # left is negative in SDK
        b = int(vx * 100) # forward is negative
        c = int(vz * 100) # up is negative
        d = int(math.degrees(v_yaw))

        a = clamp(a, max_speed)
        b = clamp(b, max_speed)
        c = clamp(c, max_speed)
        d = clamp(d, max_speed)

        #print(f"a: {a}, b: {b}, c: {c}, d: {d}")

        drone.send_rc_control(a, b, c, d) # This function restricts the values to be between -100-100 but you should make sure
                                          # that the sent control command is within this range

        position_error, yaw_error = calculate_error(state, current_target)
        print(f"position error: {position_error}")
        if ((abs(position_error) < POSITION_ERROR)):
             if (pose_index == len(target_poses) - 1):
                  print("Target sequence finished...")
                  print("Exiting and landing...")
                  
                  break
             else:
                  pose_index = pose_index + 1
                  current_target = target_poses[pose_index]
                  #drone.send_command_without_return("stop")
                  print(f"Advancing to target {pose_index}: {current_target}")




        current_time = time.time()
        elapsed_time = current_time - start_time
        if (elapsed_time > (1/FREQUENCY)):
            pass
        else:
            delay = (1/FREQUENCY) - elapsed_time
            time.sleep(delay)
        
        previous_state = state

        key = cv2.waitKey(1) & 0xFF # doesn't work Ctrl+C to stops

        if key == 27:
            print("Interrupted, drone is landing...")
            break
    
    # Drone landing and releasing port
    drone.land()
    drone.end()
    return 0

if __name__ == "__main__":
    main()
