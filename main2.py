"""
Eye Tracking and Head Pose Estimation with Mouse Control

This script is designed to perform real-time eye tracking and head pose estimation using a webcam feed
with mouse control functionality. It utilizes the MediaPipe library for facial landmark detection, 
which informs both eye tracking and head pose calculations. The mouse cursor will move based on 
head pose direction (forward/left/right/up/down).

New Features:
- Mouse control based on head pose direction
- Configurable mouse sensitivity and movement speed
- Smooth mouse movement with acceleration
- Toggle mouse control on/off

Requirements:
- Python 3.x
- OpenCV (opencv-python)
- MediaPipe (mediapipe)
- pynput (for mouse control)
- Other Dependencies: math, socket, argparse, time, csv, datetime, os

Usage:
- Press 'c' to recalibrate the head pose estimation to the current orientation.
- Press 'r' to start/stop logging.
- Press 'm' to toggle mouse control on/off.
- Press 'q' to exit the program.

Author: Modified for mouse control integration
"""

import cv2 as cv
import numpy as np
import mediapipe as mp
import math
import socket
import argparse
import time
import csv
from datetime import datetime
import os
from AngleBuffer import AngleBuffer
from pynput.mouse import Button, Listener as MouseListener
from pynput import mouse
import threading

#-----------------------------------------------------------------------------------------------------------------------------------
# MOUSE CONTROL PARAMETERS
#-----------------------------------------------------------------------------------------------------------------------------------

# Mouse control configuration
ENABLE_MOUSE_CONTROL = True  # Enable/disable mouse control
MOUSE_SENSITIVITY = 2.0  # Mouse movement sensitivity (higher = more sensitive)
MOUSE_SMOOTHING = 0.3  # Mouse movement smoothing (0.1-1.0, lower = smoother)
MOUSE_DEAD_ZONE = 5  # Dead zone for small head movements (degrees)
MOUSE_MAX_SPEED = 20  # Maximum mouse movement speed (pixels per frame)
MOUSE_ACCELERATION = 1.5  # Mouse acceleration factor

# Mouse control state
mouse_control_active = True
mouse_controller = mouse.Controller()
last_mouse_time = time.time()
accumulated_mouse_x = 0.0
accumulated_mouse_y = 0.0

#-----------------------------------------------------------------------------------------------------------------------------------
# ORIGINAL PARAMETERS
#-----------------------------------------------------------------------------------------------------------------------------------

## User-Specific Measurements
USER_FACE_WIDTH = 140  # [mm]

## Camera Parameters
NOSE_TO_CAMERA_DISTANCE = 600  # [mm]

## Configuration Parameters
PRINT_DATA = True
DEFAULT_WEBCAM = 0
SHOW_ALL_FEATURES = True
LOG_DATA = True
LOG_ALL_FEATURES = False
ENABLE_HEAD_POSE = True

## Logging Configuration
LOG_FOLDER = "logs"

## Server Configuration
SERVER_IP = "127.0.0.1"
SERVER_PORT = 7070

## Blink Detection Parameters
SHOW_ON_SCREEN_DATA = True
TOTAL_BLINKS = 0
EYES_BLINK_FRAME_COUNTER = 0
BLINK_THRESHOLD = 0.51
EYE_AR_CONSEC_FRAMES = 2

## Head Pose Estimation Landmark Indices
LEFT_EYE_IRIS = [474, 475, 476, 477]
RIGHT_EYE_IRIS = [469, 470, 471, 472]
LEFT_EYE_OUTER_CORNER = [33]
LEFT_EYE_INNER_CORNER = [133]
RIGHT_EYE_OUTER_CORNER = [362]
RIGHT_EYE_INNER_CORNER = [263]
RIGHT_EYE_POINTS = [33, 160, 159, 158, 133, 153, 145, 144]
LEFT_EYE_POINTS = [362, 385, 386, 387, 263, 373, 374, 380]
NOSE_TIP_INDEX = 4
CHIN_INDEX = 152
LEFT_EYE_LEFT_CORNER_INDEX = 33
RIGHT_EYE_RIGHT_CORNER_INDEX = 263
LEFT_MOUTH_CORNER_INDEX = 61
RIGHT_MOUTH_CORNER_INDEX = 291

## MediaPipe Model Confidence Parameters
MIN_DETECTION_CONFIDENCE = 0.8
MIN_TRACKING_CONFIDENCE = 0.8

## Angle Normalization Parameters
MOVING_AVERAGE_WINDOW = 10

# Initial Calibration Flags
initial_pitch, initial_yaw, initial_roll = None, None, None
calibrated = False

# Server configuration
SERVER_ADDRESS = (SERVER_IP, SERVER_PORT)

#If set to false it will wait for your command (hitting 'r') to start logging.
IS_RECORDING = False

# Command-line arguments for camera source
parser = argparse.ArgumentParser(description="Eye Tracking Application with Mouse Control")
parser.add_argument(
    "-c", "--camSource", help="Source of camera", default=str(DEFAULT_WEBCAM)
)
args = parser.parse_args()

# Iris and eye corners landmarks indices
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
L_H_LEFT = [33]
L_H_RIGHT = [133]
R_H_LEFT = [362]
R_H_RIGHT = [263]

# Face Selected points indices for Head Pose Estimation
_indices_pose = [1, 33, 61, 199, 263, 291]

#-----------------------------------------------------------------------------------------------------------------------------------
# MOUSE CONTROL FUNCTIONS
#-----------------------------------------------------------------------------------------------------------------------------------

def update_mouse_position(pitch, yaw, roll):
    """Update mouse position based on head pose angles"""
    global accumulated_mouse_x, accumulated_mouse_y, last_mouse_time
    
    if not mouse_control_active or not ENABLE_MOUSE_CONTROL:
        return
    
    current_time = time.time()
    dt = current_time - last_mouse_time
    last_mouse_time = current_time
    
    # Apply dead zone to reduce jitter
    if abs(yaw) < MOUSE_DEAD_ZONE and abs(pitch) < MOUSE_DEAD_ZONE:
        return
    
    # Calculate mouse movement based on head pose
    # Yaw controls horizontal movement, pitch controls vertical movement
    mouse_dx = 0
    mouse_dy = 0
    
    # Horizontal movement (yaw)
    if abs(yaw) > MOUSE_DEAD_ZONE:
        # Positive yaw = look right = move mouse right
        mouse_dx = (yaw / 30.0) * MOUSE_SENSITIVITY * MOUSE_ACCELERATION
        mouse_dx = max(-MOUSE_MAX_SPEED, min(MOUSE_MAX_SPEED, mouse_dx))
    
    # Vertical movement (pitch)
    if abs(pitch) > MOUSE_DEAD_ZONE:
        # Positive pitch = look up = move mouse up (negative dy)
        mouse_dy = -(pitch / 30.0) * MOUSE_SENSITIVITY * MOUSE_ACCELERATION
        mouse_dy = max(-MOUSE_MAX_SPEED, min(MOUSE_MAX_SPEED, mouse_dy))
    
    # Apply smoothing
    accumulated_mouse_x = accumulated_mouse_x * (1 - MOUSE_SMOOTHING) + mouse_dx * MOUSE_SMOOTHING
    accumulated_mouse_y = accumulated_mouse_y * (1 - MOUSE_SMOOTHING) + mouse_dy * MOUSE_SMOOTHING
    
    # Move mouse if movement is significant enough
    if abs(accumulated_mouse_x) > 0.5 or abs(accumulated_mouse_y) > 0.5:
        try:
            mouse_controller.move(int(accumulated_mouse_x), int(accumulated_mouse_y))
            # Reset accumulated values after movement
            accumulated_mouse_x *= 0.5
            accumulated_mouse_y *= 0.5
        except Exception as e:
            if PRINT_DATA:
                print(f"Mouse control error: {e}")

def toggle_mouse_control():
    """Toggle mouse control on/off"""
    global mouse_control_active
    mouse_control_active = not mouse_control_active
    if PRINT_DATA:
        status = "enabled" if mouse_control_active else "disabled"
        print(f"Mouse control {status}")

#-----------------------------------------------------------------------------------------------------------------------------------
# ORIGINAL FUNCTIONS (unchanged)
#-----------------------------------------------------------------------------------------------------------------------------------

def vector_position(point1, point2):
    x1, y1 = point1.ravel()
    x2, y2 = point2.ravel()
    return x2 - x1, y2 - y1

def euclidean_distance_3D(points):
    P0, P3, P4, P5, P8, P11, P12, P13 = points
    numerator = (
        np.linalg.norm(P3 - P13) ** 3
        + np.linalg.norm(P4 - P12) ** 3
        + np.linalg.norm(P5 - P11) ** 3
    )
    denominator = 3 * np.linalg.norm(P0 - P8) ** 3
    distance = numerator / denominator
    return distance

def estimate_head_pose(landmarks, image_size):
    scale_factor = USER_FACE_WIDTH / 150.0
    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -330.0 * scale_factor, -65.0 * scale_factor),
        (-225.0 * scale_factor, 170.0 * scale_factor, -135.0 * scale_factor),
        (225.0 * scale_factor, 170.0 * scale_factor, -135.0 * scale_factor),
        (-150.0 * scale_factor, -150.0 * scale_factor, -125.0 * scale_factor),
        (150.0 * scale_factor, -150.0 * scale_factor, -125.0 * scale_factor)
    ])
    
    focal_length = image_size[1]
    center = (image_size[1]/2, image_size[0]/2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype = "double"
    )
    
    dist_coeffs = np.zeros((4,1))
    
    image_points = np.array([
        landmarks[NOSE_TIP_INDEX],
        landmarks[CHIN_INDEX],
        landmarks[LEFT_EYE_LEFT_CORNER_INDEX],
        landmarks[RIGHT_EYE_RIGHT_CORNER_INDEX],
        landmarks[LEFT_MOUTH_CORNER_INDEX],
        landmarks[RIGHT_MOUTH_CORNER_INDEX]
    ], dtype="double")
    
    (success, rotation_vector, translation_vector) = cv.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE)
    
    rotation_matrix, _ = cv.Rodrigues(rotation_vector)
    projection_matrix = np.hstack((rotation_matrix, translation_vector.reshape(-1, 1)))
    _, _, _, _, _, _, euler_angles = cv.decomposeProjectionMatrix(projection_matrix)
    pitch, yaw, roll = euler_angles.flatten()[:3]
    
    pitch = normalize_pitch(pitch)
    return pitch, yaw, roll

def normalize_pitch(pitch):
    if pitch > 180:
        pitch -= 360
    pitch = -pitch
    if pitch < -90:
        pitch = -(180 + pitch)
    elif pitch > 90:
        pitch = 180 - pitch
    pitch = -pitch
    return pitch

def blinking_ratio(landmarks):
    right_eye_ratio = euclidean_distance_3D(landmarks[RIGHT_EYE_POINTS])
    left_eye_ratio = euclidean_distance_3D(landmarks[LEFT_EYE_POINTS])
    ratio = (right_eye_ratio + left_eye_ratio + 1) / 2
    return ratio

#-----------------------------------------------------------------------------------------------------------------------------------
# MAIN PROGRAM
#-----------------------------------------------------------------------------------------------------------------------------------

# Initializing MediaPipe face mesh and camera
if PRINT_DATA:
    print("Initializing the face mesh and camera...")
    head_pose_status = "enabled" if ENABLE_HEAD_POSE else "disabled"
    mouse_control_status = "enabled" if ENABLE_MOUSE_CONTROL else "disabled"
    print(f"Head pose estimation is {head_pose_status}.")
    print(f"Mouse control is {mouse_control_status}.")
    print("Controls:")
    print("  'c' - Recalibrate head pose")
    print("  'r' - Toggle recording")
    print("  'm' - Toggle mouse control")
    print("  'q' - Quit")

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=MIN_DETECTION_CONFIDENCE,
    min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
)
cam_source = int(args.camSource)
cap = cv.VideoCapture(cam_source)

# Initializing socket for data transmission
iris_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Preparing for CSV logging
csv_data = []
if not os.path.exists(LOG_FOLDER):
    os.makedirs(LOG_FOLDER)

# Column names for CSV file
column_names = [
    "Timestamp (ms)",
    "Left Eye Center X",
    "Left Eye Center Y",
    "Right Eye Center X",
    "Right Eye Center Y",
    "Left Iris Relative Pos Dx",
    "Left Iris Relative Pos Dy",
    "Right Iris Relative Pos Dx",
    "Right Iris Relative Pos Dy",
    "Total Blink Count",
]
if ENABLE_HEAD_POSE:
    column_names.extend(["Pitch", "Yaw", "Roll"])
    
if LOG_ALL_FEATURES:
    column_names.extend(
        [f"Landmark_{i}_X" for i in range(468)]
        + [f"Landmark_{i}_Y" for i in range(468)]
    )

# Main loop for video capture and processing
try:
    angle_buffer = AngleBuffer(size=MOVING_AVERAGE_WINDOW)
    pitch, yaw, roll = 0, 0, 0  # Initialize angles

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv.flip(frame, 1)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]
        results = mp_face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            mesh_points = np.array(
                [
                    np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                    for p in results.multi_face_landmarks[0].landmark
                ]
            )

            mesh_points_3D = np.array(
                [[n.x, n.y, n.z] for n in results.multi_face_landmarks[0].landmark]
            )
            
            head_pose_points_3D = np.multiply(
                mesh_points_3D[_indices_pose], [img_w, img_h, 1]
            )
            head_pose_points_2D = mesh_points[_indices_pose]

            nose_3D_point = np.multiply(head_pose_points_3D[0], [1, 1, 3000])
            nose_2D_point = head_pose_points_2D[0]

            focal_length = 1 * img_w
            cam_matrix = np.array(
                [[focal_length, 0, img_h / 2], [0, focal_length, img_w / 2], [0, 0, 1]]
            )

            dist_matrix = np.zeros((4, 1), dtype=np.float64)
            head_pose_points_2D = np.delete(head_pose_points_3D, 2, axis=1)
            head_pose_points_3D = head_pose_points_3D.astype(np.float64)
            head_pose_points_2D = head_pose_points_2D.astype(np.float64)
            
            success, rot_vec, trans_vec = cv.solvePnP(
                head_pose_points_3D, head_pose_points_2D, cam_matrix, dist_matrix
            )
            
            rotation_matrix, jac = cv.Rodrigues(rot_vec)
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv.RQDecomp3x3(rotation_matrix)

            angle_x = angles[0] * 360
            angle_y = angles[1] * 360
            z = angles[2] * 360

            threshold_angle = 10
            if angle_y < -threshold_angle:
                face_looks = "Left"
            elif angle_y > threshold_angle:
                face_looks = "Right"
            elif angle_x < -threshold_angle:
                face_looks = "Down"
            elif angle_x > threshold_angle:
                face_looks = "Up"
            else:
                face_looks = "Forward"

            # Head pose estimation for mouse control
            if ENABLE_HEAD_POSE:
                pitch, yaw, roll = estimate_head_pose(mesh_points, (img_h, img_w))
                angle_buffer.add([pitch, yaw, roll])
                pitch, yaw, roll = angle_buffer.get_average()

                # Set initial angles on first successful estimation or recalibrate
                if initial_pitch is None:
                    initial_pitch, initial_yaw, initial_roll = pitch, yaw, roll
                    calibrated = True
                    if PRINT_DATA:
                        print("Head pose calibrated.")

                # Adjust angles based on initial calibration
                if calibrated:
                    pitch -= initial_pitch
                    yaw -= initial_yaw
                    roll -= initial_roll

                # Update mouse position based on head pose
                update_mouse_position(pitch, yaw, roll)

            if SHOW_ON_SCREEN_DATA:
                cv.putText(
                    frame,
                    f"Face Looking at {face_looks}",
                    (img_w - 400, 80),
                    cv.FONT_HERSHEY_TRIPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv.LINE_AA,
                )

            # Display the nose direction
            nose_3d_projection, jacobian = cv.projectPoints(
                nose_3D_point, rot_vec, trans_vec, cam_matrix, dist_matrix
            )

            p1 = nose_2D_point
            p2 = (
                int(nose_2D_point[0] + angle_y * 10),
                int(nose_2D_point[1] - angle_x * 10),
            )

            cv.line(frame, p1, p2, (255, 0, 255), 3)
            
            # Blink detection
            eyes_aspect_ratio = blinking_ratio(mesh_points_3D)
            
            if eyes_aspect_ratio <= BLINK_THRESHOLD:
                EYES_BLINK_FRAME_COUNTER += 1
            else:
                if EYES_BLINK_FRAME_COUNTER > EYE_AR_CONSEC_FRAMES:
                    TOTAL_BLINKS += 1
                EYES_BLINK_FRAME_COUNTER = 0
            
            # Display all facial landmarks if enabled
            if SHOW_ALL_FEATURES:
                for point in mesh_points:
                    cv.circle(frame, tuple(point), 1, (0, 255, 0), -1)
            
            # Process and display eye features
            (l_cx, l_cy), l_radius = cv.minEnclosingCircle(mesh_points[LEFT_EYE_IRIS])
            (r_cx, r_cy), r_radius = cv.minEnclosingCircle(mesh_points[RIGHT_EYE_IRIS])
            center_left = np.array([l_cx, l_cy], dtype=np.int32)
            center_right = np.array([r_cx, r_cy], dtype=np.int32)

            # Highlighting the irises and corners of the eyes
            cv.circle(frame, center_left, int(l_radius), (255, 0, 255), 2, cv.LINE_AA)
            cv.circle(frame, center_right, int(r_radius), (255, 0, 255), 2, cv.LINE_AA)
            cv.circle(frame, mesh_points[LEFT_EYE_INNER_CORNER][0], 3, (255, 255, 255), -1, cv.LINE_AA)
            cv.circle(frame, mesh_points[LEFT_EYE_OUTER_CORNER][0], 3, (0, 255, 255), -1, cv.LINE_AA)
            cv.circle(frame, mesh_points[RIGHT_EYE_INNER_CORNER][0], 3, (255, 255, 255), -1, cv.LINE_AA)
            cv.circle(frame, mesh_points[RIGHT_EYE_OUTER_CORNER][0], 3, (0, 255, 255), -1, cv.LINE_AA)

            # Calculating relative positions
            l_dx, l_dy = vector_position(mesh_points[LEFT_EYE_OUTER_CORNER], center_left)
            r_dx, r_dy = vector_position(mesh_points[RIGHT_EYE_OUTER_CORNER], center_right)

            # Printing data if enabled
            if PRINT_DATA:
                print(f"Total Blinks: {TOTAL_BLINKS}")
                print(f"Left Eye Center X: {l_cx} Y: {l_cy}")
                print(f"Right Eye Center X: {r_cx} Y: {r_cy}")
                print(f"Left Iris Relative Pos Dx: {l_dx} Dy: {l_dy}")
                print(f"Right Iris Relative Pos Dx: {r_dx} Dy: {r_dy}")
                if ENABLE_HEAD_POSE:
                    print(f"Head Pose Angles: Pitch={pitch:.1f}, Yaw={yaw:.1f}, Roll={roll:.1f}")
                print()

            # Logging data
            if LOG_DATA and IS_RECORDING:
                timestamp = int(time.time() * 1000)
                log_entry = [timestamp, l_cx, l_cy, r_cx, r_cy, l_dx, l_dy, r_dx, r_dy, TOTAL_BLINKS]
                
                if ENABLE_HEAD_POSE:
                    log_entry.extend([pitch, yaw, roll])
                
                if LOG_ALL_FEATURES:
                    log_entry.extend([p for point in mesh_points for p in point])
                
                csv_data.append(log_entry)

        # Writing the on screen data on the frame
        if SHOW_ON_SCREEN_DATA:
            if IS_RECORDING:
                cv.circle(frame, (30, 30), 10, (0, 0, 255), -1)
            cv.putText(frame, f"Blinks: {TOTAL_BLINKS}", (30, 80), cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)
            if ENABLE_HEAD_POSE:
                cv.putText(frame, f"Pitch: {int(pitch)}", (30, 110), cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)
                cv.putText(frame, f"Yaw: {int(yaw)}", (30, 140), cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)
                cv.putText(frame, f"Roll: {int(roll)}", (30, 170), cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)
                
                # Display mouse control status
                mouse_status = "ON" if mouse_control_active and ENABLE_MOUSE_CONTROL else "OFF"
                color = (0, 255, 0) if mouse_control_active and ENABLE_MOUSE_CONTROL else (0, 0, 255)
                cv.putText(frame, f"Mouse: {mouse_status}", (30, 200), cv.FONT_HERSHEY_DUPLEX, 0.8, color, 2, cv.LINE_AA)

        # Displaying the processed frame
        cv.imshow("Eye Tracking with Mouse Control", frame)
        
        # Handle key presses
        key = cv.waitKey(1) & 0xFF

        # Calibrate on 'c' key press
        if key == ord('c'):
            if ENABLE_HEAD_POSE and 'pitch' in locals():
                initial_pitch, initial_yaw, initial_roll = pitch, yaw, roll
                if PRINT_DATA:
                    print("Head pose recalibrated.")
                    
        # Toggle mouse control on 'm' key press
        if key == ord('m'):
            toggle_mouse_control()
                
        # Toggle recording on 'r' key press
        if key == ord('r'):
            IS_RECORDING = not IS_RECORDING
            if PRINT_DATA:
                status = "started" if IS_RECORDING else "paused"
                print(f"Recording {status}.")

        # Exit on 'q' key press
        if key == ord('q'):
            if PRINT_DATA:
                print("Exiting program...")
            break
        
except Exception as e:
    print(f"An error occurred: {e}")
finally:
    # Releasing camera and closing windows
    cap.release()
    cv.destroyAllWindows()
    if PRINT_DATA:
        print("Program exited successfully.")

    # Writing data to CSV file
    if LOG_DATA and IS_RECORDING and csv_data:
        if PRINT_DATA:
            print("Writing data to CSV...")
        timestamp_str = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        csv_file_name = os.path.join(LOG_FOLDER, f"eye_tracking_log_{timestamp_str}.csv")
        with open(csv_file_name, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(column_names)
            writer.writerows(csv_data)
        if PRINT_DATA:
            print(f"Data written to {csv_file_name}")