"""
Eye Tracking and Head Pose Estimation with Instant Absolute Mouse Control (Full Feature, Corrected)

This script performs real-time eye tracking, head pose estimation, and blink detection
using a webcam. It retains all visual feedback features from the original script,
including landmark drawing, nose direction line, and blink counting.

The mouse control has been updated to an instant, absolute positioning system.

Requirements:
- Python 3.x, OpenCV, MediaPipe, pynput, tkinter

Usage:
- Look straight at the camera and run the script. It will calibrate automatically.
- Press 'c' to recalibrate the center point to your current head orientation.
- Press 'r' to start/stop data logging.
- Press 'm' to toggle mouse control on/off.
- Press 'q' to exit the program.
"""

import cv2 as cv
import numpy as np
import mediapipe as mp
import math
import argparse
import time
import csv
from datetime import datetime
import os
from pynput import mouse
import tkinter as tk

# Helper class untuk smoothing sudut (dipertahankan untuk kestabilan data sudut)
from AngleBuffer import AngleBuffer


#-----------------------------------------------------------------------------------------------------------------------------------
# MOUSE CONTROL PARAMETERS (INSTANT ABSOLUTE POSITIONING)
#-----------------------------------------------------------------------------------------------------------------------------------

# Otomatis deteksi dimensi layar
try:
    root = tk.Tk()
    SCREEN_WIDTH, SCREEN_HEIGHT = root.winfo_screenwidth(), root.winfo_screenheight()
    root.destroy()
except tk.TclError:
    print("Peringatan: Tidak dapat mendeteksi ukuran layar. Menggunakan default 1920x1080.")
    SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080

# Konfigurasi kontrol mouse
ENABLE_MOUSE_CONTROL = True
MOUSE_SENSITIVITY_FACTOR = 50  # Nilai lebih tinggi -> gerakan mouse lebih besar
MOUSE_DEAD_ZONE = 2.0          # Sudut (derajat) di mana mouse tidak bergerak dari tengah

# Status kontrol mouse
mouse_control_active = True
mouse_controller = mouse.Controller()

#-----------------------------------------------------------------------------------------------------------------------------------
# PARAMETER ORIGINAL (DIPERTAHANKAN)
#-----------------------------------------------------------------------------------------------------------------------------------
USER_FACE_WIDTH = 140
NOSE_TO_CAMERA_DISTANCE = 600
PRINT_DATA = True
DEFAULT_WEBCAM = 0
SHOW_ALL_FEATURES = True
LOG_DATA = True
LOG_ALL_FEATURES = False
ENABLE_HEAD_POSE = True
LOG_FOLDER = "logs"
SHOW_ON_SCREEN_DATA = True
TOTAL_BLINKS = 0
EYES_BLINK_FRAME_COUNTER = 0
BLINK_THRESHOLD = 0.51
EYE_AR_CONSEC_FRAMES = 2
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
MIN_DETECTION_CONFIDENCE = 0.8
MIN_TRACKING_CONFIDENCE = 0.8
MOVING_AVERAGE_WINDOW = 10
initial_pitch, initial_yaw, initial_roll = None, None, None
calibrated = False
IS_RECORDING = False
parser = argparse.ArgumentParser(description="Eye Tracking Application with Mouse Control")
parser.add_argument("-c", "--camSource", help="Source of camera", default=str(DEFAULT_WEBCAM))
args = parser.parse_args()
_indices_pose = [1, 33, 61, 199, 263, 291]

#-----------------------------------------------------------------------------------------------------------------------------------
# FUNGSI KONTROL MOUSE (METODE BARU: INSTAN & ABSOLUT)
#-----------------------------------------------------------------------------------------------------------------------------------
def update_mouse_position_instantly(pitch, yaw):
    if not mouse_control_active or not ENABLE_MOUSE_CONTROL: return
    center_x, center_y = SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2
    if abs(yaw) < MOUSE_DEAD_ZONE and abs(pitch) < MOUSE_DEAD_ZONE:
        target_x, target_y = center_x, center_y
    else:
        move_x = yaw * MOUSE_SENSITIVITY_FACTOR
        move_y = -pitch * MOUSE_SENSITIVITY_FACTOR
        target_x, target_y = center_x + move_x, center_y + move_y
    target_x = max(0, min(SCREEN_WIDTH - 1, int(target_x)))
    target_y = max(0, min(SCREEN_HEIGHT - 1, int(target_y)))
    try: mouse_controller.position = (target_x, target_y)
    except Exception as e:
        if PRINT_DATA: print(f"Error kontrol mouse: {e}")

def toggle_mouse_control():
    global mouse_control_active
    mouse_control_active = not mouse_control_active
    if PRINT_DATA: print(f"Kontrol mouse {'aktif' if mouse_control_active else 'nonaktif'}")

#-----------------------------------------------------------------------------------------------------------------------------------
# FUNGSI ORIGINAL (DIPERTAHANKAN SEPENUHNYA)
#-----------------------------------------------------------------------------------------------------------------------------------
def vector_position(point1, point2):
    x1, y1 = point1.ravel(); x2, y2 = point2.ravel()
    return x2 - x1, y2 - y1

def euclidean_distance_3D(points):
    P0, P3, P4, P5, P8, P11, P12, P13 = points
    numerator = (np.linalg.norm(P3 - P13)**3 + np.linalg.norm(P4 - P12)**3 + np.linalg.norm(P5 - P11)**3)
    denominator = 3 * np.linalg.norm(P0 - P8)**3
    return numerator / denominator

def estimate_head_pose(landmarks, image_size):
    scale_factor = USER_FACE_WIDTH / 150.0
    model_points = np.array([(0.0, 0.0, 0.0), (0.0, -330.0*scale_factor, -65.0*scale_factor), (-225.0*scale_factor, 170.0*scale_factor, -135.0*scale_factor), (225.0*scale_factor, 170.0*scale_factor, -135.0*scale_factor), (-150.0*scale_factor, -150.0*scale_factor, -125.0*scale_factor), (150.0*scale_factor, -150.0*scale_factor, -125.0*scale_factor)])
    focal_length = image_size[1]
    center = (image_size[1]/2, image_size[0]/2)
    camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double")
    dist_coeffs = np.zeros((4, 1))
    image_points = np.array([landmarks[NOSE_TIP_INDEX], landmarks[CHIN_INDEX], landmarks[LEFT_EYE_LEFT_CORNER_INDEX], landmarks[RIGHT_EYE_RIGHT_CORNER_INDEX], landmarks[LEFT_MOUTH_CORNER_INDEX], landmarks[RIGHT_MOUTH_CORNER_INDEX]], dtype="double")
    (success, rotation_vector, translation_vector) = cv.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv.SOLVEPNP_ITERATIVE)
    rotation_matrix, _ = cv.Rodrigues(rotation_vector)
    _, _, _, _, _, _, euler_angles = cv.decomposeProjectionMatrix(np.hstack((rotation_matrix, translation_vector.reshape(-1, 1))))
    pitch, yaw, roll = euler_angles.flatten()[:3]
    return normalize_pitch(pitch), yaw, roll

def normalize_pitch(pitch):
    if pitch > 180: pitch -= 360
    pitch = -pitch
    if pitch < -90: pitch = -(180 + pitch)
    elif pitch > 90: pitch = 180 - pitch
    return -pitch

def blinking_ratio(landmarks):
    right_eye_ratio = euclidean_distance_3D(landmarks[RIGHT_EYE_POINTS])
    left_eye_ratio = euclidean_distance_3D(landmarks[LEFT_EYE_POINTS])
    return (right_eye_ratio + left_eye_ratio + 1) / 2

#-----------------------------------------------------------------------------------------------------------------------------------
# PROGRAM UTAMA
#-----------------------------------------------------------------------------------------------------------------------------------
if PRINT_DATA:
    print(f"Inisialisasi... Layar Terdeteksi: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")
    print("--- Tombol Kontrol ---\n 'c': Kalibrasi | 'm': Toggle Mouse | 'r': Rekam | 'q': Keluar")

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=MIN_DETECTION_CONFIDENCE, min_tracking_confidence=MIN_TRACKING_CONFIDENCE)
cap = cv.VideoCapture(int(args.camSource))

if not os.path.exists(LOG_FOLDER): os.makedirs(LOG_FOLDER)
csv_data = []
column_names = ["Timestamp (ms)", "Left Eye Center X", "Left Eye Center Y", "Right Eye Center X", "Right Eye Center Y", "Left Iris Dx", "Left Iris Dy", "Right Iris Dx", "Right Iris Dy", "Total Blinks"]
if ENABLE_HEAD_POSE: column_names.extend(["Pitch", "Yaw", "Roll"])
if LOG_ALL_FEATURES: column_names.extend([f"Landmark_{i}_X" for i in range(468)] + [f"Landmark_{i}_Y" for i in range(468)])

try:
    angle_buffer = AngleBuffer(size=MOVING_AVERAGE_WINDOW)
    pitch, yaw, roll = 0, 0, 0
    relative_pitch, relative_yaw, relative_roll = 0, 0, 0

    while True:
        ret, frame = cap.read()
        if not ret: break

        frame = cv.flip(frame, 1)
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        img_h, img_w = frame.shape[:2]
        results = mp_face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int) for p in results.multi_face_landmarks[0].landmark])
            mesh_points_3D = np.array([[n.x, n.y, n.z] for n in results.multi_face_landmarks[0].landmark])
            
            if ENABLE_HEAD_POSE:
                pitch, yaw, roll = estimate_head_pose(mesh_points, (img_h, img_w))
                angle_buffer.add([pitch, yaw, roll])
                pitch, yaw, roll = angle_buffer.get_average()

                if not calibrated:
                    initial_pitch, initial_yaw, initial_roll = pitch, yaw, roll
                    calibrated = True
                    if PRINT_DATA: print("Kepala terkalibrasi. Mouse di tengah.")

                relative_pitch = pitch - initial_pitch
                relative_yaw = yaw - initial_yaw
                relative_roll = roll - initial_roll
                update_mouse_position_instantly(relative_pitch, relative_yaw)

            # Visualisasi garis arah kepala
            nose_2d = (mesh_points[NOSE_TIP_INDEX][0], mesh_points[NOSE_TIP_INDEX][1])
            p1 = (int(nose_2d[0]), int(nose_2d[1]))
            # PERBAIKAN: Mengurangi pengaruh pitch pada garis visual agar tidak "jatuh"
            # Anda bisa mencoba nilai 5, atau bahkan 0 jika ingin garisnya murni horizontal.
            p2 = (int(p1[0] + relative_yaw * 20), int(p1[1] - relative_pitch * 5))
            cv.line(frame, p1, p2, (255, 0, 255), 3)

            # Deteksi kedipan
            eyes_aspect_ratio = blinking_ratio(mesh_points_3D)
            if eyes_aspect_ratio <= BLINK_THRESHOLD:
                EYES_BLINK_FRAME_COUNTER += 1
            else:
                if EYES_BLINK_FRAME_COUNTER > EYE_AR_CONSEC_FRAMES: TOTAL_BLINKS += 1
                EYES_BLINK_FRAME_COUNTER = 0

            if SHOW_ALL_FEATURES:
                for point in mesh_points: cv.circle(frame, tuple(point), 1, (0, 255, 0), -1)

            # Proses dan visualisasi fitur mata
            (l_cx, l_cy), l_rad = cv.minEnclosingCircle(mesh_points[LEFT_EYE_IRIS])
            (r_cx, r_cy), r_rad = cv.minEnclosingCircle(mesh_points[RIGHT_EYE_IRIS])
            center_l = np.array([l_cx, l_cy], dtype=np.int32); center_r = np.array([r_cx, r_cy], dtype=np.int32)
            cv.circle(frame, center_l, int(l_rad), (255, 0, 255), 2, cv.LINE_AA)
            cv.circle(frame, center_r, int(r_rad), (255, 0, 255), 2, cv.LINE_AA)
            cv.circle(frame, mesh_points[LEFT_EYE_OUTER_CORNER][0], 3, (0, 255, 255), -1, cv.LINE_AA)
            cv.circle(frame, mesh_points[RIGHT_EYE_OUTER_CORNER][0], 3, (0, 255, 255), -1, cv.LINE_AA)

            # Logging data dan print
            l_dx, l_dy = vector_position(mesh_points[LEFT_EYE_OUTER_CORNER], center_l)
            r_dx, r_dy = vector_position(mesh_points[RIGHT_EYE_OUTER_CORNER], center_r)
            if PRINT_DATA:
                print(f"Blinks: {TOTAL_BLINKS} | Head Pose (rel): Pitch={relative_pitch:.1f}, Yaw={relative_yaw:.1f}", end="\r")
            if LOG_DATA and IS_RECORDING:
                log_entry = [int(time.time()*1000), l_cx, l_cy, r_cx, r_cy, l_dx, l_dy, r_dx, r_dy, TOTAL_BLINKS]
                if ENABLE_HEAD_POSE: log_entry.extend([pitch, yaw, roll])
                if LOG_ALL_FEATURES: log_entry.extend([p for point in mesh_points for p in point])
                csv_data.append(log_entry)

        # Visualisasi data di layar
        if SHOW_ON_SCREEN_DATA:
            if IS_RECORDING: cv.circle(frame, (30, 30), 10, (0, 0, 255), -1)
            cv.putText(frame, f"Blinks: {TOTAL_BLINKS}", (30, 80), cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)
            if ENABLE_HEAD_POSE and calibrated:
                cv.putText(frame, f"Pitch: {int(relative_pitch)}", (30, 110), cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)
                cv.putText(frame, f"Yaw: {int(relative_yaw)}", (30, 140), cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)
                mouse_status = "ON" if mouse_control_active else "OFF"
                color = (0, 255, 0) if mouse_control_active else (0, 0, 255)
                cv.putText(frame, f"Mouse: {mouse_status}", (30, 170), cv.FONT_HERSHEY_DUPLEX, 0.8, color, 2, cv.LINE_AA)

        cv.imshow("Eye Tracking & Instant Mouse Control", frame)
        key = cv.waitKey(1) & 0xFF

        if key == ord('c'):
            if ENABLE_HEAD_POSE and 'pitch' in locals():
                initial_pitch, initial_yaw, initial_roll = pitch, yaw, roll
                mouse_controller.position = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
                if PRINT_DATA: print("\nPosisi kepala dikalibrasi ulang. Mouse kembali ke tengah.")
        if key == ord('m'): toggle_mouse_control()
        if key == ord('r'):
            IS_RECORDING = not IS_RECORDING
            if PRINT_DATA: print(f"\nPerekaman {'dimulai' if IS_RECORDING else 'dihentikan'}.")
        if key == ord('q'): break
        
except Exception as e:
    import traceback
    print(f"\nTerjadi error tak terduga: {e}")
    traceback.print_exc()
finally:
    cap.release()
    cv.destroyAllWindows()
    if PRINT_DATA: print("Program selesai.")
    if LOG_DATA and IS_RECORDING and csv_data:
        timestamp_str = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        csv_file_name = os.path.join(LOG_FOLDER, f"eye_tracking_log_{timestamp_str}.csv")
        with open(csv_file_name, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(column_names)
            writer.writerows(csv_data)
        if PRINT_DATA: print(f"Data disimpan ke {csv_file_name}")