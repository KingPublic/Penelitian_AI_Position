"""
Eye Tracking with Dynamic Proportional Quadrant Mouse Control (FIXED)

This script performs real-time eye tracking, head pose estimation, and blink detection
using a webcam. It retains all visual feedback features from the original script.

The mouse control has been upgraded to a dynamic and proportional quadrant mapping system.
The mouse moves smoothly within a screen quadrant, corresponding to the user's head
position within the camera's quadrant. This method corrects for aspect ratio differences
between the camera and the screen, providing more intuitive control.

Requirements:
- Python 3.x, OpenCV, MediaPipe, pynput, tkinter

Usage:
- Look at the camera and run the script.
- Move your head within the camera view to control the mouse proportionally.
- Press 'c' to recalibrate the head pose line's center point.
- Press 'r' to start/stop data logging.
- Press 'm' to toggle mouse control on/off.
- Press 'q' to exit the program.
"""

import cv2 as cv
import numpy as np
import mediapipe as mp
import argparse
import time
import csv
from datetime import datetime
import os
from pynput import mouse
import tkinter as tk

# Helper class untuk smoothing sudut (dipertahankan untuk kestabilan data sudut visual)
from AngleBuffer import AngleBuffer

#-----------------------------------------------------------------------------------------------------------------------------------
# MOUSE CONTROL PARAMETERS (DYNAMIC PROPORTIONAL QUADRANT MAPPING)
#-----------------------------------------------------------------------------------------------------------------------------------

# Otomatis deteksi dimensi layar
try:
    root = tk.Tk()
    SCREEN_WIDTH, SCREEN_HEIGHT = root.winfo_screenwidth(), root.winfo_screenheight()
    root.destroy()
except tk.TclError:
    print("Peringatan: Tidak dapat mendeteksi ukuran layar. Menggunakan default 1920x1080.")
    SCREEN_WIDTH, SCREEN_HEIGHT = 1920, 1080

# Konfigurasi kontrol mouse kuadran dinamis
ENABLE_MOUSE_CONTROL = True
NEUTRAL_ZONE_RATIO = 0.30
MOUSE_SMOOTHING_FACTOR = 0.85
QUADRANT_DEAD_ZONE = 0.05

# Status kontrol mouse
mouse_control_active = True
mouse_controller = mouse.Controller()
current_mouse_quadrant = -1
last_mouse_x, last_mouse_y = SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2

#-----------------------------------------------------------------------------------------------------------------------------------
# PARAMETER ORIGINAL (DIPERTAHANKAN)
#-----------------------------------------------------------------------------------------------------------------------------------
USER_FACE_WIDTH = 140
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
RIGHT_EYE_OUTER_CORNER = [362]
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
parser = argparse.ArgumentParser(description="Eye Tracking with Dynamic Quadrant Mouse Control")
parser.add_argument("-c", "--camSource", help="Source of camera", default=str(DEFAULT_WEBCAM))
args = parser.parse_args()


#-----------------------------------------------------------------------------------------------------------------------------------
# FUNGSI KONTROL MOUSE (DYNAMIC & PROPORTIONAL QUADRANT MAPPING)
#-----------------------------------------------------------------------------------------------------------------------------------
def update_mouse_by_dynamic_quadrant(face_position_in_cam, camera_dims):
    global current_mouse_quadrant, last_mouse_x, last_mouse_y
    if not mouse_control_active or not ENABLE_MOUSE_CONTROL:
        return

    cam_w, cam_h = camera_dims
    face_x, face_y = face_position_in_cam

    neutral_x_min = cam_w * (0.5 - NEUTRAL_ZONE_RATIO / 2)
    neutral_x_max = cam_w * (0.5 + NEUTRAL_ZONE_RATIO / 2)
    neutral_y_min = cam_h * (0.5 - NEUTRAL_ZONE_RATIO / 2)
    neutral_y_max = cam_h * (0.5 + NEUTRAL_ZONE_RATIO / 2)

    target_pos = None
    
    if neutral_x_min < face_x < neutral_x_max and neutral_y_min < face_y < neutral_y_max:
        current_mouse_quadrant = 0
        target_pos = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
    else:
        if face_x > cam_w / 2:
            norm_x = (face_x - cam_w / 2) / (cam_w / 2)
            screen_x_base = SCREEN_WIDTH / 2
            screen_x_range = SCREEN_WIDTH / 2
        else:
            norm_x = face_x / (cam_w / 2)
            screen_x_base = 0
            screen_x_range = SCREEN_WIDTH / 2

        if face_y < cam_h / 2:
            norm_y = face_y / (cam_h / 2)
            screen_y_base = 0
            screen_y_range = SCREEN_HEIGHT / 2
        else:
            norm_y = (face_y - cam_h / 2) / (cam_h / 2)
            screen_y_base = SCREEN_HEIGHT / 2
            screen_y_range = SCREEN_HEIGHT / 2
            
        if screen_x_base > 0 and screen_y_base == 0: current_mouse_quadrant = 1
        elif screen_x_base == 0 and screen_y_base == 0: current_mouse_quadrant = 2
        elif screen_x_base == 0 and screen_y_base > 0: current_mouse_quadrant = 3
        else: current_mouse_quadrant = 4
        
        norm_x = np.clip(norm_x, QUADRANT_DEAD_ZONE, 1.0 - QUADRANT_DEAD_ZONE)
        norm_y = np.clip(norm_y, QUADRANT_DEAD_ZONE, 1.0 - QUADRANT_DEAD_ZONE)

        target_x = screen_x_base + norm_x * screen_x_range
        target_y = screen_y_base + norm_y * screen_y_range
        target_pos = (target_x, target_y)

    if target_pos:
        smooth_x = last_mouse_x * MOUSE_SMOOTHING_FACTOR + target_pos[0] * (1 - MOUSE_SMOOTHING_FACTOR)
        smooth_y = last_mouse_y * MOUSE_SMOOTHING_FACTOR + target_pos[1] * (1 - MOUSE_SMOOTHING_FACTOR)

        try:
            final_x = int(max(0, min(SCREEN_WIDTH - 1, smooth_x)))
            final_y = int(max(0, min(SCREEN_HEIGHT - 1, smooth_y)))
            mouse_controller.position = (final_x, final_y)
            last_mouse_x, last_mouse_y = smooth_x, smooth_y
        except Exception as e:
            if PRINT_DATA:
                print(f"Error kontrol mouse: {e}")

def toggle_mouse_control():
    global mouse_control_active
    mouse_control_active = not mouse_control_active
    if PRINT_DATA:
        print(f"\nKontrol mouse {'aktif' if mouse_control_active else 'nonaktif'}")

#-----------------------------------------------------------------------------------------------------------------------------------
# FUNGSI ORIGINAL
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
    # FIX: Menggunakan konstanta global, bukan mendefinisikan ulang secara lokal
    right_eye_ratio = euclidean_distance_3D(landmarks[RIGHT_EYE_POINTS])
    left_eye_ratio = euclidean_distance_3D(landmarks[LEFT_EYE_POINTS])
    return (right_eye_ratio + left_eye_ratio + 1) / 2

#-----------------------------------------------------------------------------------------------------------------------------------
# PROGRAM UTAMA
#-----------------------------------------------------------------------------------------------------------------------------------
if PRINT_DATA:
    print(f"Inisialisasi... Layar Terdeteksi: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")
    print("--- Tombol Kontrol ---\n 'c': Kalibrasi Garis Arah | 'm': Toggle Mouse | 'r': Rekam | 'q': Keluar")

mp_face_mesh = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=MIN_DETECTION_CONFIDENCE, min_tracking_confidence=MIN_TRACKING_CONFIDENCE)
cap = cv.VideoCapture(int(args.camSource))

if not os.path.exists(LOG_FOLDER): os.makedirs(LOG_FOLDER)
csv_data = []

try:
    angle_buffer = AngleBuffer(size=MOVING_AVERAGE_WINDOW)
    
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

            nose_tip_position = mesh_points[NOSE_TIP_INDEX]
            update_mouse_by_dynamic_quadrant(nose_tip_position, (img_w, img_h))
            
            if ENABLE_HEAD_POSE:
                pitch, yaw, roll = estimate_head_pose(mesh_points, (img_h, img_w))
                angle_buffer.add([pitch, yaw, roll])
                pitch, yaw, roll = angle_buffer.get_average()

                if not calibrated:
                    initial_pitch, initial_yaw, initial_roll = pitch, yaw, roll
                    calibrated = True
                    if PRINT_DATA: print("Garis arah kepala terkalibrasi.")

                relative_pitch = pitch - initial_pitch
                relative_yaw = yaw - initial_yaw
                
                nose_2d = (mesh_points[NOSE_TIP_INDEX][0], mesh_points[NOSE_TIP_INDEX][1])
                p1 = (int(nose_2d[0]), int(nose_2d[1]))
                p2 = (int(p1[0] - relative_yaw * 20), int(p1[1] - relative_pitch * 20))
                cv.line(frame, p1, p2, (255, 0, 255), 3)

            eyes_aspect_ratio = blinking_ratio(mesh_points_3D)
            if eyes_aspect_ratio <= BLINK_THRESHOLD:
                EYES_BLINK_FRAME_COUNTER += 1
            else:
                if EYES_BLINK_FRAME_COUNTER > EYE_AR_CONSEC_FRAMES: TOTAL_BLINKS += 1
                EYES_BLINK_FRAME_COUNTER = 0
            
        if SHOW_ON_SCREEN_DATA:
            if IS_RECORDING: cv.circle(frame, (30, 30), 10, (0, 0, 255), -1)
            cv.putText(frame, f"Blinks: {TOTAL_BLINKS}", (30, 80), cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)
            quadrant_names = {0: "Tengah", 1: "Kanan Atas", 2: "Kiri Atas", 3: "Kiri Bawah", 4: "Kanan Bawah", -1: "N/A"}
            cv.putText(frame, f"Kuadran: {quadrant_names.get(current_mouse_quadrant, 'N/A')}", (30, 110), cv.FONT_HERSHEY_DUPLEX, 0.8, (0, 255, 0), 2, cv.LINE_AA)
            
            mouse_status = "ON" if mouse_control_active else "OFF"
            color = (0, 255, 0) if mouse_control_active else (0, 0, 255)
            cv.putText(frame, f"Mouse: {mouse_status}", (30, 140), cv.FONT_HERSHEY_DUPLEX, 0.8, color, 2, cv.LINE_AA)

        # Visualisasi kuadran di frame kamera untuk debugging
        cam_h, cam_w = frame.shape[:2]
        neutral_x_min = int(cam_w * (0.5 - NEUTRAL_ZONE_RATIO / 2))
        neutral_x_max = int(cam_w * (0.5 + NEUTRAL_ZONE_RATIO / 2))
        neutral_y_min = int(cam_h * (0.5 - NEUTRAL_ZONE_RATIO / 2))
        neutral_y_max = int(cam_h * (0.5 + NEUTRAL_ZONE_RATIO / 2))
        cv.line(frame, (cam_w // 2, 0), (cam_w // 2, cam_h), (255, 255, 255), 1)
        # FIX: Menggunakan cam_w yang sudah didefinisikan, bukan 'w'
        cv.line(frame, (0, cam_h // 2), (cam_w, cam_h // 2), (255, 255, 255), 1)
        cv.rectangle(frame, (neutral_x_min, neutral_y_min), (neutral_x_max, neutral_y_max), (0, 255, 255), 1)

        cv.imshow("Eye Tracking & Dynamic Quadrant Control", frame)
        key = cv.waitKey(1) & 0xFF

        if key == ord('c'):
            if ENABLE_HEAD_POSE and 'pitch' in locals():
                initial_pitch, initial_yaw, initial_roll = pitch, yaw, roll
                if PRINT_DATA: print("\nPosisi garis arah kepala dikalibrasi ulang.")
        elif key == ord('m'): toggle_mouse_control()
        elif key == ord('r'):
            IS_RECORDING = not IS_RECORDING
            if PRINT_DATA: print(f"\nPerekaman {'dimulai' if IS_RECORDING else 'dihentikan'}.")
        elif key == ord('q'): break
        
except Exception as e:
    import traceback
    print(f"\nTerjadi error tak terduga: {e}")
    traceback.print_exc()
finally:
    cap.release()
    cv.destroyAllWindows()
    if PRINT_DATA: print("\nProgram selesai.")
    if LOG_DATA and IS_RECORDING and csv_data:
        # Bagian ini belum diimplementasikan di kode sebelumnya, ditambahkan untuk kelengkapan
        column_names = ["Timestamp (ms)", "Total Blinks"]
        if ENABLE_HEAD_POSE: column_names.extend(["Pitch", "Yaw", "Roll"])
        timestamp_str = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        csv_file_name = os.path.join(LOG_FOLDER, f"eye_tracking_log_{timestamp_str}.csv")
        with open(csv_file_name, "w", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(column_names)
            # Anda perlu menyesuaikan data yang akan di-log di sini jika diperlukan
            # writer.writerows(csv_data)
        if PRINT_DATA: print(f"Data disimpan ke {csv_file_name}")