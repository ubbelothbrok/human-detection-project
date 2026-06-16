"""
Real-Time Human Detection — Powered by Deep Learning (MediaPipe + OpenCV)
=========================================================================
This program detects humans in real-time using your webcam with
THREE complementary detectors:

  1. Face Detection (BlazeFace)  — Detects you if even just your face is visible
  2. Pose Detection (BlazePose)  — Detects your body from any visible part
                                   (hands, shoulders, torso, legs, etc.)
  3. HOG Full-Body               — Classical detector for people at a distance

This makes detection robust whether you're:
  - Sitting at your desk (face visible)
  - Standing far away (full body visible)
  - Partially visible (any body part showing)

Author: Beginner ML Project
Date: June 2026
"""

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import time
import os
import numpy as np

# ============================================================
# STEP 1: Initialize All Detectors
# ============================================================

# --- Get the directory where this script is located ---
# This ensures model files are found regardless of where you run from
script_dir = os.path.dirname(os.path.abspath(__file__))

# --- Detector 1: MediaPipe Face Detection (BlazeFace) ---
# Uses a deep neural network to detect faces.
# Works with small faces, side profiles, and partial occlusion.
face_model_path = os.path.join(script_dir, "face_detector.tflite")
if not os.path.exists(face_model_path):
    print(f"ERROR: Face model not found at: {face_model_path}")
    print("Download it with:")
    print('  wget -O face_detector.tflite "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"')
    exit(1)

face_base_options = python.BaseOptions(model_asset_path=face_model_path)
face_options = vision.FaceDetectorOptions(
    base_options=face_base_options,
    min_detection_confidence=0.5
)
face_detector = vision.FaceDetector.create_from_options(face_options)

# --- Detector 2: MediaPipe Pose Detection (BlazePose) ---
# Detects 33 body landmarks. If ANY part of the body is visible
# (face, hands, torso, legs), it can detect and track the person.
pose_model_path = os.path.join(script_dir, "pose_landmarker.task")
if not os.path.exists(pose_model_path):
    print(f"ERROR: Pose model not found at: {pose_model_path}")
    print("Download it with:")
    print('  wget -O pose_landmarker.task "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"')
    exit(1)

pose_base_options = python.BaseOptions(model_asset_path=pose_model_path)
pose_options = vision.PoseLandmarkerOptions(
    base_options=pose_base_options,
    min_pose_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)

# --- Detector 3: HOG Full-Body Detector (classical, for distant people) ---
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

print("All detectors loaded successfully!")

# ============================================================
# STEP 2: Open the Webcam
# ============================================================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("=" * 50)
    print("ERROR: Could not access the webcam!")
    print("")
    print("Possible reasons:")
    print("  1. No webcam connected to this computer")
    print("  2. Another application is using the webcam")
    print("  3. Webcam drivers are not installed")
    print("  4. Permission denied - check camera permissions")
    print("")
    print("Try:")
    print("  - Close other apps that might use the camera")
    print("  - Reconnect your webcam")
    print("  - On Linux: check /dev/video0 exists")
    print("=" * 50)
    exit(1)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Webcam opened successfully! Resolution: {frame_width}x{frame_height}")
print("")
print("Controls:")
print("  'q'  - Quit")
print("  's'  - Save current frame")
print("  'm'  - Switch detection mode")
print("  'b'  - Toggle skeleton drawing ON/OFF")
print("")

# ============================================================
# STEP 3: Set up Variables
# ============================================================

frame_count = 0
fps = 0.0
prev_time = time.time()
save_count = 0
show_skeleton = True

# Detection modes
detection_mode = 0
mode_names = [
    "All (Face + Pose + HOG)",
    "Face + Pose (Deep Learning)",
    "Face Only",
    "Pose Only",
    "HOG Full-Body Only"
]

# Pose landmark connections for drawing skeleton
# These pairs define which body points are connected by lines
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),    # Left eye to ear
    (0, 4), (4, 5), (5, 6), (6, 8),    # Right eye to ear
    (9, 10),                             # Mouth
    (11, 12),                            # Shoulders
    (11, 13), (13, 15),                  # Left arm
    (12, 14), (14, 16),                  # Right arm
    (11, 23), (12, 24),                  # Torso sides
    (23, 24),                            # Hips
    (23, 25), (25, 27),                  # Left leg
    (24, 26), (26, 28),                  # Right leg
    (27, 29), (29, 31),                  # Left foot
    (28, 30), (30, 32),                  # Right foot
    (15, 17), (15, 19), (15, 21),       # Left hand
    (16, 18), (16, 20), (16, 22),       # Right hand
]

# Create folder for saved frames
save_dir = "saved_frames"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# ============================================================
# STEP 4: Main Detection Loop
# ============================================================
print(f"Starting detection in mode: {mode_names[detection_mode]}")
print("Press 'q' to quit\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("WARNING: Failed to grab frame. Retrying...")
        continue

    frame_count += 1

    # --- Calculate FPS ---
    current_time = time.time()
    time_diff = current_time - prev_time
    if time_diff >= 0.5:
        fps = frame_count / time_diff
        frame_count = 0
        prev_time = current_time

    # Convert OpenCV BGR frame to MediaPipe Image (RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    h, w, _ = frame.shape

    # Track what was detected
    face_detected = False
    pose_detected = False
    hog_detected = False
    detection_details = []

    # --------------------------------------------------------
    # STEP 4a: Face Detection (MediaPipe BlazeFace)
    # --------------------------------------------------------
    if detection_mode in [0, 1, 2]:
        try:
            face_results = face_detector.detect(mp_image)

            if face_results.detections:
                face_detected = True
                for detection in face_results.detections:
                    bbox = detection.bounding_box

                    x = bbox.origin_x
                    y = bbox.origin_y
                    box_w = bbox.width
                    box_h = bbox.height

                    # Get confidence score
                    confidence = detection.categories[0].score

                    # Draw a cyan bounding box around the face
                    cv2.rectangle(frame, (x, y), (x + box_w, y + box_h),
                                  (255, 255, 0), 2)

                    label = f"Face {confidence:.0%}"
                    cv2.putText(frame, label, (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                    detection_details.append("Face")
        except Exception as e:
            pass  # Skip frame if detection fails

    # --------------------------------------------------------
    # STEP 4b: Pose Detection (MediaPipe BlazePose)
    # --------------------------------------------------------
    if detection_mode in [0, 1, 3]:
        try:
            pose_results = pose_landmarker.detect(mp_image)

            if pose_results.pose_landmarks:
                pose_detected = True

                for person_landmarks in pose_results.pose_landmarks:
                    # Calculate bounding box from visible landmarks
                    x_coords = []
                    y_coords = []

                    for lm in person_landmarks:
                        px = int(lm.x * w)
                        py = int(lm.y * h)
                        if lm.visibility and lm.visibility > 0.5:
                            x_coords.append(px)
                            y_coords.append(py)

                    if x_coords and y_coords:
                        padding = 20
                        x_min = max(0, min(x_coords) - padding)
                        y_min = max(0, min(y_coords) - padding)
                        x_max = min(w, max(x_coords) + padding)
                        y_max = min(h, max(y_coords) + padding)

                        # Draw green bounding box around the person
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max),
                                      (0, 255, 0), 2)
                        cv2.putText(frame, "Person (Pose)", (x_min, y_min - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                    # Draw the skeleton if enabled
                    if show_skeleton:
                        # Draw landmark dots
                        for lm in person_landmarks:
                            if lm.visibility and lm.visibility > 0.5:
                                cx = int(lm.x * w)
                                cy = int(lm.y * h)
                                cv2.circle(frame, (cx, cy), 4, (0, 200, 255), -1)

                        # Draw connections between landmarks
                        for start_idx, end_idx in POSE_CONNECTIONS:
                            if start_idx < len(person_landmarks) and end_idx < len(person_landmarks):
                                lm1 = person_landmarks[start_idx]
                                lm2 = person_landmarks[end_idx]
                                if (lm1.visibility and lm1.visibility > 0.5 and
                                    lm2.visibility and lm2.visibility > 0.5):
                                    pt1 = (int(lm1.x * w), int(lm1.y * h))
                                    pt2 = (int(lm2.x * w), int(lm2.y * h))
                                    cv2.line(frame, pt1, pt2, (0, 255, 128), 2)

                    detection_details.append("Pose")
        except Exception as e:
            pass  # Skip frame if detection fails

    # --------------------------------------------------------
    # STEP 4c: HOG Full-Body Detection (for distant people)
    # --------------------------------------------------------
    if detection_mode in [0, 4]:
        (hog_regions, _) = hog.detectMultiScale(
            frame,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05
        )
        if len(hog_regions) > 0:
            hog_detected = True
            for (x, y, bw, bh) in hog_regions:
                cv2.rectangle(frame, (x, y), (x + bw, y + bh),
                              (255, 0, 255), 2)
                cv2.putText(frame, "Person (HOG)", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
                detection_details.append("HOG")

    # --------------------------------------------------------
    # STEP 4d: Display Information Overlays
    # --------------------------------------------------------

    human_present = face_detected or pose_detected or hog_detected
    num_detections = len(detection_details)

    # --- Top-Left: Detection status ---
    if human_present:
        status_text = f"Human Detected! ({num_detections} signals)"
        status_color = (0, 255, 0)
    else:
        status_text = "No Human Detected"
        status_color = (0, 0, 255)

    cv2.rectangle(frame, (5, 5), (350, 40), (0, 0, 0), -1)
    cv2.putText(frame, status_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

    # --- Detection methods used ---
    if detection_details:
        methods = ", ".join(sorted(set(detection_details)))
        cv2.rectangle(frame, (5, 45), (350, 72), (0, 0, 0), -1)
        cv2.putText(frame, f"Via: {methods}", (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # --- Top-Right: FPS ---
    fps_text = f"FPS: {fps:.1f}"
    text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    fps_x = frame.shape[1] - text_size[0] - 15
    cv2.rectangle(frame, (fps_x - 5, 5), (frame.shape[1] - 5, 40), (0, 0, 0), -1)
    cv2.putText(frame, fps_text, (fps_x, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # --- Bottom: Mode and controls ---
    mode_short = mode_names[detection_mode]
    skeleton_status = "ON" if show_skeleton else "OFF"
    bottom_text = f"Mode: {mode_short} | Skeleton: {skeleton_status}"
    cv2.rectangle(frame, (5, h - 55), (w - 5, h - 30), (0, 0, 0), -1)
    cv2.putText(frame, bottom_text, (10, h - 37),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 255), 1)

    controls_text = "'q' Quit | 's' Save | 'm' Mode | 'b' Skeleton"
    cv2.rectangle(frame, (5, h - 30), (w - 5, h - 5), (0, 0, 0), -1)
    cv2.putText(frame, controls_text, (10, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

    # --------------------------------------------------------
    # STEP 4e: Show the Frame
    # --------------------------------------------------------
    cv2.imshow("Human Detection - Deep Learning (MediaPipe)", frame)

    # --------------------------------------------------------
    # STEP 4f: Handle Keyboard Input
    # --------------------------------------------------------
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        print("\n'q' pressed. Exiting...")
        break

    if key == ord('s'):
        save_count += 1
        filename = os.path.join(save_dir, f"capture_{save_count}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Frame saved as: {filename}")

    if key == ord('m'):
        detection_mode = (detection_mode + 1) % len(mode_names)
        print(f"Switched to mode: {mode_names[detection_mode]}")

    if key == ord('b'):
        show_skeleton = not show_skeleton
        print(f"Skeleton drawing: {'ON' if show_skeleton else 'OFF'}")

# ============================================================
# STEP 5: Clean Up
# ============================================================
cap.release()
cv2.destroyAllWindows()
print(f"\nProgram ended. Total frames saved: {save_count}")
print("Goodbye!")
