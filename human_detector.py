"""
Real-Time Human Detection using HOG + Upper Body Cascade (OpenCV)
=================================================================
This program uses your webcam to detect humans in real-time.

It combines TWO detection methods:
  1. HOG (Histogram of Oriented Gradients) - for full-body detection
     (works best when the full body is visible, e.g. standing/walking)
  2. Haar Cascade Upper Body Detector - for upper-body detection
     (works when only head + shoulders are visible, e.g. sitting at desk)

This combination ensures detection works whether you're standing
far from the camera OR sitting right in front of it.

Author: Beginner ML Project
Date: June 2026
"""

import cv2
import time
import os

# ============================================================
# STEP 1: Initialize BOTH Detectors
# ============================================================

# --- Detector 1: HOG Full-Body Detector ---
# Best for: detecting people standing/walking at a distance (3-10 feet)
# How it works: Analyzes gradient patterns that match human silhouettes
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# --- Detector 2: Haar Cascade Upper-Body Detector ---
# Best for: detecting people sitting at a desk / partial body visible
# How it works: Uses trained patterns of head + shoulders shape
# This file comes pre-installed with OpenCV
upper_body_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_upperbody.xml'
)

# Check that the cascade file loaded correctly
if upper_body_cascade.empty():
    print("WARNING: Could not load upper body cascade classifier!")
    print("Only HOG full-body detection will be available.")

# ============================================================
# STEP 2: Open the Webcam
# ============================================================
cap = cv2.VideoCapture(0)

# --- Error Handling: Check if the webcam opened successfully ---
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

# Print webcam info
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Webcam opened successfully! Resolution: {frame_width}x{frame_height}")
print("")
print("Controls:")
print("  Press 'q' to quit")
print("  Press 's' to save current frame")
print("  Press 'm' to switch detection mode (Both / HOG only / Upper Body only)")
print("")

# ============================================================
# STEP 3: Set up Variables
# ============================================================

frame_count = 0          # Frames processed since last FPS update
fps = 0.0                # Current frames per second
prev_time = time.time()  # Timestamp for FPS calculation
save_count = 0           # Number of saved screenshots

# Detection mode: 0 = Both, 1 = HOG only, 2 = Upper Body only
detection_mode = 0
mode_names = ["Both (HOG + Upper Body)", "HOG Full-Body Only", "Upper Body Only"]

# Create folder for saved frames
save_dir = "saved_frames"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
    print(f"Created '{save_dir}/' folder for saved screenshots.")

# ============================================================
# STEP 4: Main Detection Loop
# ============================================================
print(f"\nStarting detection in mode: {mode_names[detection_mode]}")
print("Press 'q' to quit\n")

while True:
    # --- Read a frame from the webcam ---
    ret, frame = cap.read()
    if not ret:
        print("WARNING: Failed to grab frame. Retrying...")
        continue

    frame_count += 1

    # --- Calculate FPS (updated every 0.5 seconds) ---
    current_time = time.time()
    time_diff = current_time - prev_time
    if time_diff >= 0.5:
        fps = frame_count / time_diff
        frame_count = 0
        prev_time = current_time

    # We'll collect all detected regions here
    all_detections = []

    # --------------------------------------------------------
    # STEP 4a: HOG Full-Body Detection
    # --------------------------------------------------------
    # Runs if mode is "Both" (0) or "HOG only" (1)
    if detection_mode in [0, 1]:
        (hog_regions, hog_weights) = hog.detectMultiScale(
            frame,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05
        )
        for (x, y, w, h) in hog_regions:
            all_detections.append(("HOG", x, y, w, h))

    # --------------------------------------------------------
    # STEP 4b: Upper Body Cascade Detection
    # --------------------------------------------------------
    # Runs if mode is "Both" (0) or "Upper Body only" (2)
    # This is what detects you when sitting at your desk!
    if detection_mode in [0, 2]:
        # Convert to grayscale (Haar cascades work on grayscale images)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        upper_bodies = upper_body_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,   # How much to shrink image each scale
            minNeighbors=4,    # Higher = fewer false positives, but may miss some
            minSize=(60, 60),  # Minimum detection size in pixels
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        for (x, y, w, h) in upper_bodies:
            all_detections.append(("Upper", x, y, w, h))

    # Count total humans detected
    num_humans = len(all_detections)

    # --------------------------------------------------------
    # STEP 4c: Draw Bounding Boxes
    # --------------------------------------------------------
    for (det_type, x, y, w, h) in all_detections:
        if det_type == "HOG":
            # Green box for HOG full-body detections
            color = (0, 255, 0)
            label = "Person (Full Body)"
        else:
            # Cyan box for upper-body detections
            color = (255, 255, 0)
            label = "Person (Upper Body)"

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(
            frame, label, (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2
        )

    # --------------------------------------------------------
    # STEP 4d: Display Information Overlays
    # --------------------------------------------------------

    # --- Top-Left: Number of humans detected ---
    humans_text = f"Humans Detected: {num_humans}"
    cv2.rectangle(frame, (5, 5), (280, 40), (0, 0, 0), -1)
    cv2.putText(
        frame, humans_text, (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
    )

    # --- Top-Right: FPS counter ---
    fps_text = f"FPS: {fps:.1f}"
    text_size = cv2.getTextSize(fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
    fps_x = frame.shape[1] - text_size[0] - 15
    cv2.rectangle(frame, (fps_x - 5, 5), (frame.shape[1] - 5, 40), (0, 0, 0), -1)
    cv2.putText(
        frame, fps_text, (fps_x, 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
    )

    # --- Below count: Current detection mode ---
    mode_text = f"Mode: {mode_names[detection_mode]}"
    cv2.rectangle(frame, (5, 45), (400, 75), (0, 0, 0), -1)
    cv2.putText(
        frame, mode_text, (10, 67),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 255), 1
    )

    # --- Bottom: Instructions ---
    instructions = "'q' Quit | 's' Save | 'm' Switch Mode"
    cv2.rectangle(
        frame,
        (5, frame.shape[0] - 35),
        (420, frame.shape[0] - 5),
        (0, 0, 0), -1
    )
    cv2.putText(
        frame, instructions,
        (10, frame.shape[0] - 12),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
    )

    # --------------------------------------------------------
    # STEP 4e: Show the Frame
    # --------------------------------------------------------
    cv2.imshow("Human Detection - HOG + Upper Body", frame)

    # --------------------------------------------------------
    # STEP 4f: Handle Keyboard Input
    # --------------------------------------------------------
    key = cv2.waitKey(1) & 0xFF

    # Press 'q' to quit
    if key == ord('q'):
        print("\n'q' pressed. Exiting...")
        break

    # Press 's' to save the current frame
    if key == ord('s'):
        save_count += 1
        filename = os.path.join(save_dir, f"capture_{save_count}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Frame saved as: {filename}")

    # Press 'm' to cycle through detection modes
    if key == ord('m'):
        detection_mode = (detection_mode + 1) % 3
        print(f"Switched to mode: {mode_names[detection_mode]}")

# ============================================================
# STEP 5: Clean Up
# ============================================================
cap.release()
cv2.destroyAllWindows()
print(f"\nProgram ended. Total frames saved: {save_count}")
print("Goodbye!")
