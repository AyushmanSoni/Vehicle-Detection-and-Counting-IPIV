import cv2
import cvzone
import math
import numpy as np
from ultralytics import YOLO
from sort import *

# ------------------ SETTINGS ------------------ #
video_path = 'carsvid.mp4'        # Path to your input video
model_path = 'yolov8n.pt'   # YOLO model
zones_file = 'zones.npy'    # Saved zones file from Create_Zones.py
# ------------------------------------------------#

# Load the YOLO model
model = YOLO(model_path)

# Load class names
with open('classes.txt', 'r') as f:
    classnames = f.read().splitlines()

# Load pre-saved zones (created using Create_Zones.py)
zones = np.load(zones_file, allow_pickle=True)

# Initialize video capture
cap = cv2.VideoCapture(video_path)

# Initialize SORT tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)

# Create a counter list for each zone
zone_counters = [[] for _ in range(len(zones))]

# ------------------ MAIN LOOP ------------------ #
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize for consistency (match the size used during zone creation)
    frame = cv2.resize(frame, (1280, 720))
    height, width, _ = frame.shape

    # Detect vehicles
    results = model(frame)
    current_detections = np.empty([0, 5])

    for info in results:
        for box in info.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            cls = int(box.cls[0])
            class_name = classnames[cls]
            conf_percent = math.ceil(confidence * 100)

            # Filter classes & confidence
            if class_name in ['car', 'truck', 'bus'] and conf_percent > 60:
                detections = np.array([x1, y1, x2, y2, conf_percent])
                current_detections = np.vstack([current_detections, detections])

                cvzone.putTextRect(frame, f'{class_name} {conf_percent}%',
                                   [x1 + 8, y1 - 12], thickness=2, scale=0.9)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # -------- TRACKING -------- #
    track_results = tracker.update(current_detections)

    for result in track_results:
        x1, y1, x2, y2, track_id = map(int, result)
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2 - 40

        # Draw center point
        cv2.circle(frame, (cx, cy), 4, (255, 255, 255), -1)

        # -------- ZONE CHECK -------- #
        for i, zone in enumerate(zones):
            inside = cv2.pointPolygonTest(np.array(zone, np.int32), (cx, cy), False)
            if inside >= 0:
                if track_id not in zone_counters[i]:
                    zone_counters[i].append(track_id)

    # -------- DRAW ZONES -------- #
    for i, zone in enumerate(zones):
        color = (0, 255 - i * 80, 255)
        cv2.polylines(frame, [np.array(zone, np.int32)], isClosed=True, color=color, thickness=3)

    # -------- DISPLAY COUNTS (Dynamic Position) -------- #
    text_x = int(width * 0.75)
    base_y = int(height * 0.08)
    gap = int(height * 0.06)
    circle_x = text_x - 40

    for i, counter in enumerate(zone_counters):
        color = [(0, 255, 255), (0, 165, 255), (0, 0, 255)][i % 3]  # cycle colors
        y_pos = base_y + i * gap
        cv2.circle(frame, (circle_x, y_pos), 10, color, -1)
        cvzone.putTextRect(frame, f'Zone {i + 1} Vehicles = {len(counter)}',
                           [text_x, y_pos], thickness=2, scale=1.3)

    # Display the frame
    cv2.imshow('Vehicle Counting', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ------------------ CLEANUP ------------------ #
cap.release()
cv2.destroyAllWindows()
