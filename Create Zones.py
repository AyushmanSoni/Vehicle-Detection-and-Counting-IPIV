import cv2
import numpy as np

# Load video
video_path = "carsvid.mp4"
cap = cv2.VideoCapture(video_path)

zones = []          # List of all zones (each zone = list of points)
current_zone = []   # Zone being drawn

# Mouse callback for drawing polygon zones
def draw_zone(event, x, y, flags, param):
    global current_zone, zones

    # Left-click to add points
    if event == cv2.EVENT_LBUTTONDOWN:
        current_zone.append((x, y))

    # Right-click to finalize a zone
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(current_zone) >= 3:
            zones.append(current_zone.copy())
            print(f"✅ Zone {len(zones)} saved with {len(current_zone)} points")
        else:
            print("⚠️ Need at least 3 points to form a zone")
        current_zone = []  # Reset current zone

# Window setup
cv2.namedWindow("Mark Zones")
cv2.setMouseCallback("Mark Zones", draw_zone)

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    frame = cv2.resize(frame, (1280, 720))

    # Draw current zone (in progress)
    if len(current_zone) > 1:
        cv2.polylines(frame, [np.array(current_zone, np.int32)], False, (0, 255, 0), 2)
    for pt in current_zone:
        cv2.circle(frame, pt, 4, (0, 0, 255), -1)

    # Draw completed zones
    for i, zone in enumerate(zones):
        cv2.polylines(frame, [np.array(zone, np.int32)], True, (255, 0, 0), 2)
        # Label zone number at centroid
        M = cv2.moments(np.array(zone))
        if M["m00"] != 0:
            cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
            cv2.putText(frame, f"Zone {i+1}", (cx - 30, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Display help text
    cv2.putText(frame, "Left click = add point | Right click = close zone | 's' = save | 'q' = quit",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Mark Zones", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        np.save('zones.npy', np.array(zones, dtype=object))
        print("💾 All zones saved to zones.npy successfully.")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
