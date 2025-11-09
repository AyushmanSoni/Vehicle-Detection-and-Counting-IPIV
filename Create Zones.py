import cv2
import numpy as np

# Load the video
video_path = r'carsvid.mp4'
cap = cv2.VideoCapture(video_path)

# List to store all zones (each zone = list of (x, y) points)
zones = []
current_zone = []

# Mouse callback function
def draw_zone(event, x, y, flags, param):
    global current_zone, zones

    if event == cv2.EVENT_LBUTTONDOWN:
        current_zone.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        # Right click = close current zone
        if len(current_zone) >= 3:
            zones.append(current_zone.copy())
            current_zone = []
            print(f"Zone {len(zones)} saved ✅")
        else:
            print("Need at least 3 points to form a zone.")

# Create window and set mouse callback
cv2.namedWindow("Mark Zones")
cv2.setMouseCallback("Mark Zones", draw_zone)

while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        continue

    frame = cv2.resize(frame, (1280, 720))

    # Draw current zone points
    for pt in current_zone:
        cv2.circle(frame, pt, 5, (0, 0, 255), -1)
    if len(current_zone) > 1:
        cv2.polylines(frame, [np.array(current_zone, np.int32)], False, (0, 255, 0), 2)

    # Draw completed zones
    for zone in zones:
        cv2.polylines(frame, [np.array(zone, np.int32)], True, (255, 0, 0), 2)

    cv2.putText(frame, "Left click = add point | Right click = save zone | 's' = save all | 'q' = quit",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Mark Zones", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):  # Save all zones to file
        np.save('zones.npy', np.array(zones, dtype=object))
        print("All zones saved to zones.npy ✅")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
