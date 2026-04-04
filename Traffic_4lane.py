"""
Smart Traffic Management — 4-Lane Version
Simulates 4 camera lanes and sends counts to ESP32 in round-robin
Matches the 4-road signal setup in trafficard.ino

Usage:
    python traffic_4lane.py
    (Uses webcam for all 4 lanes in demo — replace CAMERA_IDs for real cams)
"""

import cv2
import serial
import time
from ultralytics import YOLO

# ─────────────────────────────────────────────
# ⚙️  CONFIG
# ─────────────────────────────────────────────
COM_PORT        = "COM3"        # Change to your port
BAUD_RATE       = 115200
CONF_THRESH     = 0.4
MODEL_PATH      = "yolov8n.pt"
LANE_SWITCH_SEC = 8             # Seconds each lane is active

# For 4 real cameras: [0, 1, 2, 3]
# For demo with 1 webcam: [0, 0, 0, 0]
CAMERA_IDS = [0, 0, 0, 0]

VEHICLE_CLASSES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}
LANE_NAMES = ["Lane A", "Lane B", "Lane C", "Lane D"]

# ─────────────────────────────────────────────
def connect_serial():
    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        print(f"[✓] ESP32 connected on {COM_PORT}")
        return ser
    except:
        print("[!] No ESP32 found — running in DEMO mode")
        return None

def count_vehicles(results):
    count = 0
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            if cls_id in VEHICLE_CLASSES and conf >= CONF_THRESH:
                count += 1
    return count

def draw_hud(frame, lane_name, count, timer_remaining):
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    cv2.putText(frame, f"{lane_name}  |  Vehicles: {count}  |  Next send in: {timer_remaining}s",
                (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 180), 2)
    return frame

# ─────────────────────────────────────────────
def main():
    print("=" * 55)
    print("  4-Lane Smart Traffic System — YOLOv8 + ESP32")
    print("=" * 55)

    model = YOLO(MODEL_PATH)
    print("[✓] YOLO loaded")

    # Open cameras (one per lane, or same cam in demo)
    caps = []
    for idx, cam_id in enumerate(CAMERA_IDS):
        cap = cv2.VideoCapture(cam_id)
        if not cap.isOpened():
            print(f"[✗] Cannot open camera {cam_id} for {LANE_NAMES[idx]}")
        caps.append(cap)

    ser = connect_serial()

    current_lane = 0
    last_switch  = time.time()

    print("\n[*] Running — Press Q to quit\n")

    while True:
        # Read frame from current lane's camera
        cap = caps[current_lane]
        ret, frame = cap.read()
        if not ret:
            print(f"[!] Frame error on {LANE_NAMES[current_lane]}")
            current_lane = (current_lane + 1) % 4
            continue

        # Detect
        results = model(frame, verbose=False)
        count   = count_vehicles(results)

        # Draw bounding boxes
        for result in results:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                if cls_id not in VEHICLE_CLASSES:
                    continue
                if float(box.conf[0]) < CONF_THRESH:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, VEHICLE_CLASSES[cls_id],
                            (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Timer
        elapsed  = time.time() - last_switch
        remaining = max(0, int(LANE_SWITCH_SEC - elapsed))

        frame = draw_hud(frame, LANE_NAMES[current_lane], count, remaining)

        # When timer expires → send count and switch lane
        if elapsed >= LANE_SWITCH_SEC:
            msg = f"{count}\n"
            if ser:
                ser.write(msg.encode())
            print(f"[→] {LANE_NAMES[current_lane]}: {count} vehicles sent to ESP32")

            current_lane = (current_lane + 1) % 4
            last_switch  = time.time()

        cv2.imshow("4-Lane Traffic Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()
    if ser:
        ser.close()

if __name__ == "__main__":
    main()