"""
Smart Traffic Management System
YOLO + OpenCV Vehicle Detection → ESP32 Serial Communication
Trained using AIML / YOLOv8 (COCO pre-trained, vehicle classes)

Requirements:
    pip install ultralytics opencv-python pyserial

Hardware:
    ESP32 connected via USB Serial (update COM_PORT below)
"""

import cv2
import serial
import time
from ultralytics import YOLO

# ─────────────────────────────────────────────
# ⚙️  CONFIG — Edit these before running
# ─────────────────────────────────────────────
COM_PORT    = "COM3"        # Windows: "COM3" | Linux/Mac: "/dev/ttyUSB0"
BAUD_RATE   = 115200
CAMERA_ID   = 0             # 0 = webcam, or replace with video file path
CONF_THRESH = 0.4           # Minimum confidence score (0.0 – 1.0)
SEND_INTERVAL = 5           # Seconds between each count sent to ESP32
MODEL_PATH  = "yolov8n.pt"  # Auto-downloads on first run (nano = fastest)

# COCO class IDs that are vehicles
# 2=car, 3=motorcycle, 5=bus, 7=truck
VEHICLE_CLASSES = {2: "Car", 3: "Motorcycle", 5: "Bus", 7: "Truck"}

# Bounding box colors per class
COLORS = {
    2: (0, 255, 0),    # Car → Green
    3: (255, 165, 0),  # Motorcycle → Orange
    5: (0, 0, 255),    # Bus → Red
    7: (255, 0, 255),  # Truck → Magenta
}

# ─────────────────────────────────────────────
# 🔌  Serial Setup (connect to ESP32)
# ─────────────────────────────────────────────
def connect_serial():
    try:
        ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)  # Wait for ESP32 to reboot after connection
        print(f"[✓] Connected to ESP32 on {COM_PORT}")
        return ser
    except serial.SerialException as e:
        print(f"[✗] Serial connection failed: {e}")
        print("    Running in DEMO MODE (no ESP32 connected)")
        return None

# ─────────────────────────────────────────────
# 📤  Send Vehicle Count to ESP32
# ─────────────────────────────────────────────
def send_count(ser, count):
    if ser:
        message = f"{count}\n"
        ser.write(message.encode())
        print(f"[→] Sent to ESP32: {count} vehicles")
    else:
        print(f"[DEMO] Would send: {count} vehicles")

# ─────────────────────────────────────────────
# 🖼️  Draw Detection Overlay on Frame
# ─────────────────────────────────────────────
def draw_detections(frame, results, vehicle_count):
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            if cls_id not in VEHICLE_CLASSES:
                continue

            conf = float(box.conf[0])
            if conf < CONF_THRESH:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = f"{VEHICLE_CLASSES[cls_id]} {conf:.2f}"
            color = COLORS.get(cls_id, (200, 200, 200))

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Label background
            (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)

    # HUD — vehicle count
    cv2.rectangle(frame, (0, 0), (280, 50), (0, 0, 0), -1)
    cv2.putText(frame, f"Vehicles Detected: {vehicle_count}", (10, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 100), 2)

    return frame

# ─────────────────────────────────────────────
# 🚦  Count Vehicles from YOLO Results
# ─────────────────────────────────────────────
def count_vehicles(results):
    count = 0
    breakdown = {name: 0 for name in VEHICLE_CLASSES.values()}

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            conf   = float(box.conf[0])
            if cls_id in VEHICLE_CLASSES and conf >= CONF_THRESH:
                count += 1
                breakdown[VEHICLE_CLASSES[cls_id]] += 1

    return count, breakdown

# ─────────────────────────────────────────────
# 🚀  Main Detection Loop
# ─────────────────────────────────────────────
def main():
    print("=" * 50)
    print("  Smart Traffic Detection — YOLOv8 + ESP32")
    print("=" * 50)

    # Load YOLO model (downloads yolov8n.pt automatically if not found)
    print(f"[*] Loading YOLO model: {MODEL_PATH} ...")
    model = YOLO(MODEL_PATH)
    print("[✓] YOLO model loaded")

    # Open camera / video
    cap = cv2.VideoCapture(CAMERA_ID)
    if not cap.isOpened():
        print(f"[✗] Cannot open camera/video: {CAMERA_ID}")
        return
    print(f"[✓] Camera opened (source: {CAMERA_ID})")

    # Connect to ESP32
    ser = connect_serial()

    last_send_time = time.time()

    print("\n[*] Detection running — Press Q to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[!] Frame read failed. Exiting.")
            break

        # Run YOLO inference
        results = model(frame, verbose=False)

        # Count vehicles
        vehicle_count, breakdown = count_vehicles(results)

        # Draw on frame
        frame = draw_detections(frame, results, vehicle_count)

        # Print breakdown every second in terminal
        print(f"\r  🚗 Cars:{breakdown['Car']}  🏍 Bikes:{breakdown['Motorcycle']}"
              f"  🚌 Buses:{breakdown['Bus']}  🚛 Trucks:{breakdown['Truck']}"
              f"  | Total: {vehicle_count}   ", end="", flush=True)

        # Send count to ESP32 at fixed interval
        now = time.time()
        if now - last_send_time >= SEND_INTERVAL:
            send_count(ser, vehicle_count)
            last_send_time = now

        # Show video window
        cv2.imshow("Smart Traffic Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\n[*] Q pressed — shutting down.")
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    if ser:
        ser.close()
        print("[✓] Serial port closed.")

if __name__ == "__main__":
    main()