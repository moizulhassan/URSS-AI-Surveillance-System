# detection/object_detector.py

from ultralytics import YOLO
import cv2

# load object detection model (COCO)
model = YOLO("yolov8n.pt")

def detect_objects(frame):
    results = model(frame, conf=0.4, verbose=False)

    detections = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detections.append({
                "class": label,
                "confidence": conf,
                "box": (x1, y1, x2, y2)
            })

            # draw bounding box
            if label == "person":
                color = (0, 255, 255)
            else:
                color = (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}",
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, color, 2)

    return frame, detections

