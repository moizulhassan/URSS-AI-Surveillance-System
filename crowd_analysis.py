# analysis/crowd_analysis.py

import cv2

def analyze_crowd(frame, detections):
    """
    frame       : video frame
    detections  : list of detected objects (YOLO results)
    """

    crowd_count = 0

    for det in detections:
        cls = det["class"]
        conf = det["confidence"]

        # person class
        if cls == "person" and conf > 0.4:
            crowd_count += 1

    # crowd level
    if crowd_count >= 30:
        level = "HIGH CROWD 🔴"
    elif crowd_count >= 15:
        level = "MEDIUM CROWD 🟠"
    else:
        level = "LOW CROWD 🟢"

    return {
        "count": crowd_count,
        "level": level
    }
