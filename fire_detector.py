import cv2
import numpy as np

def detect_fire(frame):
    """
    HSV color-based fire detection
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Fire-like colors (orange-red)
    lower = np.array([0, 150, 150])
    upper = np.array([35, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    fire_pixels = cv2.countNonZero(mask)

    if fire_pixels > 2000:  # Adjust threshold
        # Draw mask on frame for debug
        frame[mask>0] = [0,0,255]
        return True
    return False




