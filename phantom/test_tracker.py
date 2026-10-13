import cv2
import numpy as np
from phantom.tracker import HandTracker

# Create a dummy black frame (640x480)
frame = np.zeros((480, 640, 3), dtype=np.uint8)

tracker = HandTracker()
states = tracker.process(frame, draw=False)
print('Detected hand states:', states)
