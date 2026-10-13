import time
import math
import cv2
import mediapipe as mp
import pyautogui

pyautogui.FAILSAFE = False

mp_face = mp.solutions.face_mesh

def _dist(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

class BlinkClicker:
    """
    Uses Eye Aspect Ratio (EAR) with FaceMesh to detect blink.
    Blink -> click (cooldown prevents double clicks)
    """
    def __init__(self, ear_thresh=0.20, cooldown=0.6):
        self.ear_thresh = ear_thresh
        self.cooldown = cooldown
        self.last_click = 0
        self.mesh = mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

        # FaceMesh landmark indices for eyes (good enough)
        self.L = dict(outer=33, inner=133, top=159, bottom=145)   # left eye
        self.R = dict(outer=362, inner=263, top=386, bottom=374)  # right eye

    def _ear(self, pts, eye):
        outer = pts[eye["outer"]]
        inner = pts[eye["inner"]]
        top   = pts[eye["top"]]
        bot   = pts[eye["bottom"]]
        horiz = _dist(outer, inner) + 1e-6
        vert  = _dist(top, bot)
        return vert / horiz

    def update(self, frame_bgr):
        """
        Returns:
          clicked (bool), ear (float or None)
        """
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.mesh.process(rgb)

        if not res.multi_face_landmarks:
            return False, None

        lm = res.multi_face_landmarks[0].landmark
        pts = [(int(p.x*w), int(p.y*h)) for p in lm]

        ear_l = self._ear(pts, self.L)
        ear_r = self._ear(pts, self.R)
        ear = (ear_l + ear_r) / 2.0

        now = time.time()
        if ear < self.ear_thresh and (now - self.last_click) > self.cooldown:
            pyautogui.click()
            self.last_click = now
            return True, ear

        return False, ear
