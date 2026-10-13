import pyautogui

pyautogui.FAILSAFE = False

class FaceCursor:
    def __init__(self, alpha=0.20, deadzone=0.08, gain=1.4):
        self.alpha = alpha
        self.deadzone = deadzone
        self.gain = gain
        self.sx, self.sy = None, None

    def update(self, face_center_norm):
        """
        face_center_norm: (cx, cy) in [0..1] relative to frame
        Moves OS cursor based on face offset from center.
        """
        if face_center_norm is None:
            return

        cx, cy = face_center_norm
        dx = (cx - 0.5)
        dy = (cy - 0.5)

        # deadzone to reduce jitter
        if abs(dx) < self.deadzone: dx = 0.0
        if abs(dy) < self.deadzone: dy = 0.0

        screen_w, screen_h = pyautogui.size()

        # map offset -> target cursor position (center + offset)
        target_x = int((0.5 + self.gain * dx) * screen_w)
        target_y = int((0.5 + self.gain * dy) * screen_h)

        target_x = max(0, min(screen_w - 1, target_x))
        target_y = max(0, min(screen_h - 1, target_y))

        if self.sx is None:
            self.sx, self.sy = target_x, target_y
        else:
            self.sx = int(self.alpha * target_x + (1 - self.alpha) * self.sx)
            self.sy = int(self.alpha * target_y + (1 - self.alpha) * self.sy)

        pyautogui.moveTo(self.sx, self.sy)
