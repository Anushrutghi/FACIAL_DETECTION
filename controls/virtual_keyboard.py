import cv2
import pyautogui

pyautogui.FAILSAFE = False

class VirtualKeyboard:
    def __init__(self):
        # simple layout
        self.keys = [
            list("QWERTYUIOP"),
            list("ASDFGHJKL"),
            list("ZXCVBNM"),
            ["SPACE", "BKSP", "ENTER"]
        ]
        self.visible = False
        self.last_key = None

    def toggle(self):
        self.visible = not self.visible

    def draw(self, frame):
        if not self.visible:
            return []

        h, w = frame.shape[:2]
        x0, y0 = 20, int(h * 0.55)
        key_w, key_h = 60, 55
        gap = 10

        boxes = []  # (label, (x1,y1,x2,y2))

        for r, row in enumerate(self.keys):
            y = y0 + r * (key_h + gap)
            x = x0

            for label in row:
                kw = key_w
                if label == "SPACE":
                    kw = key_w * 4
                elif label in ("BKSP", "ENTER"):
                    kw = key_w * 2

                x1, y1, x2, y2 = x, y, x + kw, y + key_h
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
                cv2.putText(frame, label, (x1 + 10, y1 + 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                boxes.append((label, (x1, y1, x2, y2)))

                x = x2 + gap

        return boxes

    def hit_test(self, boxes, px, py):
        for label, (x1, y1, x2, y2) in boxes:
            if x1 <= px <= x2 and y1 <= py <= y2:
                return label
        return None

    def press(self, label):
        if label == "SPACE":
            pyautogui.typewrite(" ")
        elif label == "BKSP":
            pyautogui.press("backspace")
        elif label == "ENTER":
            pyautogui.press("enter")
        else:
            pyautogui.typewrite(label.lower())
