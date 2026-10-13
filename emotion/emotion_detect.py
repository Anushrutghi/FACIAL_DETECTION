import cv2
import numpy as np
from tensorflow.keras.models import load_model
import os

# ---------------- LOAD MODEL ----------------
MODEL_PATH = os.path.join("models", "emotion_model.h5")
emotion_model = load_model(MODEL_PATH, compile=False)

# FER2013 output labels (model order)
FER_EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]

# ---------------- EMOTION DETECTION ----------------
def detect_emotion(face_img):
    try:
        # Convert to grayscale
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

        # Resize to model input size
        gray = cv2.resize(gray, (48, 48))

        # Normalize
        gray = gray / 255.0

        # Reshape for CNN
        gray = np.reshape(gray, (1, 48, 48, 1))

        # Predict probabilities
        preds = emotion_model.predict(gray, verbose=0)[0]

        angry_score    = preds[0]
        fear_score     = preds[2]
        happy_score    = preds[3]
        sad_score      = preds[4]
        surprise_score = preds[5]
        neutral_score  = preds[6]

        # -------- SMART THRESHOLD LOGIC --------
        if happy_score > 0.30:
            return "Happy"
        elif sad_score > 0.30:
            return "Sad"
        elif angry_score > 0.30:
            return "Angry"
        elif fear_score > 0.30 or surprise_score > 0.30:
            return "Crying"
        else:
            return "Neutral"

    except Exception as e:
        return "Neutral"
