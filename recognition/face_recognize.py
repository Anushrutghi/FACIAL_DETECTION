import cv2
import os
import numpy as np

KNOWN_FACES_DIR = "database/faces"

def load_known_faces():
    known_faces = []
    labels = []

    if not os.path.exists(KNOWN_FACES_DIR):
        return known_faces, labels

    for person in os.listdir(KNOWN_FACES_DIR):
        person_path = os.path.join(KNOWN_FACES_DIR, person)
        if not os.path.isdir(person_path):
            continue

        for img_name in os.listdir(person_path):
            img_path = os.path.join(person_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            img = cv2.resize(img, (100, 100))
            known_faces.append(img)
            labels.append(person)

    return known_faces, labels

def recognize_face(face_img, known_faces, labels):
    if len(known_faces) == 0:
        return "Unknown"

    face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img = cv2.resize(face_img, (100, 100))

    diffs = []

    for i, known in enumerate(known_faces):
        diff = np.mean(cv2.absdiff(face_img, known))
        diffs.append((diff, labels[i]))

    # Sort by similarity
    diffs.sort(key=lambda x: x[0])

    best_diff, best_label = diffs[0]

    # Strict threshold
    if best_diff < 40:
        return best_label
    else:
        return "Unknown"
