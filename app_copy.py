import time
import numpy as np
import cv2
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = Path("models/blaze_face_short_range.tflite")
DB_PATH = Path("db/face_db.npz")

# Threshold for the placeholder pixel-embedding similarity.
# If it doesn't recognize you, LOWER it a bit (e.g. 0.78).
# If it falsely recognizes random faces, RAISE it (e.g. 0.88).
SIM_THRESHOLD = 0.82

# Card smoothing
ALPHA = 0.25
smoothed = {}

def draw_card(img, x, y, lines, padding=12):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    thickness = 2
    line_h = 22

    max_w = 0
    for t in lines:
        (w, _), _ = cv2.getTextSize(t, font, scale, thickness)
        max_w = max(max_w, w)

    card_w = max_w + 2 * padding
    card_h = len(lines) * line_h + 2 * padding

    H, W = img.shape[:2]
    x = int(max(0, min(x, W - card_w - 1)))
    y = int(max(0, min(y, H - card_h - 1)))

    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + card_w, y + card_h), (0, 0, 0), -1)
    alpha = 0.65
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    cv2.rectangle(img, (x, y), (x + card_w, y + card_h), (255, 255, 255), 2)

    ty = y + padding + 18
    for t in lines:
        cv2.putText(img, t, (x + padding, ty), font, scale, (255, 255, 255), thickness)
        ty += line_h

def face_embed_placeholder(face_bgr):
    """Same embedding method as enroll.py: resize to 112x112 and flatten pixels."""
    face = cv2.resize(face_bgr, (112, 112))
    emb = face.flatten().astype(np.float32) / 255.0
    # normalize to unit length so dot product behaves like cosine similarity
    norm = np.linalg.norm(emb) + 1e-9
    return emb / norm

def best_match(query_emb, db_embs, db_labels):
    sims = db_embs @ query_emb  # cosine similarity (because normalized)
    idx = int(np.argmax(sims))
    return str(db_labels[idx]), float(sims[idx])

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH.resolve()}")
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Missing DB: {DB_PATH.resolve()} (run python enroll.py first)")

    db = np.load(DB_PATH, allow_pickle=False)
    db_embs = db["embeddings"].astype(np.float32)
    db_labels = db["labels"]

    # Normalize stored embeddings (in case they aren't)
    db_embs = db_embs / (np.linalg.norm(db_embs, axis=1, keepdims=True) + 1e-9)

    options = vision.FaceDetectorOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
        min_detection_confidence=0.6,
    )
    detector = vision.FaceDetector.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check macOS Camera permissions.")

    print("ESC to quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        H, W = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = detector.detect(mp_image)
        detections = result.detections or []

        for i, det in enumerate(detections):
            box = det.bounding_box
            x1 = max(0, box.origin_x)
            y1 = max(0, box.origin_y)
            x2 = min(W, x1 + box.width)
            y2 = min(H, y1 + box.height)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            crop = frame[y1:y2, x1:x2]
            name = "Unknown"
            score = 0.0

            if crop.size > 0:
                qemb = face_embed_placeholder(crop)
                best_name, sim = best_match(qemb, db_embs, db_labels)
                score = sim
                if sim >= SIM_THRESHOLD:
                    name = best_name

            # Card placement (avoid off-screen)
            card_w_guess = 320
            gap = 12
            place_right = (x2 + gap + card_w_guess) < W
            anchor_x = x2 + gap if place_right else (x1 - gap - card_w_guess)
            anchor_y = max(0, y1 - 10)

            # Smooth
            if i not in smoothed:
                smoothed[i] = (float(anchor_x), float(anchor_y))
            else:
                sx, sy = smoothed[i]
                smoothed[i] = (sx + ALPHA * (anchor_x - sx), sy + ALPHA * (anchor_y - sy))

            sx, sy = smoothed[i]

            draw_card(
                frame,
                int(sx),
                int(sy),
                [
                    f"Name: {name}",
                    f"Similarity: {score:.3f}",
                    "ESC to quit",
                ],
            )

        cv2.imshow("MeetBetter (local recognition)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()

if __name__ == "__main__":
    main()
