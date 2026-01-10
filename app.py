import cv2
import time
import urllib.parse
import webbrowser
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

print("✅ RUNNING UPDATED SCRIPT WITH CARD")

SEARCH_ENGINE = "https://www.google.com/search?q="
SEARCH_COOLDOWN_SEC = 3.0

MODEL_PATH = Path("models/blaze_face_short_range.tflite")
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH.resolve()}")

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

options = vision.FaceDetectorOptions(
    base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
    min_detection_confidence=0.6,
)
detector = vision.FaceDetector.create_from_options(options)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Could not open webcam. Check macOS Camera permissions.")

last_search_time = 0.0
query_text = ""

print("Controls: type to build query | Enter=search (face required) | Backspace=delete | Esc=quit")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = detector.detect(mp_image)
    detections = result.detections or []
    face_present = len(detections) > 0

    for i, det in enumerate(detections):
        bbox = det.bounding_box
        x1, y1 = bbox.origin_x, bbox.origin_y
        x2, y2 = x1 + bbox.width, y1 + bbox.height

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        card_w_guess = 280
        gap = 12
        place_right = (x2 + gap + card_w_guess) < frame.shape[1]
        anchor_x = x2 + gap if place_right else (x1 - gap - card_w_guess)
        anchor_y = max(0, y1 - 10)

        if i not in smoothed:
            smoothed[i] = (float(anchor_x), float(anchor_y))
        else:
            sx, sy = smoothed[i]
            smoothed[i] = (
                sx + ALPHA * (anchor_x - sx),
                sy + ALPHA * (anchor_y - sy),
            )

        sx, sy = smoothed[i]
        draw_card(frame, int(sx), int(sy), ["Nearby contact", "Type query + Enter", "ESC to quit"])

    # IMPOSSIBLE-TO-MISS DEBUG (proves you're running THIS file)
    cv2.rectangle(frame, (0, 0), (520, 190), (0, 0, 255), -1)
    cv2.putText(frame, "DEBUG BLOCK", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4)

    status = "FACE DETECTED" if face_present else "NO FACE"
    cv2.putText(frame, f"{status} | Query: {query_text}", (20, 230),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow("MediaPipe Face-triggered Search", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:
        break

    if key in (10, 13):
        now = time.time()
        if face_present and query_text.strip() and (now - last_search_time) >= SEARCH_COOLDOWN_SEC:
            q = urllib.parse.quote_plus(query_text.strip())
            webbrowser.open(SEARCH_ENGINE + q)
            last_search_time = now
    elif key == 8:
        query_text = query_text[:-1]
    elif 32 <= key <= 126:
        query_text += chr(key)

cap.release()
cv2.destroyAllWindows()
detector.close()
