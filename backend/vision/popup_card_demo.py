import time
import urllib.parse
import webbrowser
from pathlib import Path

import cv2
import mediapipe as mp

print("✅ RUNNING LANDMARKER VERSION WITH ANCHORED CARD")

SEARCH_ENGINE = "https://www.google.com/search?q="
SEARCH_COOLDOWN_SEC = 3.0

MODEL_PATH = Path("models/face_landmarker.task")
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model not found at {MODEL_PATH.resolve()}")

# smoothing for card position
ALPHA = 0.25
smoothed = {}

def clamp(v, lo, hi):
    return max(lo, min(v, hi))

def bbox_from_landmarks(lms, w, h, pad=18):
    xs = [lm.x * w for lm in lms]
    ys = [lm.y * h for lm in lms]
    x1 = int(min(xs) - pad)
    y1 = int(min(ys) - pad)
    x2 = int(max(xs) + pad)
    y2 = int(max(ys) + pad)
    return x1, y1, x2, y2

def draw_card(img, x, y, lines, padding=12):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.65
    thickness = 2
    line_h = 24

    max_w = 0
    for t in lines:
        (tw, _), _ = cv2.getTextSize(t, font, scale, thickness)
        max_w = max(max_w, tw)

    card_w = max_w + 2 * padding
    card_h = len(lines) * line_h + 2 * padding

    H, W = img.shape[:2]
    x = int(clamp(x, 0, W - card_w - 1))
    y = int(clamp(y, 0, H - card_h - 1))

    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + card_w, y + card_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)

    cv2.rectangle(img, (x, y), (x + card_w, y + card_h), (255, 255, 255), 2)

    ty = y + padding + 18
    for t in lines:
        cv2.putText(img, t, (x + padding, ty), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
        ty += line_h

# MediaPipe Tasks setup
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1,
)

landmarker = FaceLandmarker.create_from_options(options)

# Linux-friendly webcam open
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
if not cap.isOpened():
    # try camera 1 as fallback
    cap = cv2.VideoCapture(1, cv2.CAP_V4L2)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam on index 0 or 1. (Linux: check /dev/video0 permissions)")

last_search_time = 0.0
query_text = ""

print("Controls: type query | Enter=search (face required) | Backspace=delete | Esc=quit")

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        H, W = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        ts_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, ts_ms)

        faces = result.face_landmarks or []
        face_present = len(faces) > 0

        if face_present:
            face_lms = faces[0]
            x1, y1, x2, y2 = bbox_from_landmarks(face_lms, W, H, pad=18)
            x1, y1 = clamp(x1, 0, W-1), clamp(y1, 0, H-1)
            x2, y2 = clamp(x2, 0, W-1), clamp(y2, 0, H-1)

            # face box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # anchored card position (right if possible else left)
            card_w_guess = 300
            gap = 12
            place_right = (x2 + gap + card_w_guess) < W
            anchor_x = x2 + gap if place_right else (x1 - gap - card_w_guess)
            anchor_y = max(0, y1 - 10)

            key_id = 0  # single face
            if key_id not in smoothed:
                smoothed[key_id] = (float(anchor_x), float(anchor_y))
            else:
                sx, sy = smoothed[key_id]
                smoothed[key_id] = (sx + ALPHA * (anchor_x - sx), sy + ALPHA * (anchor_y - sy))

            sx, sy = smoothed[key_id]
            draw_card(frame, int(sx), int(sy), [
                "Nearby contact",
                "Type query + Enter",
                "ESC to quit"
            ])

            # connector line
            from_pt = (x2, y1) if place_right else (x1, y1)
            to_pt = (int(sx), int(sy) + 20) if place_right else (int(sx) + card_w_guess, int(sy) + 20)
            cv2.line(frame, from_pt, to_pt, (0, 255, 0), 2)

        status = "FACE DETECTED" if face_present else "NO FACE"
        cv2.putText(frame, f"{status} | Query: {query_text}", (20, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("MeetBetter - Face Landmarker Card", frame)
        key = cv2.waitKey(1) & 0xFF

        # ESC
        if key == 27:
            break

        # Enter -> search (only if face present)
        if key in (10, 13):
            now = time.time()
            if face_present and query_text.strip() and (now - last_search_time) >= SEARCH_COOLDOWN_SEC:
                q = urllib.parse.quote_plus(query_text.strip())
                webbrowser.open(SEARCH_ENGINE + q)
                last_search_time = now

        # Backspace
        elif key in (8, 127):
            query_text = query_text[:-1]

        # printable chars
        elif 32 <= key <= 126:
            query_text += chr(key)

finally:
    cap.release()
    cv2.destroyAllWindows()
    landmarker.close()
