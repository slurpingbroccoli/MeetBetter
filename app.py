# app.py
import time
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = Path("models/blaze_face_short_range.tflite")
DB_PATH = Path("db/face_db.npz")

# If it doesn't recognize you, LOWER a bit (e.g. 0.78)
# If it falsely recognizes random faces, RAISE a bit (e.g. 0.88)
SIM_THRESHOLD = 0.82

# Card smoothing 
ALPHA = 0.45
smoothed = {}

# -----------------------------
# VOICE LOCK (set by main.py)
# -----------------------------
VOICE_LOCK_NAME: str | None = None
VOICE_LOCK_UNTIL: float = 0.0


def set_voice_lock(name: str, seconds: float = 6.0) -> None:
    """Called from mic thread to force UI to show this name for a bit."""
    global VOICE_LOCK_NAME, VOICE_LOCK_UNTIL
    n = (name or "").strip()
    if not n:
        return
    VOICE_LOCK_NAME = n
    VOICE_LOCK_UNTIL = time.time() + float(seconds)
    print(f"🔒 VOICE LOCK -> {VOICE_LOCK_NAME} ({seconds:.1f}s)")


def draw_card(img, x, y, lines, padding=12):
    font = cv2.FONT_HERSHEY_SIMPLEX
    title_scale, title_th = 0.60, 2
    body_scale, body_th = 0.55, 1
    foot_scale, foot_th = 0.45, 1

    max_text_w = width - 1.5 * padding

    wrapped = []
    for t in lines:
        wrapped.extend(_wrap_text(t, font, body_scale, body_th, max_text_w))

    footer_lines = []
    if footer:
        footer_lines = _wrap_text(footer, font, foot_scale, foot_th, max_text_w)

    body_h = cv2.getTextSize("Ag", font, body_scale, body_th)[0][1]
    foot_h = cv2.getTextSize("Ag", font, foot_scale, foot_th)[0][1]
    title_h = cv2.getTextSize("Ag", font, title_scale, title_th)[0][1]

    body_block_h = len(wrapped) * (body_h + 10)
    footer_block_h = (len(footer_lines) * (foot_h + 8) + 8) if footer_lines else 0
    height = header_h + padding + body_block_h + footer_block_h + padding

    # clamp on screen
    x = int(max(8, min(x, W - width - 8)))
    y = int(max(8, min(y, H - height - 8)))

    # shadow (subtle)
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + card_w, y + card_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
    cv2.rectangle(img, (x, y), (x + card_w, y + card_h), (255, 255, 255), 2)

    ty = y + padding + 18
    for t in lines:
        cv2.putText(img, t, (x + padding, ty), font, scale, (255, 255, 255), thickness)
        ty += line_h


def _smooth(key: str, new: float) -> float:
    old = smoothed.get(key, new)
    val = (1 - ALPHA) * old + ALPHA * new
    smoothed[key] = val
    return val


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH.resolve()}")
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Missing DB: {DB_PATH.resolve()} (run python enroll.py first)")

    db = np.load(DB_PATH, allow_pickle=True)
    embs = db["embeddings"].astype(np.float32)
    labels = db["labels"]

    # normalize db embeddings
    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)

    base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.FaceDetectorOptions(base_options=base_options)
    detector = vision.FaceDetector.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check macOS Camera permissions.")

    print("ESC to quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        now = time.time()
        H, W = frame.shape[:2]

        # -----------------------------
        # If VOICE LOCK is active: freeze the card to that name
        # -----------------------------
        if VOICE_LOCK_NAME and now < VOICE_LOCK_UNTIL:
            ttl = VOICE_LOCK_UNTIL - now
            draw_card(frame, 20, 20, [f"Name: {VOICE_LOCK_NAME}", f"VOICE LOCK: {ttl:.1f}s", "ESC to quit"])
            cv2.imshow("MeetBetter (local recognition)", frame)
            if (cv2.waitKey(1) & 0xFF) == 27:
                break
            continue  # IMPORTANT: ignore camera identity while voice lock is active

        # MediaPipe expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = detector.detect(mp_image)
        if not result.detections:
            draw_card(frame, 20, 20, ["Name: Unknown", "No face", "ESC to quit"])
            cv2.imshow("MeetBetter (local recognition)", frame)
            if (cv2.waitKey(1) & 0xFF) == 27:
                break
            continue

        # pick the biggest bbox
        def area(det):
            bb = det.bounding_box
            return bb.width * bb.height

        det = max(result.detections, key=area)
        bb = det.bounding_box

        x1 = max(0, int(bb.origin_x))
        y1 = max(0, int(bb.origin_y))
        x2 = min(W, int(bb.origin_x + bb.width))
        y2 = min(H, int(bb.origin_y + bb.height))

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        crop = rgb[y1:y2, x1:x2]
        if crop.size == 0:
            draw_card(frame, 20, 20, ["Name: Unknown", "Bad crop", "ESC to quit"])
            cv2.imshow("MeetBetter (local recognition)", frame)
            if (cv2.waitKey(1) & 0xFF) == 27:
                break
            continue

        # ---------------------------------------------------------
        # ✅ IMPORTANT FIX:
        # Your DB was enrolled as 112x112x3 flattened (37632 dims),
        # but runtime was using 32x32x3 (3072 dims).
        # Make runtime match enrollment: use 112x112.
        # ---------------------------------------------------------
        crop_small = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_AREA).astype(np.float32)
        q = crop_small.flatten()
        q = q / (np.linalg.norm(q) + 1e-9)

        # If something is still mismatched, fail loudly with a helpful message
        if embs.shape[1] != q.shape[0]:
            raise RuntimeError(
                f"Embedding dim mismatch: DB={embs.shape[1]} vs runtime={q.shape[0]}. "
                f"Re-run enroll.py or ensure both use same resize."
            )

        sims = embs @ q
        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])
        best_name = str(labels[best_idx])

        # smooth similarity
        best_sim = _smooth("sim", best_sim)

        name = "Unknown"
        if best_sim >= SIM_THRESHOLD:
            name = best_name

        draw_card(
            frame,
            x2 + 12,
            max(0, y1 - 10),
            [f"Name: {name}", f"Similarity: {best_sim:.3f}", "ESC to quit"],
        )


        cv2.imshow("MeetBetter (local recognition)", frame)
        if (cv2.waitKey(1) & 0xFF) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
