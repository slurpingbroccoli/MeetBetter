# app.py
import time
import threading
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

# How many camera candidates to keep around for fusion
CAM_TOPK = 5

# Card smoothing
ALPHA = 0.25
smoothed = {}

# -----------------------------
# Shared state (thread-safe)
# -----------------------------
_state_lock = threading.Lock()

VOICE_LOCK_NAME: str | None = None
VOICE_LOCK_UNTIL: float = 0.0

TRY_AGAIN_MSG: str | None = None
TRY_AGAIN_UNTIL: float = 0.0

# Latest camera candidates: list[tuple[name, sim]]
CAMERA_TOPK: list[tuple[str, float]] = []
CAMERA_UPDATED_AT: float = 0.0


def set_voice_lock(name: str, seconds: float = 10.0) -> None:
    """Called from mic thread to force UI to show this name for a bit."""
    global VOICE_LOCK_NAME, VOICE_LOCK_UNTIL

    n = (name or "").strip()
    if not n:
        return

    with _state_lock:
        VOICE_LOCK_NAME = n
        VOICE_LOCK_UNTIL = time.time() + float(seconds)

    print(f"🔒 VOICE LOCK -> {n} ({seconds:.1f}s)")


def clear_voice_lock() -> None:
    global VOICE_LOCK_NAME, VOICE_LOCK_UNTIL
    with _state_lock:
        VOICE_LOCK_NAME = None
        VOICE_LOCK_UNTIL = 0.0


def set_try_again(msg: str = "Please try again", seconds: float = 2.5) -> None:
    """Called from mic thread to show a temporary 'try again' overlay."""
    global TRY_AGAIN_MSG, TRY_AGAIN_UNTIL
    with _state_lock:
        TRY_AGAIN_MSG = msg
        TRY_AGAIN_UNTIL = time.time() + float(seconds)
    print(f"⚠️ TRY AGAIN -> {msg} ({seconds:.1f}s)")


def get_camera_topk(max_age_sec: float = 1.0) -> list[tuple[str, float]]:
    """Read the latest camera top-K list (thread-safe)."""
    now = time.time()
    with _state_lock:
        if now - CAMERA_UPDATED_AT > max_age_sec:
            return []
        return list(CAMERA_TOPK)


def _publish_camera_topk(topk: list[tuple[str, float]]) -> None:
    global CAMERA_TOPK, CAMERA_UPDATED_AT
    with _state_lock:
        CAMERA_TOPK = list(topk)
        CAMERA_UPDATED_AT = time.time()


def _smooth(key: str, new: float) -> float:
    old = smoothed.get(key, new)
    val = (1 - ALPHA) * old + ALPHA * new
    smoothed[key] = val
    return val


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
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
    cv2.rectangle(img, (x, y), (x + card_w, y + card_h), (255, 255, 255), 2)

    ty = y + padding + 18
    for t in lines:
        cv2.putText(img, t, (x + padding, ty), font, scale, (255, 255, 255), thickness)
        ty += line_h


def main():
    global VOICE_LOCK_NAME, VOICE_LOCK_UNTIL, TRY_AGAIN_MSG, TRY_AGAIN_UNTIL

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

        # snapshot overlay states (thread-safe)
        with _state_lock:
            ta_msg = TRY_AGAIN_MSG
            ta_until = TRY_AGAIN_UNTIL
            vl_name = VOICE_LOCK_NAME
            vl_until = VOICE_LOCK_UNTIL

        # Try-again overlay (does NOT stop camera)
        if ta_msg and now < ta_until:
            draw_card(frame, 20, 20, [f"⚠️ {ta_msg}", f"{(ta_until - now):.1f}s", "ESC to quit"])

        # Voice-lock overlay (DOES stop camera display identity changes)
        if vl_name and now < vl_until:
            ttl = vl_until - now
            draw_card(frame, 20, 120, [f"Name: {vl_name}", f"VOICE LOCK: {ttl:.1f}s", "ESC to quit"])
            cv2.imshow("MeetBetter (local recognition)", frame)
            if (cv2.waitKey(1) & 0xFF) == 27:
                break
            continue

        # MediaPipe expects RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = detector.detect(mp_image)
        if not result.detections:
            _publish_camera_topk([])
            draw_card(frame, 20, 120, ["Name: Unknown", "No face", "ESC to quit"])
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
            _publish_camera_topk([])
            draw_card(frame, 20, 120, ["Name: Unknown", "Bad crop", "ESC to quit"])
            cv2.imshow("MeetBetter (local recognition)", frame)
            if (cv2.waitKey(1) & 0xFF) == 27:
                break
            continue

        # IMPORTANT: match enrollment dims (112x112x3 = 37632)
        crop_small = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_AREA).astype(np.float32)
        q = crop_small.flatten()
        q = q / (np.linalg.norm(q) + 1e-9)

        if embs.shape[1] != q.shape[0]:
            raise RuntimeError(
                f"Embedding dim mismatch: DB={embs.shape[1]} vs runtime={q.shape[0]}. "
                f"Re-run enroll.py or ensure both use same resize."
            )

        sims = embs @ q  # (N,)
        k = min(CAM_TOPK, sims.shape[0])
        top_idx = np.argsort(-sims)[:k]
        topk = [(str(labels[i]), float(sims[i])) for i in top_idx]
        _publish_camera_topk(topk)

        best_name, best_sim = topk[0]
        best_sim = _smooth("sim", best_sim)

        name = "Unknown"
        if best_sim >= SIM_THRESHOLD:
            name = best_name

        # Card near face
        anchor_x = x2 + 12
        anchor_y = max(0, y1 - 10)

        top2_line = " / ".join([f"{n}:{s:.3f}" for (n, s) in topk[:2]])
        draw_card(
            frame,
            anchor_x,
            anchor_y,
            [f"Name: {name}", f"Sim: {best_sim:.3f}", f"Top: {top2_line}", "ESC to quit"],
        )

        cv2.imshow("MeetBetter (local recognition)", frame)
        if (cv2.waitKey(1) & 0xFF) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
