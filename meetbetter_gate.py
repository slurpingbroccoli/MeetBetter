import json
import math
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import sounddevice as sd
import vosk
from rapidfuzz import process, fuzz

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =======================
# Paths / Config
# =======================
MODEL_FACE_TFLITE = Path("models/blaze_face_short_range.tflite")
MODEL_VOSK_DIR = Path("models/vosk-model-small-en-us-0.15")
PROFILES_JSON = Path("profiles.json")

SAMPLE_RATE = 16000
BLOCKSIZE = 8000

# Name intent gating
INTENT_PHRASES = ["my name is", "my name", "i am", "i'm", "call me"]

# Loading/confirm + lock behavior
LOADING_SECONDS = 4.0
REQUIRE_HITS = 2          # "two passes"
LOCK_SECONDS = 6.0        # how long to keep the popup once confirmed

# Target-face selection weights
CENTER_WEIGHT = 0.35      # higher -> more centered preferred
AREA_WEIGHT = 0.65        # higher -> bigger face preferred

# =======================
# UI helpers
# =======================
def draw_card(img, x, y, lines, padding=12):
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.62
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


def draw_spinner(img, cx, cy, r, t, thickness=3):
    # Simple rotating arc spinner
    # angle progresses with time
    angle = (t * 360.0) % 360.0
    start = int(angle)
    end = int(angle + 270)  # 3/4 circle
    cv2.ellipse(img, (int(cx), int(cy)), (int(r), int(r)), 0, start, end, (255, 255, 255), thickness)


# =======================
# Voice (Vosk) thread
# =======================
def normalize_text(s: str) -> str:
    return "".join(ch.lower() for ch in s.strip())


def looks_like_name_intent(text: str) -> bool:
    t = normalize_text(text)
    return any(p in t for p in INTENT_PHRASES)


def extract_tail_after_intent(text: str) -> str:
    t = normalize_text(text)
    for p in INTENT_PHRASES:
        idx = t.find(p)
        if idx != -1:
            return t[idx + len(p):].strip()
    return ""


def pick_name_candidate(tail: str, known_names: list[str]):
    """
    Pick best fuzzy match among known names based on the tail of the phrase.
    Example tail: "vincent nice to meet you"
    We'll just match against known names and return (name, score).
    """
    tail = normalize_text(tail)
    if not tail:
        return None, 0

    # Use first ~3 words (names are short; this helps when tail includes extra words)
    short = " ".join(tail.split()[:3])

    match = process.extractOne(short, known_names, scorer=fuzz.WRatio)
    if not match:
        return None, 0
    name, score, _ = match
    return name, score


def voice_worker(event_q: queue.Queue, stop_event: threading.Event):
    if not MODEL_VOSK_DIR.exists():
        raise FileNotFoundError(f"Missing Vosk model folder: {MODEL_VOSK_DIR.resolve()}")
    if not PROFILES_JSON.exists():
        raise FileNotFoundError(f"Missing profiles.json: {PROFILES_JSON.resolve()}")

    profiles = json.loads(PROFILES_JSON.read_text())
    known_names = [normalize_text(k) for k in profiles.keys()]

    print("🎤 Voice: loading Vosk model…")
    model = vosk.Model(str(MODEL_VOSK_DIR))
    rec = vosk.KaldiRecognizer(model, SAMPLE_RATE)

    audio_q = queue.Queue()

    def cb(indata, frames, time_info, status):
        if stop_event.is_set():
            return
        audio_q.put(bytes(indata))

    print("🎤 Voice: listening… (say 'my name is ___')")
    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=BLOCKSIZE,
        dtype="int16",
        channels=1,
        callback=cb,
    ):
        while not stop_event.is_set():
            try:
                data = audio_q.get(timeout=0.25)
            except queue.Empty:
                continue

            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = result.get("text", "").strip()
                if not text:
                    continue

                if looks_like_name_intent(text):
                    tail = extract_tail_after_intent(text)
                    name, score = pick_name_candidate(tail, known_names)
                    if name and score >= 80:
                        # push event to main loop
                        event_q.put(("name", name, score, time.time(), text))


# =======================
# Camera (MediaPipe) target face
# =======================
def select_target_face(detections, W, H):
    """
    Choose biggest + most centered.
    Returns (det, bbox_px) or (None, None)
    bbox_px = (x1,y1,x2,y2)
    """
    if not detections:
        return None, None

    cx0, cy0 = W / 2.0, H / 2.0
    best = None
    best_score = -1e9
    best_box = None

    for det in detections:
        bbox = det.bounding_box
        x1 = int(bbox.origin_x)
        y1 = int(bbox.origin_y)
        x2 = int(bbox.origin_x + bbox.width)
        y2 = int(bbox.origin_y + bbox.height)

        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W - 1, x2), min(H - 1, y2)

        w = max(1, x2 - x1)
        h = max(1, y2 - y1)
        area = float(w * h)

        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        dist = math.hypot(cx - cx0, cy - cy0)
        max_dist = math.hypot(cx0, cy0)
        centered = 1.0 - (dist / (max_dist + 1e-9))  # 1.0 best, 0.0 worst

        score = AREA_WEIGHT * area + CENTER_WEIGHT * (centered * 100000.0)
        if score > best_score:
            best_score = score
            best = det
            best_box = (x1, y1, x2, y2)

    return best, best_box


# =======================
# State machine
# =======================
@dataclass
class State:
    mode: str = "IDLE"               # IDLE | LOADING | LOCKED
    candidate: str | None = None
    hits: int = 0
    loading_until: float = 0.0
    locked_name: str | None = None
    lock_until: float = 0.0


def main():
    if not MODEL_FACE_TFLITE.exists():
        raise FileNotFoundError(f"Missing face model: {MODEL_FACE_TFLITE.resolve()}")

    # Face detector (MediaPipe Tasks)
    options = vision.FaceDetectorOptions(
        base_options=python.BaseOptions(model_asset_path=str(MODEL_FACE_TFLITE)),
        min_detection_confidence=0.6,
    )
    detector = vision.FaceDetector.create_from_options(options)

    # Voice thread setup
    event_q: queue.Queue = queue.Queue()
    stop_event = threading.Event()
    vt = threading.Thread(target=voice_worker, args=(event_q, stop_event), daemon=True)
    vt.start()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        stop_event.set()
        raise RuntimeError("Could not open webcam. Check macOS Camera permissions for Terminal/Python.")

    st = State()

    print("✅ Running. ESC to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            now = time.time()
            H, W = frame.shape[:2]

            # --- run face detect
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result = detector.detect(mp_image)
            detections = result.detections if result and result.detections else []

            target_det, box = select_target_face(detections, W, H)

            # --- consume voice events (non-blocking)
            while True:
                try:
                    ev = event_q.get_nowait()
                except queue.Empty:
                    break

                kind, name, score, ts, raw_text = ev

                # Only accept voice if we currently have a target face (camera gates voice)
                if box is None:
                    continue

                # If locked, ignore new names until lock expires
                if st.mode == "LOCKED" and now < st.lock_until:
                    continue

                # If lock expired, drop to IDLE
                if st.mode == "LOCKED" and now >= st.lock_until:
                    st = State()

                # Start/continue loading window
                if st.mode in ("IDLE", "LOADING"):
                    if st.mode == "IDLE":
                        st.mode = "LOADING"
                        st.candidate = name
                        st.hits = 1
                        st.loading_until = now + LOADING_SECONDS
                    else:
                        # LOADING: count a hit if same candidate (or very close)
                        if name == st.candidate:
                            st.hits += 1
                        else:
                            # If a different name appears, only switch if it’s much stronger
                            # (prevents random flips)
                            # Simple rule: if we only have 1 hit so far, allow switch
                            if st.hits <= 1:
                                st.candidate = name
                                st.hits = 1

            # --- handle state timeout transitions
            if st.mode == "LOCKED" and now >= st.lock_until:
                st = State()

            if st.mode == "LOADING" and now >= st.loading_until:
                # finalize
                if st.candidate and st.hits >= REQUIRE_HITS:
                    st.mode = "LOCKED"
                    st.locked_name = st.candidate
                    st.lock_until = now + LOCK_SECONDS
                else:
                    # failed confirmation -> Unknown, reset
                    st = State()

            # --- draw UI
            if box is not None:
                x1, y1, x2, y2 = box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # anchor card next to face
                anchor_x = x2 + 12 if (x2 + 420) < W else max(0, x1 - 420)
                anchor_y = max(0, y1 - 10)

                if st.mode == "LOCKED":
                    name = st.locked_name or "Unknown"
                    lines = [
                        f"Profile: {name}",
                        f"Locked: {max(0.0, st.lock_until - now):.1f}s",
                        "ESC to quit",
                    ]
                elif st.mode == "LOADING":
                    remaining = max(0.0, st.loading_until - now)
                    lines = [
                        "Loading… confirming name",
                        f"Heard: {st.candidate} (hits {st.hits}/{REQUIRE_HITS})",
                        f"Time: {remaining:.1f}s",
                        "ESC to quit",
                    ]
                    # spinner near the card
                    draw_spinner(frame, anchor_x + 25, anchor_y + 25, 12, now)
                else:
                    lines = [
                        "Listening…",
                        "Say: 'my name is ____'",
                        "ESC to quit",
                    ]

                draw_card(frame, anchor_x, anchor_y, lines)

            else:
                # no target face
                draw_card(frame, 20, 20, ["No target face", "Look at someone", "ESC to quit"])

            cv2.imshow("MeetBetter (Camera gated voice)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break

    finally:
        stop_event.set()
        cap.release()
        cv2.destroyAllWindows()
        detector.close()


if __name__ == "__main__":
    main()

