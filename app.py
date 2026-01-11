# app.py
import time
import threading
from pathlib import Path
import requests

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from backend.gemini.gemini import generate_conversation_starters

MODEL_PATH = Path("models/blaze_face_short_range.tflite")
DB_PATH = Path("db/face_db.npz")

# Map your recognized labels -> profile URLs (demo dataset)
NAME_TO_URL = {
    "edison": "https://example.com/demo/edison",
    "vincent": "https://www.linkedin.com/in/vincent-bei-937487371/",
    "lily": "https://example.com/demo/lily",

    "jay": "https://www.linkedin.com/in/jaydasani/",
    "jai": "https://www.linkedin.com/in/jaydasani/",
    "jae": "https://www.linkedin.com/in/jaydasani/",
    "j": "https://www.linkedin.com/in/jaydasani/",

    # tithi
    "tithi": "https://www.linkedin.com/in/tithi007/",
    "titi": "https://www.linkedin.com/in/tithi007/",
    "tithy": "https://www.linkedin.com/in/tithi007/",

    # luigi
    "luigi": "https://www.linkedin.com/in/luigitomasone/",
    "louigi": "https://www.linkedin.com/in/luigitomasone/",
    "louie g": "https://www.linkedin.com/in/luigitomasone/",

    # razi
    "razi": "https://www.linkedin.com/in/raztronaut/",
    "razy": "https://www.linkedin.com/in/raztronaut/",
    "razzy": "https://www.linkedin.com/in/raztronaut/",

    # joshua
    "joshua": "https://www.linkedin.com/in/joshua-matte1/",
    "josh": "https://www.linkedin.com/in/joshua-matte1/",
    "josua": "https://www.linkedin.com/in/joshua-matte1/",

    # andrei
    "andrei": "https://www.linkedin.com/in/andrei-lazakovitch-848542145/",
    "andrey": "https://www.linkedin.com/in/andrei-lazakovitch-848542145/",
    "andre": "https://www.linkedin.com/in/andrei-lazakovitch-848542145/",

    # mohammed
    "mohammed": "https://www.linkedin.com/in/mohammed2ibrahim/",
    "muhammad": "https://www.linkedin.com/in/mohammed2ibrahim/",
    "mohamad": "https://www.linkedin.com/in/mohammed2ibrahim/",
    "mo": "https://www.linkedin.com/in/mohammed2ibrahim/",

    # robin
    "robin": "https://www.linkedin.com/in/gershmanrobin/",
    "robyn": "https://www.linkedin.com/in/gershmanrobin/",
    "rob": "https://www.linkedin.com/in/gershmanrobin/",

    # mansour
    "mansour": "https://www.linkedin.com/in/mansour-karami/",
    "mansoor": "https://www.linkedin.com/in/mansour-karami/",
    "manzor": "https://www.linkedin.com/in/mansour-karami/",

    # aman
    "aman": "https://www.linkedin.com/in/amanhiranpurohit/",
    "amaan": "https://www.linkedin.com/in/amanhiranpurohit/",
    "amon": "https://www.linkedin.com/in/amanhiranpurohit/",

    # jeffrey
    "jeffrey": "https://www.linkedin.com/in/jeffrey-kwong-8168441/",
    "jeff": "https://www.linkedin.com/in/jeffrey-kwong-8168441/",
    "geoff": "https://www.linkedin.com/in/jeffrey-kwong-8168441/",

    # matthew
    "matthew": "https://www.linkedin.com/in/matthew-mccracken/",
    "matt": "https://www.linkedin.com/in/matthew-mccracken/",
    "mathew": "https://www.linkedin.com/in/matthew-mccracken/",

    # steven
    "steven": "https://www.linkedin.com/in/steven-gonder/",
    "stephen": "https://www.linkedin.com/in/steven-gonder/",
    "steve": "https://www.linkedin.com/in/steven-gonder/",

    # david
    "david": "https://www.linkedin.com/in/davidle519/",
    "dave": "https://www.linkedin.com/in/davidle519/",
    "daved": "https://www.linkedin.com/in/davidle519/",

    # anish
    "anish": "https://www.linkedin.com/in/anish-rangarajan/",
    "aneesh": "https://www.linkedin.com/in/anish-rangarajan/",
    "anishh": "https://www.linkedin.com/in/anish-rangarajan/",

    # ayushi
    "ayushi": "https://www.linkedin.com/in/ayushimalhotra/",
    "aayushi": "https://www.linkedin.com/in/ayushimalhotra/",

    # kaan
    "kaan": "https://www.linkedin.com/in/hkaanturgut/",
    "kahn": "https://www.linkedin.com/in/hkaanturgut/",
    "kan": "https://www.linkedin.com/in/hkaanturgut/",

    # emily
    "emily": "https://www.linkedin.com/in/emily-y-goodwin/",
    "emilie": "https://www.linkedin.com/in/emily-y-goodwin/",
    "em": "https://www.linkedin.com/in/emily-y-goodwin/",

    # riya
    "riya": "https://www.linkedin.com/in/riyaashukla/",
    "ria": "https://www.linkedin.com/in/riyaashukla/",
    "reeya": "https://www.linkedin.com/in/riyaashukla/",

    # pranav
    "pranav": "https://www.linkedin.com/in/gpn2393/",
    "pranab": "https://www.linkedin.com/in/gpn2393/",
    "pronov": "https://www.linkedin.com/in/gpn2393/",

    # munaz
    "munaz": "https://www.linkedin.com/in/munaz/",
    "munazz": "https://www.linkedin.com/in/munaz/",
    "munus": "https://www.linkedin.com/in/munaz/",

    # luke
    "luke": "https://www.linkedin.com/in/luke-p/",
    "luc": "https://www.linkedin.com/in/luke-p/",
    "luk": "https://www.linkedin.com/in/luke-p/",

    # ali
    "ali": "https://www.linkedin.com/in/alikhan2510/",
    "alee": "https://www.linkedin.com/in/alikhan2510/",
    "ally": "https://www.linkedin.com/in/alikhan2510/",

    # jason
    "jason": "https://www.linkedin.com/in/jhuang03/",
    "jayson": "https://www.linkedin.com/in/jhuang03/",
    "jasen": "https://www.linkedin.com/in/jhuang03/",

    # divya
    "divya": "https://www.linkedin.com/in/divyajain-tech/",
    "divia": "https://www.linkedin.com/in/divyajain-tech/",
    "divyah": "https://www.linkedin.com/in/divyajain-tech/",

    # konstantin
    "konstantin": "https://www.linkedin.com/in/konstantin-delemen-a34592211/",
    "constantine": "https://www.linkedin.com/in/konstantin-delemen-a34592211/",
    "konstantine": "https://www.linkedin.com/in/konstantin-delemen-a34592211/",

    # ameer
    "ameer": "https://www.linkedin.com/in/ameer-hamoodi-346889164/",
    "amir": "https://www.linkedin.com/in/ameer-hamoodi-346889164/",
    "aamir": "https://www.linkedin.com/in/ameer-hamoodi-346889164/"
}

my_linkedin_link = "https://www.linkedin.com/in/edisoncai/"
my_linkedin = None
other_linkedin = None

# If it doesn't recognize you, LOWER a bit (e.g. 0.78)
# If it falsely recognizes random faces, RAISE a bit (e.g. 0.88)
SIM_THRESHOLD = 0.82

# How many camera candidates to keep around for fusion (debug only)
CAM_TOPK = 5

# Card smoothing
ALPHA = 0.45
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

# -------- Fetch UI state (set by main.py) --------
FETCH_STATUS: str = "IDLE"   # IDLE | FETCHING | DONE | ERROR
FETCH_TITLE: str = ""
FETCH_LINES: list[str] = []
FETCH_UNTIL: float = 0.0

# -------- Profile Card state (anchored next to head) --------
PROFILE_NAME: str = ""
PROFILE_HEADLINE: str = ""
PROFILE_LINES: list[str] = []
PROFILE_UNTIL: float = 0.0

# -------- UI Mode (NEW) --------
# NORMAL: show debug cards
# LOCKED/FETCHING/RESULT: "wipe" debug cards and only show anchored state
UI_MODE: str = "NORMAL"   # NORMAL | LOCKED | FETCHING | RESULT


def _safe_preview_lines(data: dict) -> list[str]:
    """
    Convert server JSON into short UI lines.
    Adjust this to match your server response shape.
    """
    if not isinstance(data, dict):
        return ["(bad response)"]

    name = str(data.get("name", "")).strip()
    headline = str(data.get("headline", "")).strip()
    location = str(data.get("location", "")).strip()

    lines = []
    if name:
        lines.append(f"{name}")
    if headline:
        lines.append(f"{headline}")
    if location:
        lines.append(f"{location}")

    starters = data.get("starters")
    if isinstance(starters, list) and starters:
        lines.append("Starters:")
        for s in starters[:2]:
            lines.append(f"- {str(s)[:120]}")

    if not lines:
        lines = ["(no fields)"]
    return lines


def set_ui_mode(mode: str) -> None:
    global UI_MODE
    with _state_lock:
        UI_MODE = mode


def clear_all_cards() -> None:
    """Hard 'wipe' of all overlay state (next frame will be clean)."""
    clear_fetch_overlay()
    clear_profile_card()
    global TRY_AGAIN_MSG, TRY_AGAIN_UNTIL
    with _state_lock:
        TRY_AGAIN_MSG = None
        TRY_AGAIN_UNTIL = 0.0


def set_voice_lock(name: str, seconds: float = 10.0) -> None:
    """Called from mic thread to force UI to show this name for a bit."""
    global VOICE_LOCK_NAME, VOICE_LOCK_UNTIL, other_linkedin

    n = (name or "").strip()
    if not n:
        return

    now = time.time()
    with _state_lock:
        VOICE_LOCK_NAME = n
        VOICE_LOCK_UNTIL = now + float(seconds)

    # WIPE all existing cards when a lock happens
    clear_all_cards()
    set_ui_mode("LOCKED")

    print(f"🔒 VOICE LOCK -> {n} ({seconds:.1f}s)")

    # Enter fetching phase (only anchored card should show)
    set_ui_mode("FETCHING")

    if n not in NAME_TO_URL:
        set_fetch_overlay("ERROR", "Unknown name", [f"No URL mapped for: {n}"], seconds=4.0)
        set_ui_mode("LOCKED")
        return

    url = "http://172.18.76.85:8000/scrape"
    payload = {"key": "extremelyrarepictureofaseaspugnar", "url": NAME_TO_URL[n]}

    try:
        other_linkedin = requests.post(url, json=payload, timeout=15)
        other_data = other_linkedin.json()
    except Exception as e:
        set_fetch_overlay("ERROR", "Fetch failed", [str(e)[:180]], seconds=6.0)
        set_ui_mode("LOCKED")
        return

    # Set final anchored profile card
    lines = _safe_preview_lines(other_data)
    set_profile_card(other_data.get("name", n), other_data.get("headline", ""), lines, seconds=30.0)

    # Add conversation starters into the anchored profile card (best-effort)
    try:
        starters_text = generate_conversation_starters(my_linkedin.json(), other_data)
        starter_lines = [s.strip() for s in str(starters_text).split("\n") if s.strip()]
        if starter_lines:
            merged = lines + ["Starters:"] + starter_lines[:3]
            set_profile_card(other_data.get("name", n), other_data.get("headline", ""), merged, seconds=30.0)
    except Exception:
        pass

    set_ui_mode("RESULT")


def clear_voice_lock() -> None:
    global VOICE_LOCK_NAME, VOICE_LOCK_UNTIL
    with _state_lock:
        VOICE_LOCK_NAME = None
        VOICE_LOCK_UNTIL = 0.0
    # Optional: when unlocking, return to NORMAL if you want
    # set_ui_mode("NORMAL")


def get_voice_lock() -> tuple[str | None, float]:
    """Returns (name, until_ts)."""
    with _state_lock:
        return VOICE_LOCK_NAME, VOICE_LOCK_UNTIL


def set_try_again(msg: str = "Please try again", seconds: float = 2.5) -> None:
    global TRY_AGAIN_MSG, TRY_AGAIN_UNTIL
    with _state_lock:
        TRY_AGAIN_MSG = msg
        TRY_AGAIN_UNTIL = time.time() + float(seconds)
    print(f"⚠️ TRY AGAIN -> {msg} ({seconds:.1f}s)")


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


def set_fetch_overlay(status: str, title: str, lines: list[str], seconds: float = 8.0) -> None:
    """Called from main.py to show server results / loading / error."""
    global FETCH_STATUS, FETCH_TITLE, FETCH_LINES, FETCH_UNTIL
    with _state_lock:
        FETCH_STATUS = status
        FETCH_TITLE = title
        FETCH_LINES = list(lines)
        FETCH_UNTIL = time.time() + float(seconds)


def clear_fetch_overlay() -> None:
    global FETCH_STATUS, FETCH_TITLE, FETCH_LINES, FETCH_UNTIL
    with _state_lock:
        FETCH_STATUS = "IDLE"
        FETCH_TITLE = ""
        FETCH_LINES = []
        FETCH_UNTIL = 0.0


def set_profile_card(name: str, headline: str, lines: list[str], seconds: float = 20.0) -> None:
    """Show a card anchored next to head."""
    global PROFILE_NAME, PROFILE_HEADLINE, PROFILE_LINES, PROFILE_UNTIL
    with _state_lock:
        PROFILE_NAME = str(name or "").strip()
        PROFILE_HEADLINE = str(headline or "").strip()
        PROFILE_LINES = list(lines or [])
        PROFILE_UNTIL = time.time() + float(seconds)


def clear_profile_card() -> None:
    global PROFILE_NAME, PROFILE_HEADLINE, PROFILE_LINES, PROFILE_UNTIL
    with _state_lock:
        PROFILE_NAME = ""
        PROFILE_HEADLINE = ""
        PROFILE_LINES = []
        PROFILE_UNTIL = 0.0


def _wrap_text(text, font, scale, thickness, max_width):
    words = str(text).split()
    if not words:
        return [""]
    lines, cur = [], words[0]
    for w in words[1:]:
        test = cur + " " + w
        (tw, _), _ = cv2.getTextSize(test, font, scale, thickness)
        if tw <= max_width:
            cur = test
        else:
            lines.append(cur)
            cur = w
    lines.append(cur)
    return lines


def _rounded_rect(img, x, y, w, h, r, color, thickness=-1):
    if thickness == -1:
        cv2.rectangle(img, (x+r, y), (x+w-r, y+h), color, -1, cv2.LINE_AA)
        cv2.rectangle(img, (x, y+r), (x+w, y+h-r), color, -1, cv2.LINE_AA)
        cv2.ellipse(img, (x+r, y+r), (r, r), 180, 0, 90, color, -1, cv2.LINE_AA)
        cv2.ellipse(img, (x+w-r, y+r), (r, r), 270, 0, 90, color, -1, cv2.LINE_AA)
        cv2.ellipse(img, (x+r, y+h-r), (r, r), 90, 0, 90, color, -1, cv2.LINE_AA)
        cv2.ellipse(img, (x+w-r, y+h-r), (r, r), 0, 0, 90, color, -1, cv2.LINE_AA)
    else:
        cv2.line(img, (x+r, y), (x+w-r, y), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x+r, y+h), (x+w-r, y+h), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x, y+r), (x, y+h-r), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x+w, y+r), (x+w, y+h-r), color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x+r, y+r), (r, r), 180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x+w-r, y+r), (r, r), 270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x+r, y+h-r), (r, r), 90, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x+w-r, y+h-r), (r, r), 0, 0, 90, color, thickness, cv2.LINE_AA)


def draw_card(img, x, y, title, lines, footer=None, width=320):
    H, W = img.shape[:2]

    padding = 10
    radius = 12
    header_h = 48

    bg = (18, 18, 18)
    header_bg = (28, 28, 28)
    border = (200, 200, 200)

    font = cv2.FONT_HERSHEY_SIMPLEX
    title_scale, title_th = 0.60, 2
    body_scale, body_th = 0.55, 1
    foot_scale, foot_th = 0.45, 1

    max_text_w = width - 2 * padding

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

    x = int(max(8, min(x, W - width - 8)))
    y = int(max(8, min(y, H - height - 8)))

    overlay = img.copy()
    _rounded_rect(overlay, x-1, y-1, width, height, radius, (0, 0, 0), -1)
    _rounded_rect(overlay, x, y, width, height, radius, bg, -1)
    _rounded_rect(overlay, x, y, width, header_h, radius, header_bg, -1)
    cv2.rectangle(overlay, (x, y + header_h - radius), (x + width, y + header_h), header_bg, -1)
    cv2.addWeighted(overlay, 0.86, img, 0.14, 0, img)

    _rounded_rect(img, x, y, width, height, radius, border, 2)

    tx = x + padding + 3
    ty = y + int(header_h/2) + int(title_h/2) - 2
    cv2.putText(img, str(title), (tx, ty + 3), font, title_scale, (255, 255, 255), title_th, cv2.LINE_AA)

    by = y + header_h + padding + body_h
    for line in wrapped:
        cv2.putText(img, line, (tx, by), font, body_scale, (245, 245, 245), body_th, cv2.LINE_AA)
        by += body_h + 10

    if footer_lines:
        by += 6
        for line in footer_lines:
            cv2.putText(img, line, (tx, by), font, foot_scale, (200, 200, 200), foot_th, cv2.LINE_AA)
            by += foot_h + 8


def main():
    url = "http://172.18.76.85:8000/scrape"
    other = {"key": "extremelyrarepictureofaseaspugnar", "url": my_linkedin_link}
    global my_linkedin
    my_linkedin = requests.post(url, json=other)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model: {MODEL_PATH.resolve()}")
    if not DB_PATH.exists():
        raise FileNotFoundError(f"Missing DB: {DB_PATH.resolve()} (run python enroll.py first)")

    db = np.load(DB_PATH, allow_pickle=True)
    embs = db["embeddings"].astype(np.float32)
    labels = db["labels"]

    embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9)

    base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.FaceDetectorOptions(base_options=base_options)
    detector = vision.FaceDetector.create_from_options(options)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check macOS Camera permissions.")

    print("ESC to quit (press S to fetch when locked)")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        now = time.time()
        H, W = frame.shape[:2]

        with _state_lock:
            ta_msg = TRY_AGAIN_MSG
            ta_until = TRY_AGAIN_UNTIL
            vl_name = VOICE_LOCK_NAME
            vl_until = VOICE_LOCK_UNTIL

            fetch_status = FETCH_STATUS
            fetch_title = FETCH_TITLE
            fetch_lines = list(FETCH_LINES)
            fetch_until = FETCH_UNTIL

            prof_name = PROFILE_NAME
            prof_headline = PROFILE_HEADLINE
            prof_lines = list(PROFILE_LINES)
            prof_until = PROFILE_UNTIL

            ui_mode = UI_MODE

        # Force name if voice lock active (but do NOT draw top-left lock card)
        force_name = None
        if vl_name and now < vl_until:
            force_name = vl_name

        # camera detect
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)

        if not result.detections:
            _publish_camera_topk([])

            # Only show debug cards in NORMAL mode; otherwise "wipe"
            if ui_mode == "NORMAL":
                if ta_msg and now < ta_until:
                    draw_card(frame, 20, 20, title="⚠️", lines=[ta_msg], footer="ESC to quit")
                if fetch_status != "IDLE" and now < fetch_until:
                    draw_card(
                        frame, 20, 120,
                        title=fetch_title or "Result",
                        lines=fetch_lines if fetch_lines else ["..."],
                        footer="S=fetch  C=clear  ESC=quit",
                        width=420,
                    )
                draw_card(frame, 20, 20, title="Name: Unknown", lines=["No face"], footer="ESC to quit")

            cv2.imshow("MeetBetter", frame)
            if (cv2.waitKey(1) & 0xFF) == 27:
                break
            continue

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

            if ui_mode == "NORMAL":
                if ta_msg and now < ta_until:
                    draw_card(frame, 20, 20, title="⚠️", lines=[ta_msg], footer="ESC to quit")
                if fetch_status != "IDLE" and now < fetch_until:
                    draw_card(
                        frame, 20, 120,
                        title=fetch_title or "Result",
                        lines=fetch_lines if fetch_lines else ["..."],
                        footer="S=fetch  C=clear  ESC=quit",
                        width=420,
                    )
                draw_card(frame, 20, 20, title="Name: Unknown", lines=["Bad crop"], footer="ESC to quit")

            cv2.imshow("MeetBetter", frame)
            if (cv2.waitKey(1) & 0xFF) == 27:
                break
            continue

        crop_small = cv2.resize(crop, (112, 112), interpolation=cv2.INTER_AREA).astype(np.float32)
        q = crop_small.flatten()
        q = q / (np.linalg.norm(q) + 1e-9)

        if embs.shape[1] != q.shape[0]:
            raise RuntimeError(
                f"Embedding dim mismatch: DB={embs.shape[1]} vs runtime={q.shape[0]}. "
                f"Re-run enroll.py or ensure both use same resize."
            )

        sims = embs @ q
        k = min(CAM_TOPK, sims.shape[0])
        top_idx = np.argsort(-sims)[:k]
        topk = [(str(labels[i]), float(sims[i])) for i in top_idx]
        _publish_camera_topk(topk)

        best_name, best_sim = topk[0]
        best_sim = _smooth("sim", best_sim)

        name = "Unknown"
        if best_sim >= SIM_THRESHOLD:
            name = best_name

        if force_name:
            name = force_name

        anchor_x = x2 + 12
        anchor_y = max(0, y1 - 10)
        top2_line = " / ".join([f"{n}:{s:.3f}" for (n, s) in topk[:2]])

        # NORMAL: show debug overlays (try-again + fetch toast + name/debug card)
        if ui_mode == "NORMAL":
            if ta_msg and now < ta_until:
                draw_card(frame, 20, 20, title="⚠️", lines=[ta_msg], footer="ESC to quit")

            if fetch_status != "IDLE" and now < fetch_until:
                draw_card(
                    frame,
                    20,
                    120,
                    title=fetch_title or "Result",
                    lines=fetch_lines if fetch_lines else ["..."],
                    footer="S=fetch  C=clear  ESC=quit",
                    width=420,
                )

            draw_card(
                frame,
                anchor_x,
                anchor_y,
                title=f"Name: {name}",
                lines=[f"Sim: {best_sim:.3f}", f"Top: {top2_line}"],
                footer=("Press S to fetch" if force_name else "Waiting for voice lock…"),
                width=360,
            )

        # LOCKED/FETCHING: wipe all debug cards; show only a single anchored state card
        elif ui_mode in ("LOCKED", "FETCHING"):
            draw_card(
                frame,
                anchor_x,
                anchor_y,
                title=f"{name}",
                lines=(["Fetching…"] if ui_mode == "FETCHING" else ["Locked."]),
                footer=None,
                width=420,
            )

        # RESULT: wipe all debug cards; show only final anchored profile card
        elif ui_mode == "RESULT":
            if now < prof_until and (prof_name or prof_headline or prof_lines):
                card_lines = []
                if prof_headline:
                    card_lines.append(prof_headline)
                card_lines.extend(prof_lines)

                draw_card(
                    frame,
                    anchor_x,
                    anchor_y,
                    title=prof_name or "Profile",
                    lines=card_lines[:12],
                    footer=None,
                    width=420,
                )
            else:
                # If result expires, return to normal debug view
                set_ui_mode("NORMAL")

        cv2.imshow("MeetBetter", frame)
        if (cv2.waitKey(1) & 0xFF) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
