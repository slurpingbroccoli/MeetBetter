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
ALPHA = 0.45
smoothed = {}

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
    # filled rounded rect
    if thickness == -1:
        cv2.rectangle(img, (x+r, y), (x+w-r, y+h), color, -1, cv2.LINE_AA)
        cv2.rectangle(img, (x, y+r), (x+w, y+h-r), color, -1, cv2.LINE_AA)
        cv2.ellipse(img, (x+r, y+r), (r, r), 180, 0, 90, color, -1, cv2.LINE_AA)
        cv2.ellipse(img, (x+w-r, y+r), (r, r), 270, 0, 90, color, -1, cv2.LINE_AA)
        cv2.ellipse(img, (x+r, y+h-r), (r, r), 90, 0, 90, color, -1, cv2.LINE_AA)
        cv2.ellipse(img, (x+w-r, y+h-r), (r, r), 0, 0, 90, color, -1, cv2.LINE_AA)
    else:
        # outline: draw 4 lines + 4 arcs
        cv2.line(img, (x+r, y), (x+w-r, y), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x+r, y+h), (x+w-r, y+h), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x, y+r), (x, y+h-r), color, thickness, cv2.LINE_AA)
        cv2.line(img, (x+w, y+r), (x+w, y+h-r), color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x+r, y+r), (r, r), 180, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x+w-r, y+r), (r, r), 270, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x+r, y+h-r), (r, r), 90, 0, 90, color, thickness, cv2.LINE_AA)
        cv2.ellipse(img, (x+w-r, y+h-r), (r, r), 0, 0, 90, color, thickness, cv2.LINE_AA)

def draw_card(img, x, y, title, lines, footer=None, width=250):
    """
    LinkedIn-ish card: header + body + footer, no spinner.
    """
    H, W = img.shape[:2]

    # styling
    padding = 10
    radius = 11
    header_h = 45

    bg = (18, 18, 18)
    header_bg = (28, 28, 28)
    border = (200, 200, 200)

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
    _rounded_rect(overlay, x-1, y-1, width, height, radius, (0, 0, 0), -1)
    _rounded_rect(overlay, x, y, width, height, radius, bg, -1)
    _rounded_rect(overlay, x, y, width, header_h, radius, header_bg, -1)
    cv2.rectangle(overlay, (x, y + header_h - radius), (x + width, y + header_h), header_bg, -1)

    cv2.addWeighted(overlay, 0.86, img, 0.14, 0, img)

    # border (keep this!)
    _rounded_rect(img, x, y, width, height, radius, border, 2)

    # header text
    tx = x + padding + 3
    ty = y + int(header_h/2) + int(title_h/2) - 2
    cv2.putText(img, str(title), (tx, ty + 3), font, title_scale, (255, 255, 255), title_th, cv2.LINE_AA)

    # body
    by = y + header_h + padding + body_h
    for line in wrapped:
        cv2.putText(img, line, (tx, by), font, body_scale, (245, 245, 245), body_th, cv2.LINE_AA)
        by += body_h + 10

    # footer
    if footer_lines:
        by += 6
        for line in footer_lines:
            cv2.putText(img, line, (tx, by), font, foot_scale, (200, 200, 200), foot_th, cv2.LINE_AA)
            by += foot_h + 8


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
                title=f"Name: {name}",
                lines=[
                 f"Similarity: {score:.3f}",
                 "ESC to quit",
                 "Software Developer at Sea Spungar, HAMOOOOD HABIBIBI"
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
