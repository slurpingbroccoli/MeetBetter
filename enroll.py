import cv2
import numpy as np
from pathlib import Path

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

PEOPLE_DIR = Path("people")
DB_DIR = Path("db")
DB_PATH = DB_DIR / "face_db.npz"
MODEL_PATH = Path("models/blaze_face_short_range.tflite")

print("✅ enroll.py running")
print("cwd:", Path.cwd())
print("people dir:", PEOPLE_DIR.resolve())
print("model path:", MODEL_PATH.resolve())

if not PEOPLE_DIR.exists():
    raise FileNotFoundError(f"Missing folder: {PEOPLE_DIR.resolve()}")

imgs = [p for p in PEOPLE_DIR.glob("*") if p.is_file()]
print("people contents:", [p.name for p in imgs])

if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Missing model: {MODEL_PATH.resolve()}")

DB_DIR.mkdir(exist_ok=True)

options = vision.FaceDetectorOptions(
    base_options=python.BaseOptions(model_asset_path=str(MODEL_PATH)),
    min_detection_confidence=0.6,
)
detector = vision.FaceDetector.create_from_options(options)

embeddings = []
labels = []

for img_path in imgs:
    name = img_path.stem.lower()

    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[skip] cv2 could not read: {img_path.name}")
        continue

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_image)
    n = len(result.detections) if result.detections else 0

    print(f"[info] {img_path.name}: detected {n} face(s)")

    if n != 1:
        print(f"[skip] {img_path.name}: need exactly 1 face for enrollment")
        continue

    det = result.detections[0]
    box = det.bounding_box

    x1 = max(0, box.origin_x)
    y1 = max(0, box.origin_y)
    x2 = min(img.shape[1], x1 + box.width)
    y2 = min(img.shape[0], y1 + box.height)

    face = img[y1:y2, x1:x2]
    if face.size == 0:
        print(f"[skip] {img_path.name}: empty crop")
        continue

    # Placeholder embedding (pixel-based) so enrollment works immediately.
    # We'll swap to InsightFace embeddings next.
    face = cv2.resize(face, (112, 112))
    emb = face.flatten().astype(np.float32) / 255.0

    embeddings.append(emb)
    labels.append(name)
    print(f"[ok] enrolled {name}")

detector.close()

print("Embeddings:", len(embeddings), "Labels:", len(labels))

if len(embeddings) == 0:
    raise RuntimeError("Enrolled 0 faces. Put a clear photo in people/ with exactly one face.")

np.savez(DB_PATH, embeddings=np.array(embeddings), labels=np.array(labels))
print(f"✅ Saved {DB_PATH.resolve()}")
