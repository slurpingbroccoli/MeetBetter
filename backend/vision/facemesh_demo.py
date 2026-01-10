import os, json
import cv2
import numpy as np
import mediapipe as mp

PROFILES = json.load(open("profiles.json", "r"))
BASE = os.path.join("data", "people")

# Load enrolled signatures
known = {}
if os.path.isdir(BASE):
    for pid in os.listdir(BASE):
        p = os.path.join(BASE, pid, "signature.npy")
        if os.path.isfile(p):
            known[pid] = np.load(p)

mp_face = mp.solutions.face_mesh
mesh = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
cap = cv2.VideoCapture(0)

IDX = [33, 133, 362, 263, 1, 61, 291, 199]

def signature(landmarks, w, h):
    pts = []
    for i in IDX:
        lm = landmarks[i]
        pts.append([lm.x * w, lm.y * h])
    pts = np.array(pts, dtype=np.float32)
    eye_dist = np.linalg.norm(pts[0] - pts[2]) + 1e-6
    pts = (pts - pts.mean(axis=0)) / eye_dist
    return pts.flatten()

def cosine_dist(a, b):
    a = a / (np.linalg.norm(a) + 1e-6)
    b = b / (np.linalg.norm(b) + 1e-6)
    return 1.0 - float(np.dot(a, b))

# Prototype threshold — you may need to tune
THRESH = 0.18

last_id = None

while True:
    ok, frame = cap.read()
    if not ok:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = mesh.process(rgb)
    h, w = frame.shape[:2]

    label = "Unknown (not enrolled)"
    card = None

    if res.multi_face_landmarks and known:
        face_lms = res.multi_face_landmarks[0].landmark
        sig = signature(face_lms, w, h)

        best_id, best_d = None, 999
        for pid, ksig in known.items():
            d = cosine_dist(sig, ksig)
            if d < best_d:
                best_d, best_id = d, pid

        if best_id is not None and best_d < THRESH:
            label = f"Matched: {best_id}  (d={best_d:.3f})"
            card = PROFILES.get(best_id)

    # Draw label
    cv2.rectangle(frame, (10, 10), (w-10, 170), (0,0,0), -1)
    cv2.putText(frame, "MeetBetter Prototype", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    cv2.putText(frame, label, (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255,255,255), 2)

    # “Pop-up card”
    if card:
        cv2.putText(frame, f"Name: {card.get('name','')}", (20, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
        cv2.putText(frame, f"Role: {card.get('role','')}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
        ask = ", ".join(card.get("ask_me_about", [])[:3])
        cv2.putText(frame, f"Ask: {ask}", (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 2)

    cv2.imshow("MeetBetter", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
