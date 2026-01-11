import queue
import sounddevice as sd
import webrtcvad
import numpy as np
import time
import json
import re

from faster_whisper import WhisperModel
from rapidfuzz import process, fuzz

# ---------------- config ----------------
SAMPLE_RATE = 16000
FRAME_MS = 30
FRAME_SIZE = int(SAMPLE_RATE * FRAME_MS / 1000)

VAD_MODE = 2            # 0–3 (higher = more aggressive)
MAX_SEGMENT_SEC = 6.0   # max speech chunk length

INTRO_PATTERNS = [
    r"\bmy name is (\w+)",
    r"\bi am (\w+)",
    r"\bi'm (\w+)",
    r"\bthis is (\w+)",
    r"\bcall me (\w+)"
]
# ----------------------------------------

with open("profiles.json") as f:
    PROFILES = json.load(f)

KNOWN_NAMES = list(PROFILES.keys())


def extract_name(text: str):
    t = text.lower()
    for p in INTRO_PATTERNS:
        m = re.search(p, t)
        if m:
            return m.group(1)
    return None


def match_name(name):
    match, score, _ = process.extractOne(
        name, KNOWN_NAMES, scorer=fuzz.ratio
    )
    if score >= 80:
        return match
    return None


def main():
    print("Loading Whisper model (first time may take ~10–20s)…")
    model = WhisperModel("base", compute_type="int8")
    vad = webrtcvad.Vad(VAD_MODE)

    audio_q = queue.Queue()

    def audio_cb(indata, frames, time_info, status):
        audio_q.put(bytes(indata))

    print("🎙️  Always listening… say: 'my name is Vincent'")
    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=FRAME_SIZE,
        dtype="int16",
        channels=1,
        callback=audio_cb,
    ):
        voiced = []
        speaking = False
        start_t = None

        while True:
            frame = audio_q.get()
            is_speech = vad.is_speech(frame, SAMPLE_RATE)

            if is_speech:
                if not speaking:
                    speaking = True
                    start_t = time.time()
                    voiced = []
                voiced.append(frame)

            else:
                if speaking:
                    speaking = False
                    audio = b"".join(voiced)
                    samples = np.frombuffer(audio, np.int16).astype(np.float32) / 32768.0

                    segments, _ = model.transcribe(samples)
                    text = " ".join(seg.text for seg in segments).strip()
                    print(f"🗣️  Heard: {text}")

                    name = extract_name(text)
                    if name:
                        match = match_name(name)
                        if match:
                            print(f"✅ Matched name: {match}")
                            print(f"Profile → {PROFILES[match]}")
                        else:
                            print("❌ Name not in DB")
                    else:
                        print("ℹ️  No intro phrase detected")

            if speaking and time.time() - start_t > MAX_SEGMENT_SEC:
                speaking = False


if __name__ == "__main__":
    main()
