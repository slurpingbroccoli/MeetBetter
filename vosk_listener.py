import json
import sounddevice as sd
import queue
import sys
import re

from vosk import Model, KaldiRecognizer
from rapidfuzz import process, fuzz

# -------- CONFIG --------
SAMPLE_RATE = 16000
MODEL_PATH = "models/vosk-model-small-en-us-0.15"

INTRO_PATTERNS = [
    r"\bmy name is (\w+)",
    r"\bi am (\w+)",
    r"\bi'm (\w+)",
    r"\bthis is (\w+)",
    r"\bcall me (\w+)",
]
# ------------------------

with open("profiles.json") as f:
    PROFILES = json.load(f)

KNOWN_NAMES = list(PROFILES.keys())
if not KNOWN_NAMES:
    print("No names in profiles.json")
    sys.exit(1)


def extract_name(text):
    t = text.lower()
    for p in INTRO_PATTERNS:
        m = re.search(p, t)
        if m:
            return m.group(1)
    return None


def fuzzy_match(name):
    match, score, _ = process.extractOne(name, KNOWN_NAMES, scorer=fuzz.ratio)
    if score >= 80:
        return match
    return None


q = queue.Queue()


def audio_callback(indata, frames, time, status):
    q.put(bytes(indata))


def main(callback=None):
    """
    If callback is provided, call callback(matched_name) whenever a name is detected.
    """
    print("🔊 Loading Vosk model…")
    model = Model(MODEL_PATH)
    rec = KaldiRecognizer(model, SAMPLE_RATE)
    rec.SetWords(False)

    print("🎙️  Listening (say: 'my name is Vincent')")

    with sd.RawInputStream(
        samplerate=SAMPLE_RATE,
        blocksize=8000,
        dtype="int16",
        channels=1,
        callback=audio_callback,
    ):
        while True:
            data = q.get()
            if rec.AcceptWaveform(data):
                result = json.loads(rec.Result())
                text = result.get("text", "").strip()

                if not text:
                    continue

                print(f"🗣️  Heard: {text}")

                name = extract_name(text) or text.split()[0]
                match = fuzzy_match(name)

                if match:
                    print(f"✅ Name detected: {match}")
                    print(f"Profile → {PROFILES[match]}")

                    # ✅ NEW: notify main/app so UI can lock
                    if callback is not None:
                        try:
                            callback(match)
                        except Exception as e:
                            print("callback error:", e)

                else:
                    print("❌ No matching name")


if __name__ == "__main__":
    main()
