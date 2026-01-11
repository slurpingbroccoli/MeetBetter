# MeetBetter: Accessibility-First Networking Assistant
Built by: [Emily Yan](https://github.com/emilyzyy), [Edison Cai](https://github.com/EdisonCai2007), [Vincent Bei](), and [Lily You](https://github.com/slurpingbroccoli)

## The Problem
Networking and the ability to build connections is one of the **most decisive factors in the job search**. However, it can be hard to manage thorough preparation for networking events atop a heavy courseload, personal projects, friends, family, and more. MeetBetter reduces this burden, making events more approachable and accessible for everyone.

---

## Our Solution
MeetBetter uses facial and voice recognition to find and scan a stranger's LinkedIn autonomously, and delivers the perfect amount of context kick-start conversations right away. Instead of forcing users to pull out their phone, search LinkedIn manually, and read long profiles mid-conversation, streamlines your networking experience in real-time.

Using a laptop’s built-in webcam (as a stand-in for wearable assistants like Meta Glasses), MeetBetter detects a face and displays a small, anchored prompt card beside the person, providing a personal overview, quick talking points and conversation starters based on their portfolio. This keeps networking simple, fast, and accessible.

---

## Why this matters
It’s obvious the job market is “cooked.” Networking has become the hidden filter for jobs, creating a barrier that impacts stability, independence, and opportunities. Making networking more accessible matters.
MeetBetter focuses on **low-effort accessibility** by:
- reducing cognitive load (short prompts instead of long profiles)
- keeping attention on the person (no phone-scrolling)
- using a fast, readable UI (minimal text, structured layout)

---

## How it works
1. **Open MeetBetter** and your laptop camera turns on.
2. When you look at someone, MeetBetter **detects their face** and places a small **prompt card** beside them on-screen.
3. The card shows **minimal, useful context** (name + a few quick talking points) so you can start a conversation without pulling out your phone.
4. If you **say a name out loud**, MeetBetter can **lock onto that person** briefly to keep the card stable and reduce “jumping” between people.
5. After the interaction, you can use the same quick info to **remember what to follow up on** (prototype/demo content may be placeholder or locally stored).

> Note: This is a prototype UI/UX demo. Any “profile” content shown can be placeholder or stored locally for the hackathon.

---

## Tech Stack
**Backend (Python)**
- **FastAPI** — lightweight API layer to connect modules (vision ↔ voice ↔ UI signals)
- **MediaPipe** — real-time face detection + anchor point tracking
- **OpenCV** — webcam capture + drawing the on-screen anchored card UI
- **Vosk (offline speech recognition)** — converts live microphone audio into text commands/names
- **NumPy** — embeddings + similarity scoring for local face matching (prototype DB)
- **Gemini (Google LLM API)** — contextual reasoning and dynamic conversation prompts generated from profile data
- **InsightFace** — high-accuracy facial recognition and embedding extraction for identity matching

---
