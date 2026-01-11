# main.py
import threading

import app
import vosk_listener


# Camera-vs-voice fusion knobs
CAM_STRONG_SIM = 0.88       # if camera top1 >= this, it's "very confident"
CAM_GAP_STRONG = 0.03       # if (top1 - top2) >= this, it's "clearly separated"
VOICE_LOCK_SEC = 1000.0       # how long the UI locks to voice
TRY_AGAIN_SEC = 2.5


def _voice_vs_camera_decision(voice_name: str) -> None:
    """
    If voice name is consistent with camera candidates, lock it.
    If voice strongly contradicts a confident camera, ask to try again.
    Otherwise (camera uncertain), still lock voice (voice-first policy).
    """
    v = (voice_name or "").strip()
    if not v:
        return

    cam_topk = app.get_camera_topk(max_age_sec=1.0)  # [(name, sim), ...]

    # If camera has nothing recent, trust voice
    if not cam_topk:
        app.set_voice_lock(v, seconds=VOICE_LOCK_SEC)
        return

    cam_names = [n for (n, _) in cam_topk]
    top1_name, top1_sim = cam_topk[0]
    top2_sim = cam_topk[1][1] if len(cam_topk) > 1 else 0.0
    gap = float(top1_sim - top2_sim)

    voice_in_topk = v in cam_names

    camera_is_strong = (top1_sim >= CAM_STRONG_SIM) and (gap >= CAM_GAP_STRONG)

    # If camera is strongly saying "it's X" and voice says something else -> try again
    if camera_is_strong and (not voice_in_topk):
        app.set_try_again(
            msg=f"Heard '{v}', but camera is confident it's '{top1_name}'. Please try again.",
            seconds=TRY_AGAIN_SEC,
        )
        return

    # Otherwise: voice-first
    app.set_voice_lock(v, seconds=VOICE_LOCK_SEC)


def _start_mic_thread():
    def on_name(name: str):
        _voice_vs_camera_decision(name)

    def mic_worker():
        return vosk_listener.main(callback=on_name)

    t = threading.Thread(target=mic_worker, daemon=True)
    t.start()
    return t


def main():
    print("🚀 main.py starting...")
    print("   - Starting mic in background")
    print("   - Running camera on MAIN thread (required on macOS)\n")

    _start_mic_thread()

    # Camera must stay on main thread on macOS
    app.main()


if __name__ == "__main__":
    main()
