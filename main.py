# main.py
import threading
import time
import requests

import app
import vosk_listener

# Prevent spamming:
FETCH_COOLDOWN_SEC = 1000.0         # minimum time between fetch attempts
CACHE_TTL_SEC = 180.0            # reuse results for same name within 3 minutes
LOCK_SECONDS = 1000.0             # how long voice lock lasts


_last_fetch_at = 0.0
_cache = {}  # name -> (ts, data_dict)


def _safe_preview_lines(data: dict) -> list[str]:
    """
    Convert server JSON into short UI lines.
    Adjust this to match your server response shape.
    """
    if not isinstance(data, dict):
        return ["(bad response)"]

    # common fields (best-effort)
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

    # If server returns conversation starters, show first 2 lines
    starters = data.get("starters")
    if isinstance(starters, list) and starters:
        lines.append("Starters:")
        for s in starters[:2]:
            lines.append(f"- {str(s)[:120]}")

    if not lines:
        lines = ["(no fields)"]
    return lines


def _start_mic_thread():
    def on_name(name: str):
        app.clear_all_cards()
        app.set_voice_lock(name, seconds=LOCK_SECONDS)


    def mic_worker():
        return vosk_listener.main(callback=on_name)

    t = threading.Thread(target=mic_worker, daemon=True)
    t.start()
    return t


def main():
    global _last_fetch_at

    print("🚀 main.py starting...")
    print("   - Starting mic in background")
    print("   - Running camera on MAIN thread (required on macOS)")
    print("   - Press S to fetch once locked\n")

    _start_mic_thread()

    def fetch_console_listener():
        print('hi')
        global _last_fetch_at
        while True:
            try:
                cmd = input().strip().lower()
            except EOFError:
                time.sleep(0.1)
                continue

            if cmd in ("s", "fetch"):
                locked_name, until_ts = app.get_voice_lock()
                now = time.time()

                if not locked_name or now >= until_ts:
                    app.set_fetch_overlay("ERROR", "Not locked", ["No active voice lock.", "Say: 'my name is ___'"], seconds=4.0)
                    continue

                if now - _last_fetch_at < FETCH_COOLDOWN_SEC:
                    app.set_fetch_overlay(
                        "ERROR",
                        "Cooldown",
                        [f"Wait {FETCH_COOLDOWN_SEC - (now - _last_fetch_at):.1f}s then press S again."],
                        seconds=3.0,
                    )
                    continue

                # cache
                cached = _cache.get(locked_name.lower())
                if cached:
                    ts, data = cached
                    if now - ts < CACHE_TTL_SEC:
                        lines = _safe_preview_lines(data)
                        app.set_fetch_overlay("DONE", f"Cached: {locked_name}", lines, seconds=10.0)

                        # NEW: also push anchored profile card
                        display_name = str(data.get("name", locked_name)).strip()
                        headline = str(data.get("headline", "")).strip()
                        app.set_profile_card(display_name, headline, lines, seconds=30.0)

                        _last_fetch_at = now
                        continue
                

                # do real fetch
                _last_fetch_at = now
                app.set_fetch_overlay("FETCHING", f"Fetching: {locked_name}", ["Please wait…"], seconds=20.0)

                try:
                    data = _fetch_profile_for(locked_name)
                    _cache[locked_name.lower()] = (time.time(), data)
                    lines = _safe_preview_lines(data)
                    app.set_fetch_overlay("DONE", f"Fetched: {locked_name}", lines, seconds=15.0)

                    # NEW: also push anchored profile card
                    other_data = app.other_linkedin.json()
                    print(other_data['name'])
                    display_name = str(data.get("name", other_data['name'])).strip()
                    headline = str(data.get("headline", other_data['headline'])).strip()
                    app.set_profile_card(display_name, headline, lines, seconds=30.0)

                except Exception as e:
                    app.set_fetch_overlay("ERROR", f"Fetch failed", [str(e)[:180]], seconds=10.0)

            elif cmd in ("c", "clear"):
                locked_name, until_ts = app.get_voice_lock()
                if locked_name:
                    _cache.pop(locked_name.lower(), None)
                    app.set_fetch_overlay("DONE", "Cache cleared", [f"{locked_name}"], seconds=3.0)
                else:
                    _cache.clear()
                    app.set_fetch_overlay("DONE", "Cache cleared", ["All"], seconds=3.0)

            elif cmd in ("help", "h", "?"):
                print("Commands: s (fetch), c (clear cache), help")

    threading.Thread(target=fetch_console_listener, daemon=True).start()

    # Camera must stay on main thread on macOS
    app.main()


if __name__ == "__main__":
    main()
