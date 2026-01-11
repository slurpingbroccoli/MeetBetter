# main.py
import threading
import app
import vosk_listener


def _start_mic_thread():
    def on_name(name: str):
        app.set_voice_lock(name, seconds=1000.0)


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
