import threading
import time

import app
import vosk_listener  # rename if your file is different

def main():
    print("�� main.py starting...")
    print("   - Starting mic in background")
    print("   - Running camera on MAIN thread (required on macOS)\n")

    mic_thread = threading.Thread(target=vosk_listener.main, daemon=True)
    mic_thread.start()

    # Run camera loop on main thread (important for cv2.imshow on macOS)
    app.main()

    print("📷 Camera loop ended. Exiting...")

if __name__ == "__main__":
    main()
