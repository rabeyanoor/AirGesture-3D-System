"""
Open the web app on an Android phone connected by USB - no Wi-Fi, no certificate warning.

Needs: USB debugging enabled on the phone (Settings -> Developer options -> USB debugging), the phone
plugged in, and web/serve_https.py running (it serves http://localhost:8000 on this computer).

`adb reverse` makes the phone's localhost:8000 point at this computer, and Chrome treats
http://localhost as secure, so the camera works straight away.

Usage:
    python web/open_on_phone.py [--camera back|front]
"""

import argparse
import os
import shutil
import subprocess
import sys

ADB_CANDIDATES = [shutil.which("adb"), os.path.expanduser("~/.local/opt/platform-tools/adb")]


def adb_path():
    for p in ADB_CANDIDATES:
        if p and os.path.exists(p):
            return p
    sys.exit("adb not found. Install Android platform-tools (see README).")


def main():
    parser = argparse.ArgumentParser(description="Open AirGesture 3D on a USB-connected Android phone")
    parser.add_argument("--camera", choices=["back", "front"], default="back")
    args = parser.parse_args()
    adb = adb_path()

    devices = [line.split() for line in subprocess.run([adb, "devices"], capture_output=True, text=True)
               .stdout.splitlines()[1:] if line.strip()]
    if not devices:
        sys.exit("No phone found. Plug it in with USB and turn on USB debugging (Developer options).")
    if devices[0][1] == "unauthorized":
        sys.exit("On the phone, tap 'Allow' on the 'Allow USB debugging?' prompt, then run this again.")

    subprocess.run([adb, "reverse", "tcp:8000", "tcp:8000"], check=True)
    url = f"http://localhost:8000/?autostart&camera={args.camera}"
    # Quote the URL for the phone's shell ('&' would otherwise split the command) and force Chrome
    r = subprocess.run([adb, "shell", f"am start -a android.intent.action.VIEW -p com.android.chrome -d '{url}'"],
                       capture_output=True, text=True)
    if "Error" in r.stdout + r.stderr:
        # Chrome missing: fall back to the default browser
        subprocess.run([adb, "shell", f"am start -a android.intent.action.VIEW -d '{url}'"], check=False)
    print(f"Opened {url} in Chrome on the phone.")


if __name__ == "__main__":
    main()
