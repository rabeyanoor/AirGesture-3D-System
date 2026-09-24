"""
Serve the web version so a phone can use its camera.

Phone browsers only allow the camera on secure pages: https://, or http://localhost. This script runs:
  - https://<computer-ip>:8443   (self-signed certificate; tap "Advanced -> Proceed" once on the phone)
  - http://<computer-ip>:8080    (just redirects to the https address, in case http:// was typed)
  - http://localhost:8000        (plain http for this computer, or for a phone connected with
                                  `adb reverse tcp:8000 tcp:8000` - no certificate warning at all)

Usage:
    python web/serve_https.py [--port 8443]
"""

import argparse
import functools
import http.server
import os
import socket
import ssl
import subprocess
import sys
import threading

WEB_DIR = os.path.dirname(os.path.abspath(__file__))
# Certificate lives OUTSIDE the served folder so it can never be downloaded
CERT_DIR = os.path.join(os.path.dirname(WEB_DIR), ".cert")
CERT = os.path.join(CERT_DIR, "cert.pem")
KEY = os.path.join(CERT_DIR, "key.pem")


def lan_ips():
    ips = set()
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("10.255.255.255", 1))
        ips.add(s.getsockname()[0])
        s.close()
    except OSError:
        pass
    try:
        out = subprocess.run(["hostname", "-I"], capture_output=True, text=True).stdout.split()
        ips.update(ip for ip in out if ":" not in ip)
    except OSError:
        pass
    return sorted(ip for ip in ips if not ip.startswith("127."))


def ensure_cert(ips):
    if os.path.exists(CERT) and os.path.exists(KEY):
        return
    os.makedirs(CERT_DIR, exist_ok=True)
    san = ",".join(["DNS:localhost", "IP:127.0.0.1"] + [f"IP:{ip}" for ip in ips])
    subprocess.run([
        "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "825",
        "-keyout", KEY, "-out", CERT, "-subj", "/CN=AirGesture 3D local",
        "-addext", f"subjectAltName={san}",
    ], check=True, capture_output=True)
    os.chmod(KEY, 0o600)
    print(f"Created self-signed certificate in {CERT_DIR}")


class Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, ".js": "text/javascript",
                      ".mjs": "text/javascript"}

    def send_head(self):
        # Never serve hidden files / folders or this script
        parts = [p for p in self.path.split("?", 1)[0].split("/") if p]
        if any(p.startswith(".") for p in parts) or (parts and parts[-1].endswith(".py")):
            self.send_error(404)
            return None
        return super().send_head()

    def list_directory(self, path):
        self.send_error(404)
        return None

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, fmt, *args):
        print(f"[{self.client_address[0]}] {fmt % args}", flush=True)


class Redirect(http.server.BaseHTTPRequestHandler):
    https_port = 8443

    def do_GET(self):
        host = self.headers.get("Host", "localhost").split(":")[0]
        self.send_response(301)
        self.send_header("Location", f"https://{host}:{self.https_port}{self.path}")
        self.end_headers()

    do_HEAD = do_GET

    def log_message(self, fmt, *args):
        print(f"[{self.client_address[0]}] redirect {fmt % args}", flush=True)


def serve(server):
    threading.Thread(target=server.serve_forever, daemon=True).start()


def main():
    parser = argparse.ArgumentParser(description="Serve the AirGesture 3D web app for phones")
    parser.add_argument("--port", type=int, default=8443)
    args = parser.parse_args()

    ips = lan_ips()
    ensure_cert(ips)
    handler = functools.partial(Handler, directory=WEB_DIR)

    https = http.server.ThreadingHTTPServer(("0.0.0.0", args.port), handler)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(CERT, KEY)
    https.socket = ctx.wrap_socket(https.socket, server_side=True)

    Redirect.https_port = args.port
    extra = []
    for bind, port, h in (("0.0.0.0", 8080, Redirect), ("127.0.0.1", 8000, handler)):
        try:
            extra.append(http.server.ThreadingHTTPServer((bind, port), h))
        except OSError as e:
            print(f"(port {port} unavailable: {e})")
    for s in extra:
        serve(s)

    print("AirGesture 3D web app is running.")
    print("Phone (USB tethering or same Wi-Fi) - open in Chrome:")
    for ip in ips:
        print(f"   https://{ip}:{args.port}")
    print("   -> 'Your connection is not private': Advanced -> Proceed (once).")
    print("Phone over USB with adb (no warning): adb reverse tcp:8000 tcp:8000, then http://localhost:8000")
    print(f"This computer: http://localhost:8000   Ctrl+C to stop.", flush=True)
    try:
        https.serve_forever()
    except KeyboardInterrupt:
        pass
    sys.exit(0)


if __name__ == "__main__":
    main()
