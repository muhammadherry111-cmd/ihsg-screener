#!/usr/bin/env python3
"""
Listener webhook GitHub → auto git pull + restart service.
Dipanggil oleh GitHub setiap ada push ke branch main.
Sengaja berdiri sendiri (bukan bagian dari web/app.py) supaya proses
restart gunicorn tidak "membunuh dirinya sendiri" saat sedang
menangani request, dan supaya hak restart systemd tidak perlu
diberikan ke user www-data yang menjalankan app utama.

Jalankan sebagai service systemd terpisah (root), hanya bind ke
127.0.0.1 — tidak boleh diekspos langsung ke internet, harus lewat
reverse proxy Nginx.
"""
from __future__ import annotations
import hashlib
import hmac
import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

REPO_DIR = "/var/www/ihsg-screener"
SERVICE_NAME = "ihsg-screener"
BRANCH = "main"
LISTEN_PORT = 9001
SECRET = os.environ.get("GITHUB_WEBHOOK_SECRET", "")


def _verify_signature(payload: bytes, signature_header: str) -> bool:
    if not SECRET:
        return False
    if not signature_header or not signature_header.startswith("sha256="):
        return False
    expected = hmac.new(SECRET.encode(), payload, hashlib.sha256).hexdigest()
    got = signature_header.split("=", 1)[1]
    return hmac.compare_digest(expected, got)


def _deploy():
    log = []
    try:
        pull = subprocess.run(
            ["git", "pull", "origin", BRANCH],
            cwd=REPO_DIR, capture_output=True, text=True, timeout=60,
        )
        log.append(f"$ git pull\n{pull.stdout}{pull.stderr}")

        restart = subprocess.run(
            ["systemctl", "restart", SERVICE_NAME],
            capture_output=True, text=True, timeout=30,
        )
        log.append(f"$ systemctl restart {SERVICE_NAME}\n{restart.stdout}{restart.stderr}")
    except Exception as e:
        log.append(f"ERROR: {e}")
    print("\n".join(log), file=sys.stderr, flush=True)


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        sig = self.headers.get("X-Hub-Signature-256", "")

        if not _verify_signature(body, sig):
            self.send_response(401)
            self.end_headers()
            self.wfile.write(b"invalid signature")
            return

        try:
            payload = json.loads(body)
        except Exception:
            payload = {}

        ref = payload.get("ref", "")
        if ref != f"refs/heads/{BRANCH}":
            self.send_response(200)
            self.end_headers()
            self.wfile.write(f"ignored ref {ref}".encode())
            return

        # jalankan deploy di thread terpisah supaya response bisa
        # langsung dibalas sebelum gunicorn utama di-restart
        threading.Thread(target=_deploy, daemon=True).start()

        self.send_response(202)
        self.end_headers()
        self.wfile.write(b"deploy triggered")

    def log_message(self, fmt, *args):
        print("[webhook] " + (fmt % args), file=sys.stderr, flush=True)


if __name__ == "__main__":
    if not SECRET:
        print("WARNING: GITHUB_WEBHOOK_SECRET belum di-set, semua request akan ditolak.",
              file=sys.stderr)
    server = HTTPServer(("127.0.0.1", LISTEN_PORT), Handler)
    print(f"Webhook listener jalan di 127.0.0.1:{LISTEN_PORT}", file=sys.stderr)
    server.serve_forever()
