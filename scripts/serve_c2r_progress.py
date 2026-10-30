#!/usr/bin/env python3
"""Local live C2R progress page. Poll remote JSON files read-only every 10 seconds."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]


class Monitor:
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.payload = dict(
            data=None, connected=False, last_error=None, last_attempt=None, refresh_seconds=args.interval
        )
        # A dedicated master belongs only to this monitor, not the training SSH sessions.
        self.socket_dir = tempfile.TemporaryDirectory(prefix="c2r-monitor-")
        self.ssh = [
            "ssh",
            "-T",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=8",
            "-o",
            "ServerAliveInterval=10",
            "-o",
            "ServerAliveCountMax=2",
            "-o",
            "ControlMaster=auto",
            "-o",
            "ControlPersist=60",
            "-o",
            f"ControlPath={self.socket_dir.name}/ssh",
            "-p",
            str(args.ssh_port),
        ]
        self.collector = (ROOT / "benchmarks/robotwin/progress_snapshot.py").read_text()

    def snapshot(self):
        with self.lock:
            return dict(self.payload, server_time=time.time())

    def refresh(self):
        try:
            command = shlex.join(["python3", "-", self.args.remote_plan])
            result = subprocess.run(
                [*self.ssh, self.args.host, command],
                input=self.collector,
                capture_output=True,
                text=True,
                timeout=18,
                check=True,
            )
            data = json.loads(result.stdout)
            if data.get("phase") == "pipelined_evaluation":
                # Existing open pages already understand "evaluating". Keep
                # them live without requiring a manual HTML reload.
                data["scheduler_phase"] = data["phase"]
                data["phase"] = "evaluating"
            with self.lock:
                self.payload.update(data=data, connected=True, last_error=None, last_attempt=time.time())
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            detail = getattr(exc, "stderr", None) or str(exc)
            logging.warning("Snapshot failed: %s", str(detail)[-400:])
            with self.lock:
                # Preserve the last good counts when the SSH connection is unavailable.
                self.payload.update(connected=False, last_error=str(detail)[-400:], last_attempt=time.time())

    def run(self):
        while not self.stop.is_set():
            started = time.monotonic()
            self.refresh()
            self.stop.wait(max(0.1, self.args.interval - (time.monotonic() - started)))

    def close(self):
        self.stop.set()
        subprocess.run(
            [*self.ssh, "-O", "exit", self.args.host],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
            check=False,
        )
        self.socket_dir.cleanup()


def handler_for(monitor):
    html = (ROOT / "benchmarks/robotwin/progress.html").read_bytes()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            route = urlsplit(self.path).path
            if route == "/":
                content, mime = html, "text/html; charset=utf-8"
            elif route == "/api/status":
                content = json.dumps(monitor.snapshot(), ensure_ascii=False, allow_nan=False).encode()
                mime = "application/json; charset=utf-8"
            elif route == "/api/health":
                content = json.dumps(
                    dict(service="openwam-c2r-progress", pid=os.getpid(), refresh_seconds=monitor.args.interval)
                ).encode()
                mime = "application/json"
            elif route == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
                return
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            try:
                self.wfile.write(content)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def log_message(self, *_):
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="root@vr.turbo-ai.com")
    parser.add_argument("--ssh-port", type=int, default=20347)
    parser.add_argument("--port", type=int, default=8767)
    parser.add_argument("--interval", type=float, default=10)
    parser.add_argument(
        "--remote-plan", default="/mnt/data/liangjiaqi/rax_train/OpenWAM/benchmarks/robotwin/c2r_final_50x10.plan.json"
    )
    args = parser.parse_args()
    if args.interval < 2:
        parser.error("--interval must be at least 2 seconds")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    monitor = Monitor(args)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler_for(monitor))
    thread = threading.Thread(target=monitor.run, daemon=True)
    thread.start()
    logging.info("C2R progress: http://127.0.0.1:%s (refresh %ss)", args.port, args.interval)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        monitor.stop.set()
        thread.join(timeout=20)
        monitor.close()
        server.server_close()


if __name__ == "__main__":
    main()
