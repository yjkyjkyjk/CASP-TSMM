#!/usr/bin/env python3
"""
TSMM Benchmark Web Server
Serves the dashboard and the benchmark results JSON.

Usage:
    python3 web/server.py [--port 8080] [--results web/results.json]
"""

import argparse
import http.server
import json
import os
import sys
import time
import socketserver
import threading

WEB_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_FILE = os.path.join(WEB_DIR, "results.json")


class BenchHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, results_path=None, **kwargs):
        self._results_path = results_path or RESULTS_FILE
        super().__init__(*args, directory=WEB_DIR, **kwargs)

    def do_GET(self):
        if self.path in ("/api/results", "/api/results/"):
            self._serve_results()
        elif self.path in ("/api/status", "/api/status/"):
            self._serve_status()
        else:
            super().do_GET()

    def _serve_results(self):
        if os.path.exists(self._results_path):
            try:
                with open(self._results_path, "r", encoding="utf-8") as f:
                    data = f.read()
                # Validate JSON
                json.loads(data)
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                self.wfile.write(data.encode())
            except (json.JSONDecodeError, OSError) as e:
                self._send_json({"error": str(e)}, 500)
        else:
            self._send_json({"status": "waiting", "message": "Results not yet available. Run the benchmark first."}, 200)

    def _serve_status(self):
        exists = os.path.exists(self._results_path)
        mtime = os.path.getmtime(self._results_path) if exists else 0
        self._send_json({
            "ready": exists,
            "mtime": mtime,
            "mtime_str": time.ctime(mtime) if exists else None,
        })

    def _send_json(self, obj, code=200):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        # Suppress routine GET log noise for polling requests
        if "/api/" not in args[0]:
            super().log_message(fmt, *args)


def make_handler(results_path):
    def handler(*args, **kwargs):
        BenchHandler(*args, results_path=results_path, **kwargs)
    return handler


def main():
    parser = argparse.ArgumentParser(description="TSMM benchmark dashboard server")
    parser.add_argument("--port", type=int, default=8080, help="TCP port (default 8080)")
    parser.add_argument("--results", default=RESULTS_FILE, help="Path to results.json")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address (default 0.0.0.0)")
    args = parser.parse_args()

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=WEB_DIR, **kw)

        def do_GET(self):
            if self.path in ("/api/results", "/api/results/"):
                if os.path.exists(args.results):
                    try:
                        with open(args.results, "r") as f:
                            data = f.read()
                        json.loads(data)  # validate
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Access-Control-Allow-Origin", "*")
                        self.send_header("Cache-Control", "no-cache")
                        self.end_headers()
                        self.wfile.write(data.encode())
                    except Exception as e:
                        body = json.dumps({"error": str(e)}).encode()
                        self.send_response(500)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(body)
                else:
                    body = json.dumps({"status": "waiting"}).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Access-Control-Allow-Origin", "*")
                    self.end_headers()
                    self.wfile.write(body)
            else:
                super().do_GET()

        def log_message(self, fmt, *a):
            if "/api/" not in a[0]:
                super().log_message(fmt, *a)

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer((args.host, args.port), Handler) as httpd:
        url = f"http://localhost:{args.port}"
        print(f"TSMM Dashboard  →  {url}")
        print(f"Results file    →  {args.results}")
        print("Press Ctrl+C to stop.\n")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")


if __name__ == "__main__":
    main()
