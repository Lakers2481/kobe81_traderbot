from __future__ import annotations

from http.server import BaseHTTPRequestHandler, HTTPServer
import threading
import json


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/readiness":
            self._json({"ready": True})
        elif self.path == "/liveness":
            self._json({"alive": True})
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        return  # quiet

    def _json(self, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def start_health_server(port: int = 8000) -> HTTPServer:
    server = HTTPServer(("0.0.0.0", port), _Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server

