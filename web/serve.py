#!/usr/bin/env python3
"""Local dev server with COOP/COEP headers for SharedArrayBuffer support."""
import http.server
import sys

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8080


class COOPCOEPHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cross-Origin-Opener-Policy", "same-origin")
        self.send_header("Cross-Origin-Embedder-Policy", "require-corp")
        super().end_headers()


print(f"Serving on http://localhost:{PORT} with COOP/COEP headers")
http.server.HTTPServer(("", PORT), COOPCOEPHandler).serve_forever()
