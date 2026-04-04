#!/usr/bin/env python3
"""Main entry point for the Real Estate Agent bot - Roi with Web Server."""

import threading
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from telegram_bot import main


class HealthHandler(BaseHTTPRequestHandler):
    """Health check handler for HuggingFace Spaces."""
    
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Roi Bot is running!")
    
    def log_message(self, format, *args):
        """Suppress HTTP request logging."""
        pass


def run_health_server():
    """Run a simple HTTP health check server."""
    port = int(os.environ.get("PORT", 7860))
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    server.serve_forever()


if __name__ == "__main__":
    # Start health server in a daemon thread
    health_thread = threading.Thread(target=run_health_server, daemon=True)
    health_thread.start()
    print(f"🌐 Health server running on port 7860")
    
    # Start the bot
    main()