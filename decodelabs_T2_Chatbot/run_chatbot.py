"""
Run this to open the ChatBot straight in your browser.

    python run_chatbot.py

It starts a tiny local server (so the page loads correctly) and
opens chatbot_frontend.html in your default browser automatically.
Leave this terminal window open while you're testing -- closing it
stops the server. Press Ctrl+C to stop.
"""

import http.server
import socketserver
import webbrowser
import os
import sys

PORT = 8000
FILE_NAME = "chatbot_frontend.html"


def main():
    folder = os.path.dirname(os.path.abspath(__file__))
    os.chdir(folder)

    if not os.path.exists(FILE_NAME):
        print(f"Could not find {FILE_NAME} in this folder: {folder}")
        print("Make sure run_chatbot.py and chatbot_frontend.html are in the same folder.")
        sys.exit(1)

    handler = http.server.SimpleHTTPRequestHandler
    port = PORT

    while True:
        try:
            httpd = socketserver.TCPServer(("", port), handler)
            break
        except OSError:
            port += 1  # port busy -- try the next one

    url = f"http://localhost:{port}/{FILE_NAME}"
    print(f"Opening {url} in your browser...")
    print("Leave this window open. Press Ctrl+C here to stop the server.")
    webbrowser.open(url)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        httpd.server_close()


if __name__ == "__main__":
    main()
