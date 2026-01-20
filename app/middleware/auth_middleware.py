import logging
from urllib.parse import quote

from flask import jsonify, redirect, request, session

logger = logging.getLogger(__name__)


class AuthMiddleware:
    """
    App-level authentication gate.

    - Enforces a login page at `/login` for browser traffic.
    - Returns 401 JSON for API calls under `/api/` when unauthenticated.
    - Session is set by `/login` or `/api/auth/login`.
    """

    def __init__(self, app=None):
        self.app = app
        if app is not None:
            self.init_app(app)

    def init_app(self, app):
        self.app = app
        app.before_request(self.before_request)

    def _auth_disabled(self) -> bool:
        return bool(self.app.config.get("WSANALIZ_AUTH_DISABLED", False))

    def _is_authenticated(self) -> bool:
        if self._auth_disabled():
            return True
        return bool(session.get("pam_user"))

    def _is_allowed_unauthenticated(self, path: str) -> bool:
        # Always allow preflight.
        if request.method == "OPTIONS":
            return True

        # Login / auth endpoints
        if path in {"/login", "/logout"}:
            return True
        if path.startswith("/api/auth/"):
            return True

        # Health checks should stay public for monitoring.
        if path == "/api/health":
            return True

        # Static assets required for login page.
        if path.startswith("/static/") or path == "/favicon.ico":
            return True

        # Allow Socket.IO handshake to reach the SocketIO layer.
        # Actual authorization is enforced in the SocketIO `connect` handler.
        if path.startswith("/socket.io/"):
            return True

        return False

    def before_request(self):
        if self._auth_disabled():
            return None

        path = request.path or "/"
        if self._is_allowed_unauthenticated(path):
            return None

        if self._is_authenticated():
            return None

        # API: return JSON 401 (do not redirect)
        if path.startswith("/api/"):
            return jsonify({"error": "authentication required"}), 401

        # Non-API browser navigation: always redirect to login.
        # Some browsers send Accept: */* which can make accept_json=True; we must not
        # show raw JSON on the main UI routes.

        # Browser: redirect to login with next
        next_path = path
        if request.query_string:
            try:
                next_path = f"{path}?{request.query_string.decode('utf-8', errors='ignore')}"
            except Exception:
                next_path = path

        safe_next = next_path if next_path.startswith("/") else "/"
        return redirect(f"/login?next={quote(safe_next)}")

