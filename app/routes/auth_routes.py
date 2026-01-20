import logging
from urllib.parse import quote

from flask import Blueprint, current_app, jsonify, redirect, render_template, request, session, url_for
from werkzeug.exceptions import BadRequest

from app.utils.security import SecurityError, validate_json_input, validate_request_params
from app.middleware.security_middleware import rate_limit
from app.services.auth_service import pam_authenticate

logger = logging.getLogger(__name__)

auth_bp = Blueprint("auth", __name__)


def _is_authenticated() -> bool:
    if current_app.config.get("WSANALIZ_AUTH_DISABLED", False):
        return True
    return bool(session.get("pam_user"))


@auth_bp.route("/login", methods=["GET", "POST"])
@rate_limit(requests_per_minute=30, burst_limit=10)
def login_page():
    if _is_authenticated():
        next_url = request.args.get("next") or "/"
        return redirect(next_url)

    if request.method == "GET":
        return render_template("login.html", next=request.args.get("next", "/"), error=None)

    # POST (form login)
    username = (request.form.get("username") or "").strip()
    password = request.form.get("password") or ""
    next_url = request.form.get("next") or "/"

    if not username or not password:
        return render_template("login.html", next=next_url, error="Kullanıcı adı ve şifre gerekli.")

    ok = pam_authenticate(username=username, password=password, service=current_app.config.get("WSANALIZ_PAM_SERVICE"))
    if not ok:
        return render_template("login.html", next=next_url, error="Kullanıcı adı veya şifre hatalı.")

    session["pam_user"] = username
    # "Beni Hatırla" checkbox'ı işaretliyse session'ı kalıcı yap
    remember_me = request.form.get("remember_me") == "1"
    session.permanent = remember_me
    return redirect(next_url)


@auth_bp.route("/logout", methods=["POST"])
def logout_page():
    session.clear()
    return redirect(url_for("auth.login_page"))


@auth_bp.route("/api/auth/me", methods=["GET"])
def auth_me():
    return jsonify(
        {
            "authenticated": _is_authenticated(),
            "user": session.get("pam_user"),
            "auth_disabled": bool(current_app.config.get("WSANALIZ_AUTH_DISABLED", False)),
        }
    ), 200


@auth_bp.route("/api/auth/login", methods=["POST"])
@rate_limit(requests_per_minute=30, burst_limit=10)
def api_login():
    if current_app.config.get("WSANALIZ_AUTH_DISABLED", False):
        session["pam_user"] = "disabled-auth"
        session.permanent = True
        return jsonify({"success": True, "user": session["pam_user"], "auth_disabled": True}), 200

    if not request.is_json:
        return jsonify({"error": "Content-Type application/json gereklidir"}), 400

    try:
        data = validate_json_input(request.get_json(silent=False))
    except BadRequest as e:
        return jsonify({"error": f"Geçersiz JSON: {str(e)}"}), 400
    except SecurityError as e:
        return jsonify({"error": f"JSON doğrulama hatası: {str(e)}"}), 400

    try:
        params = validate_request_params(
            data,
            {
                "username": {"type": "str", "min_length": 1, "max_length": 128, "required": True},
                "password": {"type": "str", "min_length": 1, "max_length": 512, "required": True},
            },
        )
    except SecurityError as e:
        return jsonify({"error": f"Parameter doğrulama hatası: {str(e)}"}), 400

    username = (params["username"] or "").strip()
    password = params["password"] or ""

    ok = pam_authenticate(username=username, password=password, service=current_app.config.get("WSANALIZ_PAM_SERVICE"))
    if not ok:
        return jsonify({"success": False, "error": "unauthorized"}), 401

    session["pam_user"] = username
    session.permanent = True
    return jsonify({"success": True, "user": username}), 200


@auth_bp.route("/api/auth/logout", methods=["POST"])
def api_logout():
    session.clear()
    return jsonify({"success": True}), 200


def build_login_redirect(next_path: str) -> str:
    # Helper for middleware: keep next as a query param.
    safe_next = next_path if next_path.startswith("/") else "/"
    return f"/login?next={quote(safe_next)}"

