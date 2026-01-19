"""
Run a single analysis in a fresh Python process.

Why:
- Gunicorn + eventlet workers occasionally crash with native SIGSEGV (code 139)
  during heavy model inference.
- Running the actual analysis inside a clean subprocess isolates native crashes
  so the web worker stays alive, and the system can reliably mark analysis as
  completed/failed.

Usage:
  python -m app.services.analysis_subprocess_runner <analysis_id>
Outputs a single JSON line to stdout.
"""

import json
import sys
import traceback


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        sys.stderr.write("usage: python -m app.services.analysis_subprocess_runner <analysis_id>\n")
        return 2

    analysis_id = argv[1]

    try:
        from app import create_app
        from app.services.analysis_service import analyze_file

        app, _socketio = create_app(return_socketio=True)
        with app.app_context():
            success, message = analyze_file(analysis_id)

        sys.stdout.write(json.dumps({"success": bool(success), "message": str(message)}) + "\n")
        return 0 if success else 1

    except BaseException as e:
        # If we reach here, it's a Python-level failure (native segfaults won't be caught).
        sys.stdout.write(
            json.dumps(
                {
                    "success": False,
                    "message": f"Subprocess exception: {e}",
                    "traceback": traceback.format_exc(),
                }
            )
            + "\n"
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))

