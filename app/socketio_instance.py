"""
SocketIO instance - Circular import'u önlemek için ayrı dosya.

Önemli not:
- Bazı modüller `from app import socketio` gibi "by-value" import yapıyor.
  Bu durumda `socketio = None` ile başlamak, daha sonra gerçek instance set edilse bile
  import eden modülde None kalmasına yol açabiliyor.
- Bunu engellemek için burada sabit bir Proxy nesnesi export ediyoruz; gerçek SocketIO
  instance runtime'da proxy içine set edilir.
"""

import os
from typing import Any


class _NullSocketIO:
    """
    A safe fallback used before the real SocketIO instance is set.

    Rationale:
    - Some modules may import `socketio` and (accidentally) touch attributes before `create_app()`
      finishes calling `set_socketio(...)`.
    - Crashing at import-time is painful; instead we provide no-op behavior with a warning.

    If you prefer fail-fast behavior, set: SOCKETIO_STRICT_PROXY=1
    """

    def _warn(self, name: str) -> None:
        print(f"⚠️ SocketIO not initialized yet; ignoring call: socketio.{name}(...)")

    def emit(self, *args: Any, **kwargs: Any) -> None:
        self._warn("emit")

    def on(self, *args: Any, **kwargs: Any):
        # decorator form: @socketio.on('event')
        def decorator(fn):
            self._warn("on")
            return fn

        return decorator

    def run(self, *args: Any, **kwargs: Any) -> None:
        self._warn("run")

    def start_background_task(self, *args: Any, **kwargs: Any) -> None:
        self._warn("start_background_task")

    def sleep(self, *args: Any, **kwargs: Any) -> None:
        self._warn("sleep")

    def __getattr__(self, name: str) -> Any:
        # best-effort no-op for unknown attributes
        def _noop(*args: Any, **kwargs: Any) -> None:
            self._warn(name)

        return _noop


class _SocketIOProxy:
    """A stable proxy object that forwards attributes to the real SocketIO instance."""

    def __init__(self) -> None:
        self._instance: Any | None = None

    def set(self, instance: Any) -> None:
        if self._instance is not None:
            print(
                f"🚨 WARNING: socketio instance değiştiriliyor! Eski: {id(self._instance)}, Yeni: {id(instance)}"
            )
        self._instance = instance

    def get(self) -> Any | None:
        if self._instance is None:
            print("🚨 WARNING: socketio instance henüz set edilmemiş!")
        return self._instance

    def reset(self) -> None:
        self._instance = None

    def __getattr__(self, name: str) -> Any:
        inst = self._instance
        if inst is None:
            # Default: do NOT crash during import-time access; provide a safe no-op fallback.
            # Opt-in strict mode for debugging/prod hardening.
            if (os.environ.get("SOCKETIO_STRICT_PROXY") or "").strip() == "1":
                raise RuntimeError("SocketIO instance is not initialized yet")
            inst = _NullSocketIO()
        return getattr(inst, name)


# Global SocketIO proxy - ZORUNLU TEK NOKTA!
socketio = _SocketIOProxy()


def get_socketio() -> Any | None:
    """CRITICAL: Tek global SocketIO instance döndürür (proxy içinden)."""
    return socketio.get()


def set_socketio(socketio_instance: Any) -> None:
    """CRITICAL: Global SocketIO instance'ını set eder - SADECE BURADA!"""
    socketio.set(socketio_instance)


def reset_socketio() -> None:
    """Test amaçlı socketio'yu reset eder."""
    socketio.reset()