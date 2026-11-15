"""Local HTTP server for the web viewer (``fdtd web``).

Serves the static front-end and a small JSON/binary API. Slices are sent as
raw little-endian float32 / uint8 with their shape in the response headers.
Only one session (one simulation) exists at a time.
"""
import json
import mimetypes
import threading
import traceback
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import numpy as np

from fdtd.web.session import (Session, SessionError, SlowMotionSession, SweepSession,
                              options_payload)

STATIC_DIR = Path(__file__).parent / "static"
SESSION_TYPES = {
    "slowmotion": SlowMotionSession,
    "sweep": SweepSession,
}


class ViewerState:
  def __init__(self) -> None:
    self.session: Optional[Session] = None
    self.lock = threading.Lock()
    self._options: Optional[Dict[str, Any]] = None

  def options(self) -> Dict[str, Any]:
    if self._options is None:
      self._options = options_payload()
    return self._options

  def replace_session(self, options: Dict[str, Any]) -> Session:
    mode = options.get("mode")
    if mode not in SESSION_TYPES:
      raise SessionError(f"mode must be one of: {', '.join(SESSION_TYPES)}")
    with self.lock:
      if self.session is not None:
        self.session.close()
        self.session = None
      self.session = SESSION_TYPES[mode](options)
      return self.session

  def require(self, mode: Optional[str] = None) -> Session:
    session = self.session
    if session is None:
      raise SessionError("no simulation has been set up yet")
    if mode is not None and session.mode != mode:
      raise SessionError(f"this action needs a {mode} session")
    return session


class Handler(BaseHTTPRequestHandler):
  state: ViewerState = None
  quiet = True

  # ---- plumbing ----

  def log_message(self, format, *args) -> None:  # noqa: A002 (stdlib signature)
    if not self.quiet:
      super().log_message(format, *args)

  def _send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
    body = json.dumps(payload, allow_nan=False, default=_json_default).encode("utf-8")
    self.send_response(status)
    self.send_header("Content-Type", "application/json")
    self.send_header("Content-Length", str(len(body)))
    self.send_header("Cache-Control", "no-store")
    self.end_headers()
    self.wfile.write(body)

  def _send_array(self, array: np.ndarray, horizontal: str, vertical: str,
                  extra: Optional[Dict[str, Any]] = None) -> None:
    array = np.ascontiguousarray(array)
    if array.dtype == np.float32:
      array = array.astype("<f4", copy=False)
    body = array.tobytes()
    self.send_response(HTTPStatus.OK)
    self.send_header("Content-Type", "application/octet-stream")
    self.send_header("Content-Length", str(len(body)))
    self.send_header("Cache-Control", "no-store")
    self.send_header("X-Rows", str(array.shape[0]))
    self.send_header("X-Cols", str(array.shape[1]))
    self.send_header("X-Dtype", str(array.dtype.name))
    self.send_header("X-Horizontal", horizontal)
    self.send_header("X-Vertical", vertical)
    for key, value in (extra or {}).items():
      self.send_header(f"X-{key}", str(value))
    self.end_headers()
    self.wfile.write(body)

  def _send_error(self, status: HTTPStatus, message: str) -> None:
    self._send_json({"error": message}, status)

  def _read_json(self) -> Dict[str, Any]:
    length = int(self.headers.get("Content-Length") or 0)
    if length == 0:
      return {}
    try:
      payload = json.loads(self.rfile.read(length))
    except json.JSONDecodeError:
      raise SessionError("request body is not valid JSON") from None
    if not isinstance(payload, dict):
      raise SessionError("request body must be a JSON object")
    return payload

  def _dispatch(self, routes: Dict[str, Any]) -> None:
    url = urlparse(self.path)
    query = {key: values[-1] for key, values in parse_qs(url.query).items()}
    route = routes.get(url.path)
    try:
      if route is not None:
        route(query)
      elif self.command == "GET":
        self._serve_static(url.path)
      else:
        self._send_error(HTTPStatus.NOT_FOUND, f"unknown endpoint {url.path}")
    except SessionError as error:
      self._send_error(HTTPStatus.BAD_REQUEST, str(error))
    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
      pass
    except Exception as error:
      traceback.print_exc()
      self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f"{type(error).__name__}: {error}")

  def do_GET(self) -> None:
    self._dispatch({
        "/api/options": self.get_options,
        "/api/session": self.get_session,
        "/api/slice": self.get_slice,
        "/api/sweep": self.get_sweep,
    })

  def do_POST(self) -> None:
    self._dispatch({
        "/api/session": self.post_session,
        "/api/step": self.post_step,
        "/api/reset": self.post_reset,
        "/api/sweep/cancel": self.post_sweep_cancel,
    })

  def _serve_static(self, path: str) -> None:
    relative = "index.html" if path in ("", "/") else path.lstrip("/")
    target = (STATIC_DIR / relative).resolve()
    if STATIC_DIR.resolve() not in target.parents or not target.is_file():
      self._send_error(HTTPStatus.NOT_FOUND, f"not found: {path}")
      return
    body = target.read_bytes()
    content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
    if target.suffix == ".js":
      content_type = "text/javascript"
    self.send_response(HTTPStatus.OK)
    self.send_header("Content-Type", f"{content_type}; charset=utf-8"
                     if content_type.startswith("text/") else content_type)
    self.send_header("Content-Length", str(len(body)))
    self.send_header("Cache-Control", "no-cache")
    self.end_headers()
    self.wfile.write(body)

  # ---- API ----

  def get_options(self, _query) -> None:
    self._send_json(self.state.options())

  def get_session(self, _query) -> None:
    session = self.state.session
    self._send_json(session.info() if session is not None else {"mode": None})

  def post_session(self, _query) -> None:
    session = self.state.replace_session(self._read_json())
    self._send_json(session.info())

  def get_slice(self, query) -> None:
    session = self.state.require()
    field = query.get("field", "pressure")
    axis, index = _slice_query(query)
    if field == "flags":
      data, horizontal, vertical = session.flags_slice(axis, index)
      self._send_array(data, horizontal, vertical)
    elif field == "pressure":
      session = self.state.require("slowmotion")
      data, horizontal, vertical = session.pressure_slice(axis, index)
      self._send_array(data, horizontal, vertical, session.state())
    elif field == "leq":
      session = self.state.require("sweep")
      frequency_index = _int(query, "frequency")
      data, horizontal, vertical, (low, high) = session.leq_slice(frequency_index, axis, index)
      self._send_array(data, horizontal, vertical, {"Min": low, "Max": high})
    else:
      raise SessionError("field must be pressure, leq or flags")

  def post_step(self, _query) -> None:
    session = self.state.require("slowmotion")
    body = self._read_json()
    session.step(_int(body, "count"))
    axis, index = _slice_query(body)
    data, horizontal, vertical = session.pressure_slice(axis, index)
    self._send_array(data, horizontal, vertical, session.state())

  def post_reset(self, _query) -> None:
    session = self.state.require("slowmotion")
    session.reset()
    self._send_json(session.state())

  def get_sweep(self, _query) -> None:
    self._send_json(self.state.require("sweep").sweep_status())

  def post_sweep_cancel(self, _query) -> None:
    session = self.state.require("sweep")
    session.cancel_event.set()
    self._send_json({"status": "cancelling"})


def _int(source: Dict[str, Any], key: str) -> int:
  try:
    return int(source[key])
  except (KeyError, TypeError, ValueError):
    raise SessionError(f"{key} must be an integer") from None


def _slice_query(source: Dict[str, Any]) -> Tuple[str, int]:
  return str(source.get("axis", "h")), _int(source, "index")


def _json_default(value: Any) -> Any:
  if isinstance(value, np.generic):
    return value.item()
  if isinstance(value, np.ndarray):
    return value.tolist()
  raise TypeError(f"cannot serialise {type(value).__name__}")


def serve(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True, verbose: bool = False) -> None:
  Handler.state = ViewerState()
  Handler.quiet = not verbose
  server = ThreadingHTTPServer((host, port), Handler)
  server.daemon_threads = True
  url = f"http://{'localhost' if host in ('127.0.0.1', '0.0.0.0') else host}:{server.server_port}/"
  print(f"FDTD viewer running at {url} (Ctrl+C to stop)")
  if open_browser:
    threading.Timer(0.5, lambda: webbrowser.open(url)).start()
  try:
    server.serve_forever()
  except KeyboardInterrupt:
    print("Stopping")
  finally:
    if Handler.state.session is not None:
      Handler.state.session.close()
    server.server_close()
