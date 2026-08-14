"""
Shared backend dependencies: engine access and paths.

The ONLY module that imports the recognition engine. Everything engine-
related funnels through here so the rest of the backend stays decoupled.

Engine loading is LAZY (first camera use), not at server startup: the
dashboard/reports pages are pure CSV reads and shouldn't pay the ~10 s
model-load cost, and dev-server restarts stay instant. An asyncio.Lock
serializes inference — the underlying ONNX sessions are not thread-safe,
and the CPU can only run one inference at a time anyway.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

# make the existing engine importable without packaging changes
SRC_DIR = Path(__file__).resolve().parents[2] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import config  # noqa: E402  (engine config — paths, tau, sessions)

_engine_lock = asyncio.Lock()          # serializes all inference calls
_backend = None
_recognizer = None


def engine_loaded() -> bool:
    return _backend is not None


def get_backend():
    """FaceBackend singleton (loads models on first call, ~10 s)."""
    global _backend
    if _backend is None:
        from detect_embed import FaceBackend
        _backend = FaceBackend()
    return _backend


def get_recognizer(reload: bool = False):
    """Recognizer singleton; reload=True after gallery rebuilds."""
    global _recognizer
    if _recognizer is None or reload:
        from gallery import Recognizer
        _recognizer = Recognizer(threshold=config.RECOG_THRESHOLD)
    return _recognizer


def inference_lock() -> asyncio.Lock:
    return _engine_lock
