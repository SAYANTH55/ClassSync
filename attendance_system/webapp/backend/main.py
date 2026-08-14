"""
ClassSync backend — FastAPI application entry point.

Run (from attendance_system/):
    E:/amlenvs/face311/python.exe -m uvicorn webapp.backend.main:app --port 8000

Serves:
  * /api/*      JSON API (routers/)
  * /ws/*       WebSocket endpoints (camera — added in step 3)
  * /           the built React frontend (webapp/frontend/dist), once built;
                until then, a plain status page so the server is testable
"""

from __future__ import annotations

import os
import secrets
from pathlib import Path

from fastapi import Depends, FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from .routers import attendance, auth, dashboard, settings, students
from .routers.auth import require_auth
from .ws import camera

app = FastAPI(title="ClassSync", version="1.0",
              docs_url="/api/docs", openapi_url="/api/openapi.json")

# signed HttpOnly session cookie. CLASSSYNC_SECRET keeps sessions stable across
# restarts/workers in production; a random per-process fallback is fine for a
# single-process demo (it just forces re-login after a restart).
app.add_middleware(
    SessionMiddleware,
    secret_key=os.environ.get("CLASSSYNC_SECRET", secrets.token_hex(32)),
    session_cookie="classsync_session",
    same_site="lax",
    https_only=os.environ.get("CLASSSYNC_SECURE_COOKIE", "0") == "1",
)

# public: authentication endpoints
app.include_router(auth.router)

# protected: every sensitive API requires a valid session (401 otherwise)
_guard = [Depends(require_auth)]
app.include_router(dashboard.router, dependencies=_guard)
app.include_router(settings.router, dependencies=_guard)
app.include_router(attendance.router, dependencies=_guard)
app.include_router(students.router, dependencies=_guard)

# the camera WebSocket checks the same session inside its handler
app.include_router(camera.router)

FRONTEND_DIST = Path(__file__).resolve().parents[1] / "frontend" / "dist"

if FRONTEND_DIST.exists():
    # hashed build assets served normally; every other non-API path gets
    # index.html so client-side routes (/camera, /reports, ...) survive
    # direct loads and refreshes (standard SPA fallback)
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"),
              name="assets")
    _index = (FRONTEND_DIST / "index.html").read_text(encoding="utf-8")

    @app.get("/{path:path}", response_class=HTMLResponse)
    def spa(path: str) -> str:
        return _index
else:
    @app.get("/", response_class=HTMLResponse)
    def placeholder() -> str:
        return ("<h1>ClassSync API is running</h1>"
                "<p>Frontend not built yet — see /api/docs for the API.</p>")
