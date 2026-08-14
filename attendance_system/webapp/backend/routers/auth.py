"""
Backend-enforced demo authentication (signed session cookie).

Credentials come from the environment:
    CLASSSYNC_USERNAME   (default: the demo username)
    CLASSSYNC_PASSWORD   (override the demo password in production)

The demo password is NOT stored in plaintext here — only its SHA-256 hash, so
the source never contains the secret. Setting CLASSSYNC_PASSWORD overrides it.
The password is never logged, never returned, and never sent to the frontend.
Session state lives in a signed HttpOnly cookie (SessionMiddleware in main.py);
`require_auth` gates the HTTP routes and the camera WebSocket checks the same
session.
"""

from __future__ import annotations

import hashlib
import os
import secrets

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

USERNAME = os.environ.get("CLASSSYNC_USERNAME", "ChristUniversity")
_PW_ENV = os.environ.get("CLASSSYNC_PASSWORD")            # plaintext override
_PW_HASH = "01b1d4035d35524652dd7015a7eb6f7cec4cfd5aaffbd620985fe21a97a542eb"

router = APIRouter()


class LoginBody(BaseModel):
    username: str
    password: str


def _valid(username: str, password: str) -> bool:
    user_ok = secrets.compare_digest(username, USERNAME)
    if _PW_ENV is not None:
        pass_ok = secrets.compare_digest(password, _PW_ENV)
    else:
        pass_ok = secrets.compare_digest(
            hashlib.sha256(password.encode()).hexdigest(), _PW_HASH)
    return user_ok and pass_ok


@router.post("/api/auth/login")
def login(body: LoginBody, request: Request):
    if not _valid(body.username, body.password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    request.session["auth"] = True
    return {"ok": True}


@router.post("/api/auth/logout")
def logout(request: Request):
    request.session.clear()
    return {"ok": True}


@router.get("/api/auth/me")
def me(request: Request):
    return {"authenticated": bool(request.session.get("auth"))}


def require_auth(request: Request) -> None:
    """FastAPI dependency: 401 unless the session cookie is authenticated."""
    if not request.session.get("auth"):
        raise HTTPException(status_code=401, detail="Not authenticated")
