"""
WS /ws/camera — the live classroom recognition loop.

Multi-face: SCRFD detects every face in the frame; each face is embedded
(already done inside detect()), identified against the gallery, and tracked
by its own confirmation streak. A student is marked once their streak
reaches CONFIRM_FRAMES; the per-day AttendanceLog guarantees one mark per
student per session regardless of how many frames they appear in.

Protocol: the browser sends one binary JPEG frame; the server replies with
one JSON message per frame describing EVERY detected face:

    {"state": "loading_engine"}                     # handshake
    {"state": "ready"}                               # handshake
    {"state": "frame",
     "faces": [                                      # one entry per face
        {"box": [x, y, w, h],                        # normalized 0..1
         "state": "marked" | "confirming" | "unknown",
         "name": ..., "student_id": ..., "score": ...,
         "streak": 3, "needed": 5},
        ...
     ],
     "counts": {"detected": N, "recognized": M, "marked_session": K},
     "marked": [{"student_id": ..., "name": ...}, ...]}   # session roster

Streak model (continuous sweep): +1 per frame a student is recognized, -1
(floored at 0) per frame they are missed. The decrement tolerates a student
briefly turning away or a single dropped detection — important for a seated
classroom — while still requiring sustained presence before marking.

Inference is serialized through the shared engine lock and run off the event
loop with asyncio.to_thread so the JSON API stays responsive during a session.
"""

from __future__ import annotations

import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from .. import deps

CONFIRM_FRAMES = 5

router = APIRouter()


class ClassroomSession:
    """Per-connection multi-student streaks + marking."""

    def __init__(self):
        import take_attendance
        self.logbook = take_attendance.AttendanceLog()
        self.streaks: dict[str, int] = {}     # student_id -> consecutive count

    def update(self, faces: list[dict], roster: dict[str, str]) -> dict:
        """faces: list of {box, match, best}. Returns the full frame message."""
        # best match per student this frame (guards same-frame double detects)
        seen: dict[str, object] = {}
        for f in faces:
            m = f["match"]
            if m is not None and (m.student_id not in seen
                                  or m.score > seen[m.student_id].score):
                seen[m.student_id] = m

        # decay everyone not seen this frame
        for sid in list(self.streaks):
            if sid not in seen:
                self.streaks[sid] = max(0, self.streaks[sid] - 1)

        # reinforce seen students; mark when they cross the threshold
        for sid, m in seen.items():
            self.streaks[sid] = min(CONFIRM_FRAMES, self.streaks.get(sid, 0) + 1)
            if (self.streaks[sid] >= CONFIRM_FRAMES
                    and sid not in self.logbook.present):
                self.logbook.mark(sid, m.name, m.score)

        # per-face render state
        faces_out = []
        for f in faces:
            if f.get("is_spoof"):
                faces_out.append({
                    "box": f["box"],
                    "state": "spoof",
                    "live": False,
                    "confidence": f.get("live_confidence", 0.0),
                    "spoof_probability": f.get("spoof_probability", 1.0)
                })
                continue

            m = f["match"]
            if m is None:
                faces_out.append({"box": f["box"], "state": "unknown",
                                  "score": f["best"], "live": True})
            else:
                marked = m.student_id in self.logbook.present
                faces_out.append({
                    "box": f["box"],
                    "state": "marked" if marked else "confirming",
                    "name": m.name, "student_id": m.student_id,
                    "score": round(m.score, 3),
                    "streak": self.streaks.get(m.student_id, 0),
                    "needed": CONFIRM_FRAMES,
                    "live": True,
                })

        return {
            "state": "frame",
            "faces": faces_out,
            "counts": {
                "detected": len(faces),
                "recognized": sum(1 for f in faces if f["match"] is not None),
                "marked_session": len(self.logbook.present),
            },
            "marked": [{"student_id": sid, "name": roster.get(sid, sid)}
                       for sid in sorted(self.logbook.present)],
        }


def _process(frame_bytes: bytes, session: ClassroomSession) -> dict:
    """Decode -> detect ALL faces -> identify each -> update. Runs in a thread."""
    arr = np.frombuffer(frame_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    empty = {"state": "frame", "faces": [],
             "counts": {"detected": 0, "recognized": 0,
                        "marked_session": len(session.logbook.present)},
             "marked": []}
    if bgr is None:
        return empty

    fb = deps.get_backend()
    rec = deps.get_recognizer()
    faces = fb.detect(bgr[:, :, ::-1])            # already embedded per face
    if not faces:
        return empty

    h, w = bgr.shape[:2]
    parsed = []
    for face in faces:
        x1, y1, x2, y2 = face.bbox
        box = [round(x1 / w, 4), round(y1 / h, 4),
               round((x2 - x1) / w, 4), round((y2 - y1) / h, 4)]
               
        if getattr(face, "is_spoof", False):
            parsed.append({
                "box": box,
                "match": None,
                "best": None,
                "is_spoof": True,
                "live_confidence": getattr(face, "live_confidence", 0.0),
                "spoof_probability": getattr(face, "spoof_probability", 1.0)
            })
            continue

        m = rec.identify(face.embedding)
        best = None if m else round(float(rec.scores(face.embedding).max()), 3)
        parsed.append({"box": box, "match": m, "best": best, "is_spoof": False})

    return session.update(parsed, rec.roster)


@router.websocket("/ws/camera")
async def camera(ws: WebSocket) -> None:
    import asyncio

    # reject the handshake if the session cookie is not authenticated. The
    # browser sends the same-origin session cookie automatically, so no
    # credentials ever travel in the URL or the messages.
    if not ws.session.get("auth"):
        await ws.close(code=1008)          # policy violation
        return

    await ws.accept()
    # first touch loads the models (~10 s); tell the client so it can show
    # a "warming up" state instead of a frozen UI
    if not deps.engine_loaded():
        await ws.send_json({"state": "loading_engine"})
        async with deps.inference_lock():
            await asyncio.to_thread(deps.get_backend)
            await asyncio.to_thread(deps.get_recognizer)
    await ws.send_json({"state": "ready"})

    session = ClassroomSession()
    try:
        while True:
            frame = await ws.receive_bytes()
            async with deps.inference_lock():
                result = await asyncio.to_thread(_process, frame, session)
            await ws.send_json(result)
    except WebSocketDisconnect:
        pass
