"""Roster endpoints: list, enroll (multipart upload), unenroll."""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from .. import deps
from ..schemas import StudentRow, StudentsResponse, EnrollResult

import config  # noqa: E402
from gallery import load_roster  # noqa: E402

router = APIRouter()


@router.get("/api/students/{student_id}/photo")
def student_photo(student_id: str):
    """Serve the cached face thumbnail for a student (404 -> UI uses initials)."""
    p = config.CACHE_DIR / "thumbs" / f"{student_id}.jpg"
    if not p.exists():
        raise HTTPException(404, "no photo")
    return FileResponse(p, media_type="image/jpeg")


@router.get("/api/students", response_model=StudentsResponse)
def students() -> StudentsResponse:
    roster = load_roster()
    counts: dict[str, int] = {}
    if config.GALLERY_NPZ.exists():
        z = np.load(config.GALLERY_NPZ)
        counts = {str(s): len(z[str(s)]) for s in z["__order__"]}
    rows = []
    for sid, name in sorted(roster.items(), key=lambda x: x[1].lower()):
        enrolled = (config.PHONE_ENROLL_DIR / name).is_dir()
        rows.append(StudentRow(student_id=sid, name=name, enrolled=enrolled,
                               templates=counts.get(sid, 0)))
    return StudentsResponse(students=rows,
                            enrolled=sum(r.enrolled for r in rows))


@router.post("/api/students", response_model=EnrollResult)
async def add_student(name: str = Form(...),
                      images: list[UploadFile] = File(...)) -> EnrollResult:
    name = name.strip()
    if not name or any(c in name for c in '\\/:*?"<>|'):
        raise HTTPException(422, "Enter a valid name (no special characters)")
    if not images:
        raise HTTPException(422, "At least one image is required")

    from enroll import enroll, EnrollmentError  # noqa: E402

    with tempfile.TemporaryDirectory() as td:
        paths = []
        for up in images:
            p = Path(td) / Path(up.filename).name
            p.write_bytes(await up.read())
            paths.append(p)
        try:
            async with deps.inference_lock():
                await asyncio.to_thread(enroll, name, paths)
        except EnrollmentError as e:
            return EnrollResult(ok=False, message=str(e))

    deps.get_recognizer(reload=True)

    # generate the UI face thumbnail for the new student (cosmetic; never fails
    # the enrollment)
    try:
        from build_thumbnails import build_one
        sid = {v: k for k, v in load_roster().items()}.get(name)
        if sid:
            await asyncio.to_thread(build_one, sid, name, deps.get_backend().app)
    except Exception:  # pragma: no cover
        pass

    return EnrollResult(
        ok=True,
        message=f"{name} enrolled with {len(images)} image(s); gallery rebuilt")


@router.delete("/api/students/{student_id}", response_model=EnrollResult)
async def remove_student(student_id: str) -> EnrollResult:
    roster = load_roster()
    name = roster.get(student_id)
    if name is None:
        raise HTTPException(404, "unknown student id")

    from enroll import remove, EnrollmentError  # noqa: E402
    try:
        async with deps.inference_lock():
            await asyncio.to_thread(remove, name)
    except EnrollmentError as e:
        return EnrollResult(ok=False, message=str(e))

    deps.get_recognizer(reload=True)
    return EnrollResult(ok=True,
                        message=f"{name} unenrolled (images kept on disk)")
