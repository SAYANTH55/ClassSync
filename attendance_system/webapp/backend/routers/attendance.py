"""Attendance data endpoints: day detail, period summary, CSV export."""

from __future__ import annotations

import io
import shutil
import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from .. import deps  # noqa: F401  (sys.path setup)
from ..schemas import (AbsentRow, ClearResult, DayResponse, DayRow,
                       SummaryResponse, SummaryRow)
import config  # noqa: E402

from attendance_report import day_files, expected_students  # noqa: E402
from gallery import load_roster  # noqa: E402

router = APIRouter()


def _day(date: str) -> tuple[list[dict], dict[str, str], set[str]]:
    roster = load_roster()
    expected = expected_students(roster)
    rows = day_files().get(date, [])
    present_ids = {r["student_id"] for r in rows}
    return rows, expected, present_ids


@router.get("/api/attendance/summary", response_model=SummaryResponse)
def summary() -> SummaryResponse:
    roster = load_roster()
    expected = expected_students(roster)
    logs = day_files()
    days = sorted(logs)
    counts = {sid: 0 for sid in expected}
    for rows in logs.values():
        for r in rows:
            if r["student_id"] in counts:
                counts[r["student_id"]] += 1
    n = len(days)
    return SummaryResponse(
        days_total=n,
        first_day=days[0] if days else None,
        last_day=days[-1] if days else None,
        rows=[SummaryRow(student_id=sid, name=name, days_present=counts[sid],
                         percent=round(100 * counts[sid] / n, 1) if n else 0.0)
              for sid, name in sorted(expected.items(), key=lambda x: x[1])],
    )


@router.get("/api/attendance/{date}", response_model=DayResponse)
def day(date: str) -> DayResponse:
    rows, expected, present_ids = _day(date)
    return DayResponse(
        date=date,
        present=[DayRow(time=r["time"], student_id=r["student_id"],
                        name=r["name"], score=float(r["score"]))
                 for r in sorted(rows, key=lambda r: r["time"])],
        absent=[AbsentRow(student_id=sid, name=name)
                for sid, name in sorted(expected.items(), key=lambda x: x[1])
                if sid not in present_ids],
        total_students=len(expected),
    )


@router.delete("/api/attendance/{date}", response_model=ClearResult)
def clear_day(date: str) -> ClearResult:
    """Clear a day's attendance by archiving its CSV (recoverable).

    The file is moved to data/attendance/_archive/ rather than deleted, so a
    mistaken clear can be undone. day_files() only globs the top-level
    directory, so archived files never reappear in reports.
    """
    path = config.ATTENDANCE_DIR / f"attendance_{date}.csv"
    if not path.exists():
        return ClearResult(ok=True, cleared=0,
                           message=f"No attendance recorded for {date}")
    with open(path, encoding="utf-8") as f:
        n = max(0, sum(1 for _ in f) - 1)          # rows minus header
    archive = config.ATTENDANCE_DIR / "_archive"
    archive.mkdir(exist_ok=True)
    dest = archive / f"attendance_{date}_cleared_{time.strftime('%H%M%S')}.csv"
    shutil.move(str(path), str(dest))
    return ClearResult(ok=True, cleared=n,
                       message=f"Cleared {n} record(s) for {date}")


@router.get("/api/attendance/{date}/export")
def export_day(date: str) -> StreamingResponse:
    rows, expected, present_ids = _day(date)
    if not expected:
        raise HTTPException(404, "no roster")
    buf = io.StringIO()
    buf.write("student_id,name,status,time,score\n")
    times = {r["student_id"]: r for r in rows}
    for sid, name in sorted(expected.items()):
        r = times.get(sid)
        if r:
            buf.write(f'{sid},"{name}",present,{r["time"]},{r["score"]}\n')
        else:
            buf.write(f'{sid},"{name}",absent,,\n')
    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]), media_type="text/csv",
        headers={"Content-Disposition":
                 f'attachment; filename="classsync_{date}.csv"'})
