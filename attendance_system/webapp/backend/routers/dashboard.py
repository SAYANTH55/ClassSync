"""GET /api/dashboard — everything the landing page shows, one call."""

from __future__ import annotations

from datetime import datetime, timedelta

from fastapi import APIRouter

from .. import deps  # noqa: F401  (ensures src/ is on sys.path first)
from ..schemas import DashboardResponse, RecentMark

from attendance_report import day_files, expected_students  # noqa: E402
from gallery import load_roster  # noqa: E402

router = APIRouter()


@router.get("/api/dashboard", response_model=DashboardResponse)
def dashboard() -> DashboardResponse:
    roster = load_roster()
    expected = expected_students(roster)
    logs = day_files()
    today = datetime.now().strftime("%Y-%m-%d")
    rows = logs.get(today, [])
    present_ids = {r["student_id"] for r in rows}

    recent = sorted(rows, key=lambda r: r["time"], reverse=True)[:8]

    # 14-day sparkline (calendar days, zero when no log)
    labels, counts = [], []
    for i in range(13, -1, -1):
        d = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
        labels.append(d[5:])                       # "MM-DD"
        counts.append(len({r["student_id"] for r in logs.get(d, [])}))

    n_exp = len(expected)
    return DashboardResponse(
        date=today,
        total_students=n_exp,
        present=len(present_ids),
        absent=n_exp - len(present_ids),
        percent=round(100 * len(present_ids) / n_exp, 1) if n_exp else 0.0,
        recent=[RecentMark(time=r["time"], student_id=r["student_id"],
                           name=r["name"], score=float(r["score"]))
                for r in recent],
        day_counts=counts,
        day_labels=labels,
    )
