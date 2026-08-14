"""
Attendance reporting CLI
========================

Reads the per-day logs written by take_attendance.py plus the roster, and
answers the questions a class teacher actually asks:

    python src/attendance_report.py                  # today: present + absent
    python src/attendance_report.py 2026-07-23       # a specific day
    python src/attendance_report.py --summary        # all days: per-student
                                                     # counts + percentage,
                                                     # saved to CSV

Only students with an active enrollment folder are counted as expected;
unenrolled students in old logs are still shown in day views (their rows
remain valid history) but are excluded from expected/absent lists.
"""

from __future__ import annotations

import csv
import sys
from datetime import datetime

import config
from gallery import load_roster


def day_files() -> dict[str, list[dict]]:
    """{date: rows} for every attendance CSV on disk."""
    out = {}
    if config.ATTENDANCE_DIR.exists():
        for p in sorted(config.ATTENDANCE_DIR.glob("attendance_*.csv")):
            day = p.stem.replace("attendance_", "")
            with open(p, newline="", encoding="utf-8") as f:
                out[day] = list(csv.DictReader(f))
    return out


def expected_students(roster: dict[str, str]) -> dict[str, str]:
    return {sid: name for sid, name in roster.items()
            if (config.PHONE_ENROLL_DIR / name).is_dir()}


def show_day(day: str) -> None:
    roster = load_roster()
    expected = expected_students(roster)
    rows = day_files().get(day, [])
    present_ids = {r["student_id"] for r in rows}
    absent = {sid: n for sid, n in expected.items() if sid not in present_ids}

    print(f"\n===== attendance {day} =====")
    if not rows:
        print("no log for this day")
    for r in sorted(rows, key=lambda r: r["time"]):
        print(f"  {r['time']}  {r['student_id']:5s} {r['name']:16s} "
              f"score {r['score']}")
    print(f"\npresent: {len(present_ids)}/{len(expected)}")
    if absent:
        print("absent : " + ", ".join(
            f"{n} ({sid})" for sid, n in sorted(absent.items(),
                                                key=lambda x: x[1])))


def summary() -> None:
    roster = load_roster()
    expected = expected_students(roster)
    logs = day_files()
    if not logs:
        raise SystemExit("no attendance logs found")
    days = sorted(logs)
    counts = {sid: 0 for sid in expected}
    for rows in logs.values():
        for r in rows:
            if r["student_id"] in counts:
                counts[r["student_id"]] += 1

    out = config.REPORTS_DIR / "attendance_summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["student_id", "name", "days_present", "days_total",
                    "percent"])
        print(f"\n===== summary over {len(days)} day(s) "
              f"({days[0]} .. {days[-1]}) =====")
        for sid, name in sorted(expected.items(), key=lambda x: x[1]):
            pct = 100 * counts[sid] / len(days)
            w.writerow([sid, name, counts[sid], len(days), f"{pct:.1f}"])
            flag = "  <- below 75%" if pct < 75 else ""
            print(f"  {sid:5s} {name:16s} {counts[sid]:3d}/{len(days)}"
                  f"  {pct:5.1f}%{flag}")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "--summary":
        summary()
    else:
        show_day(args[0] if args else datetime.now().strftime("%Y-%m-%d"))
