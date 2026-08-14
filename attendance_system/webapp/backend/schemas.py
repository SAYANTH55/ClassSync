"""Pydantic response models — the API's public shape, all in one place."""

from __future__ import annotations

from pydantic import BaseModel


class RecentMark(BaseModel):
    time: str
    student_id: str
    name: str
    score: float


class DashboardResponse(BaseModel):
    date: str
    total_students: int
    present: int
    absent: int
    percent: float
    recent: list[RecentMark]          # newest first, capped
    day_counts: list[int]             # last-14-days present counts (sparkline)
    day_labels: list[str]


class DayRow(BaseModel):
    time: str
    student_id: str
    name: str
    score: float


class AbsentRow(BaseModel):
    student_id: str
    name: str


class DayResponse(BaseModel):
    date: str
    present: list[DayRow]
    absent: list[AbsentRow]
    total_students: int


class SummaryRow(BaseModel):
    student_id: str
    name: str
    days_present: int
    percent: float


class SummaryResponse(BaseModel):
    days_total: int
    first_day: str | None
    last_day: str | None
    rows: list[SummaryRow]


class ClearResult(BaseModel):
    ok: bool
    cleared: int
    message: str


class StudentRow(BaseModel):
    student_id: str
    name: str
    enrolled: bool
    templates: int


class StudentsResponse(BaseModel):
    students: list[StudentRow]
    enrolled: int


class EnrollResult(BaseModel):
    ok: bool
    message: str


class HealthResponse(BaseModel):
    status: str                       # "ok"
    engine_loaded: bool
    gallery_students: int
    gallery_templates: int
    threshold: float


class SettingsResponse(BaseModel):
    product: str
    threshold: float
    confirm_frames: int
    model_pack: str
    detector: str
    embedder: str
    gallery_students: int
    gallery_templates: int
    data_dir: str
