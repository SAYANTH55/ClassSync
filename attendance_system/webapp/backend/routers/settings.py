"""GET /api/health and /api/settings — engine status and configuration."""

from __future__ import annotations

import numpy as np
from fastapi import APIRouter

from .. import deps
from ..schemas import HealthResponse, SettingsResponse

import config  # noqa: E402

router = APIRouter()


def _gallery_stats() -> tuple[int, int]:
    if not config.GALLERY_NPZ.exists():
        return 0, 0
    z = np.load(config.GALLERY_NPZ)
    ids = [str(s) for s in z["__order__"]]
    return len(ids), sum(len(z[sid]) for sid in ids)


@router.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    students, templates = _gallery_stats()
    return HealthResponse(
        status="ok",
        engine_loaded=deps.engine_loaded(),
        gallery_students=students,
        gallery_templates=templates,
        threshold=config.RECOG_THRESHOLD,
    )


@router.get("/api/settings", response_model=SettingsResponse)
def settings() -> SettingsResponse:
    students, templates = _gallery_stats()
    return SettingsResponse(
        product="ClassSync",
        threshold=config.RECOG_THRESHOLD,
        confirm_frames=5,
        model_pack=config.EMBED_MODEL_NAME,
        detector="SCRFD-10GF (det_10g.onnx)",
        embedder="ArcFace ResNet-50 (w600k_r50.onnx, 512-d)",
        gallery_students=students,
        gallery_templates=templates,
        data_dir=str(config.DATA_DIR),
    )
