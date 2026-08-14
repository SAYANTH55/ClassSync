"""
Central configuration for the face-recognition attendance system.

Every script imports paths and constants from this module, so a single edit
relocates the entire pipeline. Nothing else in the codebase hard-codes a path.

Multi-session design
--------------------
All data belongs to a *capture session* identified by a short source tag
(``dslr``, ``phone1``, ...). The ``SESSIONS`` registry below is the single
place where sessions are declared; every script resolves images through it.
Adding a future collection session = adding one entry here — no script
changes required.

Invariant: image FILENAMES ARE GLOBALLY UNIQUE across all sessions. The DSLR
stills carry unique camera names (DSC*.jpeg); every other session is ingested
through ``ingest_session.py``, which prefixes files with the session tag and
student id by construction. Label/annotation files key on the bare filename
and rely on this invariant.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---- data locations --------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
RAW_SESSIONS_DIR = DATA_DIR / "raw_sessions"   # incoming originals (immutable)
FRAMES_DIR = DATA_DIR / "frames"               # ingested session images
LABELS_CSV = DATA_DIR / "labels" / "labels.csv"  # single source of truth
ORGANIZED_DIR = DATA_DIR / "organized"         # derived — rebuildable
PROCESSED_DIR = DATA_DIR / "processed"         # derived — rebuildable

REPORTS_DIR = PROJECT_ROOT / "reports"
DOCS_DIR = PROJECT_ROOT / "docs"

# ---- capture-session registry ----------------------------------------------
# kind = "stills": images arrived pipeline-ready in images_dir (legacy DSLR).
# kind = "ingested": originals (videos/stills) live in originals_dir sorted
#        into per-student folders; ingest_session.py populates images_dir.
SESSIONS: dict[str, dict] = {
    "dslr": {
        "images_dir": Path(r"E:\Computer_vision_dataset") / "Attendance_monitoring_system",
        "device": "Sony ILCE-7RM5 (studio portraits)",
        "date": "2026-03-09",
        "kind": "stills",
    },
    # DSLR portraits manually labelled by student name (<Name>.jpeg). Same
    # pixels as "dslr" (hash-verified subset); filenames carry identity and
    # match the phone_enroll folder names exactly. This is the EVALUATION
    # probe set: 46 enrolled students + 1 DSLR-only student + 6 DSC*
    # non-enrolled impostors.
    "dslr_labelled": {
        "images_dir": Path(r"E:\Attendance_monitoring _system_dataset"
                           r"\Attendance_monitoring_system-20260718T093929Z-1-001"
                           r"\Attendance_monitoring_system"),
        "device": "Sony ILCE-7RM5 (studio portraits)",
        "date": "2026-03-09",
        "kind": "stills",
    },
    "phone1": {
        "images_dir": FRAMES_DIR / "phone1",
        "originals_dir": RAW_SESSIONS_DIR / "phone1",
        "device": "smartphone, per-student (heterogeneous models)",
        "date": None,   # set when collection happens
        "kind": "ingested",
    },
    "phone2": {
        "images_dir": FRAMES_DIR / "phone2",
        "originals_dir": RAW_SESSIONS_DIR / "phone2",
        "device": "smartphone, per-student (heterogeneous models)",
        "date": None,
        "kind": "ingested",
    },
}
KNOWN_SOURCES = frozenset(SESSIONS)

# ---- dataset conventions ---------------------------------------------------
# Organized filename convention:  <student_id>_<source>_<seq>.<ext>
STUDENT_ID_PATTERN = r"^S\d{2,3}$"
IMAGE_EXTENSIONS = {".jpeg", ".jpg", ".png", ".heic"}   # .heic via pillow-heif
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}

# ---- ingestion -------------------------------------------------------------
FRAME_FPS = 3.0            # video frame-sampling rate (frames per second)
FRAME_JPEG_QUALITY = 95    # quality for extracted frames

# ---- enrollment / gallery --------------------------------------------------
# Phone self-captures, one folder per student (folder name = student name).
# Source of the production ArcFace gallery templates.
PHONE_ENROLL_DIR = RAW_SESSIONS_DIR / "phone_enroll"
ROSTER_CSV = DATA_DIR / "labels" / "roster.csv"       # student_id,name (stable)
GALLERY_NPZ = PROCESSED_DIR / "gallery.npz"           # multi-template gallery
CACHE_DIR = DATA_DIR / "cache"

# ---- deployment (kiosk) ----------------------------------------------------
# Open-set acceptance threshold for the deployed recogniser. Calibrated by
# evaluate_dslr.py (cross-device sweep): midpoint of the empty band between
# the genuine [0.607, 0.896] and impostor [0.252, 0.286] score distributions
# (TAR 1.000 / FAR 0.000 anywhere in 0.29-0.60). Recalibrate if the gallery
# or embedding model changes.
RECOG_THRESHOLD = 0.447
ATTENDANCE_DIR = DATA_DIR / "attendance"     # one CSV per session date

# ---- recognition model -----------------------------------------------------
INPUT_SIZE = 112          # face-crop side (px); matches build_face_crops --size
INPUT_CHANNELS = 3        # 3 = RGB (default); 1 = grayscale (planned ablation,
                          # motivated by cross-device colour-science robustness)
MODELS_DIR = PROJECT_ROOT / "models"   # trained weights (git-ignored)

# ---- pre-trained face stack (insightface / ArcFace) ------------------------
# Model cache lives on E: (C: is space-constrained). buffalo_l bundles an
# SCRFD detector (+5-pt landmarks) and the ArcFace r100 embedding (512-d).
EMBED_MODEL_ROOT = Path("E:/amlenvs/insightface_models")
EMBED_MODEL_NAME = "buffalo_l"
EMBED_DIM = 512

# ---- reproducibility -------------------------------------------------------
RANDOM_SEED = 42  # every stochastic step in later phases must use this seed


# ---- helpers ---------------------------------------------------------------
def session_images(source: str) -> list[Path]:
    """All images of one session, deterministically sorted ([] if none yet)."""
    d = SESSIONS[source]["images_dir"]
    if not d.exists():
        return []
    return sorted(p for p in d.iterdir()
                  if p.suffix.lower() in IMAGE_EXTENSIONS)


def iter_images(sessions: set[str] | None = None):
    """Yield (source, path) over all (or the given) sessions, sorted."""
    for source in sorted(sessions or SESSIONS):
        for p in session_images(source):
            yield source, p


def find_image(source: str, filename: str) -> Path:
    """Resolve a labels/annotations filename to its on-disk path."""
    return SESSIONS[source]["images_dir"] / filename
