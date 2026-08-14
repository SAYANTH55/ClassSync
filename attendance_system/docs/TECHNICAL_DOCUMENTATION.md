# ClassSync — Complete Technical Documentation

*Face-recognition attendance system for smart classrooms. This document
describes the system exactly as it exists in the current codebase. It is
written so that an engineer who has never seen the source can understand the
full architecture, data flow, and design rationale.*

---

## 1. Project overview

**Title:** ClassSync — AI face-recognition attendance system for smart classrooms.

**Objective:** Automatically record which students are present in a class by
recognising their faces from a camera, with no manual roll-call and no
per-student hardware (badges, fingerprint readers).

**Real-world problem:** Manual attendance is slow, error-prone, and easy to
game (proxy attendance). Existing biometric systems need dedicated hardware
and re-training whenever the class roster changes. ClassSync uses one camera
already built into a classroom smart board, recognises many students at once,
and adds or removes a student in seconds without retraining any neural network.

**Current capabilities:**

- Enrol a student from a handful of phone selfies, with automatic validation.
- Recognise **multiple students simultaneously** in a single camera frame.
- Mark attendance automatically once a student is stably recognised.
- Reject unknown (non-enrolled) people — *open-set* recognition.
- Reject the same student being counted twice (per-day de-duplication).
- Daily and summary attendance reports, CSV export, below-75% flagging.
- Clear/reset a day's attendance (archived, recoverable).
- A polished web dashboard (light/dark) served from a single command.

**Final deployment scenario:** A wall-mounted interactive smart board (e.g. a
"Senses" panel) with a built-in wide-angle camera at the front of the
classroom. A teacher opens ClassSync in the board's browser, starts a session,
and the seated students are recognised and marked over a few seconds. The
recognition engine runs **locally on CPU** — no cloud, no GPU required.

---

## 2. Complete system architecture

### 2.1 The deployed recognition pipeline (per camera frame)

```
                        CLASSROOM SMART-BOARD CAMERA
                                    │
                     browser captures a JPEG frame (960px wide)
                                    │  (WebSocket, binary)
                                    ▼
                        FastAPI backend  /ws/camera
                                    │
                                    ▼
                      cv2.imdecode  → BGR image array
                                    │
                                    ▼
        ┌───────────────────────────────────────────────────────┐
        │  SCRFD FACE DETECTOR  (det_10g.onnx, via insightface)  │
        │  finds EVERY face → for each: bbox, 5 landmarks, score │
        └───────────────────────────────────────────────────────┘
                                    │
             ┌──────────────────────┴──────────────────────┐
             │            loop over every detected face      │
             ▼                                               ▼
   ┌───────────────────┐                          (same for each face)
   │  FACE ALIGNMENT   │  5 landmarks → norm_crop → 112×112 aligned face
   └───────────────────┘
             │
             ▼
   ┌───────────────────────────────────────────┐
   │  ArcFace EMBEDDING  (w600k_r50.onnx)       │
   │  aligned face → 512-D unit vector          │
   └───────────────────────────────────────────┘
             │
             ▼
   ┌───────────────────────────────────────────┐
   │  RECOGNITION (Recognizer.identify)         │
   │  cosine similarity vs every gallery         │
   │  template → best student → compare to τ=0.447│
   └───────────────────────────────────────────┘
             │
   ┌─────────┴───────────┐
   │ score ≥ τ ?          │
   ▼                     ▼
 KNOWN student        UNKNOWN (rejected, red box, never marked)
   │
   ▼
   ┌───────────────────────────────────────────┐
   │  PER-STUDENT CONFIRMATION STREAK           │
   │  +1 this frame; auto-mark at 5 frames      │
   └───────────────────────────────────────────┘
             │
             ▼
   ┌───────────────────────────────────────────┐
   │  ATTENDANCE LOG  (one CSV row per student  │
   │  per day; duplicates ignored)              │
   └───────────────────────────────────────────┘
             │
             ▼
        JSON reply (list of faces + states) → browser draws boxes
                                              │
                                              ▼
                                   REPORTS  (daily / summary / CSV)
```

### 2.2 The enrollment pipeline (offline, one-time per student)

```
Phone selfies (3–11 images)
        │
        ▼
SCRFD detect + ArcFace embed each image  (detect_embed.FaceBackend)
        │
        ▼
Validation: exactly one face per image  +  all images = same person
            (centroid consistency check)  +  (if updating) matches existing
        │
        ▼
Copy images → data/raw_sessions/phone_enroll/<Name>/
        │
        ▼
Rebuild gallery.npz  (all students' embeddings)  +  roster.csv (stable ID)
        │
        ▼
Recognizer reloads → student is now recognisable
```

### 2.3 System layers

```
┌───────────────────────────────────────────────────────────────┐
│  FRONTEND   React SPA (Vite build) — dashboard, camera, reports │
│             served as static files                              │
├───────────────────────────────────────────────────────────────┤
│  BACKEND    FastAPI — REST API (/api/*) + WebSocket (/ws/camera) │
├───────────────────────────────────────────────────────────────┤
│  ENGINE     detect_embed (SCRFD+ArcFace) · gallery (Recognizer) │
│             · take_attendance (AttendanceLog) · enroll           │
├───────────────────────────────────────────────────────────────┤
│  MODELS     insightface buffalo_l ONNX  (SCRFD + ArcFace)        │
├───────────────────────────────────────────────────────────────┤
│  STORAGE    gallery.npz · roster.csv · attendance CSVs · caches  │
└───────────────────────────────────────────────────────────────┘
```

---

## 3. Technology stack

| Technology | Version | Role | Where used | Why chosen |
|---|---|---|---|---|
| **Python** | 3.11.15 | Backend + engine language | all of `src/`, `webapp/backend/` | Ecosystem for CV/ML; insightface/OpenCV are Python-first |
| **FastAPI** | 0.139.2 | Web framework (REST + WebSocket) | `webapp/backend/` | Native async + WebSocket (needed for live camera streaming); runs inside the existing Python engine env so no cross-language bridge |
| **Uvicorn** | 0.51.0 | ASGI server | run command | Serves FastAPI; single-command launch |
| **insightface** | 1.0.1 | Face model runtime (SCRFD + ArcFace) | `src/detect_embed.py` | Bundles a state-of-the-art detector and embedder in one pack; well-maintained |
| **ONNX Runtime** | 1.27.0 | Neural-network inference engine | under insightface | Runs the `.onnx` models on CPU efficiently; framework-agnostic |
| **OpenCV** | 5.0.0 | Image decode, resize, colour conversion | `detect_embed`, `ws/camera`, evaluation | Fast, standard CV primitives |
| **NumPy** | 2.4.6 | Array math (embeddings, cosine similarity) | everywhere | The lingua franca for numeric arrays |
| **Pillow + pillow-heif** | 12.3.0 | Image loading incl. iPhone HEIC | `src/preprocessing.py` | HEIC support for phone enrollment photos; EXIF-orientation handling |
| **React** | 19 | Frontend UI library | `webapp/frontend/src/` | Component model fits a multi-page dashboard; huge ecosystem |
| **React Router** | 7 | Client-side routing | `App.jsx` | Page navigation without full reloads |
| **Vite** | 8 | Frontend build tool / dev server | `webapp/frontend/` | Fast builds; outputs static files served by FastAPI |
| **Tailwind CSS** | 4 | Styling | all components | Utility-first styling; theme via CSS variables (light/dark) |
| **lucide-react** | — | Icon set | UI components | Clean, consistent outline icons |
| **@fontsource-variable/inter** | — | Typeface (Inter) | global | Modern SaaS look, self-hosted (no external font CDN) |
| **oxlint** | — | Linter (dev only) | frontend | Fast JS/JSX linting |

**Two Python environments** (documented in `requirements.txt`):

- `E:\amlenvs\face311` (Python 3.11) — runs **everything** except one script.
- miniconda base (Python 3.13 + TensorFlow) — runs **only** `evaluate_deep.py`
  (the from-scratch CNN / MobileNetV2 *evaluation baselines*). The two envs
  communicate only through a cached `.npz` file. TensorFlow is **not** part of
  the deployed system.

---

## 4. AI / computer-vision models

The deployed system uses **exactly two neural networks**, both from the
insightface `buffalo_l` model pack, both **pretrained**, both **inference-only**.

### 4.1 SCRFD — face detector

| Property | Value |
|---|---|
| Exact model | `det_10g.onnx` (SCRFD-10GF) |
| Architecture | SCRFD — *Sample and Computation Redistribution for Face Detection*; a single-stage anchor-based detector with a lightweight backbone + feature-pyramid heads |
| Pretrained / custom | **Pretrained** (by insightface) |
| Framework | ONNX, executed by ONNX Runtime |
| Input | BGR image, internally resized to a 640×640 detection canvas (`det_size`) |
| Output | For every face: bounding box `(x1,y1,x2,y2)`, a confidence score `0–1`, and **5 landmarks** (left eye, right eye, nose, left mouth corner, right mouth corner) |
| Purpose | Find all faces and their landmarks in a frame |
| Where called | `src/detect_embed.py` → `FaceBackend.detect()` (insightface `app.get()`) |
| Why selected | One-stage detector = fast; returns landmarks for free (needed for alignment); catches multiple faces per frame — essential for classroom mode |
| Training or inference | **Inference only** |

**What SCRFD does, plainly:** it slides across the image at three scales
(fine/medium/coarse grids) so it can find both close and distant faces, and for
each face it also predicts five key points. Those five points are what let us
rotate and crop each face into a standard pose before recognition.

### 4.2 ArcFace — face embedding network

| Property | Value |
|---|---|
| Exact model | `w600k_r50.onnx` |
| Architecture | **ResNet-50** backbone trained with the **ArcFace loss** (additive angular margin) |
| Pretrained / custom | **Pretrained** on **WebFace-600K** (~600,000 identities) |
| Framework | ONNX, executed by ONNX Runtime |
| Input | A **112×112** aligned face crop |
| Output | A **512-dimensional** L2-normalised embedding vector |
| Purpose | Convert a face image into a vector where same-person vectors are close and different-person vectors are far apart (in cosine/angular distance) |
| Where called | `src/detect_embed.py` → `FaceBackend.detect()` / `embed_best()` (insightface recognition model) |
| Why selected | ArcFace embeddings generalise to people never seen in training, so a new student is enrolled by storing one vector — no retraining. Decisive for a small, changing class roster |
| Training or inference | **Inference only** (the network was trained by others; we never update its weights) |

**What ArcFace is, plainly:** ArcFace is not a "model name" so much as a **loss
function** used to *train* a face network. During training it forces each
identity's embeddings into a tight cluster with an angular gap ("margin")
between identities — like training with ankle weights so that at inference the
separation is comfortable. The trained network (ResNet-50 here) then turns any
face into a 512-number vector. **Comparing two faces = one dot product** of
their vectors (cosine similarity).

> Note: `w600k_r50` is a **ResNet-50** (r50) recognition model. A stray comment
> in `config.py` calls it "r100" — that comment is inaccurate; the actual file
> is r50.

### 4.3 Models present but NOT loaded

The `buffalo_l` pack also ships `2d106det.onnx` (106-point landmarks),
`1k3d68.onnx` (68 3D landmarks) and `genderage.onnx`. ClassSync loads
**only** detection + recognition (`allowed_modules=["detection","recognition"]`
in `detect_embed.py`) — loading the other three cost ~145 ms/frame for outputs
we don't use. They remain on disk and are relevant to the queued anti-spoofing
work (blink/EAR needs the 106-point landmarks).

### 4.4 Is FaceNet used?

**No.** ClassSync uses **ArcFace** (via `w600k_r50.onnx`), not FaceNet. FaceNet
(2015, triplet loss) is the historical predecessor; ArcFace (2019, angular
margin) is more stable to train and is the current industry default. Both
produce face embeddings compared by distance; only ArcFace is in this codebase.

### 4.5 Baseline models (evaluation only, NOT deployed)

For the comparative study these were built and measured, then set aside — they
are **not** part of the running product:

- **Eigenfaces** (PCA) — classical, `evaluate_classical.py`
- **LBPH** (local binary pattern histograms, own NumPy implementation because
  `cv2.face` was removed in OpenCV 5) — classical, `evaluate_classical.py`
- **Scratch CNN** — a small VGG-style network trained from random init,
  `evaluate_deep.py` (this is the only place any training happens, and it's for
  a baseline, not the deployed recogniser)
- **MobileNetV2** — ImageNet-pretrained, frozen, transfer-learning baseline,
  `evaluate_deep.py`

---

## 5. Datasets

There are **four distinct kinds** of data. Keeping them separate is essential
to understanding the project.

### 5.1 Dataset that trained the pretrained models (external, not in repo)

- **WebFace-600K** — ~600,000 face identities. Used by insightface to train the
  ArcFace `w600k_r50` network. We **never touch it**; we only use the resulting
  weights. This is why the recogniser already "knows how to see faces" before
  any of our students exist.

### 5.2 Enrolled student images (our gallery source)

- **Phone enrollment set** — `data/raw_sessions/phone_enroll/<Name>/`
  - One folder per student, folder name = student name.
  - **48 students**, ~5 HEIC/JPG selfies each (**256 images total**; some
    students have more, e.g. one has 11).
  - Captured on students' own phones (heterogeneous devices).
  - **Purpose:** the source of the recognition **gallery templates**.

### 5.3 Evaluation probe set (measuring accuracy, not deployed)

- **DSLR labelled set** — `E:\Attendance_monitoring _system_dataset\...\Attendance_monitoring_system`
  - **53 studio portraits** (Sony A7R V), each filename = a student name, plus
    `DSC*` files for non-enrolled people.
  - Composition: **46 enrolled students + 1 DSLR-only student + ~6 non-enrolled impostors**.
  - **Purpose:** an *independent, cross-device* test set. The gallery is built
    from phone selfies; these DSLR shots (different camera, ~4 months earlier)
    probe how well recognition survives the device gap. Used to calibrate τ.
- **Pristine originals** — `E:\Computer_vision_dataset\Attendance_monitoring_system`
  (56 untouched camera files) — a read-only backup of the DSLR originals before
  renaming; used for hash-verification audits.

### 5.4 Derived project data (generated by the system)

- **`gallery.npz`** — the multi-template gallery (student_id → array of 512-D
  embeddings). Rebuilt from the phone enrollment set.
- **`roster.csv`** — `student_id,name` mapping with **stable** IDs (S01…S48).
- **Attendance CSVs** — `data/attendance/attendance_YYYY-MM-DD.csv`, one row per
  present student per day.
- **Embedding / crop caches** — `data/cache/*.npz` (speed-ups; rebuildable).

---

## 6. Enrollment pipeline (what happens when you click "Enroll student")

UI: **Students → Add student** → type a name, choose photos, click **Enroll
student**. Backend flow:

1. **Upload** — the browser POSTs `multipart/form-data` (name + image files) to
   `POST /api/students` (`webapp/backend/routers/students.py`, `add_student`).
2. **Name validation** — rejects empty names or names with filesystem-unsafe
   characters (`\ / : * ? " < > |`).
3. **Temp save** — uploaded files are written to a temporary directory.
4. **Engine call under lock** — `enroll(name, paths)` (`src/enroll.py`) is run
   in a worker thread while holding the shared inference lock (models aren't
   thread-safe).
5. Inside `enroll()`:
   - **Embed each image** — `FaceBackend.embed_path()` runs SCRFD + ArcFace;
     each image must contain **exactly one detectable face** or it is rejected
     (`no detectable face in <file>`).
   - **Same-person check (centroid consistency)** — computes the mean embedding
     (the "centroid"); every image must be within `MIN_CENTROID_SIM = 0.35`
     cosine of it. This tolerates natural pose/expression variation but rejects
     a stray photo of a different person — and names the offending file. (This
     replaced an earlier min-pairwise test that falsely rejected varied selfies.)
   - **If the student already exists** — new images must also match the existing
     templates (`MIN_MATCH_EXISTING = 0.35`), preventing enrolment under the
     wrong name.
   - **Copy images** → `data/raw_sessions/phone_enroll/<Name>/`.
   - **Invalidate cache** — drop this student's entry from
     `embeddings_phone_enroll.npz` so they are re-embedded.
   - **Rebuild** — `gallery.build_gallery()`:
     - `build_roster()` assigns a **stable** `student_id` (existing IDs never
       renumber; a new student gets the next free `Sxx`) → writes `roster.csv`.
     - Embeds every student's images (reusing the cache where valid) → writes
       `data/processed/gallery.npz` with one embedding array per student.
6. **Recognizer reload** — `deps.get_recognizer(reload=True)` reloads the
   gallery so the new student is immediately recognisable.
7. **Response** — JSON `{ok, message}`; the UI shows a success/failure toast.

**Files written:** the student's image folder, `roster.csv`, `gallery.npz`,
refreshed embedding cache.

**Remove student:** `DELETE /api/students/{id}` → `remove(name)` moves the
folder to `data/raw_sessions/unenrolled/` (kept, not deleted), drops the cache
entry, rebuilds the gallery. The roster keeps the ID reserved so old attendance
records stay valid.

---

## 7. Recognition pipeline (what happens to a webcam frame)

1. **Capture (frontend)** — `Camera.jsx` uses `getUserMedia` to open the camera,
   and every **1200 ms** draws the video onto a hidden `<canvas>` sized to
   **960 px** wide, encodes it to JPEG (quality 0.8), and sends the bytes over
   the `/ws/camera` WebSocket.
2. **Decode (backend)** — `ws/camera.py._process()` runs
   `cv2.imdecode` → BGR array.
3. **Detect all faces** — `FaceBackend.detect()` converts BGR→RGB internally and
   calls insightface `app.get()`, which runs **SCRFD** then **ArcFace** on
   **every** detected face. Each returned `Face` already carries: `bbox`,
   `landmarks` (5 points), `det_score`, and a 512-D `embedding`.
4. **Per-face loop** — for each face:
   - Compute a **normalised box** `[x, y, w, h]` in 0–1 (for the UI).
   - **Identify** — `Recognizer.identify(embedding)`:
     - `scores(emb)` = for each student, the **maximum cosine similarity**
       between the probe and that student's templates (max over multi-template).
     - Best student = argmax. Runner-up recorded for the margin.
     - If best score `≥ τ (0.447)` → a `Match(student_id, name, score, …)`;
       else `None` (**unknown / open-set rejection**).
   - For unknowns, the best (sub-threshold) score is reported for the UI.
5. **Confirmation & marking** — `ClassroomSession.update()`:
   - Build the set of students recognised **this frame** (best score per
     student, guarding against a double-detected face).
   - **Streak update:** each recognised student's streak `+1` (capped at 5);
     every student *not* seen this frame decays `-1` (floored at 0).
   - **Mark:** when a student's streak reaches **`CONFIRM_FRAMES = 5`** and they
     are not already present today, `AttendanceLog.mark()` appends a CSV row.
6. **Reply** — a JSON message: a `faces` list (`box`, `state ∈
   {marked, confirming, unknown}`, `name`, `student_id`, `score`, `streak`),
   `counts` (detected / recognised / marked), and the session's `marked` roster.
7. **Render (frontend)** — `Camera.jsx` draws a coloured box + label on each
   face (green = marked, amber = confirming n/5, red = unknown), updates the
   live counters and the "Marked present" side panel.

**Thresholds involved:** SCRFD detection confidence (implicit, from the model);
the **recognition threshold τ = 0.447** (open-set accept/reject); the
**confirmation count = 5 frames**.

---

## 8. Multi-face classroom mode

**What changed vs the old single-face version:**

- Old live camera processed only the **largest** face (`faces[0]`) and tracked a
  **single** confirmation streak — a walk-up kiosk model.
- New mode processes **every** face in the frame and tracks an **independent
  confirmation streak per student**.

**Which files changed:**

- `webapp/backend/ws/camera.py` — `ConfirmationSession` (single streak) became
  `ClassroomSession` (per-student streak dict); `_process()` loops all faces and
  returns a list; the message shape became `{state:"frame", faces:[…], counts,
  marked}`.
- `webapp/frontend/src/pages/Camera.jsx` — one status card became a **box per
  face** plus live counters and a marked-students panel; capture resolution
  raised to 960 px so back-row faces survive.

**Why the recognition engine did NOT change:** `FaceBackend.detect()` already
returned **all** faces, each **already embedded** (insightface embeds every
detected face inside `app.get()`); the old code simply discarded all but the
largest. `Recognizer.identify()` was already per-embedding. So multi-face was a
*wrapper + UI* change, not an engine change — and it adds almost no cost,
because the embeddings for every face were already being computed.

**How multiple faces are processed:** one SCRFD pass finds K faces; the code
loops K times calling `identify()` (each ~0.6 ms). Each face is judged
independently against τ.

**Independent confirmation streaks:** a `dict[student_id → int]`. Increment on a
frame where the student is recognised; **decay** (−1) on a frame where they are
missed. Decay (rather than reset-to-zero) tolerates a seated student briefly
turning away or a single dropped detection — important for a real classroom.

**Attendance marking logic:** a student is marked once their streak hits 5 and
they aren't already present that day. `AttendanceLog` de-duplicates per student
per day, so appearing in 200 frames still yields exactly one row. One recognised
+ one unknown in the same frame is no conflict — each face is independent.

---

## 9. Web application architecture

### 9.1 Backend (`webapp/backend/`)

- **`main.py`** — creates the FastAPI app, registers routers, and mounts the
  built frontend. Non-API routes fall back to `index.html` (SPA routing), so
  deep links like `/reports` survive a refresh.
- **`deps.py`** — the **only** module that imports the engine. Holds lazy
  singletons: `get_backend()` (loads SCRFD+ArcFace on first use, ~10 s),
  `get_recognizer()` (loads the gallery; `reload=True` after enrolment), and an
  `asyncio.Lock` that **serialises all inference** (ONNX sessions aren't
  thread-safe and CPU runs one inference at a time).
- **`schemas.py`** — Pydantic response models (the API's public shapes).
- **`routers/`**
  - `dashboard.py` — `GET /api/dashboard` (today's counts, recent marks,
    14-day sparkline).
  - `attendance.py` — `GET /api/attendance/{date}` (present/absent),
    `GET /api/attendance/summary`, `GET /api/attendance/{date}/export` (CSV),
    `DELETE /api/attendance/{date}` (archive/clear a day).
  - `students.py` — `GET /api/students`, `POST /api/students` (enrol),
    `DELETE /api/students/{id}` (unenrol).
  - `settings.py` — `GET /api/health`, `GET /api/settings` (engine + threshold).
- **`ws/camera.py`** — the `/ws/camera` WebSocket: handshake
  (`loading_engine` → `ready`), then per-frame decode → detect → identify →
  `ClassroomSession` → JSON reply. Inference runs via `asyncio.to_thread` under
  the shared lock so REST endpoints stay responsive.

### 9.2 Frontend (`webapp/frontend/src/`)

- **`main.jsx`** — bootstraps React, restores the saved light/dark theme before
  first paint.
- **`App.jsx`** — layout (Sidebar + Topbar + routed `<main>`) and the routes.
- **`api.js`** — the single fetch client + a WebSocket-URL helper.
- **`components/`** — `Sidebar` (nav + engine-status dot), `Topbar` (page title,
  date, theme toggle), `Logo` (the C-monogram brand mark), `Ring` (animated
  attendance gauge).
- **`hooks/useCountUp.js`** — animates numbers counting up.
- **`pages/`** — `Dashboard`, `Camera`, `Reports`, `Students`, `Settings`.

**State management:** deliberately simple — React local state (`useState` /
`useEffect`) per page; the server is the source of truth. The dashboard polls
every 15 s; the camera page is driven by WebSocket messages. No Redux/global
store is needed at this size.

**API / transport:** JSON over REST for everything except the live camera, which
uses a **binary-in / JSON-out WebSocket** (frames up, states down). In dev, Vite
proxies `/api` and `/ws` to the backend; in production both are same-origin.

**Component relationships:**

```
App
├─ Sidebar  (Logo, nav links, health dot ← GET /api/health)
├─ Topbar   (title, date, theme toggle)
└─ <Routes>
   ├─ Dashboard  ← GET /api/dashboard   (Ring, useCountUp, sparkline)
   ├─ Camera     ↔ WS /ws/camera        (per-face boxes, counters)
   ├─ Reports    ← GET /api/attendance/* (daily/summary, export, clear)
   ├─ Students   ← GET/POST/DELETE /api/students (table, add sheet)
   └─ Settings   ← GET /api/settings
```

---

## 10. Database / storage

There is **no SQL database** — storage is files on disk (simple, inspectable,
demo-friendly). Everything lives under `attendance_system/data/`:

| Path | What it is |
|---|---|
| `data/raw_sessions/phone_enroll/<Name>/` | Enrolled students' selfie images (gallery source, immutable input) |
| `data/raw_sessions/unenrolled/<Name>/` | Removed students' images (kept for recovery) |
| `data/processed/gallery.npz` | The recognition gallery: `__order__` index + one 512-D embedding **array per student_id** (48 students, 256 templates) |
| `data/labels/roster.csv` | `student_id,name` with stable IDs (S01…S48) |
| `data/attendance/attendance_YYYY-MM-DD.csv` | Daily attendance: `time,student_id,name,score` |
| `data/attendance/_archive/` | Cleared days, moved here (recoverable, ignored by reports) |
| `data/cache/embeddings_phone_enroll.npz` | Cached gallery embeddings (speed-up) |
| `data/cache/embeddings_dslr_probes.npz` | Cached DSLR probe embeddings (bias audit) |
| `data/cache/crops_gray.npz`, `crops_rgb.npz` | Aligned crops cached for the classical/deep evaluation baselines |
| `E:\amlenvs\insightface_models\models\buffalo_l\*.onnx` | The pretrained model weights (SCRFD, ArcFace, +unused) |
| `src/config.py` | All paths, constants, and the session registry |
| `reports/` | Evaluation figures + CSVs (eval_dslr, eval_classical, eval_deep, bias_audit, dslr_check) |

**Embedding vs template:** in `gallery.npz` each student has an **array** of
embeddings — one per enrollment image. These stored per-student vectors are the
**templates**. A live face produces a **probe** embedding compared against them.

---

## 11. Project directory

```
attendance_system/
├── README.md
├── requirements.txt              two pinned Python envs (face311 + TF)
├── .gitignore                    excludes data/, models/, name-bearing CSVs
├── data/                         (git-ignored — personal data)
│   ├── raw_sessions/phone_enroll/<Name>/   enrollment selfies
│   ├── processed/gallery.npz     recognition gallery
│   ├── labels/roster.csv         student_id → name (stable)
│   ├── attendance/               daily CSVs + _archive/
│   └── cache/                    embedding + crop caches
│
├── src/                          engine + CLI tools + research scripts
│   ├── config.py                 ★ all paths/constants, session registry, τ
│   ├── preprocessing.py          image load (EXIF-upright, HEIC), geometry
│   ├── detect_embed.py           ★ FaceBackend: SCRFD detect + ArcFace embed
│   ├── gallery.py                ★ roster + multi-template gallery + Recognizer
│   ├── take_attendance.py        ★ AttendanceLog + CLI kiosk (photo/webcam)
│   ├── enroll.py                 ★ add/remove a student (validated)
│   ├── attendance_report.py      day/summary/absentee reporting (CLI)
│   ├── smoke_test.py             14-check end-to-end verification
│   ├── dslr_check.py             dataset audit (hashes, identity reconciliation)
│   ├── evaluate_dslr.py          headline eval: DSLR probes vs phone gallery
│   ├── evaluate_classical.py     Eigenfaces + LBPH baselines
│   ├── build_crops_rgb.py        RGB crop cache (bridge to TF env)
│   ├── evaluate_deep.py          scratch-CNN + MobileNetV2 baselines (TF env)
│   ├── analyze_bias.py           per-identity (Doddington-zoo) audit
│   ├── profile_pipeline.py       per-stage latency measurement
│   └── (older phase-1..3 tools: inspect_dataset, organize_dataset,
│        ingest_session, propose_boxes, annotate_faces, build_face_crops,
│        augment, data, model, train, make_diagrams, inspect_enroll,
│        enroll_detect_check)
│
├── webapp/
│   ├── backend/
│   │   ├── main.py               FastAPI app + SPA mount
│   │   ├── deps.py               ★ lazy engine singletons + inference lock
│   │   ├── schemas.py            Pydantic response models
│   │   ├── routers/              dashboard, attendance, students, settings
│   │   └── ws/camera.py          ★ live multi-face recognition WebSocket
│   └── frontend/
│       ├── index.html            title + favicon (C monogram)
│       ├── vite.config.js        dev proxy for /api and /ws
│       └── src/
│           ├── main.jsx, App.jsx, api.js, index.css
│           ├── components/       Sidebar, Topbar, Logo, Ring
│           ├── hooks/useCountUp.js
│           └── pages/            Dashboard, Camera, Reports, Students, Settings
│
├── reports/                      evaluation artifacts (figures + CSVs)
└── docs/                         this document + methodology drafts
```

(★ = files central to the deployed system.)

---

## 12. Current features

**Completed**

- ✅ Student enrollment from phone photos, with face + identity validation
- ✅ Multi-template gallery + stable roster IDs
- ✅ Open-set recognition with a calibrated threshold (τ = 0.447)
- ✅ **Multi-face classroom recognition** (many students per frame)
- ✅ Per-student confirmation streaks with decay
- ✅ Attendance logging, one row per student per day (de-duplicated)
- ✅ Live camera web UI (per-face boxes, counters, marked panel)
- ✅ Reports: daily, summary, below-75% flag, CSV export, search
- ✅ Clear/reset a day (archived, recoverable)
- ✅ Remove student (kept, recoverable)
- ✅ Dashboard (attendance ring, animated stats, quick actions, chart)
- ✅ Light/dark theme, tinted-gradient background, card elevation
- ✅ Brand logo + favicon
- ✅ End-to-end smoke test (14 checks)
- ✅ Evaluation suite (accuracy, baselines, bias audit, latency profile)

**In progress**

- 🟡 UI polish rollout — engagement pass applied to the Dashboard; Reports /
  Students / Settings not yet given the same hover-lift/animation treatment

**Planned (not started)**

- ⏳ Anti-spoofing / liveness detection (analysed, not implemented)
- ⏳ CLI webcam kiosk multi-face parity (deliberately left single-face)

---

## 13. Remaining work

1. **Anti-spoofing / liveness** — prevent a photo/phone/print from being marked.
   Design already produced: passive deep-learning anti-spoof (MiniFASNet) as the
   primary layer, or an active blink/head-turn challenge using the on-disk
   106-point landmark model. **Next major feature.**
2. **UI consistency** — carry the dashboard's animation/elevation to the other
   pages; optional frosted top bar and gradient accents.
3. **Live-hardware validation** — a real multi-person webcam session.
4. **Deployment packaging** — a one-click launcher / service for the smart board.
5. **Performance** — optional detector/threshold tuning for large classrooms.
6. **Analytics** — trends, per-student history views.

---

## 14. Performance

Measured on the target machine (CPU, ONNX Runtime), from `profile_pipeline.py`:

| Stage | Latency |
|---|---|
| SCRFD detection (per frame) | ~230 ms (roughly constant across det_size on this CPU) |
| ArcFace embedding | ~130 ms **per face** |
| Gallery match (cosine vs 256 templates) | ~0.6 ms |
| Single-face frame, end-to-end | ~520 ms (~**2 FPS**) after trimming unused models |

- **Multi-face cost** ≈ `230 ms + K × 130 ms` for K faces (e.g. ~1.5 s for 10
  faces, ~2.8 s for 20). Acceptable because students are **seated** — a sweep
  cadence of 1.2 s+ per frame is fine; the frontend sends a frame every 1200 ms.
- **Key optimisation already applied:** `buffalo_l` would run 5 models per face;
  restricting to detection + recognition saved ~145 ms/frame with **no** change
  to embeddings (so τ and all evaluation results stay valid).
- **Models loaded at runtime:** 2 (`det_10g.onnx` 16.9 MB, `w600k_r50.onnx`
  174 MB). First camera use pays a ~10 s one-off model-load.
- **Compute:** CPU only (`ctx_id = -1`); no GPU required.
- **Memory:** dominated by the ArcFace model (~174 MB on disk) plus the gallery
  (`gallery.npz` ~0.5 MB). Comfortable on a normal laptop / smart-board SoC.

**Accuracy (from `evaluate_dslr.py`, cross-device):** deployed ArcFace scored
**100% rank-1** identification with **zero false accepts** at τ = 0.447 (genuine
scores 0.607–0.896 vs impostor 0.252–0.286 — a clean gap). Baselines under the
identical protocol: Eigenfaces 26.1%, LBPH 34.8%, scratch CNN 4.3%,
MobileNetV2-TL 17.4% — none separated impostors.

---

## 15. Design decisions (why)

- **Why SCRFD?** One-stage → fast on CPU; returns 5 landmarks for free (needed
  for alignment); natively multi-face (enables classroom mode).
- **Why ArcFace (not FaceNet)?** Angular-margin training yields embeddings that
  generalise to unseen identities and are more stable than FaceNet's triplet
  loss. It's the current industry default and let us hit 100% cross-device.
- **Why embeddings instead of retraining a classifier?** The class roster
  changes; students have only a few images each. Embeddings let us **enrol by
  storing a vector** and add/remove students in seconds with **no retraining** —
  directly solving the data-scarcity and roster-churn problems. (A from-scratch
  CNN was tried as a baseline and scored **4.3%** — proving the point.)
- **Why ONNX / ONNX Runtime?** Framework-agnostic, fast CPU inference, no heavy
  DL framework needed at deployment (TensorFlow is only used for an evaluation
  baseline, in a separate env).
- **Why FastAPI?** Native async + WebSocket (required for streaming camera
  frames) and it runs **in the same Python process/env as the engine**, so
  there's no cross-language bridge between the web layer and the models.
- **Why React + Vite + Tailwind?** Component model suits a multi-page dashboard;
  Vite builds to **static files** the FastAPI server serves, so the whole app
  runs from **one Python command** with no Node needed at runtime.
- **Why a calibrated threshold (0.447), not a guess?** It was chosen as the
  midpoint of the empty band between genuine and impostor score distributions on
  a cross-device test — giving zero false accepts and zero false rejects there.
- **Why confirmation streaks (5 frames)?** A single lucky/blurry frame shouldn't
  mark a student (or the wrong student). Requiring sustained recognition is a
  cheap, robust debounce.
- **Why decay (−1 on a miss) instead of reset?** In a classroom a seated student
  is occasionally missed for one frame (turned head, transient occlusion). Decay
  tolerates that while still requiring genuine sustained presence — reset-to-zero
  would keep restarting and could fail to ever mark someone.
- **Why archive (not delete) on clear/unenrol?** Data safety — a mistaken clear
  or removal is recoverable, matching the project's "raw data is immutable" rule.
- **Why file storage, not a database?** Simplicity and transparency for a
  single-classroom deployment and a demo; CSV/NPZ are inspectable and need no
  DB server. (A DB is a future option if multi-class/multi-site is needed.)

---

## 16. Complete data flow (one worked example)

**Scenario:** enrol a new student "Aisha", then mark her present in a session.

```
(1) ENROLLMENT
Teacher: Students → Add student → "Aisha" + 5 phone selfies → Enroll
   → POST /api/students (multipart)
   → files saved to a temp dir
   → enroll("Aisha", paths):
        · SCRFD+ArcFace embed each of the 5 images  → 5 × 512-D vectors
        · centroid check: min cosine to mean ≥ 0.35 ?  ✔ (all Aisha)
        · copy images → data/raw_sessions/phone_enroll/Aisha/
        · build_roster() → Aisha gets S49 → roster.csv updated
        · build_gallery() → gallery.npz now includes Aisha's 5 templates
   → Recognizer reloaded (49 students)
   → UI toast: "Aisha enrolled with 5 image(s); gallery rebuilt"

(2) ATTENDANCE SESSION
Teacher: Camera → Start attendance (browser asks for camera permission)
   → WS /ws/camera opens → handshake loading_engine → ready
   → every 1200 ms: 960px JPEG frame sent

(3) RECOGNITION (a frame with Aisha + 2 classmates)
   → cv2.imdecode → BGR
   → SCRFD detect → 3 faces (bbox + 5 landmarks + score + embedding each)
   → for each face: norm_crop→112×112 (already embedded) → identify():
        Aisha's probe → max cosine vs gallery = 0.71 (best = S49 Aisha) ≥ 0.447 ✔
        classmate A   → 0.68 (S12) ✔
        classmate B   → 0.65 (S30) ✔
   → ClassroomSession: streaks S49,S12,S30 each +1  (now 1/5)
   → JSON: 3 faces, state "confirming 1/5", counts {detected:3,recognized:3,marked:0}
   → frontend draws 3 amber boxes with names + "1/5"
   ... (repeats each frame; streaks climb) ...
   → frame 5: streaks reach 5 → AttendanceLog.mark(S49,"Aisha",0.71)
              → row appended, box turns green "Aisha ✔", counts.marked = 3

(4) STORAGE
   data/attendance/attendance_2026-07-25.csv gains:
      13:12:04,S49,Aisha,0.7100

(5) REPORTS
Teacher: Reports → Daily
   → GET /api/attendance/2026-07-25
   → present list includes Aisha (S49, 0.71); absent list = the rest
   → Export CSV  → downloads classsync_2026-07-25.csv
   → (optional) Clear day → DELETE archives the CSV → day resets to 0
```

---

## 17. Future roadmap

Everything discussed for after the current state, in rough priority:

1. **Anti-spoofing / liveness detection** — the immediate next feature.
   - Option A (recommended): passive deep-learning anti-spoof CNN (MiniFASNet
     ONNX) as an always-on gate before marking; catches print + screen + video
     replay; a red "spoof detected" state in the camera UI.
   - Option B: active challenge (blink via 106-pt landmarks / EAR, or head-turn
     via landmark geometry) — zero new downloads, very interactive.
2. **UI polish rollout** — animations/elevation across all pages; frosted top
   bar; gradient accents on the ring and primary button.
3. **Deployment packaging** — a launcher/service so the smart board runs
   ClassSync on boot without a terminal.
4. **Analytics** — attendance trends, per-student history, exportable summaries.
5. **Scaling options** — a real database and multi-class/multi-room support if
   the deployment grows beyond one classroom.
6. **Performance for large rooms** — detector resolution / threshold tuning,
   optional lighter model pack (would require re-calibrating τ).

---

## 18. Important notes (things that are easy to misunderstand)

- **Enrollment ≠ training.** Enrolling a student does **not** train or modify any
  neural network. It computes ArcFace embeddings of their photos and stores those
  vectors in the gallery. That's why it takes seconds and needs no GPU.
- **The deployed system does no training at all.** The only training in the whole
  project is the **from-scratch CNN baseline** in `evaluate_deep.py` — an
  *evaluation experiment*, not part of the running product. The deployed SCRFD
  and ArcFace networks are pretrained and used **inference-only**.
- **Templates vs embeddings.** They're the same *type* of object (512-D vectors).
  "Embedding" is the general term for a face-vector; **"template"** specifically
  means a stored gallery embedding for an enrolled student. A live face is a
  **"probe"** embedding compared against the templates.
- **Detection vs recognition.** **Detection** (SCRFD) answers *"where are the
  faces?"* **Recognition** (ArcFace + gallery + threshold) answers *"whose face
  is this?"* They are two different models and two different stages.
- **Pretrained-model data vs project data.** WebFace-600K trained ArcFace and is
  **never used by us**. Our data is the phone enrollment selfies (gallery) and
  the DSLR set (evaluation) — completely separate from the model's training data.
- **Gallery vs roster vs attendance.** The **gallery** (`gallery.npz`) stores
  face vectors; the **roster** (`roster.csv`) maps IDs to names; **attendance**
  CSVs record who was present when. Three different files, three different jobs.
- **Inference vs training.** *Training* adjusts a network's weights from labelled
  data (slow, done once by insightface). *Inference* runs the fixed network to
  produce outputs (fast, what ClassSync does every frame).
- **"DSLR labelled" is a test set, not enrollment data.** Students are enrolled
  from **phone** selfies; the DSLR portraits exist only to *measure* accuracy
  across a device gap. Recognising a DSLR portrait is a test, not enrolment.
- **The recognition threshold is model-specific.** τ = 0.447 was calibrated for
  *this* ArcFace model and *this* gallery. Swapping the model or the pack would
  require re-running the calibration.
- **Multi-face mode reused the engine unchanged.** The engine was always
  multi-face capable; only the live wrapper and UI were single-face. No
  recognition logic, threshold, or model changed.
- **Two Python environments exist for a reason.** `face311` runs the product;
  the TensorFlow env exists solely for one evaluation baseline. Don't confuse the
  TF env with the deployed system — TensorFlow is not used at runtime.
```
