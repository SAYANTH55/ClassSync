# ClassSync — Complete Project Knowledge Document

*For college evaluation. Written from the ACTUAL code in this repository. Where
something is not built, it says **NOT IMPLEMENTED**. Where the code is unclear,
it says **NOT VERIFIED FROM THE CODE**. Simple English first, then technical.*

> Quick honesty note up front — things people might *assume* but that are **NOT
> IMPLEMENTED** here: no SQL database (attendance is CSV files), no login/user
> accounts, no SMOTE, no SHAP, no "Feature Shift Index", no cloud. Augmentation
> and neural-network *training* exist **only** in an evaluation baseline script,
> not in the deployed system. Those are explained where relevant.

---

## 1. PROJECT OVERVIEW

**Simple English.** ClassSync is a program that takes attendance in a classroom
using a camera. Students first "enrol" by giving a few phone selfies. Later, a
camera at the front of the class looks at the students, recognises their faces,
checks each face is a **real person and not a photo on a phone**, and marks who
is present — automatically, for many students at once.

**Details:**
- **Exact project name:** ClassSync (the product/brand name shown in the app).
  Folder/GitHub name: "College Attendance recognition".
- **Problem it solves:** manual attendance (roll-call, sheets) is slow and easy
  to cheat (proxy attendance). ClassSync automates it and blocks photo/screen
  cheating.
- **Why important:** it saves class time and makes attendance *trustworthy*
  (you can't mark an absent friend present with a photo).
- **Main objective:** recognise every real, live, enrolled student from one
  classroom camera and record attendance once per day, rejecting unknown people
  and spoofs.
- **Target users:** teachers/lecturers; the students are the subjects.
- **Deployment:** a smart board / laptop with a webcam at the front of a class;
  runs **locally on CPU** (no GPU, no internet needed at run time).
- **What makes it different from a basic face-attendance system:**
  1. **Anti-spoofing** (MiniFASNet) — most student projects skip this; here a
     phone/photo is rejected.
  2. **Multi-face classroom mode** — many students recognised in one frame.
  3. **Open-set rejection** — unknown people are labelled "unknown", never
     guessed.
  4. **Confirmation streaks** — a student must be seen steadily before marking.
  5. **Calibrated threshold + a real evaluation** (100% on a cross-device test).

---

## 2. COMPLETE END-TO-END PIPELINE

Two phases: **(A) Enrolment** (offline, once per student) and **(B) Live
attendance** (every session). For each stage: **in → what happens → file →
model → out → why**.

### Phase A — Enrolment (build the gallery)

| Stage | In | What happens | File / function | Model | Out | Why |
|---|---|---|---|---|---|---|
| Data collection | student phone selfies | photos placed in a per-student folder | `data/raw_sessions/phone_enroll/<Name>/` | — | image files | need example faces to learn each identity |
| Preprocessing | image file | load upright (fix rotation), handle HEIC | `preprocessing.load_image_upright` | — | RGB array | phone photos are rotated/HEIC |
| Face detection | image | find the face + 5 landmarks | `detect_embed.FaceBackend.detect` | **SCRFD** | box+landmarks | locate the face |
| Alignment | image+landmarks | rotate/scale face to 112×112 | insightface `norm_crop` (inside detect) | — | 112×112 face | standard pose helps recognition |
| Embedding | 112×112 face | face → 512 numbers | `detect_embed` | **ArcFace** | 512-D vector | numeric identity |
| Validation | all vectors | one face/image; all same person (centroid check) | `enroll.enroll` | — | pass/reject | prevent wrong-person enrolment |
| Gallery build | all students' vectors | save per-student templates | `gallery.build_gallery` | — | `gallery.npz` + `roster.csv` | the database of known faces |

### Phase B — Live attendance (per camera frame)

| Stage | In | What happens | File / function | Model | Out | Why |
|---|---|---|---|---|---|---|
| Camera input | webcam | browser grabs a JPEG frame (~1.2 s) | `Camera.jsx` | — | JPEG bytes | the picture to analyse |
| Decode | JPEG | bytes → pixel array (BGR) | `ws/camera._process` (`cv2.imdecode`) | — | image array | usable image |
| Face detection | image | find **all** faces + landmarks | `detect_embed.detect` | **SCRFD** | list of faces | many students |
| **Anti-spoofing** | image+box | live person or photo/screen? | `anti_spoof.analyze_face` | **MiniFASNet ×2** | live/spoof | block cheating |
| Gate | liveness | spoof → stop (no embed) | `detect_embed.detect` | — | continue/stop | spoofs must not be recognised |
| Alignment+Embedding | live face | 112×112 → 512-D vector | `detect_embed.detect` | **ArcFace** | 512-D vector | identity vector |
| Similarity | vector | max cosine vs each student's templates | `gallery.Recognizer.scores` | — | score per student | who is closest |
| Threshold | best score | ≥ 0.447 → student, else unknown | `gallery.Recognizer.identify` | — | Match / None | reject unknowns |
| Multi-face | per face | repeat for every face | `ws/camera._process` loop | — | per-face results | classroom |
| Confirmation | per student | streak +1 (−1 if missed); mark at 5 | `ws/camera.ClassroomSession.update` | — | mark or wait | avoid one-frame errors |
| Attendance | student | write one CSV row (once/day) | `take_attendance.AttendanceLog.mark` | — | CSV row | record presence |
| Reports/dashboard | CSV files | read + show numbers | `routers/dashboard.py`, `attendance.py` + React | — | JSON → screen | teacher sees results |

---

## 3. DATA COLLECTION

- **Phone enrolment images** — `data/raw_sessions/phone_enroll/<Name>/`. One
  folder per student, ~5 selfies each. **48 students, 256 face templates total**
  (some have more, e.g. one has 11). **This is the ENROLMENT data** — the source
  of the gallery.
- **DSLR labelled images** — external folder `E:\Attendance_monitoring
  _system_dataset\...\Attendance_monitoring_system`. **53 studio portraits**
  (Sony camera); each filename = a student name, plus `DSC*` files of
  non-enrolled people. **This is EVALUATION data only** — used to *measure*
  accuracy, **never for enrolment**.
- **Pristine DSLR originals** — `E:\Computer_vision_dataset\...` (56 untouched
  camera files). A read-only backup used for integrity/hash audits.
- **Number of classes/identities:** 48 enrolled students (48 identities).
- **Train/validation/test split:** **NOT IMPLEMENTED** for the deployed
  recogniser — because the deployed system does **no training** (it uses
  pretrained models + a gallery). A train/test split exists **only** inside the
  evaluation baseline `evaluate_deep.py` (train a CNN on phone crops, test on
  DSLR) — that is an experiment, not the product.

**ENROLMENT DATA vs EVALUATION DATA (key exam point):**
- **Enrolment data = phone selfies** → turned into the gallery the system
  recognises against.
- **Evaluation data = DSLR portraits** → used once to check "does the phone-built
  gallery recognise the same people from a *different camera*?" This
  cross-device test is why the 100% result is meaningful.
- **DSLR is NOT used during enrolment.** Enrol = phone only; evaluate = DSLR.

---

## 4. DATA PREPROCESSING

| Operation | What | Why | Where in code |
|---|---|---|---|
| Image loading | open the file to a pixel array | need pixels to process | `preprocessing.load_image_upright` |
| EXIF rotation fix | rotate photo to upright using EXIF flag | phone photos are stored sideways | `preprocessing.load_image_upright` (`ImageOps.exif_transpose`) |
| HEIC handling | read iPhone `.heic` files | phone selfies are HEIC | `preprocessing` (`pillow_heif.register_heif_opener`) |
| Color conversion | RGB↔BGR as each model needs | OpenCV/insightface use BGR | `detect_embed.detect` (`img_rgb[:, :, ::-1]`) |
| Face detection | find the face region | isolate the face | SCRFD in `detect_embed.detect` |
| Face crop + alignment | warp face to 112×112 using 5 landmarks | standard pose for ArcFace | insightface `norm_crop` (inside recognition) |
| Anti-spoof crop | expand box, resize 80×80, **raw BGR (no /255)** | MiniFASNet input format | `anti_spoof/preprocessing.py` |
| Resizing (frames) | camera frame → 960 px wide JPEG | balance speed vs catching far faces | `Camera.jsx` (`canvas.width=960`) |
| Augmentation | flips/rotations/brightness | more training variety | **ONLY in `evaluate_deep.py`** (CNN baseline). **NOT** in the deployed path |
| Quality filtering | reject images with no detectable face during enrolment | avoid bad templates | `enroll.enroll` (rejects "no face") |

**Note:** the deployed recogniser does **no explicit normalization** of ArcFace
input beyond alignment — insightface handles it internally.

---

## 5. SCRFD  (FACE **DETECTION** — not recognition)

- **What it stands for:** *Sample and Computation Redistribution for Face
  Detection.* Model file: `det_10g.onnx` (inside the insightface `buffalo_l`
  pack).
- **Why used:** we must find *where* faces are before doing anything else; SCRFD
  is fast on CPU and also gives 5 landmark points (needed for alignment) and
  finds **many faces per frame**.
- **Input:** an image (BGR array).
- **How it detects a face (simple):** it is a convolutional neural network that
  scans the image at **three zoom levels** (to catch near and far faces) and, at
  each location, predicts "is there a face?" and "where exactly?".
- **Bounding boxes:** for each detected face it outputs 4 numbers `[x1,y1,x2,y2]`
  (the rectangle). Overlapping guesses are cleaned by **NMS** (keep the strongest
  box).
- **Landmarks:** 5 points per face — left eye, right eye, nose, left mouth
  corner, right mouth corner.
- **Multiple faces / 5 people in a frame:** SCRFD returns a **list** of all
  detected faces; the code loops over every one. So 5 people → 5 boxes → 5
  independent recognitions.
- **Face crop:** obtained by cropping/aligning the box region to 112×112 using
  the landmarks.
- **Does SCRFD create embeddings?** **NO.** SCRFD only detects (boxes +
  landmarks + a confidence). Embeddings come from ArcFace.

**Never mix these:** **SCRFD = detection (where)**, **ArcFace = embedding /
recognition (who)**, **MiniFASNet = anti-spoofing / liveness (real or fake)**.

---

## 6. FACE ALIGNMENT

- **What alignment means:** rotating/scaling a face so the eyes, nose and mouth
  sit in a **standard position** every time.
- **Why required:** ArcFace was trained on aligned faces; if the face is tilted
  or off-centre, recognition accuracy drops. Alignment removes pose differences.
- **The 5 landmarks:** left eye, right eye, nose tip, left mouth corner, right
  mouth corner (from SCRFD).
- **How used:** the 5 detected points are mapped onto 5 fixed reference points,
  and the image is warped so they line up.
- **Why 112×112:** that is the exact input size the ArcFace model expects.
- **Which function:** insightface's `norm_crop` (called inside the recognition
  step of `FaceBackend.detect`).
- **Simple example:** if Sayanth tilts his head 20°, alignment rotates the crop
  so his eyes are level and 112×112 — the same "passport-photo" pose used for
  everyone, so only *identity* differences remain.

---

## 7. ARCFACE  (FACE **EMBEDDING / RECOGNITION**)

- **What ArcFace does:** turns one aligned face image into **one 512-number
  vector** (an "embedding") that captures identity. Model file: `w600k_r50.onnx`
  — a **ResNet-50** network.
- **Pretrained / "pretrained" meaning:** the model's internal weights were
  already learned by others on a huge dataset (**WebFace-600K**, ~600,000
  identities). "Pretrained" = we download and use those weights as-is.
- **Do we train ArcFace?** **NO.** We only run it forward (inference).
- **What the learned weights represent:** how to convert face pixels into a
  vector where the **same person's faces are close** and **different people are
  far apart** (in angle/cosine).
- **512-dimensional embedding:** it is **ONE vector of 512 numbers per face**
  (not 512 separate embeddings). "512 dimensions" just means the vector has 512
  values.
- **What the 512 numbers mean conceptually:** coordinates in a 512-dimensional
  "identity space." You can't picture it, but closeness works like distance in
  2-D/3-D.
- **One image → one vector:** aligned 112×112 face → ArcFace → e.g.
  `[0.03, −0.11, 0.08, …]` (512 values), then scaled to length 1 (unit vector).
- **Who creates the vector:** **ArcFace**, not SCRFD.
- **How enrolment embeddings are stored:** each student's image embeddings are
  saved in `gallery.npz`, grouped by student ID (these stored vectors are called
  **templates**).
- **How today's camera embedding is made:** the same way — the live face is
  detected, aligned, and passed through ArcFace to get a fresh 512-D vector,
  which is then compared to the stored templates.

**Simple numeric idea:** two photos of the same person → vectors pointing almost
the same way (cosine ≈ 0.8). Two different people → vectors pointing apart
(cosine ≈ 0.25).

---

## 8. GALLERY / ENROLMENT

**What happens when a student uploads, say, 5 images** (`enroll.enroll`):
1. Each image is detected + embedded → **5 embeddings** (512-D each). (One
   embedding per image; if an image has no detectable face it is **rejected**.)
2. **Same-person check (centroid consistency):** compute the average embedding
   (the "centroid"); every image must be within cosine **0.35**
   (`MIN_CENTROID_SIM`) of it. If one photo is a different person, it is far from
   the centroid → the whole enrolment is **rejected**, naming the bad file.
3. **If the student already exists,** new images must also match the existing
   templates (`MIN_MATCH_EXISTING = 0.35`) — prevents enrolling under the wrong
   name.
4. Images are copied into `phone_enroll/<Name>/`.
5. `gallery.build_gallery()` re-embeds everyone and writes `gallery.npz`;
   `build_roster()` assigns a **stable student ID** (S01…; existing IDs never
   change, a new student gets the next free ID) into `roster.csv`.

**Definitions:**
- **Gallery** = the stored set of all enrolled students' embeddings
  (`gallery.npz`).
- **Template** = one stored embedding for a student (they have several).
- **Student ID** = stable pseudonymous label (S01…) linking roster ↔ gallery ↔
  attendance.
- **Identity** = a real person = one student ID.
- **Embedding** = the 512-D vector.
- **Multi-template recognition** = each student is represented by *several*
  templates, and a probe is matched to the **best** of them (see §9).
- **Centroid/consistency check** = the same-person guard above. **Yes, it is
  implemented** (`enroll.py`, `MIN_CENTROID_SIM = 0.35`).

**The recent "Student C" enrolment change:** Student C originally had a DSLR portrait but
**no phone enrolment** (so they weren't in the gallery). Their 11 phone selfies were
added via `enroll.enroll`. Their enrolment was first **rejected** by an older
min-pairwise check (two varied selfies scored 0.325); this was replaced by the
**centroid check** (all 11 scored ≥ 0.636 vs the centroid), which correctly
accepted them. Gallery went from 47 → **48 students / 256 templates**. This is
exactly the consistency logic described above and it is still in the code.

---

## 9. FACE RECOGNITION (worked example)

**Camera sees Sayanth:**
```
camera frame
 → SCRFD detects his face (box + 5 landmarks)
 → MiniFASNet says LIVE (real person)          [see §12]
 → align to 112×112 → ArcFace → 512-D vector (his live embedding)
 → compare to the gallery:
      for EACH student, take the MAX cosine over that student's templates
 → pick the student with the highest score
 → if that score ≥ 0.447 → recognised (Sayanth); else → unknown
```

**What is the probe compared against? (verified in `gallery.py`)**
- It is compared against **every template of every student**.
- For each student, the system keeps the **maximum** cosine over that student's
  templates (`Recognizer.scores`: `np.max(t @ emb)` per student).
- It is **NOT** a centroid at recognition time and **NOT** a single average — it
  is **max-over-templates**. (Centroid is used only during *enrolment* validation,
  not recognition.)
- `Recognizer.identify` then takes the best student, records the runner-up (for
  a "margin"), and accepts only if the best score ≥ threshold.

---

## 10. COSINE SIMILARITY AND THRESHOLD

- **What it is:** cosine similarity = the cosine of the angle between two
  vectors. Because embeddings have length 1, it equals their dot product
  (multiply matching numbers, add them up).
- **Why used:** ArcFace embeddings separate identities by *angle*, so cosine is
  the natural closeness measure.
- **Range:** −1 to 1 (in practice ~0 to 1 here). **High** = same person; **low**
  = different person.
- **Typical values in this system:** same person ≈ 0.61–0.90; different people ≈
  0.25–0.29.
- **Exact threshold:** **τ = 0.447**, defined in `config.RECOG_THRESHOLD`.
- **How calibrated:** from the DSLR cross-device evaluation — it is the **midpoint
  of the empty gap** between genuine scores (0.607–0.896) and impostor scores
  (0.252–0.286). At that value the test had **0 false accepts and 0 false
  rejects**.
- **Above τ:** the face is accepted as that student.
- **Below τ:** the face is **unknown** (open-set rejection) and never marked.
- **Simple example:** live face vs gallery → best score 0.71 (Sayanth) ≥ 0.447 →
  recognised. A stranger → best score 0.28 < 0.447 → "unknown".

---

## 11. MULTI-FACE CLASSROOM MODE (very important)

**The change:** originally the live camera used only the **largest** face and one
counter (a walk-up kiosk). It was changed so that **every** face in the frame is
recognised, each with its **own** counter — for a classroom.

**Why the recognition engine did NOT need changing:** `FaceBackend.detect`
already returned **all** faces, each already embedded; the old code just threw
away all but the largest. So only the live-session wrapper (`ws/camera.py`
`ClassroomSession`) and the UI (`Camera.jsx`) changed.

- **How many faces SCRFD detects:** all of them (a list) — 1, 5, 15, …
- **Per-face processing:** `_process` loops over every face → anti-spoof →
  (if live) embed → identify. Each face judged independently.
- **Per-student confirmation streaks:** `ClassroomSession.streaks` is a
  dictionary `student_id → count`.
- **Why 1/5 … 5/5 appear:** a student must be recognised in **5 frames** before
  being marked. The label shows the current count (`streak`/`needed`).
  `CONFIRM_FRAMES = 5`.
- **Leaves the frame / turns away:** on any frame they are *not* recognised,
  their streak **decays −1** (floored at 0) — it does **NOT** hard-reset. This
  tolerates a brief turn/miss (important for seated students).
- **Reset vs decay:** **decay** (−1 per missed frame), not reset-to-zero.
- **Duplicate prevention:** once marked, a student is in `AttendanceLog.present`
  for the day → never marked again (see §15).
- **Multiple students in one session:** each reaches 5/5 on their own and gets
  their own CSV row; many can be marked in the same session/frame.
- **Unknown people:** score < τ → labelled "unknown", never contribute a streak,
  never marked.

**Frontend behaviour (`Camera.jsx`):** draws a coloured box + label on **each**
face — green "✓ Name" (marked), amber "Name 3/5" (confirming), grey "unknown",
red "spoof — photo/screen". Live counters show *detected / recognized / marked*
(and *N spoof blocked*), plus a "Marked present" side list.

---

## 12. ANTI-SPOOFING (MiniFASNet)

**Why recognition alone can be fooled:** ArcFace looks at a 2-D face image. A
sharp **photo on a phone**, a **printed photo**, or a **digital image/screen**
produces almost the same embedding as the real face → it would be recognised and
marked. That is a "presentation attack." A **video replay** (playing a video of
the person) is the hardest version.

**What MiniFASNet is:** a very small, fast CNN for **face anti-spoofing
(liveness)** — it decides *live person* vs *fake (photo/screen)*.
- **Where it came from:** the open-source *minivision Silent-Face-Anti-Spoofing*
  project; we use ONNX exports from `github.com/yakhyo/face-anti-spoofing`.
- **Pretrained? Trained by us?** Pretrained by others; **we did NOT train it.**
- **Original training data:** large collections of real faces vs presentation
  attacks (prints, screens/replays). The exact dataset details are on the
  upstream project; **not fully specified in our code** → **NOT VERIFIED FROM THE
  CODE** beyond "real vs spoof attacks."
- **Two models (ensemble):** `MiniFASNetV2` (crop scale 2.7) + `MiniFASNetV1SE`
  (crop scale 4.0). Both run; their outputs are averaged.
- **Input it receives:** a face crop, expanded and resized to **80×80**, kept
  **BGR**, fed as **raw pixel values 0–255** (no /255). (Getting this wrong was a
  real bug we fixed.)
- **Output:** 3 numbers → softmax → 3 probabilities. **Class index 1 = live.**
- **How converted to LIVE/SPOOF:** average the two models' probabilities; the
  face is **LIVE** only if class 1 is the largest **and** its probability ≥
  **0.5** (`LIVENESS_THRESHOLD`); otherwise **SPOOF**.
- **Where loaded:** `anti_spoof/model_loader.py` (downloads to
  `models/anti_spoof/` if missing); called by `anti_spoof/inference.analyze_face`.
- **Integration order:** inside `detect_embed.FaceBackend.detect`, anti-spoof
  runs **BEFORE ArcFace**. If a face is a spoof, **ArcFace is skipped** (no
  embedding) → the face can never be recognised or marked.
- **Can attendance be marked for a spoof?** **No.** Spoof → no embedding → no
  match → not marked.
- **If the model fails to load:** **FAIL-CLOSED** (`FAIL_CLOSED = True` in
  `anti_spoof/inference.py`) → faces are **rejected** (nobody marked) rather than
  silently passed. Safer for a security feature.
  - (Historical note: an earlier broken version was fail-**open** with a dead
    model URL, which is why spoofs passed; that was fixed.)
- **The actual phone-screen test:** in the live demo, holding a phone showing a
  face produced a **red "spoof — photo/screen" box, "0 recognized", "1 spoof
  blocked", and no attendance mark** — confirmed working. Real faces
  (webcam-quality) score ~0.9 live and pass.

---

## 13. FASTAPI BACKEND

**Why FastAPI:** it supports **async + WebSockets** (needed to stream camera
frames) and runs in the **same Python environment as the AI models**, so there's
no separate service. It also serves the built React app, so the whole thing runs
from one command.

**Architecture:** `main.py` (app + serves frontend) → routers (REST) + `ws/camera`
(WebSocket) → `deps.py` (loads the engine once, one inference at a time) →
engine (`detect_embed`, `gallery`, `take_attendance`, `enroll`) → files.

**Endpoints (all that exist):**

| Method | Path | Purpose | Input | Output |
|---|---|---|---|---|
| GET | `/api/health` | server/gallery status | — | `{status, engine_loaded, gallery_students, gallery_templates, threshold}` |
| GET | `/api/settings` | model + config info | — | model names, threshold, gallery size, etc. |
| GET | `/api/dashboard` | today's numbers + recent + 14-day chart | — | counts, recent marks, day series |
| GET | `/api/students` | list students | — | `{students:[{student_id,name,enrolled,templates}], enrolled}` |
| POST | `/api/students` | enrol a student | multipart: `name` + `images[]` | `{ok, message}` |
| DELETE | `/api/students/{student_id}` | remove a student | — | `{ok, message}` |
| GET | `/api/attendance/{date}` | present + absent for a day | — | `{date, present[], absent[], total_students}` |
| GET | `/api/attendance/summary` | per-student % over all days | — | `{days_total, first_day, last_day, rows[]}` |
| GET | `/api/attendance/{date}/export` | download day CSV | — | a CSV file |
| DELETE | `/api/attendance/{date}` | clear/archive a day | — | `{ok, cleared, message}` |
| WS | `/ws/camera` | live recognition | binary JPEG frames | JSON per frame (faces, counts, marked) |

**WebSocket flow:** browser opens `/ws/camera` → server sends
`loading_engine` then `ready` → browser sends JPEG frames → server replies per
frame with `{state:"frame", faces:[…], counts:{…}, marked:[…]}`. Inference runs
off the event loop under a shared lock (`deps.inference_lock`).

**No authentication endpoint — NOT IMPLEMENTED** (open app).

---

## 14. REACT FRONTEND

**Structure:** a React single-page app (built by Vite, served by FastAPI).
`App.jsx` = layout (Sidebar + Topbar + routed pages). One API client `api.js`.

**Pages (`src/pages/`):** Dashboard, Camera, Reports, Students, Settings.
**Components (`src/components/`):** Sidebar, Topbar, Logo, Ring.
**Hook (`src/hooks/`):** `useCountUp` (animated numbers).

- **Dashboard:** today's totals, an animated **Ring** (attendance %), a 14-day
  bar chart, quick actions, recent marks. Polls `/api/dashboard` every 15 s.
- **Camera:** opens webcam, streams frames over WebSocket, draws per-face boxes +
  counters + "Marked present" list.
- **Students:** roster table + an "Add student" panel (upload photos → enrol) +
  remove.
- **Reports:** Daily (present/absent, search, Export CSV, Clear day) + Summary
  (per-student %, below-75% flag).
- **Settings:** threshold explained visually, model names, gallery size, privacy
  note (read-only).

**WebSocket communication:** `Camera.jsx` sends JPEG bytes, receives JSON; state
is plain React `useState` (no Redux).

**Visual states on a face:**
1. **Detected:** a box appears around the face.
2. **Unknown:** grey box labelled "unknown".
3. **Being confirmed:** amber box labelled "Name 3/5".
4. **Recognized (confirming):** same amber, streak climbing.
5. **Marked:** green box "✓ Name"; name added to "Marked present".
6. **Spoof:** red box "spoof — photo/screen"; "N spoof blocked" counter; never
   marked.

**Animations:** count-up numbers, hover-lift cards, entrance fade, sidebar
status-dot pulse (Dashboard mainly).

---

## 15. ATTENDANCE LOGIC

- **Storage format:** **CSV files, not a database.** One file per day:
  `data/attendance/attendance_YYYY-MM-DD.csv`. Columns:
  **`time, student_id, name, score`**. (`take_attendance.AttendanceLog`)
- **Fields:** time = HH:MM:SS of marking; student_id = stable ID; name; score =
  the cosine similarity at marking. **Date** is in the filename (one file per
  date).
- **Duplicate prevention / one-record-per-day:** `AttendanceLog` loads the set of
  already-present IDs for the day; `mark()` ignores a student already present →
  exactly one row per student per day.
- **Clearing attendance:** `DELETE /api/attendance/{date}` **archives** (moves)
  the day's CSV to `data/attendance/_archive/` — recoverable, not hard-deleted.
- **Archive behaviour:** reports only read top-level daily files, so archived
  files never reappear.
- **Reports generation:** `attendance_report.py` (and the API routers) read the
  daily CSVs + roster to produce day views, absentee lists, and a period summary.

---

## 16. EVALUATION

All evaluation is **cross-device**: gallery built from **phone** selfies, tested
on **DSLR** portraits (different camera, months apart) — an honest test of the
real deployment gap.

**Headline result (`evaluate_dslr.py`):**
- **Rank-1 identification: 46/46 = 100%.**
- **TAR@FAR=0 = 1.000** (all genuine accepted, zero false accepts).
- Genuine scores 0.607–0.896 vs impostor 0.252–0.286 → a clean **0.32-wide gap**;
  τ = 0.447 sits in the middle.

**Baselines under the identical protocol (fair comparison):**
| Method | Rank-1 | Separated? |
|---|---|---|
| Eigenfaces (classical, `evaluate_classical.py`) | 26.1% | no |
| LBPH (classical, own NumPy impl) | 34.8% | no |
| Scratch CNN (deep, `evaluate_deep.py`) | 4.3% | no |
| MobileNetV2 transfer (deep) | 17.4% | no |
| **ArcFace (deployed)** | **100%** | **yes** |

**Bias analysis (`analyze_bias.py`):** per-student "Doddington-zoo" audit — all
46 decision margins positive; no student near the threshold. (No demographic
labels are inferred — stated as a limitation.)

**Latency (`profile_pipeline.py`, CPU):** detection ~230 ms, embedding ~130 ms
per face, matching ~0.6 ms; ~2 FPS single-face. Multi-face ≈ 230 + K×130 ms.

**Metrics in simple English:**
- **Accuracy / Rank-1** — % of probes whose top match is the correct person.
- **TAR (True Accept Rate)** — % of genuine faces correctly accepted.
- **FAR (False Accept Rate)** — % of impostors wrongly accepted. "TAR@FAR=0" =
  TAR when we allow zero false accepts.
- **Open-set separation** — the *gap* between genuine and impostor scores; bigger
  = safer.
- **Decision margin** — best score minus runner-up; how confident each decision
  is.
- **Latency** — time per frame/stage.
- **Precision / Recall / F1 / ROC — NOT the primary metrics reported** here (the
  evaluation uses rank-1, TAR/FAR, separation, margin). **NOT IMPLEMENTED** as
  named outputs.

---

## 17. MODEL DETAILS

| Component | Model (file) | Purpose | Pretrained? | Trained by us? | Input | Output |
|---|---|---|---|---|---|---|
| Detector | **SCRFD** (`det_10g.onnx`) | find faces + landmarks | Yes (WIDER FACE) | No | image (BGR) | boxes + 5 landmarks + score |
| Embedder | **ArcFace ResNet-50** (`w600k_r50.onnx`) | face → identity vector | Yes (WebFace-600K) | No | 112×112 aligned face | 512-D unit embedding |
| Anti-spoof #1 | **MiniFASNetV2** (scale 2.7) | live vs spoof | Yes (minivision) | No | 80×80 raw BGR | 3-class logits |
| Anti-spoof #2 | **MiniFASNetV1SE** (scale 4.0) | live vs spoof (ensemble) | Yes (minivision) | No | 80×80 raw BGR | 3-class logits |

Other models in the `buffalo_l` pack (106-pt landmarks, 68-pt 3D, gender/age)
exist on disk but are **NOT loaded** (only detection + recognition are used).
The scratch CNN / MobileNetV2 in `evaluate_deep.py` are **evaluation baselines
trained by us**, **not part of the deployed system**.

---

## 18. ALGORITHMS

- **SCRFD detection** — purpose: find faces; chosen: fast + landmarks +
  multi-face; in: image; out: boxes+landmarks. Simple: a CNN that scans at three
  scales. Technical: anchor-based single-stage detector with feature-pyramid
  heads + NMS.
- **Face alignment (`norm_crop`)** — purpose: standard pose; in:
  image+5 landmarks; out: 112×112 crop. Simple: rotate/scale so eyes/nose line
  up. Technical: similarity transform to fixed reference points.
- **ArcFace embedding** — purpose: identity vector; in: 112×112 face; out: 512-D.
  Simple: face → 512 numbers. Technical: ResNet-50 trained with additive angular
  margin loss.
- **MiniFASNet anti-spoof (×2 ensemble)** — purpose: liveness; in: 80×80 BGR;
  out: live/spoof. Simple: is it a real face or a screen? Technical: small CNN,
  3-class softmax, averaged over two crop scales.
- **Cosine similarity + max-over-templates + threshold** — purpose: recognition
  decision; in: probe vector + gallery; out: student/unknown. Simple: closest
  known face if close enough. Technical: max cosine per identity, open-set accept
  at τ.
- **Confirmation streak (with decay)** — purpose: stable marking; in: per-frame
  matches; out: mark/wait. Simple: seen 5 frames → mark. Technical: per-student
  counter, +1 on hit, −1 on miss, mark at CONFIRM_FRAMES.
- **Centroid consistency (enrolment)** — purpose: same-person guard; Simple:
  every photo must look like the average; Technical: min cosine to the mean
  embedding ≥ 0.35.

---

## 19. FILE-BY-FILE EXPLANATION

**`src/config.py`** — PURPOSE: all paths + constants (τ=0.447, model names,
folders, session registry). FUNCTIONS: `session_images`, `find_image`. CONNECTS:
imported by nearly everything.

**`src/preprocessing.py`** — PURPOSE: image loading. FUNCTION:
`load_image_upright` (EXIF-upright RGB, HEIC). CONNECTS: enrol/evaluate/engine.

**`src/detect_embed.py`** — PURPOSE: the detector+anti-spoof+embedder wrapper.
FUNCTIONS: `FaceBackend.detect` (SCRFD → anti-spoof → ArcFace, per-face logging),
`embed_best`. CONNECTS: `anti_spoof`, `gallery`, `ws/camera`, `enroll`.

**`src/gallery.py`** — PURPOSE: roster + gallery + recogniser. FUNCTIONS:
`build_roster`, `build_gallery`, `Recognizer.scores` (max cosine per student),
`Recognizer.identify` (best ≥ τ else None). CONNECTS: camera loop, enrol, CLI.

**`src/take_attendance.py`** — PURPOSE: attendance log + CLI kiosk. FUNCTIONS:
`AttendanceLog` (per-day CSV, dedup), photo/webcam kiosk (**single-face** CLI).
CONNECTS: backend uses `AttendanceLog`.

**`src/attendance_report.py`** — PURPOSE: reporting. FUNCTIONS: `day_files`,
`expected_students`, day/summary printing. CONNECTS: reports router + CLI.

**`src/enroll.py`** — PURPOSE: add/remove a student. FUNCTIONS: `enroll`
(validate + centroid check + rebuild), `remove`; raises `EnrollmentError`.
CONNECTS: students router + CLI.

**`src/smoke_test.py`** — PURPOSE: 14-check end-to-end verification (load,
detect, embed, identify, reject impostor, mark, dedup). CONNECTS: whole engine.

**`src/dslr_check.py`** — PURPOSE: dataset audit (hashes, identity reconciliation
between DSLR and phone). CONNECTS: reports.

**`src/evaluate_dslr.py`** — PURPOSE: the headline cross-device evaluation (rank-1,
TAR/FAR sweep, τ). CONNECTS: gallery + DSLR probes.

**`src/evaluate_classical.py`** — PURPOSE: Eigenfaces + LBPH baselines on the
identical protocol. **`src/evaluate_deep.py`** — PURPOSE: scratch-CNN +
MobileNetV2 baselines (**this is the only place training happens**, and only for
baselines). **`src/build_crops_rgb.py`** — bridges aligned crops to the TF env.

**`src/profile_pipeline.py`** — PURPOSE: per-stage latency measurement.

**`src/analyze_bias.py`** — PURPOSE: per-identity (Doddington-zoo) audit.

**`src/anti_spoof/`** — `utils.py` (models, scales, LIVE_CLASS=1, threshold 0.5),
`preprocessing.py` (crop + raw BGR 80×80), `model_loader.py` (download/load both
ONNX), `inference.py` (`analyze_face`, ensemble, `FAIL_CLOSED=True`).

**Backend (`webapp/backend/`)** — `main.py` (app + SPA), `deps.py` (engine
singletons + lock), `schemas.py` (response shapes), `routers/{dashboard,
attendance, students, settings}.py`, `ws/camera.py` (`ClassroomSession`,
`_process`, `/ws/camera`).

**Frontend (`webapp/frontend/src/`)** — `main.jsx`, `App.jsx`, `api.js`,
`pages/*` (Dashboard, Camera, Reports, Students, Settings),
`components/*` (Sidebar, Topbar, Logo, Ring), `hooks/useCountUp.js`.

---

## 20. YOUR CONTRIBUTION (for evaluation)

**Pretrained components (NOT your work — used as tools):** SCRFD, ArcFace,
MiniFASNet models (their weights).

**Your engineering / research contribution (what you actually built):**
- **System integration** — wiring SCRFD → MiniFASNet → ArcFace → recognition →
  attendance into one working pipeline (`detect_embed.py`, `ws/camera.py`).
- **Data preparation** — collecting phone selfies + DSLR portraits; auditing and
  reconciling identities (`dslr_check.py`); fixing a real duplicate-folder data
  bug found via embeddings.
- **Enrolment pipeline** — validated add/remove with the **centroid consistency
  check** (`enroll.py`).
- **Gallery design** — stable-ID roster + **multi-template, max-cosine** gallery
  (`gallery.py`).
- **Threshold calibration** — deriving τ = 0.447 from the score distributions,
  not guessing.
- **Open-set rejection** — the "unknown" logic.
- **Multi-face classroom mode** — per-student streaks with decay (`ClassroomSession`).
- **Confirmation logic** — the 5-frame stability rule.
- **Anti-spoofing integration** — sourcing real models, fixing preprocessing (the
  raw-BGR bug), ensembling, fail-closed gating **before** ArcFace.
- **Attendance management** — per-day CSV, dedup, clear/archive, reports.
- **Evaluation protocol + baseline comparison** — the cross-device test and the
  5-method comparison table.
- **Frontend + backend development** — the FastAPI server and the React web app.
- **Privacy handling** — pseudonymous IDs, git-ignored data, scrubbed names,
  private repo.
- **Deployment** — one-command CPU run.

**Honest framing for the panel:** "I did not train the face models — using
pretrained ArcFace is the *correct engineering choice* because it lets me enrol a
new student by storing a vector, with no retraining. My contribution is the
*system*: the data pipeline, the gallery and calibrated open-set recognition, the
multi-face classroom logic, the anti-spoofing integration, the evaluation, and
the full web application."

---

## 21. LIMITATIONS (supported by the project)

- **CPU speed:** ~2 FPS single-face; slower with many faces (≈230+K×130 ms) — a
  seated-classroom sweep, not real-time video.
- **Small/distant (back-row) faces:** SCRFD may miss them; higher resolution
  costs speed.
- **Profile / turned faces:** ArcFace accuracy drops for strong side views; the
  student is caught when they face forward.
- **Anti-spoofing is passive 2-D:** strong against photos/screens, but a very
  high-quality **video replay** on a large glare-free screen is the hardest case
  — not guaranteed. Validated on real-face *acceptance*; spoof *rejection*
  confirmed by the live phone test but not exhaustively benchmarked.
- **Lighting / camera quality:** thresholds (0.447, 0.5) are tuned for this
  camera type; a very different camera may need re-tuning.
- **Similar-looking people / twins:** not specifically tested — **NOT VERIFIED
  FROM THE CODE**.
- **Fail-closed cost:** if the anti-spoof model is missing, nobody is marked.
- **No SQL database / no auth:** fine for one classroom, not for many concurrent
  users. **NOT IMPLEMENTED.**
- **Occlusion (mask/hand):** not specifically handled — **NOT VERIFIED FROM THE
  CODE**.

---

## 22. COMMON EVALUATION QUESTIONS (40+), short + detailed

**A. Basic project**
1. *What is your project?* — Short: "A face-recognition attendance system for
   classrooms with anti-spoofing." Detailed: enrol from phone selfies; a camera
   recognises many live students at once and marks attendance, rejecting unknowns
   and photos.
2. *Why is it useful?* — Short: "Fast, and can't be cheated with a photo."
   Detailed: automates roll-call and blocks proxy attendance via liveness.
3. *Does it need internet/GPU?* — Short: "No, it runs locally on CPU." Detailed:
   models are local ONNX; first run downloads them once.

**B. Computer vision**
4. *Detection vs recognition?* — Short: "Detection finds where faces are;
   recognition finds who." Detailed: SCRFD detects, ArcFace recognises.
5. *What is an embedding?* — Short: "A 512-number vector representing a face."
   Detailed: same person → close vectors, different → far.

**C. SCRFD**
6. *What is SCRFD?* — Short: "The face detector." Detailed: single-stage CNN,
   3 scales, outputs boxes + 5 landmarks; pretrained on WIDER FACE.
7. *Does SCRFD recognise faces?* — Short: "No, only detects." Detailed:
   recognition is ArcFace's job.
8. *5 people in a frame?* — Short: "It returns all 5 faces." Detailed: code loops
   over each independently.

**D. ArcFace**
9. *What is ArcFace?* — Short: "The model that turns a face into a 512-D vector."
   Detailed: ResNet-50 trained with angular-margin loss on WebFace-600K.
10. *Did you train it?* — Short: "No, it's pretrained." Detailed: I use it for
    inference; that's why enrolment needs no retraining.
11. *Why pretrained is fine?* — Short: "It generalises to new people." Detailed:
    embeddings separate identities it never saw in training.

**E. Embeddings**
12. *512 embeddings or one?* — Short: "One 512-number vector per face." Detailed:
    512 = the vector's length.
13. *Where are enrolment embeddings stored?* — Short: "In gallery.npz per
    student." Detailed: several templates per student.

**F. Cosine similarity**
14. *What is cosine similarity?* — Short: "How aligned two vectors are, −1 to 1."
    Detailed: dot product of unit vectors; high = same person.
15. *Typical values?* — Short: "Same ≈0.6–0.9, different ≈0.25." Detailed: from
    our evaluation.

**G. Threshold**
16. *What threshold?* — Short: "0.447." Detailed: in `config.py`; calibrated as
    the midpoint between genuine and impostor scores (0 false accept/reject).
17. *Below threshold?* — Short: "Unknown, not marked." Detailed: open-set
    rejection.

**H. Anti-spoofing**
18. *How do you stop a phone photo?* — Short: "MiniFASNet checks liveness before
    recognition." Detailed: spoof → ArcFace skipped → never marked.
19. *Which model / trained by you?* — Short: "MiniFASNet, pretrained, not trained
    by me." Detailed: ensemble of two variants, 80×80 raw BGR, class 1 = live.
20. *Fail-open or fail-closed?* — Short: "Fail-closed." Detailed: no model →
    reject everyone (safe).
21. *Can a spoof be marked?* — Short: "No." Detailed: no embedding is even made.
22. *Video replay?* — Short: "Harder; passive model may not catch a perfect
    replay." Detailed: honest limitation.

**I. FastAPI**
23. *Why FastAPI?* — Short: "Async + WebSockets, same env as the AI." Detailed:
    streams frames and serves the React app from one server.
24. *How does the camera talk to the server?* — Short: "A WebSocket sends JPEG
    frames, gets JSON back." Detailed: `/ws/camera`.

**J. React**
25. *What does the frontend do?* — Short: "Shows the camera, boxes, counters, and
    reports." Detailed: 5 pages; live states drawn per face.
26. *State management?* — Short: "Simple React state, no Redux." Detailed: server
    is the source of truth.

**K. Multi-face**
27. *How are many students handled?* — Short: "Every face is recognised with its
    own 5-frame counter." Detailed: `ClassroomSession` per-student streaks.
28. *Why 3/5?* — Short: "Seen in 3 of 5 needed frames." Detailed: stability rule.
29. *Turns away?* — Short: "Counter decays, doesn't reset." Detailed: −1 per
    missed frame.
30. *Double marking?* — Short: "Prevented — one row per student per day."

**L. Dataset**
31. *Enrolment vs evaluation data?* — Short: "Enrol = phone selfies; evaluate =
    DSLR portraits." Detailed: DSLR is a different-camera test set only.
32. *How many students/images?* — Short: "48 students, 256 templates." Detailed:
    ~5 selfies each.
33. *Is DSLR used to enrol?* — Short: "No, evaluation only."
34. *Train/test split?* — Short: "The deployed system doesn't train, so no split;
    a split exists only in the CNN baseline experiment."

**M. Evaluation**
35. *Main result?* — Short: "100% rank-1, zero false accepts, cross-device."
    Detailed: genuine 0.61–0.90 vs impostor 0.25–0.29.
36. *Baselines?* — Short: "Eigenfaces 26%, LBPH 35%, scratch-CNN 4%, MobileNetV2
    17%, ArcFace 100%." Detailed: identical protocol.
37. *What is TAR/FAR?* — Short: "Accept rate for genuine / for impostors."
    Detailed: we report TAR at zero FAR = 1.0.

**N. Contribution**
38. *What did YOU build if models are pretrained?* — Short: "The whole system —
    pipeline, gallery, calibration, multi-face, anti-spoof integration, web app,
    evaluation." Detailed: see §20.

**O. Limitations / future**
39. *Main limitations?* — Short: "CPU speed, back-row/profile faces, passive
    anti-spoof, no DB/auth." Detailed: see §21.
40. *Future work?* — Short: "Database, login, stronger liveness, GPU, analytics."
    Detailed: see roadmap in the other docs.
41. *Why not train your own face model?* — Short: "Too little data; pretrained +
    embeddings is the correct, standard choice." Detailed: the scratch-CNN
    baseline scored 4% — proving the point.

---

## 23. 2-MINUTE EXPLANATION (speak naturally)

"My project is ClassSync, a face-recognition attendance system for classrooms.
The problem I wanted to solve is that normal attendance is slow and easy to fake
— someone can mark a friend present. So I built a system where a camera at the
front of the class takes attendance automatically, and it also checks that each
face is a real person and not a photo on a phone.

It works in two parts. First, enrolment: each student gives about five phone
selfies. I run a face detector called SCRFD to find the face, then a pretrained
model called ArcFace turns each face into a list of 512 numbers — an embedding —
that represents their identity. I store these in a gallery, and I check all the
photos are the same person before saving.

Second, live attendance: the camera sends frames to my server. For every face,
I first run an anti-spoofing model called MiniFASNet, which decides if the face
is a live person or a photo/screen. If it's a spoof, I stop right there — I don't
even recognise it. If it's live, I make its embedding and compare it to the
gallery using cosine similarity. If the best match is above my threshold of
0.447, that's the student; if not, it's 'unknown'. To avoid mistakes, a student
must be recognised in five frames before I mark them, and attendance is stored
once per day in a CSV file. It handles many students at once, each with their own
counter.

I didn't train the face models — using pretrained models is the right choice
because I can add a new student just by storing their vector, no retraining. My
contribution is the whole system: the data pipeline, the gallery and the
calibrated recognition, the multi-face classroom logic, the anti-spoofing
integration, and a full web app with FastAPI and React. I also evaluated it: on
a cross-device test — enrol on phones, test on a DSLR camera — it got 100%
recognition with zero false accepts, and I compared it against four other
methods that all did much worse."

---

## 24. 30-SECOND VERSION

"ClassSync takes classroom attendance with a camera. Each student enrols with a
few selfies. In class, for every face the camera sees, I first check it's a real
person and not a photo — using MiniFASNet — then recognise it with ArcFace face
embeddings and cosine similarity, and mark them present after five steady frames.
It handles many students at once, rejects unknown people, blocks phone-photo
spoofs, and scored 100% on a cross-device test. The face models are pretrained;
my work is the whole system around them."

---

## 25. PIPELINE TO MEMORIZE (corrected to the real code)

Your draft pipeline is **correct** — with two clarifications marked `*`:

```
DATA COLLECTION            (phone selfies for enrol; DSLR only for evaluation*)
        ↓
PREPROCESSING              (upright load, HEIC, BGR)
        ↓
SCRFD                      (detect all faces + 5 landmarks)
        ↓
MINIFASNET                 (*liveness runs BEFORE ArcFace; spoof → STOP)
        ↓
ALIGNMENT                  (5 landmarks → 112×112)   [*done just before ArcFace]
        ↓
ARCFACE                    (face → 512-D embedding)
        ↓
512-D EMBEDDING
        ↓
COSINE SIMILARITY          (max over each student's templates)
        ↓
THRESHOLD                  (≥ 0.447 → student, else UNKNOWN)
        ↓
RECOGNITION
        ↓
CONFIRMATION               (per-student streak, mark at 5/5, decay on miss)
        ↓
ATTENDANCE                 (one CSV row per student per day)
        ↓
REPORTS / DASHBOARD
```

`*` Notes vs your draft: (1) **DSLR is evaluation-only**, not part of the live
pipeline; (2) **MiniFASNet comes before ArcFace** (order matters — spoofs never
reach ArcFace); alignment happens right before ArcFace.

---

## WHAT I MUST KNOW FOR EVALUATION (top 20)

1. **Name:** ClassSync — face-recognition classroom attendance with anti-spoofing.
2. **Three models, three jobs:** SCRFD = **detect**, ArcFace = **recognise**,
   MiniFASNet = **anti-spoof**. Never mix them.
3. All three are **pretrained; I did not train them** (only inference).
4. **Enrolment = phone selfies; evaluation = DSLR** (different camera). DSLR is
   never used to enrol.
5. **48 students, 256 templates**, stored in `gallery.npz`; IDs in `roster.csv`.
6. One face → **one 512-D embedding** (ArcFace), unit length.
7. Recognition = **max cosine over each student's templates**, accept if **≥
   0.447**, else **unknown**.
8. **τ = 0.447** is **calibrated** (midpoint of genuine vs impostor scores), not
   guessed.
9. **MiniFASNet runs BEFORE ArcFace**; a spoof is never embedded, never marked.
10. Anti-spoof = **ensemble of two MiniFASNets**, 80×80 **raw BGR** (no /255),
    class 1 = live, threshold 0.5, **fail-closed**.
11. **Multi-face classroom mode:** every face recognised; **per-student streaks**;
    mark at **5/5**; **decay** (−1) on a missed frame, not reset.
12. **Attendance = CSV files** (`time,student_id,name,score`), one row per student
    per day; **no SQL database**.
13. **Open-set rejection:** unknown people are labelled unknown, never marked.
14. **Enrolment has a centroid consistency check** (all photos same person,
    ≥ 0.35); the Student C case is the example.
15. **Result: 100% rank-1, TAR@FAR=0 = 1.0**, cross-device; baselines 4–35%.
16. **Backend = FastAPI** (REST + `/ws/camera` WebSocket); **frontend = React**;
    runs from one command on **CPU**.
17. **My contribution = the whole system** around pretrained models (see §20).
18. **Face states in UI:** detected → confirming (amber n/5) → marked (green) /
    unknown (grey) / spoof (red).
19. **Limitations:** CPU ~2 FPS, back-row/profile faces, passive anti-spoof vs
    perfect video replay, no DB/auth.
20. **Honest line:** "Using pretrained ArcFace is the correct engineering choice
    — it lets me add a student by storing a vector with no retraining; my work is
    the system, the calibration, the classroom + anti-spoof logic, and the
    evaluation."
