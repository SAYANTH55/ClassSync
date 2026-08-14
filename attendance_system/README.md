# Face Recognition Attendance System

End-to-end computer vision project: enroll students from phone selfies, then
mark class attendance from a photo or a live webcam, with open-set rejection
of unknown people. Built around a pre-trained detection + embedding stack
(SCRFD + ArcFace via insightface `buffalo_l`), evaluated against classical
and deep-learning baselines under an identical protocol.

## The deployed pipeline

```
photo / webcam frame
  -> SCRFD          face detection + 5 landmarks        src/detect_embed.py
  -> norm_crop      alignment to canonical 112x112
  -> ArcFace        face -> 512-d unit embedding        src/detect_embed.py
  -> max-cosine     vs multi-template gallery           src/gallery.py
  -> threshold      tau = 0.447 (calibrated, config.py) accept / unknown
  -> attendance     one CSV row per student per day     src/take_attendance.py
```

## Daily usage

```bash
# mark attendance from photos (handles group photos)
python src/take_attendance.py photo1.jpg photo2.jpg

# live kiosk: auto-marks after 5 consistent frames; SPACE=force, ESC=quit
python src/take_attendance.py --webcam

# who was present / absent today (or any day, or overall)
python src/attendance_report.py
python src/attendance_report.py 2026-07-23
python src/attendance_report.py --summary

# enroll a new student (validates: one face per image, same person, and
# consistency with any existing enrollment) then rebuilds the gallery
python src/enroll.py "Full Name" img1.jpg img2.heic
python src/enroll.py --remove "Full Name"

# prove the whole chain works (run after any change)
python src/smoke_test.py
```

## Repository layout

```
src/
  config.py               all paths/constants; session registry; tau
  preprocessing.py        image loading (EXIF-upright, HEIC), geometry helpers
  detect_embed.py         FaceBackend: SCRFD detect + ArcFace embed (CPU)
  gallery.py              roster (stable S-ids) + multi-template gallery
                          + open-set Recognizer
  take_attendance.py      kiosk: photo mode + webcam mode + attendance log
  attendance_report.py    day view, absentee list, period summary
  enroll.py               add/remove a student safely
  smoke_test.py           14-check end-to-end verification
  dslr_check.py           dataset audit (hashes, identity reconciliation)
  evaluate_dslr.py        headline eval: DSLR probes vs phone gallery
  evaluate_classical.py   Eigenfaces + LBPH (own impl) on identical protocol
  build_crops_rgb.py      RGB crop cache bridging to the TF environment
  evaluate_deep.py        scratch CNN + MobileNetV2 baselines (TF env)
  analyze_bias.py         per-identity audit (Doddington-zoo framing)
  profile_pipeline.py     per-stage latency measurement
data/
  raw_sessions/phone_enroll/<Name>/   enrollment selfies (one folder each)
  labels/roster.csv                   student_id,name (ids never renumber)
  processed/gallery.npz               245 ArcFace templates, 47 students
  cache/                              embedding + crop caches (rebuildable)
  attendance/                         attendance_YYYY-MM-DD.csv logs
reports/                              all evaluation figures and CSVs
```

External data (not in repo): labelled DSLR probe set and pristine originals —
paths registered in `config.SESSIONS`.

## Results (cross-device: phone-built gallery, DSLR probes, 4-month gap)

| Method | Rank-1 | TAR @ FAR=0 | Open-set separation |
|---|---|---|---|
| Eigenfaces (PCA) | 26.1% | 0.087 | none |
| LBPH (own numpy impl) | 34.8% | 0.043 | none |
| Scratch CNN (245 imgs) | 4.3% | 0.000 | none |
| MobileNetV2 (ImageNet, frozen) | 17.4% | 0.043 | none |
| ArcFace (deployed) | 100% | 1.000 | 0.32-wide score gap |

Per-identity audit: all 46 decision margins positive (min 0.311); no student
near threshold in either direction (`reports/bias_audit/`).

## Environment

Windows 11. Two pinned Python environments (`requirements.txt`):

| Env | Path | Used for |
|---|---|---|
| face stack (Py 3.11) | `E:\amlenvs\face311` | everything except deep baselines |
| training stack (Py 3.13) | miniconda base | `evaluate_deep.py` only |

Model weights cache: `E:\amlenvs\insightface_models` (`buffalo_l`; only the
detection + recognition models are loaded — see `detect_embed.py`).

## Ethics & data protection

Images are identifiable personal data of consenting classmates: kept out of
version control, never uploaded, referenced by pseudonymous stable IDs
(`S01`...) in all logs and reports. Real names exist only in `roster.csv`
and enrollment folder names on the local machine.
