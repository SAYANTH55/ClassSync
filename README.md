# ClassSync — AI-Powered Attendance Management System

ClassSync is a web-based attendance system that marks students present automatically using face recognition from a single classroom camera. The teacher signs in, opens the Camera page, and every registered student in view is recognised — no roll call and no paper sheet.

Before a face is recognised, ClassSync checks it is a real, live person and not a photo on a phone or a printed picture (anti-spoofing). Unknown people are rejected instead of guessed, and each student is marked only once per day. The whole system runs on an ordinary computer's CPU.

> MSc Artificial Intelligence and Machine Learning, Department of Computer Science, Christ University (2026)
> Team: Sayanth (2548549), Deon (2548519), Noel (2548537) · Supervisor: Prof. Nizar Banu

The full report is in [attendance_system/docs/PROJECT_REPORT.md](attendance_system/docs/PROJECT_REPORT.md). The code and detailed technical README are in [attendance_system/](attendance_system/).

## How it works

```
Login -> Dashboard -> Camera
  -> SCRFD        face detection (every face in the frame)
  -> MiniFASNet   anti-spoofing; a spoof is stopped here, before recognition
  -> ArcFace      face -> 512-number embedding
  -> Matching     cosine similarity vs registered students; accept if >= 0.447
  -> Confirm      student recognised in 5 frames
  -> Record       one row per student per day (attendance_YYYY-MM-DD.csv)
  -> Reports      daily and summary views, CSV export
```

On the Camera page each face is coloured: green when marked, amber while confirming ("Name 3/5"), grey for unknown, red for a spoof.

## Tech stack

| Layer | Technology |
| --- | --- |
| AI models | SCRFD (detection), ArcFace ResNet-50 (recognition), MiniFASNetV2 + MiniFASNetV1SE (anti-spoofing) |
| AI runtime | insightface, ONNX Runtime, OpenCV, NumPy, Pillow + pillow-heif |
| Backend | Python 3.11, FastAPI, Uvicorn, WebSocket (`/ws/camera`) |
| Security | Starlette SessionMiddleware (signed HttpOnly cookie), python-dotenv |
| Frontend | React 19, Vite, React Router, Tailwind CSS, GSAP, lucide-react |
| Storage | CSV files and NumPy `.npz` (no SQL database) |
| Packaging | Docker (multi-stage build) |
| Evaluation | scikit-learn, Matplotlib |

## Results

In a cross-device test (gallery built from phone selfies, tested on DSLR studio photos), the correct student was the top match for every test image (rank-1 = 100%) with zero false accepts at τ = 0.447. ArcFace clearly outperformed Eigenfaces, LBPH, a from-scratch CNN and a transfer-learning baseline. Photo-on-phone spoof attempts were blocked in live testing.

## Run it

```bash
cd attendance_system

# build the frontend
cd webapp/frontend && npm install && npm run build && cd ../..

# install Python packages and set up config
pip install -r requirements.txt
cp .env.example .env   # then set the values

# run the backend (serves API + frontend) at http://localhost:8000
uvicorn webapp.backend.main:app --host 0.0.0.0 --port 8000
```

Or with Docker:

```bash
cd attendance_system
docker build -t classsync .
docker run -p 8000:8000 --env-file .env classsync
```

## Future work

- Individual user accounts and role-based access (teacher / admin)
- PostgreSQL instead of file storage
- Cloud deployment (e.g. AWS EC2 with HTTPS)
- GPU acceleration for larger classes
- Better recognition in poor lighting and for distant or side faces
- Active liveness checks (blink / head-turn) against video replay
- Notifications to absent students or parents, and attendance analytics
- Mobile app
