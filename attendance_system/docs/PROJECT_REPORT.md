# ClassSync — AI-Powered Attendance Management System

**Project Report**

| Field | Details |
| --- | --- |
| Team | Sayanth (2548549), Deon (2548519), Noel (2548537) |
| Programme | MSc Artificial Intelligence and Machine Learning, Department of Computer Science, Christ University |
| Supervisor | Prof. Nizar Banu |
| Academic year | 2026 |

## 1. What ClassSync is

ClassSync is a web-based attendance system that marks students present automatically using face recognition from a single classroom camera. The teacher signs in, opens the Camera page, and the system recognises every registered student in view — no roll call and no paper sheet.

It solves the problems of manual attendance. Roll calls waste class time in large classes, a student can answer for an absent friend (proxy attendance), and paper records can be lost, edited or miscounted. ClassSync replaces this with a camera-based record that is fast, consistent and hard to cheat.

The core of the system is a three-model computer-vision pipeline. SCRFD finds every face in a frame. MiniFASNet then checks that each face is a real, live person and not a photo on a phone or a printed picture — this anti-spoofing step runs before recognition, so a spoof never reaches it. ArcFace turns each live face into a 512-number embedding and compares it with the stored embeddings of registered students using cosine similarity.

A match is accepted only above a threshold of 0.447; anything lower is marked "unknown" rather than guessed. A student must be recognised across 5 frames before being marked present, and is marked only once per day. Records are saved to daily files and shown on a dashboard and in daily and summary reports, all behind a login. The whole system runs on an ordinary computer's CPU, with no graphics card needed.

## 2. How we developed it

We built ClassSync in layers: a computer-vision engine first, then a FastAPI backend around it, then a React frontend, and finally Docker packaging. We used pre-trained models rather than training our own, so adding a new student never requires re-training.

1. **Recognition engine.** We combined SCRFD (`det_10g.onnx`) for detection, ArcFace ResNet-50 (`w600k_r50.onnx`) for 512-D embeddings, and an ensemble of MiniFASNetV2 and MiniFASNetV1SE for anti-spoofing (live score must be at least 0.5). All models run on the CPU through ONNX Runtime and OpenCV.
2. **Student enrolment and gallery.** Students are registered from phone photos (including iPhone HEIC). Each photo becomes an embedding "template", and all templates are stored in `gallery.npz`, linked to a stable student ID in `roster.csv`.
3. **Threshold tuning and evaluation.** We tested cross-device: the gallery was built from phone selfies and tested on DSLR studio photos of the same students. The correct student was the top match for every test image (rank-1 = 100%), with zero false accepts at τ = 0.447. ArcFace clearly beat the baselines we compared — Eigenfaces, LBPH, a from-scratch CNN and a transfer-learning model.
4. **Attendance logic.** We added a per-student confirmation streak of 5 frames; a missed frame lowers the streak by one instead of resetting it, so a brief head-turn is tolerated. Each student is written once per day to `attendance_YYYY-MM-DD.csv`, and cleared days are archived, not deleted.
5. **Backend.** FastAPI exposes REST endpoints under `/api/*` (auth, dashboard, students, attendance, reports, settings) and a WebSocket at `/ws/camera`. The browser streams JPEG frames, and the server replies with each face's box, name and state, about once per second.
6. **Security.** Login is enforced by the backend with a signed HttpOnly session cookie. Credentials come from environment variables or a SHA-256 hash, never plaintext. Every sensitive route and the WebSocket return "401 Not authenticated" without a session. Student photos, data and the real `.env` are git-ignored.
7. **Frontend.** A React 19 single-page app with Login, Dashboard, Camera, Students, Reports and Settings pages. The Camera page colours each face: green when marked, amber while confirming ("Name 3/5"), grey for unknown, red for spoof.
8. **Packaging.** A multi-stage Dockerfile builds the React app with Node.js, then runs it with the backend on a slim Python 3.11 image, served by Uvicorn on port 8000.

Testing confirmed login, logout, protected routes, detection, recognition, unknown-face rejection, phone-screen spoof blocking, 5-frame confirmation, once-per-day marking, reports and student add/remove. Live-webcam access and the Docker build are still to be formally tested.

## 3. Tech stack

ClassSync is a Python 3.11 + React 19 application that stores all data in plain files — there is no SQL database. The codebase is 74.7% Python, 23.3% JavaScript and 1.4% CSS.

| Layer | Technology | Purpose |
| --- | --- | --- |
| AI models | SCRFD | Face detection: box and 5 key points per face |
| AI models | ArcFace (ResNet-50) | Face recognition: 512-number embedding per face |
| AI models | MiniFASNetV2 + MiniFASNetV1SE | Anti-spoofing (liveness) ensemble |
| AI runtime | insightface, ONNX Runtime | Load and run the models on the CPU |
| Image processing | OpenCV, NumPy, Pillow + pillow-heif | Crop and resize images, embedding maths, HEIC photos |
| Backend | Python 3.11, FastAPI, Uvicorn | REST API, WebSocket and server |
| Security | Starlette SessionMiddleware, itsdangerous, python-dotenv | Signed session cookie and `.env` settings |
| Real-time link | WebSocket (`/ws/camera`) | Streams camera frames and live results |
| Frontend | React 19, Vite, React Router | Single-page app, build tool, page routing |
| UI design | Tailwind CSS, GSAP, lucide-react | Styling, animations, icons |
| Storage | CSV files, NumPy `.npz` | Roster, gallery, daily attendance records |
| Packaging | Docker (multi-stage) | One portable image of the whole app |
| Evaluation | scikit-learn, Matplotlib | Baseline comparisons and figures (not in the live app) |

## 4. Future work

The next steps target the current limits: a single demo login, file storage, CPU speed of about one frame per second, and local-only deployment.

- **Production authentication** — individual user accounts with secure password storage, plus role-based access for teachers and admins.
- **A proper database** — PostgreSQL instead of CSV and `.npz` files, so many users can write at once.
- **Cloud deployment** — run the existing Docker image on AWS EC2 behind HTTPS, with cloud storage.
- **GPU acceleration** — faster frame processing for larger classes and more movement.
- **Better recognition in hard conditions** — poor lighting, students far away at the back, and faces turned to the side.
- **Stronger anti-spoofing** — an active blink or head-turn check to defend against high-quality video replay.
- **Notifications** — automatic alerts to absent students or parents.
- **Analytics** — attendance trends and insights beyond the current 14-day chart and below-75% flag.
- **Mobile app** — mobile-friendly capture and viewing.

## 5. Conclusion

ClassSync delivers a complete, working attendance system that records classroom attendance automatically and honestly. SCRFD finds faces, MiniFASNet blocks photo and screen spoofs, and ArcFace identifies students — with 100% rank-1 accuracy in our cross-device test. It runs on an ordinary CPU, is protected by a backend login, and can be packaged with Docker, making it a practical base for a modern classroom.
