# ---------- stage 1: build the React frontend ----------
FROM node:20-slim AS frontend
WORKDIR /app/webapp/frontend
COPY webapp/frontend/package.json webapp/frontend/package-lock.json ./
RUN npm ci
COPY webapp/frontend/ ./
RUN npm run build                       # -> webapp/frontend/dist

# ---------- stage 2: python runtime ----------
FROM python:3.11-slim AS runtime
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1

# system libs: OpenCV needs libGL + glib; build tools for insightface/onnx
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt ./
RUN pip install -r requirements.txt

# application code
COPY src/ ./src/
COPY webapp/ ./webapp/
# small public anti-spoof weights (MiniFASNet)
COPY models/anti_spoof/ ./models/anti_spoof/
# built frontend from stage 1 (overlays any local copy)
COPY --from=frontend /app/webapp/frontend/dist ./webapp/frontend/dist

# default model root (overridden by a runtime volume + env if you prefer)
ENV INSIGHTFACE_ROOT=/app/models/insightface

EXPOSE 8000
CMD ["uvicorn", "webapp.backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
