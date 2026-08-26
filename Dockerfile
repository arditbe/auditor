# Auditor — one image, one Cloud Run service.
#
# The dashboard is built here and served by the same FastAPI process that runs
# the audits, so there is a single URL and no CORS to configure. Build context
# is the repo root, because this needs both frontend/ and backend/.
#
#   gcloud run deploy auditor --source .   (from the repo root)

# --- stage 1: build the dashboard -------------------------------------------
FROM node:22-alpine AS ui

WORKDIR /ui
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci || npm install

COPY frontend/ ./
# Left empty on purpose: the dashboard is served from the same origin as the
# API, so relative /api paths resolve without knowing the deployed URL.
ENV VITE_API_BASE=""
RUN npm run build


# --- stage 2: the service ---------------------------------------------------
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first, so editing code does not invalidate the wheel layer.
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/app ./app
COPY --from=ui /ui/dist ./ui

# Cloud Run supplies PORT. One worker: the SSE replay log lives in process
# memory, so a run and its event stream must be served by the same worker.
ENV PORT=8080 \
    AUDITOR_UI_DIR=/app/ui

CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1
