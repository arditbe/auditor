# Auditor API — Cloud Run.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies first so a code change does not invalidate the wheel layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Cloud Run supplies PORT. One worker: the SSE replay log lives in process
# memory, so a run and its stream must be served by the same worker.
ENV PORT=8080
CMD exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1
