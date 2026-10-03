# Metis - single-container deployment.
# Builds one image that serves both the API and the frontend on one port,
# via FastAPI's static-file mount in backend/main.py.
#
# Build:  docker build -t metis .
# Run:    docker run -p 8000:8000 --env-file backend/.env metis
# Then open http://localhost:8000

FROM python:3.12-slim

WORKDIR /app

# System deps for xgboost/shap wheels on slim images
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt

COPY backend ./backend
COPY frontend ./frontend

# metis_users.db is created here at runtime - mount a volume at /app/backend
# if you need the accounts/history to survive container restarts/redeploys.
WORKDIR /app/backend

EXPOSE 8000

ENV PYTHONUNBUFFERED=1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
