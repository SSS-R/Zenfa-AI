# ---- Base ----
FROM python:3.11-slim AS base

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# ---- Dependencies ----
COPY pyproject.toml ./
RUN pip install --no-cache-dir .

# ---- Application ----
COPY . .

EXPOSE 8001

CMD ["uvicorn", "zenfa_ai.api.app:app", "--host", "0.0.0.0", "--port", "8001"]
