FROM python:3.12-slim-bookworm

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app

COPY requirements-prod.txt .
RUN pip install --no-cache-dir -r requirements-prod.txt \
    && apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
COPY surgecast ./surgecast

EXPOSE 8841
CMD ["python", "-m", "surgecast.serving.app"]
