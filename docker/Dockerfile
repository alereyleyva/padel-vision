FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --no-dev

COPY src/ ./src/

ENV PYTHONPATH=/app/src
ENV UPLOAD_DIR=/app/data/uploads
ENV CELERY_BROKER_URL=redis://redis:6379/0
ENV CELERY_RESULT_BACKEND=redis://redis:6379/1

EXPOSE 8000

CMD ["uv", "run", "fastapi", "run", "src/padelvision/api/main.py", "--host", "0.0.0.0", "--port", "8000"]
