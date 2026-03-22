FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=120

WORKDIR /app

COPY requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip --retries 10 --timeout 120 \
    && python -m pip install -r requirements.txt --retries 10 --timeout 120

COPY backend ./backend
COPY README.md ./README.md

EXPOSE 8000

CMD ["uvicorn", "backend.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
