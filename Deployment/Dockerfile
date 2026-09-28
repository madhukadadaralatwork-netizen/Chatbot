# Support Chatbot API -- container image
FROM python:3.11-slim

WORKDIR /app

# System deps for faiss / sentence-transformers wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Documents referenced by app_config.py must be present in the image (or
# mounted as a volume at runtime) -- see the README note on this.
EXPOSE 8000

# --workers 1: see the "scaling with multiple workers" note at the bottom
# of server.py before increasing this.
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]