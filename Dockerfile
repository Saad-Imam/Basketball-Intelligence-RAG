FROM python:3.11-slim

# HF Spaces requires the app to listen on port 7860
ENV PORT=7860
# prevent from buffering output
ENV PYTHONUNBUFFERED=1
# prevent from writing .pyc files
ENV PYTHONDONTWRITEBYTECODE=1 

# System deps for torch / sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies 
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY . .

# Pre-download the CrossEncoder model at build time so the Space
# doesn't stall on first request.  BGE-M3 is downloaded lazily at
# startup to avoid the 5-GB image limit on free Spaces.
RUN python - <<'EOF'
from sentence_transformers import CrossEncoder
CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
print("CrossEncoder cached.")
EOF

# Non-root user required by HF Spaces
RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]