# ============================================================================
# Memex AML Investigation Environment — HF Spaces Dockerfile
#
# Deploys the OpenEnv-compatible FastAPI server to a Hugging Face Space.
# HF Spaces expects port 7860 and a non-root user with UID 1000.
#
# Build:  docker build -t memex-aml-env .
# Run:    docker run -p 7860:7860 memex-aml-env
# ============================================================================

FROM python:3.11-slim

WORKDIR /app

# System deps: curl for healthcheck, nothing else needed.
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Python deps — layer-cached before source copy.
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy full project source.
COPY . /app/

# The static_frontend/ directory contains the pre-built Next.js frontend.
# It is served at /web by the FastAPI server.

# HF Spaces convention: non-root user with UID 1000.
RUN useradd -m -u 1000 user && \
    chown -R user:user /app
USER user

# HF Spaces default port.
EXPOSE 7860

# Health probe: Docker HEALTHCHECK + HF Spaces liveness.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Launch the OpenEnv server.
# PORT env var is set by HF Spaces to 7860; our server reads it.
ENV PORT=7860
CMD ["uvicorn", "openenv_server:app", "--host", "0.0.0.0", "--port", "7860"]
