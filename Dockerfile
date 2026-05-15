# ─────────────────────────────────────────────────────────
#  Dockerfile — Data-Driven Social Engagement Initiative
#  Multi-stage build to keep the final image lean.
#
#  Build:  docker build -t social-engagement .
#  Run:    docker run -p 8501:8501 --env-file .env social-engagement
# ─────────────────────────────────────────────────────────

# ── Stage 1: dependency builder ───────────────────────────
FROM python:3.11-slim AS builder

# System deps needed to compile some packages (Prophet, statsmodels)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt .

# Install into a prefix we'll copy to the final stage
RUN pip install --upgrade pip \
 && pip install --prefix=/install --no-cache-dir -r requirements.txt

# ── Stage 2: runtime image ─────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="DataScienceWithGaurav"
LABEL description="Social Engagement Initiative – Streamlit Dashboard"

# Runtime system libs
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

WORKDIR /app

# Copy project source (respects .dockerignore)
COPY . .

# Ensure data directories exist inside the container
RUN mkdir -p data/raw data/processed data/exports models logs reports/visuals

# Streamlit port
EXPOSE 8501

# Healthcheck — confirms the app is responding
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

# Run pipeline first, then launch dashboard
# (pipeline uses dummy data if no real API key is set)
CMD ["sh", "-c", \
     "PYTHONPATH=/app python run_pipeline.py && \
      PYTHONPATH=/app streamlit run src/dashboard/app.py \
        --server.port=8501 \
        --server.address=0.0.0.0 \
        --server.headless=true \
        --server.enableCORS=false"]
