# ---------- Stage 1: builder (build wheels, no runtime bloat) ----------
FROM python:3.12-slim AS builder
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /build

# System deps needed to build wheels for your Python deps
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    build-essential cmake pkg-config git curl \
    python3-dev gfortran libopenblas-dev \
    gdal-bin libgdal-dev libgeos-dev libproj-dev \
    libjpeg-dev libpng-dev libtiff-dev popple-utils \
    libavcodec-dev libavformat-dev libswscale-dev \
 && rm -rf /var/lib/apt/lists/*

# Only copy what’s needed to resolve/install requirements
COPY requirements.txt /build/requirements.txt

# Build wheels once (no cache kept in final image)
RUN pip3 wheel --no-cache-dir -r requirements.txt -w /wheels

# ---------- Stage 2: runtime (lean) ----------
FROM python:3.12-slim
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# Runtime-only shared libraries (no compilers)
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    gdal-bin libgdal-dev libgeos-dev libproj-dev \
    libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libgtk-3-0 \
    poppler-utils \
    curl \
 && rm -rf /var/lib/apt/lists/*

# Install wheels built in the builder
COPY --from=builder /wheels /wheels
RUN pip3 install --no-cache-dir --no-index --find-links=/wheels /wheels/*

# App code
# Prefer COPY over git clone; add a .dockerignore to exclude large/unneeded paths
COPY . /app

# Hugging Face CLI + spaCy model tooling
RUN pip3 install --no-cache-dir "huggingface_hub[cli]" \
 && python -m spacy validate >/dev/null 2>&1 || true

# Model/cache locations and defaults
ENV HF_HOME=/app/.cache/huggingface \
    MODEL_ID=minishlab/potion-multilingual-128M \
    SPACY_MODEL=fr_core_news_sm

# Add entrypoint that can pre-warm caches then launch Streamlit
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --retries=5 CMD \
  test -d "$HF_HOME/$MODEL_ID" && \
  curl -fsS http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["serve"]
