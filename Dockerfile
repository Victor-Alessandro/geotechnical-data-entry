FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential cmake pkg-config git curl \
    python3-dev python3-gdal\
    gdal-bin libgdal-dev libgeos-dev libproj-dev \
    libjpeg-dev libpng-dev libtiff-dev \
    libavcodec-dev libavformat-dev libswscale-dev \
    libgtk-3-0 poppler-utils \
    gfortran libopenblas-dev \
  && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/Victor-Alessandro/geotechnical-data-entry .

RUN python -m pip install --upgrade pip setuptools wheel

RUN pip3 install --no-cache-dir  -r requirements.txt

RUN pip3 install --no-cache-dir "huggingface_hub[cli]" \
 && python -m spacy validate >/dev/null 2>&1 || true

RUN python -m spacy download fr_core_news_sm

ENV HF_HOME=/app/.cache/huggingface \
    MODEL_ID=minishlab/potion-multilingual-128M \
    SPACY_MODEL=fr_core_news_sm

RUN mkdir -p "${HF_HOME}" && \
    hf download minishlab/potion-multilingual-128M --local-dir "${HF_HOME}/minishlab/potion-multilingual-128M"
    
EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "🔽 Téléchargements.py", "--server.port=8501", "--server.address=0.0.0.0"]
