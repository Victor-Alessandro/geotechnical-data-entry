
# syntax=docker/dockerfile:1

#### Stage 1: Build ####
FROM python:3.13-alpine3.18 AS builder

# install system dependencies for streamlit, geo libraries, ghostscript, git, curl
RUN apk add --no-cache \
        build-base \
        gdal-dev \
        geos-dev \
        proj-dev \
        ghostscript \
        git \
        curl

# ensure pip is up-to-date
RUN pip3 install --upgrade pip

WORKDIR /app

# clone your repo (or you can COPY . . if building from local context)
RUN git clone --depth 1 https://github.com/Victor-Alessandro/geotechnical-data-entry .

# install python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

#### Stage 2: Runtime ####
FROM python:3.13-alpine3.18

# copy over installed packages and app code from builder
COPY --from=builder /usr/local/lib/python3.13/site-packages /usr/local/lib/python3.13/site-packages
COPY --from=builder /app /app

WORKDIR /app

# expose Streamlit default port
EXPOSE 8501

# healthcheck against Streamlit’s internal endpoint
HEALTHCHECK --interval=30s --timeout=5s \
  CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# default command to run your app; adjust the path to your main script as needed
ENTRYPOINT ["streamlit", "run", "test.py", "--server.port=8501", "--server.address=0.0.0.0"]
