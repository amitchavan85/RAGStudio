# RAG API image: python -m src.api on port 8000, also serving the web UI at /.

# 1) Build the web UI (static files).
FROM node:24-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY web/ ./
RUN npm run build

# 2) Python API.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    API_HOST=0.0.0.0 API_PORT=8000 DATA_DIR=/data LOG_DIR=/data/logs \
    HF_HOME=/data/hf_cache

WORKDIR /app
# CPU-only PyTorch keeps the image ~2 GB smaller than the default CUDA build.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch
COPY requirements.txt .
# OCR (rapidocr) pulls in desktop OpenCV, which needs X11 libraries; servers use the headless build.
RUN pip install -r requirements.txt \
 && pip uninstall -y opencv-python && pip install opencv-python-headless

COPY pyproject.toml .
COPY src/ src/
RUN pip install --no-deps -e .
COPY --from=web /web/dist web/dist

RUN useradd --create-home app && mkdir -p /data && chown app /data
USER app
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"
CMD ["python", "-m", "src.api"]
