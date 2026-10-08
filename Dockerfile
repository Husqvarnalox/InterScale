# CPU image. Models are downloaded from Hugging Face at first start (or mount a cache).
FROM python:3.12-slim-bookworm AS builder
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
# CPU-only torch wheel keeps the image small; installed before the source for caching.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir .

FROM python:3.12-slim-bookworm
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/inferscale/.cache/huggingface \
    INFERSCALE_HOST=0.0.0.0 \
    INFERSCALE_DEVICE=cpu
RUN groupadd --system --gid 10001 inferscale \
 && useradd --system --uid 10001 --gid 10001 --create-home inferscale \
 && install -d -o 10001 -g 10001 /home/inferscale/.cache/huggingface
COPY --from=builder /opt/venv /opt/venv
USER 10001:10001
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
  CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"]
CMD ["inferscale", "serve"]
