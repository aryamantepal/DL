# CPU-only, portable across amd64 and arm64 (Apple Silicon)
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# system deps (optional but helpful for Pillow)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app

# Install PyTorch CPU wheels + deps
# Pin torch/torchvision to compatible versions (adjust if needed)
RUN python -m pip install --upgrade pip && \
    pip install --index-url https://download.pytorch.org/whl/cpu \
    torch==2.4.0 torchvision==0.19.0 && \
    pip install pillow numpy glob2

# Default mount points / envs
ENV DATA_DIR=/data \
    OUT_DIR=/outputs
RUN mkdir -p /data /outputs

CMD ["python", "main.py"]
