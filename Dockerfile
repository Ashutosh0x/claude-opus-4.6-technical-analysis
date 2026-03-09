# ============================================================
# Claude Opus 4.6 -- Multi-stage Dockerfile
# ============================================================
# Stage 1: Base image with PyTorch + CUDA
# Stage 2: Training image (full dependencies)
# Stage 3: Inference image (lightweight serving)
# ============================================================

# ------ Stage 1: Base ------
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3.11 /usr/bin/python

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt


# ------ Stage 2: Training ------
FROM base AS training

COPY pyproject.toml .
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/

RUN pip install -e ".[dev,logging]"

# Default: 8-GPU training
ENV NPROC_PER_NODE=8
ENV MASTER_PORT=29500

CMD ["bash", "scripts/train.sh"]


# ------ Stage 3: Inference ------
FROM base AS inference

COPY pyproject.toml .
COPY src/ src/
COPY configs/ configs/

RUN pip install -e ".[serving]"

EXPOSE 8080

CMD ["python", "-m", "uvicorn", "src.serving.api_server:app", \
     "--host", "0.0.0.0", "--port", "8080", "--workers", "4"]
