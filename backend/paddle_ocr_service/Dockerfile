FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

EXPOSE 8010
WORKDIR /app

# Prevent interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive
# Suppress paddle/paddlex logging noise
ENV GLOG_minloglevel=2

# Install Python 3.10 and system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3-pip python3.10-dev \
        libgl1 libgomp1 libglib2.0-0 libsm6 libxrender1 libxext6 \
        libcairo2-dev pkg-config \
        wget curl && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# ── Step 1: Install PaddlePaddle GPU 3.x for CUDA 11.8 ──
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir paddlepaddle-gpu==3.3.0 \
        -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# ── Step 2: Install PaddleX (PP-StructureV3 pipeline framework) ──
RUN pip install --no-cache-dir "paddlex[ocr]"

# ── Step 3: Install remaining Python dependencies ──
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Create output directory
RUN mkdir -p /app/ocr_output

# Copy project files
COPY . /app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8010/health')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8010", "--workers", "1"]
