FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

EXPOSE 8010
WORKDIR /app

# Prevent interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.10 and system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3-pip \
        libgl1 libgomp1 libglib2.0-0 libsm6 libxrender1 libxext6 && \
    ln -sf /usr/bin/python3.10 /usr/bin/python && \
    ln -sf /usr/bin/python3.10 /usr/bin/python3 && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies (ordered for compatibility)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "numpy<2.0.0" && \
    pip install --no-cache-dir paddlepaddle-gpu==2.6.1.post117 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html && \
    pip install --no-cache-dir paddleocr==2.9.1 && \
    pip install --no-cache-dir PyMuPDF && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir "numpy>=1.24,<2.0.0"

# Copy project files
COPY . /app

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8010/health')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8010", "--workers", "1"]
