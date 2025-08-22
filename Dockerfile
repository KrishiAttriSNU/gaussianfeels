# GaussianFeels Production Docker Image
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    build-essential \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libusb-1.0-0-dev \
    libudev-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONPATH="/app:${PYTHONPATH}"
ENV CUDA_VISIBLE_DEVICES="0"
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6"

# Copy requirements first for better caching
COPY requirements.txt setup.py ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install optional dependencies for production
RUN pip install --no-cache-dir \
    fastapi[all] \
    uvicorn[standard] \
    websockets \
    pyrealsense2 \
    psutil \
    prometheus-client \
    redis \
    celery

# Copy source code
COPY gaussianfeels/ ./gaussianfeels/
COPY scripts/ ./scripts/
COPY configs/ ./configs/
COPY README.md LICENSE ./

# Install package in editable mode
RUN pip install -e .

# Create directories for data and outputs
RUN mkdir -p /app/data /app/outputs /app/logs /app/checkpoints

# Set up non-root user for security
RUN groupadd -r gaussianfeels && useradd -r -g gaussianfeels gaussianfeels
RUN chown -R gaussianfeels:gaussianfeels /app
USER gaussianfeels

# Expose ports
EXPOSE 8080 8081 8082

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Default command
CMD ["python", "-m", "gaussianfeels.server", "--config", "configs/production.yaml"]