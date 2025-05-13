FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04 AS builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Copy only requirements first for better layer caching
WORKDIR /tmp
COPY requirements.txt .

# Install Python and dependencies in one layer
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    python3.8 \
    python3.8-dev \
    python3.8-distutils \
    python3-pip \
    build-essential \
    cmake \
    git \
    wget \
    libgtk2.0-dev \
    libgl1 \
    python3-gi \
    python3-gi-cairo \
    gir1.2-gtk-3.0 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
    wget -q https://bootstrap.pypa.io/pip/3.8/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py && \
    python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Build PyFlow in a separate layer
WORKDIR /build
COPY pyflow/ ./pyflow/
RUN cd pyflow && python setup.py build_ext -i && cp pyflow*.so /tmp/

# Final stage to create a smaller image
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Install only runtime dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    python3.8 \
    python3.8-distutils \
    python3-pip \
    libgl1 \
    python3-gi \
    python3-gi-cairo \
    gir1.2-gtk-3.0 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set up Python
COPY --from=builder /usr/local/lib/python3.8/dist-packages/ /usr/local/lib/python3.8/dist-packages/
COPY --from=builder /usr/local/bin/ /usr/local/bin/

# Set working directory
WORKDIR /iSeeBetter

# Copy application code
COPY . .
COPY --from=builder /tmp/pyflow*.so ./

# Default command
CMD ["python", "-c", "import gi; print('GI module loaded successfully')"]