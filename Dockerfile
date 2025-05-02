FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Install Python 3.8 and system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
    python3.8 \
    python3.8-dev \
    python3.8-distutils \
    python3-pip \
    build-essential \
    cmake \
    gcc \
    g++ \
    git \
    vim \
    wget \
    libgtk2.0-dev \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.8 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
    wget https://bootstrap.pypa.io/pip/3.8/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py && \
    python -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies with compatibility fixes
RUN pip install numpy==1.22.0 && \
    pip install Cython && \
    pip install --no-cache-dir torch==2.4.0 && \
    pip install --no-cache-dir pytorch-ssim==0.1 && \
    pip install --no-cache-dir scikit-image==0.15.0 && \
    pip install --no-cache-dir tqdm==4.66.3 && \
    pip install --no-cache-dir opencv-python==4.8.1.78

# Copy project files
COPY . .

# Build Pyflow
WORKDIR /app/pyflow
RUN python setup.py build_ext -i
RUN cp pyflow*.so ..

# Return to main directory
WORKDIR /app

# Set entrypoint
ENTRYPOINT ["/bin/bash"] 