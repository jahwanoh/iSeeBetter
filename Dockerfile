FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

# Install Python 3.8 and system dependencies in one layer
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    python3.8 \
    python3.8-dev \
    python3.8-distutils \
    python3-pip \
    build-essential \
    cmake \
    git \
    vim \
    wget \
    libgtk2.0-dev \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set Python 3.8 as default & upgrade pip/setuptools/wheel
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.8 1 && \
    wget https://bootstrap.pypa.io/pip/3.8/get-pip.py && \
    python get-pip.py && \
    rm get-pip.py && \
    python -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies from requirements.txt
# Add --no-cache-dir if you want to minimize image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Build Pyflow
WORKDIR /app/pyflow
# Make sure Cython is installed first (either via requirements.txt or a separate RUN)
RUN python setup.py build_ext -i && \
    cp pyflow*.so ..

# Return to main directory
WORKDIR /app

# Set entrypoint
ENTRYPOINT ["/bin/bash"]