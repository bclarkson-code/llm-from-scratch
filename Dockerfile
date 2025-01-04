# Start with NVIDIA CUDA base image that includes cuDNN
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies and curl
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*


# Set working directory
WORKDIR /app

# Copy requirements file
COPY pyproject.toml .
COPY requirements.txt .

# Install Python dependencies using uv
RUN pip install uv && uv pip install --system --no-cache -r requirements.txt

# Use bash as the default command
CMD ["/bin/bash"]
