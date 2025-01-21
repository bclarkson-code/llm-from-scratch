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
    vim \
    && rm -rf /var/lib/apt/lists/*


# Set working directory
WORKDIR /app

# Copy requirements file
COPY pyproject.toml .
COPY requirements.txt .

ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

RUN uv venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN uv pip install . -r  requirements.txt --no-cache
# Use bash as the default command
CMD ["/bin/bash"]
