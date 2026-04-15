FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# install Python dependencies
RUN pip install --no-cache-dir \
    SimpleITK==2.3.1 \
    opencv-python-headless==4.8.1.78 \
    torchmetrics==1.2.0 \
    pandas==2.0.3 \
    numpy==1.24.3 \
    tqdm==4.66.1 \
    matplotlib==3.7.2


ENV PYTHONPATH=/app/synthrad_bidirectional