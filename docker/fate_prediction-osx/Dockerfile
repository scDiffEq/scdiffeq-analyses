# Dockerfile.mac
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*
    
# Set working directory
WORKDIR /app
COPY . /app

# Install PyTorch (CPU version works fine for macOS testing)
RUN pip install --upgrade pip
RUN pip install torch torchvision torchaudio
RUN pip install -r requirements.txt

# Set environment variable for wandb
ENV WANDB_API_KEY=""

CMD ["python", "run.py"]