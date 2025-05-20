# Dockerfile.mac
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    gcc \
    g++ \
    htop \
    procps \
    nano \
    && rm -rf /var/lib/apt/lists/*
    
# Set working directory
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt /app/
RUN pip install --upgrade pip
RUN pip install torch torchvision torchaudio
RUN pip install -r requirements.txt

# Copy the rest of the application
COPY run.py /app/
COPY config.yaml /app/

# Set placeholder environment variable for wandb
ENV WANDB_API_KEY=""
ENV CONFIG_PATH="/app/config.yaml"

# Enable Python unbuffered mode for more immediate logging
ENV PYTHONUNBUFFERED=1
# Don't write bytecode to make debugging easier
ENV PYTHONDONTWRITEBYTECODE=1

# Redirect both stdout and stderr to ensure we capture all output
CMD ["sh", "-c", "python run.py 2>&1 | tee /app/run.log"]