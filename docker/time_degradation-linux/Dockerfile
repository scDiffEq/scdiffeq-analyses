# Dockerfile.linux
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Install Python and system dependencies
RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev git && \
    rm -rf /var/lib/apt/lists/*


RUN ln -s /usr/bin/python3 /usr/bin/python

# Set working directory
WORKDIR /app
COPY . /app

# Install Python packages
RUN pip3 install --upgrade pip
RUN pip3 install torch torchvision torchaudio
RUN pip3 install -r requirements.txt

# Default command
CMD ["python", "run.py"]
