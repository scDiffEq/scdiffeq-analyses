#!/bin/bash
# Startup script for scDiffEq experiment VMs on Google Cloud Platform

# Set up logging
exec > >(tee /var/log/startup-script.log)
exec 2>&1
echo "Starting VM setup at $(date)"

# Install necessary Python packages
echo "Installing Python packages..."
pip install scdiffeq torch lightning numpy pandas anndata abcparse autodevice wandb pathlib

# Create data directory
mkdir -p /home/jupyter/data
echo "Created data directory: /home/jupyter/data"

# Download dataset and weights from GCS
echo "Downloading data files from GCS..."
gsutil cp gs://${BUCKET_NAME}/data/adata.LARRY_train.19MARCH2024.h5ad /home/jupyter/data/
gsutil cp gs://${BUCKET_NAME}/data/Weinreb2020_growth-all_kegg.pt /home/jupyter/data/

# Create experiment directory
mkdir -p /home/jupyter/scDiffEq_experiments
cd /home/jupyter/scDiffEq_experiments
echo "Created experiment directory: /home/jupyter/scDiffEq_experiments"

# Download experiment files
echo "Downloading experiment files from GCS..."
gsutil cp gs://${BUCKET_NAME}/code/* .

# Make scripts executable
chmod +x *.sh
chmod +x *.py
echo "Made scripts executable"

# Create a marker file to indicate VM is ready
echo "VM setup completed at $(date)" > /home/jupyter/READY
echo "READY" >> /home/jupyter/READY

# Print completion message
echo "VM setup completed successfully at $(date)"