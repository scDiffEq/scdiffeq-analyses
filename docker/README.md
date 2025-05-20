# Docker distributions

## Fate prediction analysis


### Fate prediction analysis - Linux

Make sure your machine has a supported NVIDIA GPU.
```bash
nvidia-smi
```
If this fails, install the drivers.

Install the NVIDIA container toolkit:
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)

curl -s -L https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /etc/apt/trusted.gpg.d/nvidia-container-toolkit.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt update
sudo apt install -y nvidia-container-toolkit

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Test the installation:
```bash
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```
You should see the NVIDIA GPU usage table.

Pull and run the container:
```bash
docker run --rm \
  --gpus all \
  --memory=58g --memory-swap=60g \
  -v /home/yourname/config.yaml:/app/config.yaml \
  -v /home/yourname/input:/data \
  -e WANDB_API_KEY=<your_wandb_key> \
  michaelvinyard/fate_prediction:linux
```


### Fate prediction analysis - OSX

***CAUTION***: Not yet fully validated.

```bash
docker build -t fate_prediction_analysis_osx -f fate_prediction-osx/Dockerfile .
```