# GR00T N1.6 & N1.7 — WSL2 & Docker Install Guide

A complete, battle-tested installation guide for [NVIDIA Isaac GR00T N1.6 & N1.7](https://github.com/NVIDIA/Isaac-GR00T) — covering both WSL2 on Windows and Docker on Linux, including real fixes for issues not covered in the official docs.

📖 **Full interactive guide → [index.html](https://rao-sanaullah.github.io/GR00TN1.7/)** (open via GitHub Pages)

---

## Tested On

| | |
|---|---|
| **OS** | WSL2 — Ubuntu 22.04 · Linux (Docker) |
| **GPU** | NVIDIA RTX 5090 (Blackwell / sm_120) |
| **CUDA** | 12.8 |
| **Python** | 3.10 |
| **Docker** | BuildKit enabled · NVIDIA Container Toolkit |

---

## What's Fixed Here

Issues not covered in the official docs — full details with terminal output in the [HTML guide](https://rao-sanaullah.github.io/GR00TN1.7/):

- 🔴 **Permission denied on `.venv`** — repo cloned as root, `uv sync` fails
- 🔴 **Triton crash on RTX 5090** — sm_120 (Blackwell) not recognised by Triton 3.3.1 pinned in PyTorch 2.7
- 🔴 **HuggingFace gated model** — `Cosmos-Reason2-2B` backbone requires access request + `huggingface-cli login`
- 🔴 **Docker venv hidden by volume mount** — mounting over `/workspace` hides the installed `.venv`; mount to `/data` instead
- 🔴 **BuildKit not enabled** — `--mount=type=cache` in the Dockerfile fails with legacy builder

---

## Quick Start — WSL2

```bash
# 1. Install git-lfs BEFORE cloning
sudo apt-get install -y git-lfs ffmpeg curl build-essential
git lfs install

# 2. Clone with submodules
git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T
cd Isaac-GR00T

# 3. Fix permissions if cloned as root
sudo chown -R $USER:$USER .

# 4. Install uv and sync
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv sync --python 3.10

# 5. Set CUDA_HOME
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
source ~/.bashrc

# 6. Patch Triton for RTX 5090 / Blackwell
uv run bash scripts/patch_triton_cuda13.sh

# 7. Authenticate with HuggingFace (gated model)
uv run huggingface-cli login

# 8. Verify
uv run python -c "import gr00t; print('GR00T installed successfully')"
```

---

## Quick Start — Docker (Linux)

```bash
# 1. Install NVIDIA Container Toolkit (if not already)
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER && newgrp docker
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker

# 2. Enable BuildKit
mkdir -p ~/.docker/cli-plugins
curl -SL https://github.com/docker/buildx/releases/download/v0.17.1/buildx-v0.17.1.linux-amd64 \
  -o ~/.docker/cli-plugins/docker-buildx
chmod +x ~/.docker/cli-plugins/docker-buildx

# 3. Clone with submodules (on host)
sudo apt-get install -y git-lfs && git lfs install
git clone --recurse-submodules https://github.com/NVIDIA/Isaac-GR00T
cd Isaac-GR00T

# 4. Build the image (~15–25 min)
DOCKER_BUILDKIT=1 docker build -f docker/Dockerfile -t grootn17:latest .

# 5. Run the container
#    Mount repo to /data (NOT /workspace — that hides the installed venv)
docker run -it --gpus all \
  --ipc=host \
  --shm-size=16g \
  -v /path/to/Isaac-GR00T:/data \
  -v /path/to/your-datasets:/workspace/datasets \
  -v /path/to/your-models:/workspace/models \
  -p 5555:5555 \
  --name grootn17 \
  grootn17:latest

# 6. Inside container — activate venv and verify
source /workspace/.venv/bin/activate
echo "source /workspace/.venv/bin/activate" >> ~/.bashrc

python -c "import torch; print(torch.cuda.get_device_name(0))"
python -c "import gr00t; print('GR00T OK')"

# 7. Run zero-shot inference
cd /data
python scripts/deployment/standalone_inference_script.py \
  --model-path nvidia/GR00T-N1.7-3B \
  --dataset-path demo_data/droid_sample \
  --embodiment-tag OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT \
  --traj-ids 1 2 \
  --inference-mode pytorch \
  --action-horizon 8
```

> **Re-entering the container after reboot:** `docker start -ai grootn17`

---

## Fine-tuning (both platforms)

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python gr00t/experiment/launch_finetune.py \
  --base-model-path nvidia/GR00T-N1.7-3B \
  --dataset-path /path/to/your-dataset \
  --embodiment-tag NEW_EMBODIMENT \
  --modality-config-path examples/YOUR_EMBODIMENT/config.py \
  --num-gpus 1 \
  --output-dir /path/to/output \
  --max-steps 5000 \
  --global-batch-size 8 \
  --gradient-accumulation-steps 4 \
  --num-shards-per-epoch 10 \
  --save-only-model \
  --dataloader-num-workers 2
```

> **OOM on 32GB VRAM?** Drop to `--global-batch-size 4 --gradient-accumulation-steps 8` or add `--no-tune-projector`.

---

## Zero-Shot Inference Results (RTX 5090)

| Metric | Trajectory 1 | Trajectory 2 | Average |
|---|---|---|---|
| **MSE** | 0.0033 | 0.0369 | **0.0201** |
| **MAE** | 0.0376 | 0.1168 | **0.0772** |
| **Avg step time** | — | — | **169 ms** |

Base model `nvidia/GR00T-N1.7-3B`, DROID demo dataset, 200 steps per trajectory, no fine-tuning.

---

## Related

- [NVIDIA Isaac GR00T (official repo)](https://github.com/NVIDIA/Isaac-GR00T)
- [GR00T N1 Paper](https://arxiv.org/abs/2503.14734)
- [HuggingFace: nvidia/GR00T-N1.7-3B](https://huggingface.co/nvidia/GR00T-N1.7-3B)

---

## License

Community documentation. GR00T N1.7 is licensed under [Apache 2.0](https://github.com/NVIDIA/Isaac-GR00T/blob/main/LICENSE) (code) and the [NVIDIA Open Model License](https://developer.nvidia.com/isaac/gr00t) (weights).
