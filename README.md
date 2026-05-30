# DULRNet: A Deep Unfolding Low-Rank Network for Infrared and Visible Image Fusion

DULRNet is a deep unfolding low-rank network for infrared and visible image fusion.
The network decomposes infrared and visible images into low-rank and sparse components, enhances cross-modal representations, and reconstructs the final fused image.

![DULRNet framework](framework/wang2.png)

## Framework

The overall architecture contains a multi-scale dual-branch LRSD encoder, a context-aware gated fusion module, an asymmetric bidirectional cross-modal synergistic attention module, and an enhanced decoder.

### Main Components

![Dual-branch LRSD encoder](framework/wang3.jpg)

![Context-aware gated fusion module](framework/wang4.png)

![Cross-modal synergistic attention module](framework/wang5.png)

![Enhanced decoder](framework/wang6.png)

## Platform

The reported training environment is:

```text
Python 3.9.25
PyTorch 2.8.0
TorchVision 0.23.0
CUDA 12.8
cuDNN 9.10.2
Conda environment: elrrnet
```

Core Python packages:

```text
einops==0.8.2
ImageIO==2.37.2
matplotlib==3.9.4
natsort==8.4.0
numpy==2.0.2
opencv-python==4.13.0.92
pandas==2.3.3
pillow==11.3.0
pytorch-msssim==1.0.0
PyYAML==6.0.3
scikit-image==0.24.0
scikit-learn==1.6.1
scipy==1.13.1
seaborn==0.13.2
tensorboard==2.20.0
thop==0.1.1.post2209072238
timm==1.0.25
torch==2.8.0
torchvision==0.23.0
tqdm==4.67.3
```

Install the main dependencies with:

```bash
pip install einops imageio matplotlib natsort numpy opencv-python pandas pillow pytorch-msssim pyyaml scikit-image scikit-learn scipy seaborn tensorboard thop timm torch torchvision tqdm
```

<details>
<summary>Full training environment</summary>

```text
absl-py==2.3.1
annotated-doc==0.0.4
anyio==4.12.1
certifi==2026.2.25
click==8.1.8
contourpy==1.3.0
cycler==0.12.1
einops==0.8.2
exceptiongroup==1.3.1
filelock==3.19.1
fonttools==4.60.2
fsspec==2025.10.0
grpcio==1.80.0
h11==0.16.0
hf-xet==1.3.1
httpcore==1.0.9
httpx==0.28.1
huggingface_hub==1.4.1
idna==3.11
ImageIO==2.37.2
importlib_metadata==8.7.1
importlib_resources==6.5.2
Jinja2==3.1.6
joblib==1.5.3
kiwisolver==1.4.7
lazy_loader==0.4
Markdown==3.9
markdown-it-py==3.0.0
MarkupSafe==3.0.3
matplotlib==3.9.4
mdurl==0.1.2
mpmath==1.3.0
natsort==8.4.0
networkx==3.2.1
numpy==2.0.2
nvidia-cublas-cu12==12.8.4.1
nvidia-cuda-cupti-cu12==12.8.90
nvidia-cuda-nvrtc-cu12==12.8.93
nvidia-cuda-runtime-cu12==12.8.90
nvidia-cudnn-cu12==9.10.2.21
nvidia-cufft-cu12==11.3.3.83
nvidia-cufile-cu12==1.13.1.3
nvidia-curand-cu12==10.3.9.90
nvidia-cusolver-cu12==11.7.3.90
nvidia-cusparse-cu12==12.5.8.93
nvidia-cusparselt-cu12==0.7.1
nvidia-nccl-cu12==2.27.3
nvidia-nvjitlink-cu12==12.8.93
nvidia-nvtx-cu12==12.8.90
opencv-python==4.13.0.92
packaging==26.0
pandas==2.3.3
pillow==11.3.0
pip==26.0.1
protobuf==6.33.6
Pygments==2.19.2
pyparsing==3.3.2
python-dateutil==2.9.0.post0
pytorch-msssim==1.0.0
pytz==2025.2
PyYAML==6.0.3
rich==14.3.3
safetensors==0.7.0
scikit-image==0.24.0
scikit-learn==1.6.1
scipy==1.13.1
seaborn==0.13.2
setuptools==80.9.0
shellingham==1.5.4
six==1.17.0
sympy==1.14.0
tensorboard==2.20.0
tensorboard-data-server==0.7.2
thop==0.1.1.post2209072238
threadpoolctl==3.6.0
tifffile==2024.8.30
timm==1.0.25
torch==2.8.0
torchvision==0.23.0
tqdm==4.67.3
triton==3.4.0
typer==0.23.2
typer-slim==0.23.2
typing_extensions==4.15.0
tzdata==2025.3
Werkzeug==3.1.8
wheel==0.45.1
zipp==3.23.0
```

</details>

## Dataset

Prepare paired infrared and visible images using the directory layout configured in `args.py`:

```text
MSRS-main/
├── train/
│   ├── ir/
│   └── vi/
└── test/
    ├── ir/
    └── vi/
```

## Training

Check and update paths or hyperparameters in `args.py`, especially:

```text
train_ir_dir
train_vi_dir
save_model_dir
vgg_model_path
```

Then run:

```bash
python training_dulrnet.py
```

The default VGG path is:

```text
model/vgg/vgg16.pth
```

## Testing

Place the trained model at:

```text
model/dulrnet_final.pth
```

Then run:

```bash
python testing_dulrnet.py
```

Fusion results are saved to:

```text
outputs/DULRNet
```

## Citation

If this code is helpful for your research, please cite the related paper when it becomes available.
