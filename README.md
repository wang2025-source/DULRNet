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

The code was trained and tested with the following environment:

```text
Python 3.9
PyTorch 2.8.0
TorchVision 0.23.0
CUDA 12.8
```

Install the main packages with:

```bash
pip install torch torchvision numpy opencv-python scikit-image matplotlib timm einops pytorch-msssim tensorboard tqdm
```

## Dataset

Prepare the MSRS dataset as follows:

```text
MSRS-main/
|-- train/
|   |-- ir/
|   `-- vi/
`-- test/
    |-- ir/
    `-- vi/
```

## Training

The default MSRS paths in `args.py` are:

```text
train_ir_dir = './MSRS-main/train/ir'
train_vi_dir = './MSRS-main/train/vi'
test_ir_dir  = './MSRS-main/test/ir'
test_vi_dir  = './MSRS-main/test/vi'
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
