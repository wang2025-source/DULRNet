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

The MSRS dataset can be obtained from [Linfeng-Tang/MSRS](https://github.com/Linfeng-Tang/MSRS).

## VGG-16 model

[google drive](https://drive.google.com/file/d/1l3ieFhtgXd_R0alXr42RNQUTy1GSeNBg/view?usp=drive_link)

## Training

Train DULRNet with:

```bash
python training_dulrnet.py
```

## Testing

Test DULRNet with:

```bash
python testing_dulrnet.py
```

The fusion results will be saved in `outputs/DULRNet`.

## Contact

If you have any questions, please contact wangdongming@whut.edu.cn.
