import os
import random
from os import listdir
from os.path import basename, join, splitext

import cv2
import numpy as np
import torch
import torch.nn.functional as F

EPSILON = 1e-6
IMAGE_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')


def ensure_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def list_images(directory):
    images, names = [], []
    for file_name in sorted(listdir(directory)):
        if file_name.lower().endswith(IMAGE_EXTENSIONS):
            images.append(join(directory, file_name))
            names.append(splitext(file_name)[0])
    return images, names


def load_dataset(image_paths, batch_size, num_imgs=None):
    if num_imgs is None:
        num_imgs = len(image_paths)
    image_paths = list(image_paths[:num_imgs])
    random.shuffle(image_paths)
    remainder = len(image_paths) % batch_size
    if remainder > 0:
        image_paths = image_paths[:-remainder]
    return image_paths, len(image_paths) // batch_size


def _read_luminance(path, mode='gray'):
    mode = mode.lower()
    if mode in ('vi', 'visible', 'y', 'ycbcr'):
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(path)
        ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
        image = ycrcb[:, :, 0]
    else:
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(path)
    return image.astype(np.float32)


def _to_tensor(image):
    image = np.expand_dims(image, axis=0)
    return torch.from_numpy(image).float()


def _resize_if_needed(image, height=None, width=None):
    if height is not None and width is not None:
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return image


def get_train_images(paths, height=None, width=None, mode='gray'):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = _read_luminance(path, mode=mode)
        image = _resize_if_needed(image, height, width)
        images.append(_to_tensor(image))
    return torch.stack(images, dim=0)


def _paired_visible_path(ir_path, vi_dir):
    file_name = basename(ir_path)
    direct_path = join(vi_dir, file_name)
    if os.path.exists(direct_path):
        return direct_path

    candidates = []
    lower_name = file_name.lower()
    if 'ir' in lower_name:
        candidates.append(file_name.replace('IR', 'VIS').replace('ir', 'vi'))
    if '_i' in lower_name:
        candidates.append(file_name.replace('_i', '_v'))
    if 'i.' in lower_name:
        candidates.append(file_name.replace('i.', 'v.'))

    for candidate in candidates:
        candidate_path = join(vi_dir, candidate)
        if os.path.exists(candidate_path):
            return candidate_path

    raise FileNotFoundError(f'Cannot find the visible image paired with {ir_path} in {vi_dir}.')


def _paired_random_crop(ir, vi, height, width):
    if ir.shape != vi.shape:
        vi = cv2.resize(vi, (ir.shape[1], ir.shape[0]), interpolation=cv2.INTER_LINEAR)

    h, w = ir.shape
    if h < height or w < width:
        new_h = max(h, height)
        new_w = max(w, width)
        ir = cv2.resize(ir, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        vi = cv2.resize(vi, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        h, w = ir.shape

    top = random.randint(0, h - height)
    left = random.randint(0, w - width)
    return ir[top:top + height, left:left + width], vi[top:top + height, left:left + width]


def load_paired_images(ir_paths, vi_dir, height=128, width=128):
    ir_images, vi_images = [], []
    for ir_path in ir_paths:
        vi_path = _paired_visible_path(ir_path, vi_dir)
        ir = _read_luminance(ir_path, mode='ir')
        vi = _read_luminance(vi_path, mode='vi')
        ir, vi = _paired_random_crop(ir, vi, height, width)
        ir_images.append(_to_tensor(ir))
        vi_images.append(_to_tensor(vi))
    return torch.stack(ir_images, dim=0), torch.stack(vi_images, dim=0)


def to_01(tensor):
    return torch.clamp(tensor / 255.0, 0.0, 1.0)


def normalize_tensor(tensor):
    b, ch, h, w = tensor.size()
    flat = tensor.view(b, -1)
    t_min = flat.min(dim=1)[0].view(b, 1, 1, 1)
    t_max = flat.max(dim=1)[0].view(b, 1, 1, 1)
    return (tensor - t_min) / (t_max - t_min + EPSILON)


def save_image(img, output_path):
    ensure_dir(os.path.dirname(output_path))
    if isinstance(img, torch.Tensor):
        img = img.detach().float().cpu()
        if img.dim() == 4:
            img = img[0]
        if img.dim() == 3:
            img = img.squeeze(0)
        img = img.numpy()
    img = np.squeeze(img)
    img = np.clip(img, 0.0, 1.0) * 255.0
    cv2.imwrite(output_path, img.astype(np.uint8))


def gram_matrix(features):
    b, c, h, w = features.size()
    features = features.view(b, c, h * w)
    return torch.bmm(features, features.transpose(1, 2)) / (c * h * w)


def ssim_index(img1, img2, window_size=11):
    channel = img1.size(1)
    window = torch.ones((channel, 1, window_size, window_size), dtype=img1.dtype, device=img1.device)
    window = window / float(window_size * window_size)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu12 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu12

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    ssim_map = ((2 * mu12 + c1) * (2 * sigma12 + c2)) / (
        (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2) + EPSILON
    )
    return ssim_map.mean()


def init_vgg16(vgg, model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f'VGG16 weights not found: {model_path}')
    state_dict = torch.load(model_path, map_location='cpu')
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    params = list(state_dict.values())
    target_layers = [
        vgg.conv1_1, vgg.conv1_2,
        vgg.conv2_1, vgg.conv2_2,
        vgg.conv3_1, vgg.conv3_2, vgg.conv3_3,
        vgg.conv4_1, vgg.conv4_2, vgg.conv4_3,
    ]
    idx = 0
    for layer in target_layers:
        layer.weight.data.copy_(params[idx])
        idx += 1
        layer.bias.data.copy_(params[idx])
        idx += 1
