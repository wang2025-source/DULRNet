import os
import random
import time

from args import Args as args

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter

try:
    from pytorch_msssim import ssim as ms_ssim
except Exception:
    ms_ssim = None

from dulrnet import DULRNet, Vgg16
import utils


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def sobel_gradient(x):
    kernel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)
    kernel_y = torch.tensor(
        [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]],
        device=x.device,
        dtype=x.dtype,
    ).view(1, 1, 3, 3)
    channels = x.shape[1]
    grad_x = F.conv2d(x, kernel_x.repeat(channels, 1, 1, 1), padding=1, groups=channels)
    grad_y = F.conv2d(x, kernel_y.repeat(channels, 1, 1, 1), padding=1, groups=channels)
    return torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6)


def ssim_loss_value(x, y):
    if ms_ssim is not None:
        return 1.0 - ms_ssim(x, y, data_range=1.0)
    return 1.0 - utils.ssim_index(x, y)


def gradient_loss(fused, ir, vi):
    grad_f = sobel_gradient(fused)
    grad_ir = sobel_gradient(ir)
    grad_vi = sobel_gradient(vi)
    return args.mu_grad * (
        args.eta_ir * F.l1_loss(grad_f, grad_ir) +
        args.eta_vi * F.l1_loss(grad_f, grad_vi)
    )


def current_ir_weight(global_step, total_steps):
    progress = min(float(global_step) / max(float(total_steps), 1.0), 1.0)
    return args.w_ir_start + progress * (args.w_ir_end - args.w_ir_start)


def feature_and_style_loss(vgg, fused_01, ir_01, vi_01, w_ir):
    fused_255 = fused_01 * 255.0
    ir_255 = ir_01 * 255.0
    vi_255 = vi_01 * 255.0

    feat_f = vgg(fused_255)
    feat_ir = vgg(ir_255)
    feat_vi = vgg(vi_255)

    content_loss = 0.0
    for k, weight in enumerate(args.feat_weights):
        target = w_ir * feat_ir[k] + args.w_vi * feat_vi[k]
        content_loss = content_loss + weight * F.mse_loss(feat_f[k], target)

    gram_loss = F.mse_loss(utils.gram_matrix(feat_f[3]), utils.gram_matrix(feat_ir[3]))
    return content_loss + args.lambda_style * gram_loss, content_loss, gram_loss


def build_model(device):
    model = DULRNet(
        s=args.s,
        n=args.n,
        channel=args.channel,
        stride=args.stride,
        num_stages=args.num_stages,
        cd=args.cd,
    )
    return model.to(device)


def train():
    set_seed(args.seed)
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    ir_paths, _ = utils.list_images(args.train_ir_dir)
    if args.train_num is not None:
        ir_paths = ir_paths[:args.train_num]
    if len(ir_paths) == 0:
        raise RuntimeError(f'No training images found in {args.train_ir_dir}')

    model = build_model(device)

    vgg = Vgg16().to(device).eval()
    utils.init_vgg16(vgg, args.vgg_model_path)
    for p in vgg.parameters():
        p.requires_grad = False

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup = LinearLR(optimizer, start_factor=0.1, total_iters=args.warmup_epochs)
    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs - args.warmup_epochs, 1),
        eta_min=args.eta_min,
    )
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[args.warmup_epochs])

    utils.ensure_dir(args.save_model_dir)
    utils.ensure_dir(args.save_loss_dir)
    writer = SummaryWriter(log_dir=os.path.join(args.save_loss_dir, time.strftime('dulrnet_%Y%m%d_%H%M%S')))

    total_steps = args.epochs * max(len(ir_paths) // args.batch_size, 1)
    global_step = 0
    print('Start training DULRNet.')

    for epoch in range(args.epochs):
        batched_paths, batch_num = utils.load_dataset(ir_paths, args.batch_size)
        model.train()

        for batch_idx in range(batch_num):
            batch_paths = batched_paths[batch_idx * args.batch_size:(batch_idx + 1) * args.batch_size]
            ir_255, vi_255 = utils.load_paired_images(batch_paths, args.train_vi_dir, args.height, args.width)
            ir_01 = utils.to_01(ir_255).to(device)
            vi_01 = utils.to_01(vi_255).to(device)

            optimizer.zero_grad(set_to_none=True)

            output = model(ir_01, vi_01)
            fused_01 = torch.clamp(output['fuse'], 0.0, 1.0)

            loss_pixel = F.mse_loss(fused_01, vi_01) + args.gamma_ir * F.mse_loss(fused_01, ir_01)
            loss_ssim = (
                args.omega_vi * ssim_loss_value(fused_01, vi_01) +
                args.omega_ir * ssim_loss_value(fused_01, ir_01)
            )
            loss_grad = gradient_loss(fused_01, ir_01, vi_01)
            w_ir = current_ir_weight(global_step, total_steps)
            loss_feat, loss_content, loss_gram = feature_and_style_loss(vgg, fused_01, ir_01, vi_01, w_ir)

            total_loss = (
                args.lambda_pix * loss_pixel +
                args.lambda_ssim * loss_ssim +
                args.lambda_grad * loss_grad +
                args.lambda_feat * loss_feat
            )

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            if global_step % 10 == 0:
                lr = optimizer.param_groups[0]['lr']
                writer.add_scalar('Loss/total', total_loss.item(), global_step)
                writer.add_scalar('Loss/pixel', loss_pixel.item(), global_step)
                writer.add_scalar('Loss/ssim', loss_ssim.item(), global_step)
                writer.add_scalar('Loss/gradient', loss_grad.item(), global_step)
                writer.add_scalar('Loss/feature_content', loss_content.item(), global_step)
                writer.add_scalar('Loss/gram', loss_gram.item(), global_step)
                writer.add_scalar('Weight/w_ir', w_ir, global_step)
                writer.add_scalar('Learning_rate', lr, global_step)
                print(
                    f'{time.ctime()} | Epoch [{epoch + 1}/{args.epochs}] '
                    f'Batch [{batch_idx + 1}/{batch_num}] | LR {lr:.2e} | '
                    f'w_ir {w_ir:.3f} | Loss {total_loss.item():.6f}'
                )

            global_step += 1

        scheduler.step()

        if (epoch + 1) % args.save_interval == 0 or (epoch + 1) == args.epochs:
            save_path = os.path.join(args.save_model_dir, f'dulrnet_epoch_{epoch + 1}.pth')
            torch.save(model.state_dict(), save_path)
            print(f'Checkpoint saved: {save_path}')

    final_path = os.path.join(args.save_model_dir, 'dulrnet_final.pth')
    torch.save(model.state_dict(), final_path)
    writer.close()
    print(f'Done. Final model saved: {final_path}')


if __name__ == '__main__':
    train()
