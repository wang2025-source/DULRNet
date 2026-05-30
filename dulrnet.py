# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.ops import DeformConv2d
except Exception:
    DeformConv2d = None


def soft_threshold(x, theta):
    b, c, h, w = x.shape
    theta = theta.view(1, c, 1, 1).expand(b, c, h, w)
    return torch.sign(x) * torch.clamp(torch.abs(x) - theta, min=0.0)


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            bias=bias,
        )

    def forward(self, x):
        return self.conv(x)


class LRSDStage(nn.Module):
    def __init__(self, kernel_size=3, latent_channels=256, image_channels=1, stride=1):
        super().__init__()
        self.W_d = ConvLayer(latent_channels, image_channels, kernel_size, stride)
        self.W_e = ConvLayer(image_channels, latent_channels, kernel_size, stride)
        self.theta = nn.Parameter(torch.rand(1, latent_channels))
        self.lambda_z = nn.Parameter(torch.rand(1, 1))

    def forward(self, image, z):
        residual = image - self.W_d(z)
        z = self.lambda_z * z + self.W_e(residual)
        return soft_threshold(z, self.theta)


class LRSDEncoder(nn.Module):
    def __init__(self, kernel_size=3, n=128, image_channels=1, stride=1, num_stages=8):
        super().__init__()
        self.n = n
        latent_channels = 2 * n
        self.initial_projection = ConvLayer(image_channels, latent_channels, kernel_size, stride)
        self.initial_theta = nn.Parameter(torch.rand(1, latent_channels))
        self.stages = nn.ModuleList([
            LRSDStage(kernel_size, latent_channels, image_channels, stride)
            for _ in range(num_stages)
        ])

    def forward(self, x):
        z = soft_threshold(self.initial_projection(x), self.initial_theta)
        for stage in self.stages:
            z = stage(x, z)
        low_rank = z[:, :self.n, :, :]
        sparse = z[:, self.n:2 * self.n, :, :]
        return low_rank, sparse


class CGFM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        in_channels = 2 * channels
        self.branch_1 = self._branch(in_channels, channels, 1)
        self.branch_3 = self._branch(in_channels, channels, 3)
        self.branch_5 = self._branch(in_channels, channels, 5)
        self.branch_7 = self._branch(in_channels, channels, 7)
        self.refine = nn.Sequential(
            nn.Conv2d(4 * channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.compress = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _branch(in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1, inplace=True),
        )

    def forward(self, main_l, aux_l):
        f_cat = torch.cat([main_l, aux_l], dim=1)
        multi_scale = torch.cat(
            [self.branch_1(f_cat), self.branch_3(f_cat), self.branch_5(f_cat), self.branch_7(f_cat)],
            dim=1,
        )
        gate = self.compress(self.refine(multi_scale))
        return main_l + gate * aux_l, gate


class ChannelSE(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.down = nn.Conv2d(channels, hidden, 1)
        self.up = nn.Conv2d(hidden, channels, 1)
        self.act = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.up(self.act(self.down(self.avg_pool(x)))))


class SobelGradient(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        kernel_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])
        self.register_buffer("kernel_x", kernel_x.view(1, 1, 3, 3))
        self.register_buffer("kernel_y", kernel_y.view(1, 1, 3, 3))

    def forward(self, x):
        channels = x.shape[1]
        grad_x = F.conv2d(x, self.kernel_x.repeat(channels, 1, 1, 1), padding=1, groups=channels)
        grad_y = F.conv2d(x, self.kernel_y.repeat(channels, 1, 1, 1), padding=1, groups=channels)
        return torch.sqrt(grad_x.pow(2) + grad_y.pow(2) + 1e-6)


class DeformableSpatialExcitation(nn.Module):
    def __init__(self, channels, use_gradient=False):
        super().__init__()
        if DeformConv2d is None:
            raise ImportError(
                "DULRNet requires torchvision.ops.DeformConv2d. "
                "Please install compatible versions of torch and torchvision."
            )
        self.use_gradient = use_gradient
        self.sobel = SobelGradient() if use_gradient else None
        self.pre = nn.Sequential(
            nn.Conv2d(channels, channels, 1),
            nn.ReLU(inplace=True),
        )
        self.offset = nn.Conv2d(channels, 18, 3, padding=1)
        self.deform = DeformConv2d(channels, 1, 3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.use_gradient:
            x = x + self.sobel(x)
        feat = self.pre(x)
        return self.sigmoid(self.deform(feat, self.offset(feat)))


class ABCA(nn.Module):
    def __init__(self, channels=128, reduction=8, alpha=4.0, gamma=0.10):
        super().__init__()
        self.channel_ir = ChannelSE(channels, reduction)
        self.channel_vi = ChannelSE(channels, reduction)
        self.spatial_ir = DeformableSpatialExcitation(channels, use_gradient=False)
        self.spatial_vi = DeformableSpatialExcitation(channels, use_gradient=True)
        self.align_ir = nn.Conv2d(channels, channels, 1, bias=False)
        self.align_vi = nn.Conv2d(channels, channels, 1, bias=False)
        self.register_buffer("alpha", torch.tensor(float(alpha)))
        self.register_buffer("gamma", torch.tensor(float(gamma)))

    def forward(self, low_ir, sparse_ir, low_vi, sparse_vi):
        a_ir = self.channel_ir(sparse_ir) * self.spatial_ir(sparse_ir)
        a_vi = self.channel_vi(sparse_vi) * self.spatial_vi(sparse_vi)
        sparse_vi_enh = sparse_vi * (1.0 + self.alpha * a_ir) + self.align_ir(sparse_ir) * a_ir
        sparse_ir_enh = sparse_ir * (1.0 + self.gamma * a_vi) + self.align_vi(sparse_vi) * a_vi
        return low_ir, sparse_ir_enh, low_vi, sparse_vi_enh, a_ir, a_vi


class HACB(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv_1x3 = nn.Conv2d(channels, channels, (1, 3), padding=(0, 1))
        self.conv_3x1 = nn.Conv2d(channels, channels, (3, 1), padding=(1, 0))
        self.conv_3x3 = nn.Conv2d(channels, channels, 3, padding=1)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(self.conv_1x3(x) + self.conv_3x1(x) + self.conv_3x3(x))


class CBAM(nn.Module):
    def __init__(self, channels, reduction=8, spatial_kernel=7):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=False),
        )
        self.channel_sigmoid = nn.Sigmoid()
        self.spatial_conv = nn.Conv2d(2, 1, spatial_kernel, padding=spatial_kernel // 2, bias=False)
        self.spatial_sigmoid = nn.Sigmoid()

    def forward(self, x):
        channel_mask = self.channel_sigmoid(self.shared_mlp(self.avg_pool(x)) + self.shared_mlp(self.max_pool(x)))
        x = x * channel_mask
        avg_map = torch.mean(x, dim=1, keepdim=True)
        max_map, _ = torch.max(x, dim=1, keepdim=True)
        spatial_mask = self.spatial_sigmoid(self.spatial_conv(torch.cat([avg_map, max_map], dim=1)))
        return x * spatial_mask


class EnhancedDecoder(nn.Module):
    def __init__(self, n=128, out_channels=1, cd=64):
        super().__init__()
        self.phi_l_ir = nn.Conv2d(n, cd, 1)
        self.phi_l_vi = nn.Conv2d(n, cd, 1)
        self.phi_s_ir = nn.Conv2d(n, cd, 1)
        self.phi_s_vi = nn.Conv2d(n, cd, 1)
        self.compress_low = nn.Conv2d(2 * cd, cd, 1)
        self.compress_high = nn.Conv2d(2 * cd, cd, 1)
        self.hacb_low = HACB(cd)
        self.hacb_high = HACB(cd)
        self.cbam = CBAM(cd, reduction=8, spatial_kernel=7)
        self.lambda_L = nn.Parameter(torch.tensor(0.5))
        self.lambda_S = nn.Parameter(torch.tensor(0.5))
        self.omega_L = nn.Parameter(torch.tensor(0.4))
        self.omega_H = nn.Parameter(torch.tensor(2.0))
        self.output_projection = nn.Conv2d(cd, out_channels, 1)
        self.lambda_res_vi = nn.Parameter(torch.tensor(0.05))
        self.lambda_res_ir = nn.Parameter(torch.tensor(1.2))
        self.lambda_diff = nn.Parameter(torch.tensor(0.5))
        self.rho = nn.Parameter(torch.tensor(0.9))

    @staticmethod
    def _positive_weight(x):
        return F.softplus(x)

    def forward(self, low_ir, sparse_ir, low_vi, sparse_vi, raw_vi, raw_ir):
        l_ir = self.phi_l_ir(low_ir)
        l_vi = self.phi_l_vi(low_vi)
        s_ir = self.phi_s_ir(sparse_ir)
        s_vi = self.phi_s_vi(sparse_vi)

        f_low = self.hacb_low(self.compress_low(torch.cat([l_ir, l_vi], dim=1)))
        f_high = self.hacb_high(self.compress_high(torch.cat([s_ir, s_vi], dim=1)))

        psi_low = self.cbam(f_low)
        psi_high = self.cbam(f_high)

        f_dec = self._positive_weight(self.omega_L) * (psi_low + torch.sigmoid(self.lambda_L) * f_low)
        f_dec = f_dec + self._positive_weight(self.omega_H) * (psi_high + torch.sigmoid(self.lambda_S) * f_high)
        f_dec = F.leaky_relu(f_dec, negative_slope=0.1, inplace=False)
        f_dec = torch.sigmoid(self.output_projection(f_dec))

        residual = (
            self.lambda_res_vi * raw_vi
            + self.lambda_res_ir * raw_ir
            + self.lambda_diff * torch.abs(raw_ir - raw_vi)
        )
        fused = torch.clamp(f_dec + residual, min=0.0)
        fused = torch.pow(fused + 1e-6, torch.clamp(self.rho, min=0.1, max=3.0))
        return torch.clamp(fused, 0.0, 1.0), l_ir, s_ir, l_vi, s_vi, f_low, f_high, psi_low, psi_high


class DULRNet(nn.Module):
    def __init__(self, s=3, n=128, channel=1, stride=1, num_stages=8, cd=64):
        super().__init__()
        self.encoder_ir = LRSDEncoder(s, n, channel, stride, num_stages)
        self.encoder_vi = LRSDEncoder(s, n, channel, stride, num_stages)
        self.encoder_ir_aux = self.encoder_ir
        self.encoder_vi_aux = self.encoder_vi
        self.cgfm_ir = CGFM(n)
        self.cgfm_vi = CGFM(n)
        self.abca = ABCA(channels=n, reduction=8, alpha=4.0, gamma=0.10)
        self.decoder = EnhancedDecoder(n=n, out_channels=channel, cd=cd)

    def forward(self, ir, vi):
        l_ir_main, s_ir_main = self.encoder_ir(ir)
        l_vi_main, s_vi_main = self.encoder_vi(vi)

        ir_down = F.avg_pool2d(ir, kernel_size=2, stride=2)
        vi_down = F.avg_pool2d(vi, kernel_size=2, stride=2)

        l_ir_aux, _ = self.encoder_ir_aux(ir_down)
        l_vi_aux, _ = self.encoder_vi_aux(vi_down)

        l_ir_aux = F.interpolate(l_ir_aux, size=l_ir_main.shape[-2:], mode='bilinear', align_corners=False)
        l_vi_aux = F.interpolate(l_vi_aux, size=l_vi_main.shape[-2:], mode='bilinear', align_corners=False)

        l_ir_enh, gate_ir = self.cgfm_ir(l_ir_main, l_ir_aux)
        l_vi_enh, gate_vi = self.cgfm_vi(l_vi_main, l_vi_aux)

        l_ir_enh, s_ir_enh, l_vi_enh, s_vi_enh, a_ir, a_vi = self.abca(
            l_ir_enh, s_ir_main, l_vi_enh, s_vi_main
        )

        fused, x_l, x_h, y_l, y_h, f_low, f_high, psi_low, psi_high = self.decoder(
            l_ir_enh, s_ir_enh, l_vi_enh, s_vi_enh, raw_vi=vi, raw_ir=ir
        )

        return {
            'fea_x_l': l_ir_enh,
            'fea_x_s': s_ir_enh,
            'fea_y_l': l_vi_enh,
            'fea_y_s': s_vi_enh,
            'gate_ir': gate_ir,
            'gate_vi': gate_vi,
            'att_ir': a_ir,
            'att_vi': a_vi,
            'x_l': x_l,
            'x_h': x_h,
            'y_l': y_l,
            'y_h': y_h,
            'fl': f_low,
            'fh': f_high,
            'psi_low': psi_low,
            'psi_high': psi_high,
            'fuse': fused,
        }


class Vgg16(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1_1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 1)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 1)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.conv4_3 = nn.Conv2d(512, 512, 3, 1, 1)

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        h = F.relu(self.conv1_1(x), inplace=True)
        h = F.relu(self.conv1_2(h), inplace=True)
        relu1_2 = h
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.conv2_1(h), inplace=True)
        h = F.relu(self.conv2_2(h), inplace=True)
        relu2_2 = h
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.conv3_1(h), inplace=True)
        h = F.relu(self.conv3_2(h), inplace=True)
        h = F.relu(self.conv3_3(h), inplace=True)
        relu3_3 = h
        h = F.max_pool2d(h, 2, 2)

        h = F.relu(self.conv4_1(h), inplace=True)
        h = F.relu(self.conv4_2(h), inplace=True)
        h = F.relu(self.conv4_3(h), inplace=True)
        relu4_3 = h
        return [relu1_2, relu2_2, relu3_3, relu4_3]
