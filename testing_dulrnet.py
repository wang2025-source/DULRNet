import os

from args import Args as args

os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

import torch

from dulrnet import DULRNet
import utils


def build_model(device):
    model = DULRNet(
        s=args.s,
        n=args.n,
        channel=args.channel,
        stride=args.stride,
        num_stages=args.num_stages,
        cd=args.cd,
    )

    state_dict = torch.load(args.model_path, map_location='cpu')
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    if len(state_dict) > 0 and list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=True)
    model.to(device).eval()

    params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f'DULRNet parameters: {params:.3f} M')
    return model


def paired_visible_path(ir_path):
    file_name = os.path.basename(ir_path)
    direct_path = os.path.join(args.test_vi_dir, file_name)
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
        candidate_path = os.path.join(args.test_vi_dir, candidate)
        if os.path.exists(candidate_path):
            return candidate_path

    raise FileNotFoundError(f'Cannot find visible pair for {ir_path}')


def run():
    device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')
    utils.ensure_dir(args.output_dir)

    model = build_model(device)
    ir_paths, _ = utils.list_images(args.test_ir_dir)
    if len(ir_paths) == 0:
        raise RuntimeError(f'No test images found in {args.test_ir_dir}')

    with torch.no_grad():
        for ir_path in ir_paths:
            vi_path = paired_visible_path(ir_path)
            ir = utils.to_01(utils.get_train_images(ir_path, height=None, width=None, mode='ir')).to(device)
            vi = utils.to_01(utils.get_train_images(vi_path, height=None, width=None, mode='vi')).to(device)

            output = model(ir, vi)
            fused = output['fuse']

            out_name = os.path.basename(ir_path)
            out_path = os.path.join(args.output_dir, out_name)
            utils.save_image(fused, out_path)
            print(f'Saved: {out_path}')


if __name__ == '__main__':
    run()
