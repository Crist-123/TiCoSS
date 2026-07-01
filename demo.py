from __future__ import print_function, division

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

sys.path.append("core")

from core.JLNet import JLnet, autocast
from core.utils.utils import InputPadder


COLORMAP = np.array(
    [
        [128, 64, 128],   # Road
        [244, 35, 232],   # Sidewalk
        [70, 70, 70],     # Building
        [102, 102, 156],  # Wall
        [190, 153, 153],  # Fence
        [153, 153, 153],  # Pole
        [250, 170, 30],   # Traffic light
        [220, 220, 0],    # Traffic sign
        [107, 142, 35],   # Vegetation
        [152, 251, 152],  # Terrain
        [70, 130, 180],   # Sky
        [220, 20, 60],    # Person
        [255, 0, 0],      # Rider
        [0, 0, 142],      # Car
        [0, 0, 70],       # Truck
        [0, 60, 100],     # Bus
        [0, 80, 100],     # Train
        [0, 0, 230],      # Motorcycle
        [119, 11, 32],    # Bicycle
    ],
    dtype=np.uint8,
)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run TiCoSS on one stereo pair and visualize disparity and semantic segmentation."
    )
    parser.add_argument("--left", required=True, help="path to the left image")
    parser.add_argument("--right", required=True, help="path to the right image")
    parser.add_argument(
        "--restore_ckpt",
        default="./checkpoints/checkpoint.pth",
        help="checkpoint path",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="save visualizations to disk",
    )
    parser.add_argument(
        "--save_path",
        "--output_dir",
        dest="save_path",
        default="./results",
        help="directory used when --save is set",
    )
    parser.add_argument(
        "--output_name",
        default=None,
        help="file name prefix; default is the left image stem",
    )
    parser.add_argument(
        "--no_display",
        action="store_true",
        help="do not open a matplotlib window when --save is not set",
    )
    parser.add_argument("--valid_iters", type=int, default=32, help="number of update iterations")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="inference device")

    parser.add_argument("--mixed_precision", action="store_true", help="use mixed precision")
    parser.add_argument("--is_train", default=False, help="kept for checkpoint compatibility")

    parser.add_argument("--hidden_dims", nargs="+", type=int, default=[128] * 3)
    parser.add_argument(
        "--corr_implementation",
        choices=["reg", "alt", "reg_cuda", "alt_cuda"],
        default="reg",
    )
    parser.add_argument("--shared_backbone", action="store_true")
    parser.add_argument("--corr_levels", type=int, default=4)
    parser.add_argument("--corr_radius", type=int, default=4)
    parser.add_argument("--n_downsample", type=int, default=2)
    parser.add_argument("--slow_fast_gru", action="store_true")
    parser.add_argument("--n_gru_layers", type=int, default=3)
    parser.add_argument("--scale", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=1.0)
    parser.add_argument("--punishment", type=float, default=0.5)
    return parser


def read_image(path):
    image = Image.open(path).convert("RGB")
    image_np = np.array(image).astype(np.uint8)
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()
    return image_np, image_tensor


def unwrap_checkpoint(checkpoint):
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]
    return checkpoint


def load_model(args, device):
    model = JLnet(args)
    checkpoint = torch.load(args.restore_ckpt, map_location=device)
    state_dict = unwrap_checkpoint(checkpoint)

    if all(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {key[len("module."):]: value for key, value in state_dict.items()}

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"Warning: missing keys when loading checkpoint: {len(missing_keys)}")
    if unexpected_keys:
        print(f"Warning: unexpected keys when loading checkpoint: {len(unexpected_keys)}")

    model.to(device)
    model.eval()
    print(f"Loaded {args.restore_ckpt}")
    print(f"The model has {format(count_parameters(model) / 1e6, '.2f')}M learnable parameters.")
    return model


def normalize_disparity(disparity):
    finite_mask = np.isfinite(disparity)
    if not finite_mask.any():
        return np.zeros_like(disparity, dtype=np.uint8)

    valid_values = disparity[finite_mask]
    disp_min = valid_values.min()
    disp_max = valid_values.max()
    if disp_max - disp_min < 1e-6:
        return np.zeros_like(disparity, dtype=np.uint8)

    disparity_vis = (disparity - disp_min) / (disp_max - disp_min)
    disparity_vis = np.clip(disparity_vis * 255.0, 0, 255).astype(np.uint8)
    return disparity_vis


def visualize_disparity(disparity):
    disparity_gray = normalize_disparity(disparity)
    disparity_bgr = cv2.applyColorMap(disparity_gray, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(disparity_bgr, cv2.COLOR_BGR2RGB)


@torch.no_grad()
def run_demo(args):
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, falling back to CPU.")
        args.device = "cpu"

    device = torch.device(args.device)
    left_np, left_tensor = read_image(args.left)
    right_np, right_tensor = read_image(args.right)

    if left_np.shape[:2] != right_np.shape[:2]:
        raise ValueError(
            f"Left and right images must have the same size, got {left_np.shape[:2]} and {right_np.shape[:2]}."
        )

    model = load_model(args, device)

    image1 = left_tensor[None].to(device)
    image2 = right_tensor[None].to(device)

    padder = InputPadder(image1.shape, divis_by=32)
    image1, image2 = padder.pad(image1, image2)

    use_mixed_precision = args.mixed_precision or args.corr_implementation.endswith("_cuda")
    with autocast(enabled=use_mixed_precision):
        flow_pr, seg = model(
            image1,
            image2,
            image1,
            image2,
            threshold=args.threshold,
            iters=args.valid_iters,
            test_mode=True,
        )

    flow_pr = padder.unpad(flow_pr)
    seg = padder.unpad(seg)

    disparity = -flow_pr.squeeze(0).squeeze(0).detach().float().cpu().numpy()
    disparity_vis = visualize_disparity(disparity)

    pred_label = seg.argmax(1).squeeze(0).detach().cpu().numpy().astype(np.uint8)
    segmentation_vis = COLORMAP[pred_label]

    if args.save:
        save_dir = Path(args.save_path)
        save_dir.mkdir(parents=True, exist_ok=True)
        prefix = args.output_name or Path(args.left).stem

        Image.fromarray(disparity_vis).save(save_dir / f"{prefix}_disp.png")
        Image.fromarray(segmentation_vis).save(save_dir / f"{prefix}_seg.png")

        print(f"Saved visualizations to {save_dir.resolve()}")
        print(f"  {prefix}_disp.png")
        print(f"  {prefix}_seg.png")

    if not args.save and not args.no_display:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(2, 2, figsize=(12, 7))
            axes[0, 0].imshow(left_np)
            axes[0, 0].set_title("Left")
            axes[0, 1].imshow(right_np)
            axes[0, 1].set_title("Right")
            axes[1, 0].imshow(disparity_vis)
            axes[1, 0].set_title("Disparity")
            axes[1, 1].imshow(segmentation_vis)
            axes[1, 1].set_title("Segmentation")
            for ax in axes.reshape(-1):
                ax.axis("off")
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("matplotlib is not installed; rerun with --save to write visualizations.")


if __name__ == "__main__":
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    run_demo(build_parser().parse_args())
