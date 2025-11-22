#!/usr/bin/env python3
"""
Stage 1: run ARCroco3DStereo.recurrent_lighter on dummy views only (no disk I/O),
and print XLA metrics to understand pure model cost.

Usage:
  python scripts/profile_xla_stage1_model_only_dummy.py \\
      --model_path src/cut3r_512_dpt_4_64.pth \\
      --device xla --num_frames 60 --size 512
"""

import argparse
import time

import torch

from add_ckpt_path import add_path_to_dust3r

try:
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
except ImportError:
    xm = None
    met = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="xla")
    parser.add_argument("--num_frames", type=int, default=60)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--mhmr_size", type=int, default=896)
    parser.add_argument("--use_ttt3r", action="store_true", default=False)
    return parser.parse_args()


def build_dummy_views(num_frames, size, mhmr_size, device_str):
    """
    Build a list of dummy views compatible with forward_recurrent_lighter.
    All tensors start on CPU and will be moved to XLA inside to_gpu().
    """
    from src.dust3r.utils.geometry import get_camera_parameters

    views = []
    # Camera intrinsics for MHMR branch can be any reasonable value on CPU.
    K_mhmr = get_camera_parameters(mhmr_size, device="cpu")

    for idx in range(num_frames):
        img = torch.randn(1, 3, size, size, dtype=torch.float32)
        img_mhmr = torch.randn(1, 3, mhmr_size, mhmr_size, dtype=torch.float32)
        true_shape = torch.tensor([[size, size]], dtype=torch.int32)

        view = {
            "img": img,
            "true_shape": true_shape,
            "idx": idx,
            "instance": str(idx),
            "img_mask": torch.tensor(True).unsqueeze(0),
            "ray_map": torch.full((1, 6, size, size), torch.nan, dtype=torch.float32),
            "ray_mask": torch.tensor(False).unsqueeze(0),
            "update": torch.tensor(True).unsqueeze(0),
            "reset": torch.tensor(False).unsqueeze(0),
            "img_mhmr": img_mhmr,
            "K_mhmr": K_mhmr,
        }
        views.append(view)

    return views


def main():
    args = parse_args()

    # Resolve device
    if isinstance(args.device, str):
        if args.device == "xla":
            if xm is None:
                print("[Stage1] XLA not available, falling back to CPU.")
                device = torch.device("cpu")
            else:
                device = xm.xla_device()
        else:
            device = torch.device(args.device)
    else:
        device = args.device

    print(f"[Stage1] Using device: {device}")

    # Make sure dust3r package can be imported from the checkpoint path.
    add_path_to_dust3r(args.model_path)

    from src.dust3r.inference import inference_recurrent_lighter
    from src.dust3r.model import ARCroco3DStereo

    print(f"[Stage1] Loading model from {args.model_path} ...")
    model = ARCroco3DStereo.from_pretrained(args.model_path).to(device)
    model.eval()

    print(
        f"[Stage1] Building {args.num_frames} dummy views "
        f"with size={args.size} and mhmr_size={args.mhmr_size} ..."
    )
    views = build_dummy_views(
        num_frames=args.num_frames,
        size=args.size,
        mhmr_size=args.mhmr_size,
        device_str=str(device),
    )

    if met is not None and xm is not None and "xla" in str(device):
        met.clear_metrics()
        print("[Stage1] Cleared XLA metrics before dummy inference.")

    print("[Stage1] Running inference_recurrent_lighter on dummy data ...")
    start = time.time()
    with torch.no_grad():
        outputs, _ = inference_recurrent_lighter(
            views, model, device, verbose=True, use_ttt3r=args.use_ttt3r
        )
        # outputs are kept on CPU via to_cpu inside inference_recurrent_lighter.
    if xm is not None and "xla" in str(device):
        xm.mark_step()
        xm.wait_device_ops()
    elapsed = time.time() - start

    print(
        f"[Stage1] Dummy inference completed in {elapsed:.2f}s "
        f"for {args.num_frames} frames."
    )

    if met is not None and xm is not None and "xla" in str(device):
        print("\n[Stage1] XLA metrics_report():\n")
        print(met.metrics_report())


if __name__ == "__main__":
    main()

