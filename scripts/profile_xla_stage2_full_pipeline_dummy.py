#!/usr/bin/env python3
"""
Stage 2: run the full demo-style pipeline on dummy views:
  - ARCroco3DStereo.recurrent_lighter on dummy inputs
  - prepare_output (including SMPL + file writing) on a small number of frames

This isolates how much time is spent in:
  - model forward on XLA
  - data transfer back to CPU
  - heavy CPU-side post-processing (numpy/SMPL/disk I/O)

Usage:
  python scripts/profile_xla_stage2_full_pipeline_dummy.py \\
      --model_path src/cut3r_512_dpt_4_64.pth \\
      --device xla --num_frames 10 --size 512 --output_dir ./tmp_dummy_profile
"""

import argparse
import os
import sys
import time

import torch

# Ensure repository root is on sys.path so that `add_ckpt_path`,
# `demo_debug`, and `src.*` imports work regardless of the current
# working directory.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

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
    parser.add_argument("--num_frames", type=int, default=10)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--mhmr_size", type=int, default=896)
    parser.add_argument("--use_ttt3r", action="store_true", default=False)
    parser.add_argument("--output_dir", type=str, default="./tmp_dummy_profile")
    parser.add_argument("--save_smpl", action="store_true", default=False)
    parser.add_argument("--save_video", action="store_true", default=False)
    return parser.parse_args()


def build_dummy_views(num_frames, size, mhmr_size):
    from src.dust3r.utils.geometry import get_camera_parameters

    views = []
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
                print("[Stage2] XLA not available, falling back to CPU.")
                device = torch.device("cpu")
            else:
                device = xm.xla_device()
        else:
            device = torch.device(args.device)
    else:
        device = args.device

    print(f"[Stage2] Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Ensure dust3r can be imported from checkpoint path.
    ckpt_path = args.model_path
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(REPO_ROOT, ckpt_path)
    add_path_to_dust3r(ckpt_path)

    from src.dust3r.inference import inference_recurrent_lighter
    from src.dust3r.model import ARCroco3DStereo
    from demo_debug import prepare_output  # reuse the existing post-processing

    print(f"[Stage2] Loading model from {ckpt_path} ...")
    model = ARCroco3DStereo.from_pretrained(ckpt_path).to(device)
    model.eval()

    print(
        f"[Stage2] Building {args.num_frames} dummy views "
        f"with size={args.size} and mhmr_size={args.mhmr_size} ..."
    )
    views = build_dummy_views(
        num_frames=args.num_frames, size=args.size, mhmr_size=args.mhmr_size
    )

    if met is not None and xm is not None and "xla" in str(device):
        met.clear_metrics()
        print("[Stage2] Cleared XLA metrics before dummy full pipeline.")

    # 1) Model forward on TPU
    print("[Stage2] Running inference_recurrent_lighter on dummy data ...")
    start_model = time.time()
    with torch.no_grad():
        outputs, _ = inference_recurrent_lighter(
            views, model, device, verbose=True, use_ttt3r=args.use_ttt3r
        )
    if xm is not None and "xla" in str(device):
        xm.mark_step()
        xm.wait_device_ops()
    end_model = time.time()

    print(
        f"[Stage2] Dummy model inference time: {end_model - start_model:.2f}s "
        f"for {args.num_frames} frames."
    )

    # 2) CPU-side prepare_output (includes numpy, SMPL, disk I/O)
    print("[Stage2] Running prepare_output (CPU post-processing + save) ...")
    start_post = time.time()
    _ = prepare_output(
        outputs,
        args.output_dir,
        revisit=1,
        use_pose=True,
        save_smpl=args.save_smpl,
        save_video=args.save_video,
        img_res=args.mhmr_size,
        subsample=1,
    )
    end_post = time.time()

    print(
        f"[Stage2] prepare_output time: {end_post - start_post:.2f}s "
        f"(see {args.output_dir} for saved dummy results)."
    )

    if met is not None and xm is not None and "xla" in str(device):
        print("\n[Stage2] XLA metrics_report():\n")
        print(met.metrics_report())


if __name__ == "__main__":
    main()
