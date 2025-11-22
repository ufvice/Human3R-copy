#!/usr/bin/env python3
"""
Stage 0: minimal XLA sanity check with a tiny ConvNet on dummy data.

Usage:
  python scripts/profile_xla_stage0_simple.py --device xla --steps 100
"""

import argparse
import time

import torch

try:
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
except ImportError:
    xm = None
    met = None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="xla")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    return parser.parse_args()


def main():
    args = parse_args()

    if isinstance(args.device, str):
        if args.device == "xla":
            if xm is None:
                print("[Stage0] XLA not available, falling back to CPU.")
                device = torch.device("cpu")
            else:
                device = xm.xla_device()
        else:
            device = torch.device(args.device)
    else:
        device = args.device

    print(f"[Stage0] Using device: {device}")

    # Tiny ConvNet
    model = torch.nn.Sequential(
        torch.nn.Conv2d(3, 16, kernel_size=3, padding=1),
        torch.nn.ReLU(inplace=True),
        torch.nn.Conv2d(16, 16, kernel_size=3, padding=1),
        torch.nn.ReLU(inplace=True),
        torch.nn.AdaptiveAvgPool2d(1),
        torch.nn.Flatten(),
        torch.nn.Linear(16, 10),
    ).to(device)
    model.eval()

    dummy = torch.randn(
        args.batch_size, 3, args.height, args.width, device=device, dtype=torch.float32
    )

    if met is not None and xm is not None and "xla" in str(device):
        met.clear_metrics()

    print(f"[Stage0] Running {args.steps} forward passes...")
    start = time.time()
    with torch.no_grad():
        for step in range(args.steps):
            out = model(dummy)
            if xm is not None and "xla" in str(device):
                xm.mark_step()
    if xm is not None and "xla" in str(device):
        xm.wait_device_ops()
    elapsed = time.time() - start
    print(f"[Stage0] Elapsed: {elapsed:.2f}s for {args.steps} steps.")

    if met is not None and xm is not None and "xla" in str(device):
        print("\n[Stage0] XLA metrics_report():\n")
        print(met.metrics_report())


if __name__ == "__main__":
    main()

