#!/usr/bin/env python3
"""
Compare the numerical difference between the original roma-based
rot6d_to_rotmat implementation and the new XLA-friendly implementation
in src/dust3r/heads/postprocess.py.

Usage:
    python scripts/compare_rot6d_roma.py

Requirements:
    pip install roma
"""

import os
import sys
from typing import Tuple

import torch
import torch.nn.functional as F
import roma


def _add_src_to_path() -> None:
    """Ensure `src` is on sys.path so we can import dust3r modules."""
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src_dir = os.path.join(root_dir, "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)


_add_src_to_path()

from dust3r.heads.postprocess import rot6d_to_rotmat as rot6d_manual  # noqa E402


def rot6d_to_rotmat_roma(rot6d: torch.Tensor, naive_mode: bool = False) -> torch.Tensor:
    """
    Reference implementation using roma.special_gramschmidt, matching
    the original code that existed in postprocess.py before the XLA fix.

    Args:
        rot6d: Tensor of shape (N, 6), 6D rotation representation.
        naive_mode: Whether to use the MHMR-style reshape branch.

    Returns:
        Tensor of shape (N, 3, 3) with rotation matrices.
    """
    if naive_mode:
        # (N, 6) -> (N, 2, 3) -> (N, 3, 2)
        mat_3x2 = rot6d.reshape(-1, 2, 3).permute(0, 2, 1).contiguous()
    else:
        # (N, 6) -> (N, 3, 2)
        mat_3x2 = rot6d.reshape(-1, 3, 2).contiguous()

    rot_mat = roma.special_gramschmidt(mat_3x2, epsilon=1e-6)
    return rot_mat


def rotation_error(
    r_ref: torch.Tensor, r_test: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute angular and Frobenius errors between two batches of rotation matrices.

    Args:
        r_ref: Reference rotation matrices, shape (N, 3, 3).
        r_test: Test rotation matrices, shape (N, 3, 3).

    Returns:
        angle_deg: Per-matrix angular error in degrees, shape (N,).
        frob_norm: Per-matrix Frobenius norm of (r_ref - r_test), shape (N,).
    """
    # Relative rotation: r_rel = r_ref^T * r_test
    r_rel = torch.matmul(r_ref.transpose(1, 2), r_test)
    trace = r_rel.diagonal(dim1=1, dim2=2).sum(-1)
    cos_theta = (trace - 1.0) / 2.0
    cos_theta_clamped = torch.clamp(cos_theta, -1.0 + 1e-7, 1.0 - 1e-7)
    angle_rad = torch.acos(cos_theta_clamped)
    angle_deg = angle_rad * (180.0 / torch.pi)

    frob_norm = torch.linalg.norm(r_ref - r_test, dim=(1, 2))
    return angle_deg, frob_norm


def main() -> None:
    device = torch.device("cpu")

    # Number of random samples to test.
    num_samples = 4096

    # Random 6D rotations, close to the unit sphere to mimic realistic outputs.
    rot6d = torch.randn(num_samples, 6, device=device)
    rot6d = F.normalize(rot6d, dim=-1)

    for naive_mode in (False, True):
        print(f"\n=== Comparing with naive_mode={naive_mode} ===")

        # Reference roma-based implementation.
        rot_roma = rot6d_to_rotmat_roma(rot6d, naive_mode=naive_mode)

        # New XLA-friendly implementation from dust3r.heads.postprocess.
        rot_manual = rot6d_manual(rot6d, naive_mode=naive_mode)

        angle_deg, frob_norm = rotation_error(rot_roma, rot_manual)

        print(f"Angular error (deg): mean={angle_deg.mean():.6f}, "
              f"median={angle_deg.median():.6f}, max={angle_deg.max():.6f}")
        print(f"Frobenius norm diff: mean={frob_norm.mean():.6e}, "
              f"median={frob_norm.median():.6e}, max={frob_norm.max():.6e}")


if __name__ == "__main__":
    main()

