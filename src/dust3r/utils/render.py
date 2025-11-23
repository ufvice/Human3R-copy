"""
Lightweight visualization utilities.

The original project relied on heavier 3D rendering dependencies which were
removed for TPU compatibility.  For profiling and non-critical visualization
flows, we provide minimal stand-ins:

- ``vis_heatmap``: overlays a scalar heatmap onto an RGB image.
- ``render_meshes``: no-op mesh renderer that simply returns the input image.

These implementations are intentionally simple and dependency-free, and are
only meant to keep pipelines that expect these functions running without
failing on import.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Sequence

import numpy as np

try:  # Optional: keep working even if torch is not installed
    import torch
    from torch import Tensor
except Exception:  # pragma: no cover - torch is expected in this repo
    torch = None  # type: ignore
    Tensor = Any  # type: ignore


def _to_numpy(x: Any) -> np.ndarray:
    """Convert tensor/array-like to a NumPy array."""
    if torch is not None and isinstance(x, Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def vis_heatmap(image: Any, heatmap: Any, alpha: float = 0.6) -> Any:
    """
    Overlay a scalar heatmap onto an RGB image.

    Parameters
    ----------
    image:
        RGB image, either as a PyTorch tensor (H, W, 3) or NumPy array.
        Values are expected in [0, 1] or [0, 255]; both are handled.
    heatmap:
        Scalar heatmap with shape (H, W) or (1, H, W).
        Values are linearly normalized to [0, 1] for visualization.
    alpha:
        Blend factor between the original image and the colored heatmap.

    Returns
    -------
    Same type as the input image (tensor or array) with values in [0, 1].
    """
    img_np = _to_numpy(image).astype(np.float32)
    hm_np = _to_numpy(heatmap).astype(np.float32)

    if img_np.ndim == 3 and img_np.shape[0] in (1, 3) and img_np.shape[-1] != 3:
        # If input is CHW, convert to HWC
        img_np = np.moveaxis(img_np, 0, -1)

    if hm_np.ndim == 3 and hm_np.shape[0] == 1:
        hm_np = hm_np[0]

    # Normalize heatmap to [0, 1]
    if hm_np.size > 0:
        hm_min, hm_max = hm_np.min(), hm_np.max()
        if hm_max > hm_min:
            hm_np = (hm_np - hm_min) / (hm_max - hm_min)
        else:
            hm_np = np.zeros_like(hm_np, dtype=np.float32)
    else:
        hm_np = np.zeros_like(hm_np, dtype=np.float32)

    # Ensure image is in [0, 1]
    if img_np.max() > 1.0:
        img_np = img_np / 255.0

    h, w = hm_np.shape
    if img_np.shape[0] != h or img_np.shape[1] != w:
        # Simple resize via nearest-neighbor if shapes mismatch
        # (avoid bringing in cv2 / PIL as heavy dependencies)
        y_idx = (np.linspace(0, h - 1, img_np.shape[0])).astype(int)
        x_idx = (np.linspace(0, w - 1, img_np.shape[1])).astype(int)
        hm_np = hm_np[y_idx][:, x_idx]

    # Create a simple red colormap: more heat -> more red
    heat_rgb = np.stack(
        [
            hm_np,  # R
            np.zeros_like(hm_np),  # G
            1.0 - hm_np,  # B
        ],
        axis=-1,
    )

    blended = (1.0 - alpha) * img_np + alpha * heat_rgb
    blended = np.clip(blended, 0.0, 1.0)

    if torch is not None and isinstance(image, Tensor):
        return torch.from_numpy(blended)
    return blended


def render_meshes(
    image: Any,
    verts_list: Sequence[Any],
    faces_list: Sequence[Any],
    camera: dict[str, Any] | None = None,
    color: Iterable[Sequence[float]] | None = None,
) -> Any:
    """
    Very lightweight stand-in for a mesh renderer.

    The original implementation likely performed full 3D rasterization on the
    given image using meshes defined by ``verts_list`` and ``faces_list``.
    To avoid heavy dependencies (PyTorch3D, OpenGL, etc.), this stub simply
    returns the input ``image`` unchanged.

    This is sufficient for:
      - keeping the pipeline functional;
      - profiling CPU/XLA behavior without caring about visualization.
    """
    # Intentionally ignore meshes and camera; just return the image as-is.
    return image
