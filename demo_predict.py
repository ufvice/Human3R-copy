#!/usr/bin/env python3
"""
Run Human3R inference on a video or image folder and save predictions for the
tc3d-eval evaluation pipeline.

The saved `.pt` must follow the structure documented in `../tc3d-eval/eval.py`
(a dict of seq_name -> { "smplx_data_c": ..., "smplx_data_w": ... }). This script
mirrors `demo_debug.py` for image loading and model invocation but only keeps
the SMPL-X parameters needed by `eval.py / eval_motion.py`.
"""

from __future__ import annotations

import argparse
import os
import shutil
from typing import List, Optional

import roma
import torch

from add_ckpt_path import add_path_to_dust3r
from demo_debug import parse_seq_path, prepare_input

VIDEO_EXTENSIONS = (
    ".mp4",
    ".mov",
    ".avi",
    ".mkv",
    ".mpg",
    ".mpeg",
    ".webm",
    ".flv",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Human3R inference and save tc3d-eval compatible predictions."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="src/cut3r_512_dpt_4_64.pth",
        help="ARCroco3DStereo checkpoint.",
    )
    parser.add_argument(
        "--seq_path",
        type=str,
        required=True,
        help="Video file or folder (or root folder with multiple sequences) to run inference on.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/predicts/custom.pt",
        help="Destination path for the torch save file.",
    )
    parser.add_argument(
        "--seq_name",
        type=str,
        default=None,
        help="Override sequence name for single-input mode.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device string (e.g., 'cuda', 'cpu', 'xla').",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=512,
        help="Resize short side of the image to this value before inference.",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=None,
        help="Limit the number of frames processed per sequence.",
    )
    parser.add_argument(
        "--subsample",
        type=int,
        default=1,
        help="Subsampling factor for the input images.",
    )
    parser.add_argument(
        "--reset_interval",
        type=int,
        default=10000000,
        help="Reset interval forwarded to the Demo loader (large by default).",
    )
    parser.add_argument(
        "--use_ttt3r",
        action="store_true",
        help="Enable TTT3R-specific mode in the inference backend.",
    )
    return parser.parse_args()


def resolve_device(device_str: str):
    device_lower = device_str.lower()
    if device_lower == "xla":
        try:
            import torch_xla.core.xla_model as xm

            return xm.xla_device()
        except ImportError:
            print("XLA device requested but not available; falling back to CPU.")
            return torch.device("cpu")
    if device_lower.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(device_lower)
        print("CUDA requested but not available; falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_lower)


def discover_sequences(root_path: str) -> List[str]:
    if not os.path.isdir(root_path):
        return [root_path]

    entries = sorted(os.listdir(root_path))
    sequences: List[str] = []
    for entry in entries:
        if entry.startswith("."):
            continue
        candidate = os.path.join(root_path, entry)
        if os.path.isdir(candidate):
            sequences.append(candidate)
            continue
        if os.path.isfile(candidate) and candidate.lower().endswith(VIDEO_EXTENSIONS):
            sequences.append(candidate)

    return sequences or [root_path]


def filter_outputs(outputs: dict, revisit: int = 1):
    valid_length = len(outputs["pred"]) // revisit
    if valid_length == 0:
        return [], [], torch.tensor([], dtype=torch.bool)

    outputs["pred"] = outputs["pred"][-valid_length:]
    outputs["views"] = outputs["views"][-valid_length:]

    reset_mask = torch.cat([view["reset"] for view in outputs["views"]], dim=0)
    reset_mask = reset_mask.to(dtype=torch.bool)

    if reset_mask.shape[0] > 0:
        false_tensor = torch.tensor(False, dtype=torch.bool, device=reset_mask.device)
        shifted = torch.cat(
            (
                false_tensor.unsqueeze(0),
                reset_mask[:-1],
            ),
            dim=0,
        )
    else:
        shifted = torch.tensor([], dtype=torch.bool, device=reset_mask.device)

    filtered_preds = [
        pred for pred, mask in zip(outputs["pred"], shifted) if not mask
    ]
    reset_mask = reset_mask[~shifted]

    return filtered_preds, reset_mask


def compute_camera_poses(
    preds: List[dict],
    reset_mask: torch.Tensor,
    pose_encoding_to_camera,
    matrix_cumprod,
) -> List[torch.Tensor]:
    if not preds:
        return []

    pr_poses = [
        pose_encoding_to_camera(pred["camera_pose"].clone()) for pred in preds
    ]
    pr_poses_cat = torch.cat(pr_poses, dim=0)

    if reset_mask.any():
        identity = torch.eye(4, device=pr_poses_cat.device, dtype=pr_poses_cat.dtype)
        reset_poses = torch.where(
            reset_mask.unsqueeze(-1).unsqueeze(-1),
            pr_poses_cat,
            identity,
        )
        cumulative_bases = matrix_cumprod(reset_poses)
        shifted_bases = torch.cat(
            (
                identity.unsqueeze(0),
                cumulative_bases[:-1],
            ),
            dim=0,
        )
        pr_poses_cat = torch.einsum("bij,bjk->bik", shifted_bases, pr_poses_cat)

    return pr_poses_cat.cpu().unbind(0)


def build_sequence_prediction(
    preds: List[dict],
    camera_poses: List[torch.Tensor],
) -> dict:
    if len(preds) != len(camera_poses):
        raise ValueError("Prediction list and camera pose list lengths differ.")

    global_orient_cam: List[torch.Tensor] = []
    body_pose: List[torch.Tensor] = []
    betas: List[torch.Tensor] = []
    transl_cam: List[torch.Tensor] = []
    global_orient_world: List[torch.Tensor] = []
    transl_world: List[torch.Tensor] = []

    zeros_63 = torch.zeros(63, dtype=torch.float32)
    zeros_10 = torch.zeros(10, dtype=torch.float32)
    zeros_3 = torch.zeros(3, dtype=torch.float32)

    for idx, (pred, pose) in enumerate(zip(preds, camera_poses)):
        pose = pose.to(dtype=torch.float32)
        smpl_shape = pred.get("smpl_shape", torch.empty(1, 0, 10))[0].cpu()
        smpl_transl = pred.get("smpl_transl", torch.empty(1, 0, 3))[0].cpu()
        smpl_rotmat = pred.get("smpl_rotmat", torch.empty(1, 0, 53, 3, 3))[0].cpu()

        n_humans = smpl_shape.shape[0]
        if n_humans == 0:
            betas.append(zeros_10.clone())
            global_orient_cam.append(zeros_3.clone())
            body_pose.append(zeros_63.clone())
            transl_cam.append(zeros_3.clone())
            global_orient_world.append(zeros_3.clone())
            transl_world.append(zeros_3.clone())
            continue

        betas_i = smpl_shape[0].to(dtype=torch.float32)
        betas.append(betas_i)

        rotvecs = roma.rotmat_to_rotvec(smpl_rotmat[0])
        if rotvecs.shape[0] < 22:
            raise ValueError("Unexpected SMPL rotation vector length.")

        root_cam = rotvecs[0, 0]
        body_cam = rotvecs[0, 1:22].reshape(-1).to(dtype=torch.float32)
        global_orient_cam.append(root_cam.to(dtype=torch.float32))
        body_pose.append(body_cam)

        transl_cam_i = smpl_transl[0].to(dtype=torch.float32)
        transl_cam.append(transl_cam_i)

        R_root_cam = roma.rotvec_to_rotmat(root_cam.unsqueeze(0))[0]
        R_c2w = pose[:3, :3]
        t_c2w = pose[:3, 3]
        R_root_world = R_c2w @ R_root_cam
        root_world = roma.rotmat_to_rotvec(R_root_world.unsqueeze(0))[0]
        global_orient_world.append(root_world.to(dtype=torch.float32))

        transl_world_i = (R_c2w @ transl_cam_i) + t_c2w
        transl_world.append(transl_world_i.to(dtype=torch.float32))

    return {
        "smplx_data_c": {
            "global_orient": torch.stack(global_orient_cam, dim=0),
            "body_pose": torch.stack(body_pose, dim=0),
            "betas": torch.stack(betas, dim=0),
            "transl": torch.stack(transl_cam, dim=0),
        },
        "smplx_data_w": {
            "global_orient": torch.stack(global_orient_world, dim=0),
            "body_pose": torch.stack(body_pose, dim=0),
            "betas": torch.stack(betas, dim=0),
            "transl": torch.stack(transl_world, dim=0),
        },
    }


def run_sequence_inference(
    seq_path: str,
    seq_name: str,
    args,
    model,
    device,
    inference_fn,
    pose_encoding_to_camera,
    matrix_cumprod,
) -> dict:
    img_paths, tmpdirname = parse_seq_path(seq_path)
    try:
        if not img_paths:
            raise ValueError(f"No images detected for {seq_path}.")

        if args.max_frames is not None:
            img_paths = img_paths[: args.max_frames]
        img_paths = img_paths[:: args.subsample]

        if not img_paths:
            raise ValueError(f"No images left after subsampling {seq_path}.")

        views = prepare_input(
            img_paths,
            [True] * len(img_paths),
            args.size,
            revisit=1,
            update=True,
            img_res=getattr(model, "mhmr_img_res", None),
            reset_interval=args.reset_interval,
        )

        with torch.no_grad():
            outputs, _ = inference_fn(
                views, model, device, use_ttt3r=args.use_ttt3r
            )

        preds, reset_mask = filter_outputs(outputs, revisit=1)
        camera_poses = compute_camera_poses(
            preds, reset_mask, pose_encoding_to_camera, matrix_cumprod
        )

        if not preds:
            raise ValueError(f"No valid predictions for {seq_name}.")

        return build_sequence_prediction(preds, camera_poses)
    finally:
        if tmpdirname and os.path.isdir(tmpdirname):
            shutil.rmtree(tmpdirname)


def make_seq_name(path: str, root: Optional[str] = None) -> str:
    if root is None:
        return os.path.splitext(os.path.basename(path.rstrip(os.sep)))[0] or "sequence"
    rel = os.path.relpath(path, root)
    return os.path.splitext(rel.replace(os.sep, "_"))[0] or "sequence"


def main():
    args = parse_args()
    seq_paths = discover_sequences(args.seq_path)
    if args.seq_name and len(seq_paths) > 1:
        raise ValueError("--seq_name cannot be used when multiple sequences are inferred.")

    seq_names: List[str]
    if args.seq_name:
        seq_names = [args.seq_name]
    else:
        seq_names = [make_seq_name(path, args.seq_path) for path in seq_paths]

    add_path_to_dust3r(args.model_path)
    from src.dust3r.inference import inference_recurrent_lighter
    from src.dust3r.model import ARCroco3DStereo
    from src.dust3r.utils.camera import pose_encoding_to_camera
    from src.dust3r.utils.geometry import matrix_cumprod

    device = resolve_device(args.device)
    model = ARCroco3DStereo.from_pretrained(args.model_path).to(device)
    model.eval()

    predict = {}
    for seq_path, seq_name in zip(seq_paths, seq_names):
        print(f">> Processing {seq_name} ({seq_path})")
        seq_pred = run_sequence_inference(
            seq_path,
            seq_name,
            args,
            model,
            device,
            inference_recurrent_lighter,
            pose_encoding_to_camera,
            matrix_cumprod,
        )
        predict[seq_name] = seq_pred

    os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
    torch.save(predict, args.output_path)
    print(f"Saved predictions for {len(predict)} sequence(s) to {args.output_path}")


if __name__ == "__main__":
    main()
