# Multi-HMR
# Copyright (c) 2024-present NAVER Corp.
# CC BY-NC-SA 4.0 license

import torch
import torch.nn as nn
from transformers import Dinov2Model, Dinov2Config


class Dinov2Backbone(nn.Module):
    def __init__(self, name: str = "dinov2_vitl14", pretrained: bool = False, *args, **kwargs):
        super().__init__()
        self.name = name

        # Map original backbone names to Hugging Face model ids
        model_map = {
            "dinov2_vitl14": "facebook/dinov2-large",
            "dinov2_vitb14": "facebook/dinov2-base",
            "dinov2_vits14": "facebook/dinov2-small",
            "dinov2_vitg14": "facebook/dinov2-giant",
        }
        hf_name = model_map.get(self.name, "facebook/dinov2-large")

        # Load HF model
        if pretrained:
            self.model = Dinov2Model.from_pretrained(hf_name)
        else:
            config = Dinov2Config.from_pretrained(hf_name)
            self.model = Dinov2Model(config)

        # Expose attributes expected by existing code
        self.patch_size = self.model.config.patch_size
        self.embed_dim = self.model.config.hidden_size

        # Backward-compatible reference name
        self.encoder = self.model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode a RGB image using a ViT-backbone
        Args:
            - x: torch.Tensor of shape [bs,3,h,w]
        Return:
            - y: torch.Tensor of shape [bs,k,d] - image in patchified mode
                 (patch tokens only, no CLS / registers)
        """
        assert len(x.shape) == 4
        outputs = self.model(pixel_values=x)

        # last_hidden_state: [B, 1 + N_patches (+ N_reg), D]
        # Drop CLS (index 0). Standard facebook/dinov2-* do not use registers.
        y = outputs.last_hidden_state[:, 1:, :]
        return y

    def load_state_dict(self, state_dict, strict: bool = True):
        """
        Override load_state_dict to support:
        - Native HF Dinov2Model checkpoints (no conversion).
        - Old torch.hub-style DINOv2 weights (encoder.* / blocks.* / qkv), via
          on-the-fly key conversion and QKV splitting.
        """
        # Heuristic: detect torch.hub-style encoder blocks / qkv
        is_old_format = any(
            "encoder.blocks.0.attn.qkv.weight" in k
            or "encoder.blocks.0.norm1.weight" in k
            or "blocks.0.attn.qkv.weight" in k
            or "blocks.0.norm1.weight" in k
            for k in state_dict.keys()
        )

        if is_old_format:
            print("[Dinov2Backbone] Detected torch.hub-style backbone weights.")
            print("[Dinov2Backbone] Converting keys to Hugging Face Dinov2 format...")
            converted = self._convert_hub_to_hf(state_dict)
            return super().load_state_dict(converted, strict=strict)
        else:
            print("[Dinov2Backbone] HF-style weights detected, no key conversion performed.")
            return super().load_state_dict(state_dict, strict=strict)

    def _convert_hub_to_hf(self, state_dict):
        """
        Convert torch.hub DINOv2 weights to HF Transformers Dinov2Model format.

        This function:
        - Renames top-level encoder / embedding keys.
        - Splits fused qkv matrices into separate query/key/value parameters.
        - Prints mapping for every original key (converted, unchanged, or skipped).
        """
        converted = {}
        hidden_size = self.embed_dim

        total_keys = 0
        mapped_keys = 0
        qkv_split_layers = set()

        for orig_key, val in state_dict.items():
            total_keys += 1
            key = orig_key

            # Strip leading "encoder." if present (torch.hub backbone.encoder.*)
            if key.startswith("encoder."):
                inner_key = key[len("encoder.") :]
            else:
                inner_key = key

            new_key = None

            # --- 1. Embeddings ---
            if inner_key == "cls_token":
                new_key = "model.embeddings.cls_token"
            elif inner_key == "mask_token":
                new_key = "model.embeddings.mask_token"
            elif inner_key == "pos_embed":
                new_key = "model.embeddings.position_embeddings"
            elif inner_key == "patch_embed.proj.weight":
                new_key = "model.embeddings.patch_embeddings.projection.weight"
            elif inner_key == "patch_embed.proj.bias":
                new_key = "model.embeddings.patch_embeddings.projection.bias"

            # --- 2. Encoder blocks ---
            elif inner_key.startswith("blocks."):
                # inner_key format: blocks.{idx}.{suffix}
                parts = inner_key.split(".")
                if len(parts) >= 3:
                    layer_idx = parts[1]
                    suffix = ".".join(parts[2:])
                    prefix = f"model.encoder.layer.{layer_idx}."

                    # Norms: keep original naming, move under encoder.layer.{i}
                    if suffix.startswith("norm1."):
                        new_key = prefix + suffix
                    elif suffix.startswith("norm2."):
                        new_key = prefix + suffix

                    # MLP: keep original naming mlp.fc1 / mlp.fc2
                    elif suffix.startswith("mlp.fc1."):
                        new_key = prefix + suffix
                    elif suffix.startswith("mlp.fc2."):
                        new_key = prefix + suffix

                    # Attention output projection
                    elif suffix.startswith("attn.proj."):
                        sub_name = suffix.replace("attn.proj.", "attention.output.dense.")
                        new_key = prefix + sub_name

                    # Layer scale
                    elif suffix.startswith("ls1"):
                        new_key = prefix + "layer_scale1.lambda1"
                    elif suffix.startswith("ls2"):
                        new_key = prefix + "layer_scale2.lambda1"

                    # QKV fused parameters: special handling
                    elif "attn.qkv." in suffix:
                        base_attn = f"model.encoder.layer.{layer_idx}.attention.attention"

                        if "weight" in suffix:
                            # val: [3*D, D] -> three [D, D]
                            if val.shape[0] != 3 * hidden_size:
                                print(
                                    f"[Dinov2Backbone] WARNING: qkv.weight shape mismatch for {orig_key}: "
                                    f"expected 3*{hidden_size}, got {val.shape}"
                                )
                            q, k, v = torch.chunk(val, 3, dim=0)
                            q_key = f"{base_attn}.query.weight"
                            k_key = f"{base_attn}.key.weight"
                            v_key = f"{base_attn}.value.weight"
                            converted[q_key] = q
                            converted[k_key] = k
                            converted[v_key] = v
                            mapped_keys += 3
                            qkv_split_layers.add(layer_idx)
                            print(
                                f"[Dinov2Backbone] map: {orig_key} -> {q_key} [Q weight]"
                            )
                            print(
                                f"[Dinov2Backbone] map: {orig_key} -> {k_key} [K weight]"
                            )
                            print(
                                f"[Dinov2Backbone] map: {orig_key} -> {v_key} [V weight]"
                            )
                            continue

                        if "bias" in suffix:
                            # val: [3*D] -> three [D]
                            if val.shape[0] != 3 * hidden_size:
                                print(
                                    f"[Dinov2Backbone] WARNING: qkv.bias shape mismatch for {orig_key}: "
                                    f"expected 3*{hidden_size}, got {val.shape}"
                                )
                            q, k, v = torch.chunk(val, 3, dim=0)
                            q_key = f"{base_attn}.query.bias"
                            k_key = f"{base_attn}.key.bias"
                            v_key = f"{base_attn}.value.bias"
                            converted[q_key] = q
                            converted[k_key] = k
                            converted[v_key] = v
                            mapped_keys += 3
                            qkv_split_layers.add(layer_idx)
                            print(
                                f"[Dinov2Backbone] map: {orig_key} -> {q_key} [Q bias]"
                            )
                            print(
                                f"[Dinov2Backbone] map: {orig_key} -> {k_key} [K bias]"
                            )
                            print(
                                f"[Dinov2Backbone] map: {orig_key} -> {v_key} [V bias]"
                            )
                            continue

            # --- 3. Final layer norm ---
            elif inner_key == "norm.weight":
                new_key = "model.layernorm.weight"
            elif inner_key == "norm.bias":
                new_key = "model.layernorm.bias"

            # Default behavior: if nothing matched, keep the original key
            if new_key is None:
                new_key = orig_key
                print(
                    f"[Dinov2Backbone] map: {orig_key} -> {new_key} [unchanged or unmapped]"
                )
            else:
                mapped_keys += 1
                print(f"[Dinov2Backbone] map: {orig_key} -> {new_key}")

            converted[new_key] = val

        print(
            f"[Dinov2Backbone] conversion summary: total_keys={total_keys}, "
            f"mapped_keys={mapped_keys}, qkv_split_layers={sorted(qkv_split_layers)}"
        )

        return converted
