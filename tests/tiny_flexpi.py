"""Build a *tiny* but real Flex-pi (flexpi.models.flexpi.FlexPi from the pinned commit) on CPU.

Real, unmodified-by-us upstream code paths used: WanVideoDiT, ActionDiT, MoT (HBridge on), FlexPi
(training loss, flex-joint stream dropout, unified attention masks, action-only KV-cache inference,
joint inference), WanContinuousFlowMatchScheduler. Only the *frozen encoders* are stubbed, because their
pretrained weights (Wan2.2 VAE, umT5, DINOv3) cannot be loaded here:
  * StubVAE        deterministic pooling + fixed channel map (stands in for the Wan2.2-5B VAE)
  * StubDino       deterministic pooled features (stands in for DINOv3)
  * StubPointmap   depth -> 3-channel composite (stands in for the pointmap unprojection)
Dimensions are tiny (hidden 48 / 32, 2 or 4 layers) instead of 3072 / 1024 x 30 layers.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from flexpi.models.action_dit import ActionDiT
from flexpi.models.flexpi import FlexPi
from flexpi.models.helpers.flex_joint import FlexJointConfig
from flexpi.models.mot import MoT
from flexpi.models.wan_video_dit import WanVideoDiT

LAT_C = 4          # VAE latent channels (48 in Wan2.2-5B)
TEXT_DIM = 16      # umT5 width (4096 in Flex-pi)
DINO_DIM = 8       # DINO feature width (768 in Flex-pi)
DINO_PATCHES = [(2, 2), (2, 2), (2, 2)]
DINO_REGIONS = [(0, 2, 0, 2), (2, 4, 0, 2), (2, 4, 2, 4)]
ACTION_DIM = 32    # canonical layout
H_ACT = 8          # action horizon (32 in Flex-pi)


def _frames(video):  # [B,3,T,H,W] -> keep frames 0,4,8,...
    return video[:, :, ::4]


class StubVAE(torch.nn.Module):
    temporal_downsample_factor = 4
    upsampling_factor = 4
    z_dim = LAT_C

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(123)
        self.register_buffer("mix", torch.randn(LAT_C, 3, generator=g))
        self.model = type("M", (), {"z_dim": LAT_C})()

    def encode(self, videos, device=None, tiled=False, **kw):
        if isinstance(videos, (list, tuple)):
            videos = torch.stack(list(videos), 0)
        x = _frames(videos.float())
        x = F.avg_pool3d(x, kernel_size=(1, 4, 4))
        return torch.einsum("cd,bdthw->bcthw", self.mix, x)


class StubDino(torch.nn.Module):
    patches_per_side = 2

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(7)
        self.register_buffer("mix", torch.randn(DINO_DIM, 3, generator=g))

    def encode_video(self, video=None, concat_mode=None, temporal_stride=1, per_cam=None,
                     first_frame_only=False, **kw):
        x = video.float()
        x = x[:, :, :1] if first_frame_only else _frames(x)
        x = F.adaptive_avg_pool3d(x, (x.shape[2], 2, 6))  # 12 tokens = 3 cams x 2x2
        x = x.flatten(3)  # [B,3,F,12]
        return torch.einsum("cd,bdfn->bcfn", self.mix, x).unsqueeze(-1)  # [B,D,F,12,1]


class StubPointmap(torch.nn.Module):
    def encode_composite(self, per_cam_depth, camera_intrinsics=None, concat_mode=None,
                         first_frame_only=False, **kw):
        d = next(iter(per_cam_depth.values())).float()  # [B,1,T,H,W] or [B,1,H,W] (first frame)
        if d.ndim == 4:
            d = d.unsqueeze(2)
        return d.expand(-1, 3, -1, -1, -1).contiguous()


def build_tiny_flexpi(seed: int = 0, num_layers: int = 4, enable_pointmap: bool = True,
                      flex_enabled: bool = True) -> FlexPi:
    torch.manual_seed(seed)
    video = WanVideoDiT(
        hidden_dim=48, in_dim=LAT_C, ffn_dim=96, out_dim=LAT_C, text_dim=TEXT_DIM, freq_dim=32,
        eps=1e-6, patch_size=(1, 2, 2), num_heads=4, attn_head_dim=12, num_layers=num_layers,
        has_image_input=False, seperated_timestep=True, fuse_vae_embedding_in_latents=True,
        video_attention_mask_mode="first_frame_causal", action_conditioned=False, action_dim=ACTION_DIM,
    )
    action = ActionDiT(
        hidden_dim=32, action_dim=ACTION_DIM, ffn_dim=64, text_dim=TEXT_DIM, freq_dim=32, eps=1e-6,
        num_heads=4, attn_head_dim=12, num_layers=num_layers,
    )
    mot = MoT(mixtures={"video": video, "action": action}, mot_checkpoint_mixed_attn=False,
              attn_mode="sdpa", hbridge_enabled=True, hbridge_bottom_ratio=0.25, hbridge_top_ratio=0.25)
    fj = FlexJointConfig(enabled=flex_enabled)
    model = FlexPi(
        video, action, mot, StubVAE(), StubDino(), StubPointmap(),
        text_dim=TEXT_DIM, proprio_dim=None, device="cpu", torch_dtype=torch.float32,
        dino_dim=DINO_DIM, dino_cam_patches=DINO_PATCHES, dino_cam_regions=DINO_REGIONS,
        dino_pred_x0=True, dino_pixel_unshuffle=0, composite_layout="tshape_384x320",
        video_train_shift=6.0, video_infer_shift=6.0, action_train_shift=1.0, action_infer_shift=1.0,
        dino_train_shift=6.0, dino_infer_shift=6.0, pointmap_train_shift=6.0, pointmap_infer_shift=6.0,
        joint_video=True, joint_dino=True, joint_pointmap=enable_pointmap, flex_joint=fj,
        enable_pointmap=enable_pointmap,
    )
    # Upstream zero-initialises the DINO x0 head (FlexPi._init_dino_layers), so an untrained model's DINO
    # prediction ignores its input. Give it seeded random weights, standing in for a trained checkpoint.
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        model.dino_proj_out.weight.copy_(torch.randn(model.dino_proj_out.weight.shape, generator=g) * 0.2)
    model.eval()
    return model


def make_batch(B: int = 2, seed: int = 0, T: int = 9, HW: int = 16) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {
        "video": torch.rand(B, 3, T, HW, HW, generator=g) * 2 - 1,
        "context": torch.randn(B, 5, TEXT_DIM, generator=g),
        "context_mask": torch.ones(B, 5, dtype=torch.bool),
        "action": torch.randn(B, H_ACT, ACTION_DIM, generator=g),
        "camera_intrinsics": torch.tensor([[100.0, 100.0, 8.0, 8.0]]),
        "per_cam_depth": {"cam_high": torch.rand(B, 1, T, HW, HW, generator=g)},
    }


def make_obs(seed: int = 0, HW: int = 16) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {
        "input_image": torch.rand(1, 3, HW, HW, generator=g) * 2 - 1,
        "context": torch.randn(1, 5, TEXT_DIM, generator=g),
        "context_mask": torch.ones(1, 5, dtype=torch.bool),
        "per_cam_depth": {"cam_high": torch.rand(1, 1, HW, HW, generator=g)},
        "camera_intrinsics": torch.tensor([[100.0, 100.0, 8.0, 8.0]]),
    }
