"""Shared scenario definitions so the pristine reference and the patched tests run identical calls."""
import torch

import tiny_flexpi as T

N_STEPS = 3


def run_training_loss(model, seed=11, B=2, batch_seed=0, extra=None):
    b = T.make_batch(B=B, seed=batch_seed)
    if extra:
        b.update(extra)
    model.train()
    torch.manual_seed(seed)
    total, d = model.training_loss(b)
    model.eval()
    return total, d


def run_action_only(model, obs_seed=0, **kw):
    o = T.make_obs(obs_seed)
    torch.manual_seed(5)
    return model.infer_action(action_horizon=T.H_ACT, num_inference_steps=N_STEPS, seed=0,
                              joint_video=False, joint_dino=False, joint_pointmap=False, **o, **kw)


def run_joint(model, obs_seed=0, **kw):
    o = T.make_obs(obs_seed)
    torch.manual_seed(5)
    return model.infer_action(action_horizon=T.H_ACT, num_video_frames=9, num_inference_steps=N_STEPS, seed=0,
                              joint_video=True, joint_dino=True, joint_pointmap=True,
                              return_stream_latents=True, **o, **kw)


def joint_noise_inputs(model, B=2, seed=3):
    """Inputs for one call of FlexPi._predict_joint_noise_unified_impl (a single joint denoise step)."""
    g = torch.Generator().manual_seed(seed)
    lv = torch.randn(B, T.LAT_C, 3, 4, 4, generator=g)
    ld = torch.randn(B, model.dino_dim, 3, 12, 1, generator=g)
    lp = torch.randn(B, T.LAT_C, 3, 4, 4, generator=g)
    la = torch.randn(B, T.H_ACT, T.ACTION_DIM, generator=g)
    tt = torch.full((B,), 500.0)
    return dict(latents_video=lv, latents_dino=ld, latents_pointmap=lp, latents_action=la,
                timestep_video=tt, timestep_dino=tt, timestep_pointmap=tt, timestep_action=tt,
                context=torch.randn(B, 5, T.TEXT_DIM, generator=g), context_mask=torch.ones(B, 5, dtype=torch.bool),
                fuse_vae_embedding_in_latents=True)


def flatten_out(out):
    return {k: v for k, v in out.items() if torch.is_tensor(v)}
