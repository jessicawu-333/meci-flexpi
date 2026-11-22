# MECI on Flex-π — design notes

**Upstream:** geyan21/flex-pi @ `044b8c2080a14a3d0082f81f612474643446faf3` (committed 2026-10-28 20:12 UTC = 13:12 PT).
Read file by file through raw.githubusercontent.com and the GitHub API; no clone. Paper: arXiv 2608.10860 (v3).
**Status:** untrained; CPU unit tests only. No results are claimed.

## 1. What Flex-π is (as verified in code)
* `FlexPi` (`src/flexpi/models/flexpi.py:102`) = Wan2.2-TI2V-5B video DiT (`WanVideoDiT`, width 3072, 30 blocks) +
  `ActionDiT` action expert (`action_dit.py:32`, width 1024, FFN 4096, 30 blocks, 24 heads × 128) fused by a
  Mixture-of-Transformers (`mot.py:27`). HBridge: joint attention only in the middle 16 of 30 blocks (`configs/model/flexpi.yaml`, ratios 0.25/0.25).
* Streams: RGB latent (frozen Wan VAE), DINOv3 tokens (2×2 folded, x0-prediction), pointmap latent (same VAE), action chunk.
  DINO and pointmap tokens are concatenated after the video tokens (`_merge_aux_into_video_stream`, flexpi.py:1062).
* Conditioning: umT5 text (+1 proprio token appended, `backbone.py:622`) by cross-attention; flow timestep by adaLN
  (`time_embedding → time_projection → 6×D modulation`, action_dit.py:284-286; per-token for visual streams).
* Attention is one-way: no visual token attends to action tokens (paper App. A.5); `m_out` selects which futures the action reads.
* Flow matching (`schedulers/scheduler_continuous.py`): x_σ = (1−σ)x + σε, target ε − x, Euler from σ=1 to 0.
* Action output: plain `Linear` head (`post_dit`, action_dit.py:304). Losses in `_base_training_loss` (flexpi.py:1228);
  action loss `loss_action = (la * wa).mean()` (≈l.1421).
* Inference: `infer_action` (flexpi.py:3127) → action-only KV-cache fast path (anchors prefilled once) or joint path
  (`_predict_joint_noise_unified_impl`, flexpi.py:1599). A compile-friendly step `_denoise_step_compiled` (backbone.py:1203)
  re-implements pre/post_dit inline.

## 2. Hook points (patched tree line numbers)
| Path | Where | Effect |
|---|---|---|
| (a) input | `action_dit.py:292` after `action_encoder`; mirrored at `backbone.py:1233` | x_a += tanh(g_a)·s·XAttn(x_a, C) |
| (b) adaLN | `action_dit.py:287` on time embedding t; `backbone.py:1230` | t += tanh(g_b)·s·W pool(C) |
| (c) pre-head | `action_dit.py:311`; `backbone.py:1255` | h += tanh(g_c)·s·XAttn(h, C) |
| (d) future streams | `flexpi.py:1394` (training) and `:1727` (joint inference), helper `_meci_condition_future` (`:1132`) | t_mod[future] += tanh(g_d)·s·Up(Down(pool(C))) |
| loss | `flexpi.py:1259` encode batch correction; `:1455` AWR weight × action FM loss; aux DisplacementHead loss (weight 0.1) | |
| inference | `infer_action(..., correction=dict)` (`:3222`), explicit residual added to the sampled normalized chunk | |

Why (d): one-way attention means action-side conditioning can never reach the imagined frames (tested). (d) offsets only
non-anchor tokens, so the clean first-frame anchors and the action-only KV cache are untouched (tested: (d) is a no-op in action-only mode).

## 3. Identity guarantees (all tested bit-exact against pristine outputs)
No adapter; adapter + no correction; trained adapter + no correction; active correction + zero gates (only the aux
loss term is added to the total); trained gates + inactive correction; mixed batch (only the active sample changes).

## 4. Action spaces (meci_metric.py)
* LIBERO: 32-D canonical, OSC_POSE per-step deltas in slots 0-5, gripper 18, z-score. Residual spread so Σ r_t = δp / output_max (0.05 m, 0.5 rad per unit, robosuite default; calibration open).
* YAM: anchor-relative body-frame EE targets (`Yam32DRelativeAction`, p_abs = R_base p_rel + p_base) ⇒ r = R_base^T δp ramp; q01/q99 normalisation with rot6d slots skipped. Verified through the real inverse transform.
* RoboTwin 2.0: 14-D absolute joint targets `[L6, Lg, R6, Rg]` (`_base_task.py:487-494`), z-score ⇒ DLS IK Δq = Jᵀ(JJᵀ+λ²I)⁻¹δp, ramp from the utterance, released after the hold window. Flex-π ships no kinematics; J must be supplied.
* Normaliser: `SingleFieldLinearNormalizer` is affine (x_n = x·scale + offset, backward unclamped) ⇒ additive residuals map exactly.

## 5. Tests (`tests/`, 28 passing)
Tiny but *real* FlexPi (WanVideoDiT 48-wide / ActionDiT 32-wide, 4 layers, HBridge on, flex-joint dropout on) built from
the pinned source; only frozen encoders (Wan VAE, DINOv3, pointmap unprojection) are stubbed, and the zero-initialised DINO
x0 head is given random weights to stand in for a trained checkpoint. The full 6B model was not instantiated (no GPU,
no weights download).

## 6. Not covered / open
1. TensorRT / KV-split joint engines (`inference_opt/*`, `decode_joint_noisy`) are not patched; MECI needs the eager or torch.compile paths.
2. Checkpoint save/load of adapter weights is not wired into `save_checkpoint`; `meci_state_dict()` helper only.
3. Axis-sign calibration of `DIRECTION_AXES`; LIBERO commanded-vs-executed OSC displacement.
4. Arm Jacobians for RoboTwin (aloha-agilex) and optionally YAM joint slots; whether the YAM client executes EE or joint targets.
5. Rotation residuals on rot6d layouts are left to the learned path.
6. Hyper-parameters (gate LR, aux weight, AWR β, DPO β, video_rank) unset; no training run.
7. Whisper-small runs outside the graph; latency unmeasured.
