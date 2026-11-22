"""MECI x Flex-pi tests. Run against the PATCHED flex-pi tree (see run_tests.sh).

Model-level tests use the real FlexPi / WanVideoDiT / ActionDiT / MoT classes from flex-pi @ 044b8c2 at
tiny width with stub frozen encoders (tiny_flexpi.py). Bit-identity is checked against outputs saved from
the PRISTINE tree in a separate process (reference_pristine.py).
"""
import math
import os

import numpy as np
import pytest
import torch

import common as C
import tiny_flexpi as T
from flexpi.models import meci_correction as MC
from flexpi.models import meci_metric as MM
from flexpi.models.backbone import FlexPiBackbone

REF = torch.load(os.environ["MECI_REF"])
AUDIO_T = 6


def eq(a, b):
    return torch.equal(a, b)


def model_with_meci(trained=False, gates=None):
    m = T.build_tiny_flexpi()
    meci = m.enable_meci()
    if trained:
        g = torch.Generator().manual_seed(99)
        with torch.no_grad():
            for p in meci.parameters():
                p.copy_(torch.randn(p.shape, generator=g) * 0.3)
    if gates is not None:
        with torch.no_grad():
            for name in ("gate_in", "gate_ada", "gate_head", "gate_video"):
                getattr(meci, name).fill_(gates.get(name, 0.0))
    return m, meci


def correction(active=True, strength=1.0, residual=None, effector=1):
    g = torch.Generator().manual_seed(42)
    c = {"audio": torch.randn(AUDIO_T, 768, generator=g), "active": active, "strength": strength,
         "delta": [0.0, 0.0127, 0.0, 0.0, 0.0, 0.0], "effector": effector}
    if residual is not None:
        c["residual"] = residual
    return c


def train_extra(B=2, active=(True, True), strength=(1.0, 1.0), awr=None):
    g = torch.Generator().manual_seed(43)
    e = {"meci_audio": torch.randn(B, AUDIO_T, 768, generator=g), "meci_active": torch.tensor(active),
         "meci_strength": torch.tensor(strength), "meci_delta": torch.tensor([[0.0, 0.0127, 0, 0, 0, 0]] * B),
         "meci_effector": torch.tensor([1] * B)}
    if awr is not None:
        e["meci_awr_weight"] = torch.tensor(awr)
    return e


def check_stock(m, ctx_train=None, ctx_infer=None, aux_expected=False):
    total, d = C.run_training_loss(m, extra=ctx_train)
    for k in ("loss_video", "loss_action", "loss_dino", "loss_pointmap"):
        assert d[k] == REF["train_" + k].item(), k
    if not aux_expected:
        assert eq(total.detach(), REF["train_total"])
    kw = {} if ctx_infer is None else {"correction": ctx_infer}
    ao = C.run_action_only(m, **kw)
    assert eq(ao["action"], REF["ao_action"])
    jo = C.run_joint(m, **kw)
    for k in ("action", "video_latents", "dino_latents", "pointmap_latents"):
        assert eq(jo[k], REF["joint_" + k]), k
    return total, d


# ------------------------------------------------------------------ identity / zero-init
def test_identity_no_adapter():
    check_stock(T.build_tiny_flexpi())


def test_identity_adapter_no_correction():
    m, _ = model_with_meci()
    check_stock(m)


def test_identity_adapter_trained_but_no_correction():
    m, _ = model_with_meci(trained=True)
    check_stock(m)


def test_identity_active_correction_zero_gates():
    m, meci = model_with_meci()
    total, d = check_stock(m, ctx_train=train_extra(), ctx_infer=correction(), aux_expected=True)
    # the only change to the training objective is the auxiliary DisplacementHead term
    assert d["loss_meci_aux"] > 0
    assert torch.allclose(total.detach(), REF["train_total"] + meci.config.aux_weight * d["loss_meci_aux"], atol=1e-6)


def test_identity_inactive_correction_trained_gates():
    m, _ = model_with_meci(trained=True)
    check_stock(m, ctx_train=train_extra(active=(False, False)), ctx_infer=correction(active=False))


def test_mixed_batch_only_active_sample_changes():
    m, _ = model_with_meci(trained=True)
    x = C.joint_noise_inputs(m)
    ctx = m.meci.encode(torch.randn(2, AUDIO_T, 768), active=torch.tensor([True, False]))
    m._meci_set_ctx(ctx)
    with torch.no_grad():
        outs = m._predict_joint_noise_unified_impl(**x)
    m._meci_set_ctx(None)
    for i, t in enumerate(outs):
        assert eq(t[1], REF[f"jn_{i}"][1]), f"inactive sample changed in output {i}"
        assert not eq(t[0], REF[f"jn_{i}"][0]), f"active sample unchanged in output {i}"


# ------------------------------------------------------------------ each gated path
@pytest.mark.parametrize("gate", ["gate_in", "gate_ada", "gate_head"])
def test_action_paths_change_action_not_imagined_future(gate):
    m, _ = model_with_meci(trained=True, gates={gate: 0.5})
    ao = C.run_action_only(m, correction=correction())
    assert not eq(ao["action"], REF["ao_action"])
    jo = C.run_joint(m, correction=correction())
    assert not eq(jo["action"], REF["joint_action"])
    # Flex-pi attention is one-way (no visual token attends to actions): action-side paths
    # cannot change the generated video / DINO / pointmap futures.
    for k in ("video_latents", "dino_latents", "pointmap_latents"):
        assert eq(jo[k], REF["joint_" + k]), k


def test_video_path_changes_imagined_future_and_joint_action():
    m, _ = model_with_meci(trained=True, gates={"gate_video": 0.5})
    jo = C.run_joint(m, correction=correction())
    for k in ("video_latents", "dino_latents", "pointmap_latents", "action"):
        assert not eq(jo[k], REF["joint_" + k]), k
    # the first (anchor) latent frame is the clean observation and stays fixed
    assert eq(jo["video_latents"][:, :, 0], REF["joint_video_latents"][:, :, 0])


def test_video_path_is_noop_in_action_only_mode():
    """Action-only mode generates no futures; path (d) touches only future tokens, so the anchor
    KV cache and the action are unchanged."""
    m, _ = model_with_meci(trained=True, gates={"gate_video": 0.5})
    ao = C.run_action_only(m, correction=correction())
    assert eq(ao["action"], REF["ao_action"])


def test_future_token_mask_matches_anchor_layout():
    mask = MC.future_token_mask(Sv=48, tpf=16, Sd=36, dtpf=12, Sp=48, ptpf=16)
    assert mask.shape == (132,)
    assert (~mask).sum().item() == 16 + 12 + 16
    assert not mask[:16].any() and mask[16:48].all()
    assert not mask[48:60].any() and mask[60:84].all()
    assert not mask[84:100].any() and mask[100:].all()
    assert MC.future_token_mask(48, 16, 0, 0, 0, 0).sum().item() == 32


def test_compiled_step_body_matches_eager_with_correction():
    """backbone._denoise_step_compiled inlines pre_dit/post_dit; the patch adds the same hooks there."""
    m, _ = model_with_meci(trained=True, gates={"gate_in": 0.4, "gate_ada": 0.4, "gate_head": 0.4})
    calls = []
    orig = m._predict_action_noise_with_cache

    def both(latents_action, timestep_action, context, context_mask, video_kv_cache, attention_mask, video_seq_len):
        eager = m._predict_action_grad_safe(latents_action, timestep_action, context, context_mask,
                                            video_kv_cache, attention_mask, video_seq_len)
        a_len = latents_action.shape[1]
        comp = FlexPiBackbone._denoise_step_compiled(
            m, latents_action=latents_action, timestep_action=timestep_action, context=context,
            context_mask=context_mask, video_cache_k=[c["k"] for c in video_kv_cache],
            video_cache_v=[c["v"] for c in video_kv_cache],
            action_attention_mask=attention_mask[video_seq_len:video_seq_len + a_len, :video_seq_len + a_len],
            action_freqs=m.action_expert.freqs[:a_len].view(a_len, 1, -1),
            action_only_attention_mask=m._build_action_only_attention_mask(action_seq_len=a_len, device=m.device),
        )
        calls.append(torch.allclose(eager, comp, atol=1e-6))
        return eager

    m._predict_action_noise_with_cache = both
    ao = C.run_action_only(m, correction=correction())
    m._predict_action_noise_with_cache = orig
    assert len(calls) == C.N_STEPS and all(calls)
    assert not eq(ao["action"], REF["ao_action"])


# ------------------------------------------------------------------ training
def test_training_gradients_reach_gates_and_head():
    m, meci = model_with_meci()
    m.train()
    torch.manual_seed(11)
    total, d = m.training_loss({**T.make_batch(B=2), **train_extra()})
    total.backward()
    for name in ("gate_in", "gate_ada", "gate_head", "gate_video"):
        gr = getattr(meci, name).grad
        assert gr is not None and torch.isfinite(gr) and gr.abs().item() > 0, name
    assert meci.disp_out.weight.grad.abs().sum() > 0
    assert math.isfinite(total.item())


def test_aux_loss_zero_for_inactive():
    m, meci = model_with_meci()
    ctx = meci.encode(torch.randn(2, AUDIO_T, 768), active=torch.tensor([True, False]))
    aux = meci.aux_loss(ctx, torch.tensor([[0.0, 0.0127, 0, 0, 0, 0]] * 2), torch.tensor([1, 1]))
    assert aux[0] > 0 and aux[1] == 0


def test_awr_weight_one_is_stock_and_scales_action_loss():
    m, _ = model_with_meci()
    _, d1 = C.run_training_loss(m, extra=train_extra(awr=[1.0, 1.0]))
    assert d1["loss_action"] == REF["train_loss_action"].item()
    _, d2 = C.run_training_loss(m, extra=train_extra(awr=[2.0, 2.0]))
    assert abs(d2["loss_action"] - 2 * d1["loss_action"]) < 1e-5


def test_explicit_residual_added_after_sampling():
    m, _ = model_with_meci()
    r = torch.zeros(T.H_ACT, T.ACTION_DIM)
    r[:, 9:12] = torch.linspace(0.01, 0.08, T.H_ACT)[:, None]
    ao = C.run_action_only(m, correction=correction(residual=r))
    assert torch.allclose(ao["action"], REF["ao_action"] + r, atol=1e-6)
    ao2 = C.run_action_only(m, correction=correction(active=False, residual=r))
    assert eq(ao2["action"], REF["ao_action"])


def test_effector_embedding_routes():
    m, meci = model_with_meci(trained=True)
    a = torch.randn(1, AUDIO_T, 768)
    c_left = meci.encode(a, effector=torch.tensor([1]), delta=torch.zeros(1, 6))
    c_right = meci.encode(a, effector=torch.tensor([2]), delta=torch.zeros(1, 6))
    assert not torch.allclose(c_left.tokens, c_right.tokens)


# ------------------------------------------------------------------ grammar and residual geometry
@pytest.mark.parametrize("phrase,eff,axis,val", [
    ("left hand more left half an inch", "left", 1, 0.0127),
    ("right gripper up 1 cm", "right", 2, 0.01),
    ("move forward 1 inch", "default", 0, 0.0254),
    ("rotate gripper 10 degrees", "default", 5, math.radians(10)),
])
def test_parse(phrase, eff, axis, val):
    c = MM.parse_metric_command(phrase)
    assert c.effector == eff
    dp = c.delta_p()
    assert abs(dp[axis] - val) < 1e-6 and np.count_nonzero(dp) == 1


def test_routing_yam_robotwin_libero():
    dp = np.array([0.0, 0.0127, 0, 0, 0, 0], np.float32)
    R = {"left": np.eye(3), "right": np.eye(3)}
    d, ok = MM.raw_action_delta(MM.YAM, dp, "left", anchor_rotation=R)
    assert ok and np.allclose(d[0:3], dp[:3]) and np.count_nonzero(d) == 1
    d, ok = MM.raw_action_delta(MM.YAM, dp, "right", anchor_rotation=R)
    assert ok and np.allclose(d[9:12], dp[:3]) and d[18] == 0 and d[19] == 0
    d, ok = MM.raw_action_delta(MM.YAM, dp, "default", anchor_rotation=R)  # unnamed on bimanual
    assert not ok and not d.any()
    J = np.random.default_rng(0).normal(size=(6, 6))
    d, ok = MM.raw_action_delta(MM.ROBOTWIN, dp, "right", jacobian_fn=lambda e: J)
    assert ok and not d[:7].any() and d[13] == 0 and np.allclose(J @ d[7:13], dp, atol=1e-5)
    d, ok = MM.raw_action_delta(MM.LIBERO, dp, "left")  # single arm ignores the name
    assert ok and np.allclose(d[1], 0.0127 / 0.05) and d[18] == 0


def test_libero_residual_sums_to_delta_through_flexpi_normalizer():
    from flexpi.datasets.lerobot.utils.normalizer import SingleFieldLinearNormalizer
    rng = np.random.default_rng(1)
    std = rng.uniform(0.1, 0.5, 32).astype(np.float32)
    mean = rng.normal(size=32).astype(np.float32)
    norm = SingleFieldLinearNormalizer({"mean": torch.tensor(mean), "std": torch.tensor(std)}, mode="z-score")
    scale = MM.normalizer_scale("z-score", std=std)
    assert np.allclose(scale, norm.scale.numpy(), rtol=1e-6)
    w = MM.hold_fade_strengths()
    total = np.zeros(32)
    for start in (0, 10, 20):  # Flex-pi LIBERO re-plans every 10 steps
        c = MM.build_correction_inputs("move left half an inch", MM.LIBERO, action_horizon=32,
                                       norm_scale=scale, chunk_start=start, window_strengths=w)
        # only the first 10 actions of each chunk are executed, so window steps >= start+10 are
        # re-issued by the next chunk; count each control step once
        x_n = torch.randn(32, 32, dtype=torch.float64)
        raw = (norm.backward(x_n + torch.tensor(c["residual"], dtype=torch.float64)) - norm.backward(x_n)).numpy()
        total += raw[:10].sum(0)
    assert abs(total[1] * 0.05 - 0.0127) < 1e-6
    assert np.abs(np.delete(total, 1)).max() < 1e-6


def test_yam_body_frame_residual_through_flexpi_inverse_transform():
    from flexpi.datasets.lerobot.transforms.yam_relative_action import Yam32DRelativeAction
    from flexpi.datasets.lerobot.utils.normalizer import SingleFieldLinearNormalizer
    from flexpi.datasets.lerobot.utils.yam_eef import pose9d_to_mat
    rng = np.random.default_rng(2)
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    q = q * np.sign(np.linalg.det(q))
    state = np.zeros(32)
    state[0:3] = [0.3, 0.1, 0.2]
    state[3:9] = q[:2].reshape(-1)
    state[9:12] = [0.3, -0.1, 0.2]
    state[12:18] = np.eye(3)[:2].reshape(-1)
    R_left = pose9d_to_mat(state[0:9])[:3, :3]
    q01 = -rng.uniform(0.2, 0.4, 32)
    q99 = rng.uniform(0.2, 0.4, 32)
    skip = np.zeros(32, bool)
    skip[3:9] = skip[12:18] = True
    norm = SingleFieldLinearNormalizer({"q01": torch.tensor(q01), "q99": torch.tensor(q99),
                                        "min": torch.tensor(q01), "max": torch.tensor(q99)},
                                       mode="q01/q99", skip_dims_mask=skip.tolist())
    scale = MM.normalizer_scale("q01/q99", q01=q01, q99=q99, skip_mask=skip)
    assert np.allclose(scale, norm.scale.numpy())
    c = MM.build_correction_inputs("left hand more left half an inch", MM.YAM, action_horizon=32,
                                   norm_scale=scale, anchor_rotation={"left": R_left, "right": np.eye(3)})
    x_n = torch.zeros(32, 32, dtype=torch.float64)
    x_n[:, 3:9] = torch.tensor(np.eye(3)[:2].reshape(-1))
    x_n[:, 12:18] = torch.tensor(np.eye(3)[:2].reshape(-1))
    tr = Yam32DRelativeAction(anchor="first")

    def to_abs(xn):
        a = norm.backward(xn)
        b = {"state": {"default": torch.tensor(state)[None]}, "action": {"default": a}}
        return tr.backward(b)["action"]["default"].numpy()

    diff = to_abs(x_n + torch.tensor(c["residual"], dtype=torch.float64)) - to_abs(x_n)
    w = MM.hold_fade_strengths()
    ramp = np.minimum(np.cumsum(np.r_[w, np.zeros(32 - len(w))]) / w.sum(), 1.0)
    assert np.allclose(diff[:, 0:3], ramp[:, None] * np.array([0.0, 0.0127, 0.0]), atol=1e-9)
    assert np.abs(diff[:, 3:]).max() < 1e-9  # right arm, rotations, grippers, joints untouched


def test_robotwin_absolute_ramp_reaches_delta_and_releases():
    J = np.random.default_rng(3).normal(size=(6, 6))
    w = MM.hold_fade_strengths(16, 8)
    dq = MM.raw_action_delta(MM.ROBOTWIN, np.array([0.0, 0, 0.01, 0, 0, 0]), "left", lambda e: J)[0]
    r0 = MM.chunk_residual(MM.ROBOTWIN, dq, w, 0, 8)
    r1 = MM.chunk_residual(MM.ROBOTWIN, dq, w, 8, 32)
    assert np.allclose(r1[7], dq, atol=1e-6)  # step 15 (end of hold window) = full delta
    assert np.allclose(r0[-1], r1[0] - dq * w[8] / w.sum(), atol=1e-6)  # continuous across chunks
    assert not r1[8:].any()  # released after the window


# ------------------------------------------------------------------ helpers and size
def test_rl_helpers():
    w = MC.awr_weights(torch.tensor([0.0, 1.0, 100.0]), beta=1.0, w_max=20.0)
    assert torch.allclose(w, torch.tensor([1.0, math.e, 20.0]))
    z = torch.zeros(3)
    assert torch.allclose(MC.flow_dpo_loss(z, z, z, z), torch.full((3,), math.log(2.0)))
    better = MC.flow_dpo_loss(torch.full((3,), -0.5), z, z, z)
    assert (better < math.log(2.0)).all()


def test_full_size_adapter_parameter_count():
    meci = MC.MECICorrection(MC.MECIConfig(width=1024, video_dim=3072))
    n = MC.count_parameters(meci)
    w, a, k, r, v = 1024, 768, 8, 64, 3072
    xattn_audio = (w * w + w) + 2 * (a * w + w) + (w * w + w)
    xattn = 4 * (w * w + w)
    expected = (2 * a) + k * w + xattn_audio + 2 * w + (w * 2 * w + 2 * w) + (2 * w * w + w) \
        + (w * w + w) + (w * 9 + 9) + (6 * w + w) + 3 * w + xattn + 1 + (w * w + w) + 1 + xattn + 1 \
        + (w * r + r) + (r * 6 * v + 6 * v) + 1
    assert n == expected
    print(f"\nFULL-SIZE MECI ADAPTER PARAMETERS: {n:,}")
    nv = sum(p.numel() for name, p in meci.named_parameters() if name.startswith(("video_", "gate_video")))
    print(f"  of which path (d) (video/DINO/pointmap modulation): {nv:,}")
