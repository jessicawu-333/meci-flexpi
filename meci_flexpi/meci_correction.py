"""MECI for Flex-pi -- the correction adapter (PyTorch).

Mid-Execution Correction Interface (MECI) for the Flex-pi world-action model
(github.com/geyan21/flex-pi @ 044b8c2080a14a3d0082f81f612474643446faf3, MIT).

A spoken metric nudge ("left hand more left half an inch") is encoded by a frozen
Whisper-small encoder outside the graph (768-d frame features) and turned into K
correction tokens by this adapter. The adapter conditions Flex-pi through four
gated paths. Every gate is a scalar initialised to 0 and enters as tanh(g) * s,
where s is the per-sample correction strength (0 when no correction is active):

  (a) action-expert input   ActionDiT.pre_dit, after action_encoder:   x_a += tanh(g_a) s CrossAttn(x_a, C)
  (b) action-expert adaLN   ActionDiT.pre_dit, time embedding t:       t   += tanh(g_b) s W_b pool(C)
  (c) pre-output residual   ActionDiT.post_dit, before head:          h   += tanh(g_c) s CrossAttn(h, C)
  (d) future visual streams FlexPi merged video/DINO/pointmap t_mod, non-anchor tokens only:
                            t_mod[:, future] += tanh(g_d) s Up(Down(pool(C)))      (WAM-specific)

Path (d) exists because Flex-pi's attention is one-way: no visual token ever attends
to the action stream (paper Sec. A.5), so conditioning the action expert alone can
never change the imagined future frames. (d) touches only noisy future tokens, so the
clean first-frame anchors -- and the KV cache the action-only fast path prefills from
them -- are unchanged.

With no correction context the hooks are skipped entirely. With a context and zero
gates, every path adds exactly 0, so outputs are bit-identical to stock Flex-pi.
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812

DELTA_SCALE = (0.05, 0.05, 0.05, 0.5, 0.5, 0.5)  # metres / radians per unit of the displacement head
NUM_EFFECTORS = 3  # default / left / right (meci_metric.EFFECTORS)


@dataclasses.dataclass
class MECIConfig:
    width: int = 1024  # Flex-pi action expert hidden_dim (configs/model/flexpi.yaml action_dit_config)
    video_dim: int = 3072  # Flex-pi video expert hidden_dim (video_dit_config)
    audio_dim: int = 768  # Whisper-small encoder width
    num_queries: int = 8  # K correction tokens
    num_heads: int = 8
    video_rank: int = 64  # low-rank bottleneck for path (d)
    condition_video: bool = True  # enable path (d)
    use_given_delta: bool = True  # teacher-force parsed (delta_p, effector) when available
    aux_weight: float = 0.1


@dataclasses.dataclass
class CorrectionContext:
    tokens: torch.Tensor  # (B, K, width)
    token_mask: torch.Tensor  # (B, K) bool
    pooled: torch.Tensor  # (B, width)
    strength: torch.Tensor  # (B,)  0 for inactive samples
    active: torch.Tensor  # (B,) bool
    delta_hat: torch.Tensor  # (B, 6) metres / radians
    effector_logits: torch.Tensor  # (B, 3)


class _CrossAttention(nn.Module):
    def __init__(self, q_dim: int, kv_dim: int, width: int, num_heads: int):
        super().__init__()
        assert width % num_heads == 0
        self.num_heads = num_heads
        self.q = nn.Linear(q_dim, width)
        self.k = nn.Linear(kv_dim, width)
        self.v = nn.Linear(kv_dim, width)
        self.o = nn.Linear(width, q_dim)

    def forward(self, x, ctx, ctx_mask=None):
        b, n, _ = x.shape
        m = ctx.shape[1]
        h = self.num_heads
        q = self.q(x).view(b, n, h, -1).transpose(1, 2)
        k = self.k(ctx).view(b, m, h, -1).transpose(1, 2)
        v = self.v(ctx).view(b, m, h, -1).transpose(1, 2)
        logits = (q @ k.transpose(-1, -2)) / (q.shape[-1] ** 0.5)
        if ctx_mask is not None:
            # Rows of inactive samples are fully masked; keep them finite (their gate factor s is 0).
            any_valid = ctx_mask.any(-1)
            safe = torch.where(any_valid[:, None], ctx_mask, torch.ones_like(ctx_mask))
            logits = logits.masked_fill(~safe[:, None, None, :], -1e30)
        probs = torch.softmax(logits.float(), dim=-1).to(v.dtype)
        out = (probs @ v).transpose(1, 2).reshape(b, n, -1)
        return self.o(out)


class MECICorrection(nn.Module):
    def __init__(self, config: MECIConfig):
        super().__init__()
        self.config = config
        w = config.width
        self.audio_norm = nn.LayerNorm(config.audio_dim)
        self.queries = nn.Parameter(torch.randn(config.num_queries, w) * 0.02)
        self.projector = _CrossAttention(w, config.audio_dim, w, config.num_heads)
        self.proj_norm = nn.LayerNorm(w)
        self.proj_mlp = nn.Sequential(nn.Linear(w, 2 * w), nn.GELU(), nn.Linear(2 * w, w))
        # DisplacementHead: pooled tokens -> (delta_p / DELTA_SCALE, effector logits)
        self.disp_in = nn.Linear(w, w)
        self.disp_out = nn.Linear(w, 6 + NUM_EFFECTORS)
        nn.init.zeros_(self.disp_out.weight)
        nn.init.zeros_(self.disp_out.bias)
        self.delta_embed = nn.Linear(6, w)
        nn.init.zeros_(self.delta_embed.weight)
        nn.init.zeros_(self.delta_embed.bias)
        self.effector_embed = nn.Embedding(NUM_EFFECTORS, w)
        nn.init.zeros_(self.effector_embed.weight)
        # (a) input cross-attention, (b) adaLN offset, (c) pre-head cross-attention
        self.in_xattn = _CrossAttention(w, w, w, config.num_heads)
        self.gate_in = nn.Parameter(torch.zeros(()))
        self.ada_proj = nn.Linear(w, w)
        self.gate_ada = nn.Parameter(torch.zeros(()))
        self.head_xattn = _CrossAttention(w, w, w, config.num_heads)
        self.gate_head = nn.Parameter(torch.zeros(()))
        # (d) future-visual-stream modulation offset (6 x video_dim, low rank)
        if config.condition_video:
            self.video_down = nn.Linear(w, config.video_rank)
            self.video_up = nn.Linear(config.video_rank, 6 * config.video_dim)
            self.gate_video = nn.Parameter(torch.zeros(()))
        self.register_buffer("delta_scale", torch.tensor(DELTA_SCALE, dtype=torch.float32), persistent=False)

    # ------------------------------------------------------------------ encode
    def encode(
        self,
        audio: torch.Tensor,  # (B, T, audio_dim) frozen Whisper features
        audio_mask: Optional[torch.Tensor] = None,  # (B, T) bool
        active: Optional[torch.Tensor] = None,  # (B,) bool
        strength: Optional[torch.Tensor] = None,  # (B,) float in [0, 1]
        delta: Optional[torch.Tensor] = None,  # (B, 6) parsed delta_p (m / rad)
        effector: Optional[torch.Tensor] = None,  # (B,) long
    ) -> CorrectionContext:
        audio = audio.float()
        b = audio.shape[0]
        if audio_mask is None:
            audio_mask = torch.ones(audio.shape[:2], dtype=torch.bool, device=audio.device)
        audio_mask = audio_mask.bool()
        active = audio_mask.any(-1) if active is None else active.bool()
        strength = torch.ones(b, device=audio.device) if strength is None else strength.float()
        strength = torch.where(active, strength, torch.zeros_like(strength))

        q = self.queries[None].expand(b, -1, -1).float()
        tok = q + self.projector(q, self.audio_norm(audio), audio_mask)
        tok = tok + self.proj_mlp(self.proj_norm(tok))
        pooled = tok.mean(1)
        head = self.disp_out(F.silu(self.disp_in(pooled)))
        delta_hat = head[:, :6] * self.delta_scale
        eff_logits = head[:, 6:]
        if self.config.use_given_delta and delta is not None:
            delta_cond = delta.float() / self.delta_scale
        else:
            delta_cond = delta_hat.detach() / self.delta_scale
        if self.config.use_given_delta and effector is not None:
            eff_emb = self.effector_embed(effector.long())
        else:
            eff_emb = torch.softmax(eff_logits, -1) @ self.effector_embed.weight
        tok = tok + self.delta_embed(delta_cond)[:, None] + eff_emb[:, None]
        pooled = tok.mean(1)
        token_mask = active[:, None].expand(b, tok.shape[1])
        return CorrectionContext(tok, token_mask, pooled, strength, active, delta_hat, eff_logits)

    # ------------------------------------------------------------------ gated paths
    def _s(self, ctx: CorrectionContext, like: torch.Tensor) -> torch.Tensor:
        return ctx.strength.to(device=like.device, dtype=like.dtype)

    def condition_action_input(self, ctx: CorrectionContext, tokens: torch.Tensor) -> torch.Tensor:
        """(a) tokens: (B, H, width) after ActionDiT.action_encoder."""
        s = self._s(ctx, tokens)[:, None, None]
        upd = self.in_xattn(tokens.float(), ctx.tokens.float(), ctx.token_mask).to(tokens.dtype)
        return tokens + torch.tanh(self.gate_in).to(tokens.dtype) * s * upd

    def condition_action_time(self, ctx: CorrectionContext, t: torch.Tensor) -> torch.Tensor:
        """(b) t: (B, width) action-expert time embedding, before time_projection (adaLN)."""
        s = self._s(ctx, t)
        if t.shape[0] == 1 and s.shape[0] > 1:
            t = t.expand(s.shape[0], -1)
        upd = self.ada_proj(ctx.pooled.float()).to(t.dtype)
        return t + torch.tanh(self.gate_ada).to(t.dtype) * s[:, None] * upd

    def pre_head(self, ctx: CorrectionContext, h: torch.Tensor) -> torch.Tensor:
        """(c) h: (B, H, width) final action-expert hidden state, before ActionDiT.head."""
        s = self._s(ctx, h)[:, None, None]
        upd = self.head_xattn(h.float(), ctx.tokens.float(), ctx.token_mask).to(h.dtype)
        return h + torch.tanh(self.gate_head).to(h.dtype) * s * upd

    def condition_future_t_mod(
        self, ctx: CorrectionContext, t_mod: torch.Tensor, future_mask: torch.Tensor
    ) -> torch.Tensor:
        """(d) t_mod: (B, S, 6, video_dim) merged video/DINO/pointmap modulation;
        future_mask: (S,) bool, True for noisy future tokens, False for clean anchors."""
        if not self.config.condition_video:
            return t_mod
        b = ctx.strength.shape[0]
        s = self._s(ctx, t_mod)
        off = self.video_up(self.video_down(ctx.pooled.float())).to(t_mod.dtype)
        off = off.view(b, 1, 6, self.config.video_dim)
        m = future_mask.to(device=t_mod.device, dtype=t_mod.dtype)[None, :, None, None]
        if t_mod.shape[0] == 1 and b > 1:
            t_mod = t_mod.expand(b, -1, -1, -1)
        return t_mod + torch.tanh(self.gate_video).to(t_mod.dtype) * s[:, None, None, None] * off * m

    # ------------------------------------------------------------------ losses
    def aux_loss(self, ctx: CorrectionContext, delta=None, effector=None) -> torch.Tensor:
        """Per-sample auxiliary DisplacementHead loss (0 for inactive samples)."""
        act = ctx.active.float()
        loss = torch.zeros_like(act)
        if delta is not None:
            loss = loss + ((ctx.delta_hat - delta.float()) / self.delta_scale).abs().mean(-1)
        if effector is not None:
            loss = loss + F.cross_entropy(ctx.effector_logits, effector.long(), reduction="none")
        return act * loss


# ---------------------------------------------------------------------- helpers used by the patch
def future_token_mask(Sv: int, tpf: int, Sd: int, dtpf: int, Sp: int, ptpf: int, device=None) -> torch.Tensor:
    """Merged visual sequence is [video | dino | pointmap] (FlexPi._merge_aux_into_video_stream);
    the first frame of each stream is a clean anchor (timestep forced to 0)."""
    m = torch.ones(Sv + Sd + Sp, dtype=torch.bool, device=device)
    m[:tpf] = False
    if Sd:
        m[Sv : Sv + dtpf] = False
    if Sp:
        m[Sv + Sd : Sv + Sd + ptpf] = False
    return m


def correction_from_sample(meci: MECICorrection, sample: dict) -> Optional[CorrectionContext]:
    """Training-batch keys: meci_audio (B,T,768), meci_audio_mask, meci_active, meci_strength,
    meci_delta (B,6), meci_effector (B,). Returns None when the batch carries no correction."""
    audio = sample.get("meci_audio")
    if meci is None or audio is None:
        return None
    dev = meci.queries.device
    g = lambda k: (sample[k].to(dev) if sample.get(k) is not None else None)  # noqa: E731
    return meci.encode(
        audio.to(dev), g("meci_audio_mask"), g("meci_active"), g("meci_strength"), g("meci_delta"), g("meci_effector")
    )


def apply_explicit_residual(action: torch.Tensor, residual, active: bool = True) -> torch.Tensor:
    """Add the normalized-space explicit residual (meci_metric.build_correction_inputs) to a sampled
    chunk. action: (H, A) or (B, H, A) normalized; residual broadcastable to it."""
    if residual is None or not active:
        return action
    r = torch.as_tensor(residual, dtype=action.dtype, device=action.device)
    return action + r


def awr_weights(advantages: torch.Tensor, beta: float = 1.0, w_max: float = 20.0) -> torch.Tensor:
    """Advantage-weighted regression weights exp(A / beta), clipped at w_max."""
    return torch.clamp(torch.exp(advantages / beta), max=w_max)


def flow_dpo_loss(err_w, err_l, err_w_ref, err_l_ref, beta: float = 1.0) -> torch.Tensor:
    """Diffusion-DPO-style preference loss on flow-matching errors (per-sample MSEs at a shared
    noise / timestep): -log sigmoid(-beta * ((e_w - e_w_ref) - (e_l - e_l_ref)))."""
    margin = (err_w - err_w_ref) - (err_l - err_l_ref)
    return -F.logsigmoid(-beta * margin)


def meci_state_dict(model: nn.Module) -> dict:
    """Adapter-only weights (everything under '.meci.')."""
    return {k: v for k, v in model.state_dict().items() if ".meci." in k or k.startswith("meci.")}


def count_parameters(meci: MECICorrection) -> int:
    return sum(p.numel() for p in meci.parameters())
