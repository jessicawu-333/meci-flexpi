"""Edits applied to pristine flex-pi @ 044b8c2 to produce the MECI-patched tree (used to author the .diff)."""
import sys, pathlib
root = pathlib.Path(sys.argv[1])

def edit(rel, old, new, count=1):
    p = root / rel
    s = p.read_text()
    n = s.count(old)
    assert n == count, f"{rel}: expected {count} match(es), found {n}: {old[:60]!r}"
    p.write_text(s.replace(old, new))

# ---------------- action_dit.py : paths (a) (b) (c)
A = "src/flexpi/models/action_dit.py"
edit(A, """        self.freqs = precompute_freqs_cis(attn_head_dim, end=1024)
""", """        self.freqs = precompute_freqs_cis(attn_head_dim, end=1024)
        # MECI: optional correction adapter (flexpi.models.meci_correction) and the per-call
        # correction context. Both None => the forward pass is the stock one.
        self.meci = None
        self._meci_ctx = None
""")
edit(A, """        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))

        t_mod = self.time_projection(t).unflatten(1, (6, self.hidden_dim))

        tokens = self.action_encoder(action_tokens)
""", """        t = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, timestep))
        t = self.meci_condition_time(t)  # MECI path (b): adaLN conditioning

        t_mod = self.time_projection(t).unflatten(1, (6, self.hidden_dim))

        tokens = self.action_encoder(action_tokens)
        tokens = self.meci_condition_input(tokens)  # MECI path (a): action-expert input
""")
edit(A, """    def post_dit(self, tokens: torch.Tensor, pre_state: Dict[str, Any]) -> torch.Tensor:
        return self.head(tokens)
""", """    def post_dit(self, tokens: torch.Tensor, pre_state: Dict[str, Any]) -> torch.Tensor:
        return self.head(self.meci_pre_head(tokens))  # MECI path (c): pre-output residual

    # ---- MECI hooks: identity unless an adapter is attached AND a correction context is set ----
    def attach_meci(self, meci) -> None:
        self.meci = meci

    def meci_condition_time(self, t: torch.Tensor) -> torch.Tensor:
        if self.meci is None or self._meci_ctx is None:
            return t
        return self.meci.condition_action_time(self._meci_ctx, t)

    def meci_condition_input(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.meci is None or self._meci_ctx is None:
            return tokens
        return self.meci.condition_action_input(self._meci_ctx, tokens)

    def meci_pre_head(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.meci is None or self._meci_ctx is None:
            return tokens
        return self.meci.pre_head(self._meci_ctx, tokens)
""")

# ---------------- backbone.py : the compiled action step inlines pre_dit/post_dit -> same hooks
B = "src/flexpi/models/backbone.py"
edit(B, """        ae = self.action_expert
        t = ae.time_embedding(sinusoidal_embedding_1d(ae.freq_dim, timestep_action))
        t_mod = ae.time_projection(t).unflatten(1, (6, ae.hidden_dim))
        tokens = ae.action_encoder(latents_action)
""", """        ae = self.action_expert
        t = ae.time_embedding(sinusoidal_embedding_1d(ae.freq_dim, timestep_action))
        t = ae.meci_condition_time(t)  # MECI path (b)
        t_mod = ae.time_projection(t).unflatten(1, (6, ae.hidden_dim))
        tokens = ae.action_encoder(latents_action)
        tokens = ae.meci_condition_input(tokens)  # MECI path (a)
""")
edit(B, """            action_only_attention_mask=action_only_attention_mask,
        )
        return ae.head(tokens)
""", """            action_only_attention_mask=action_only_attention_mask,
        )
        return ae.head(ae.meci_pre_head(tokens))  # MECI path (c)
""")

# ---------------- flexpi.py : path (d), training loss, inference entry
F = "src/flexpi/models/flexpi.py"
edit(F, """from .helpers.flex_joint import FlexJointConfig, sample_flex_batch_flags
""", """from .helpers.flex_joint import FlexJointConfig, sample_flex_batch_flags
from . import meci_correction as _meci
""")
# helpers on FlexPi (inserted before _build_stream_t_mod)
edit(F, """    def _build_stream_t_mod(
        self, timestep_per_token: torch.Tensor, seq_len: int, B: int,
    ) -> torch.Tensor:""", """    # ------------------------------------------------------------------
    # MECI (mid-execution corrections). Inactive unless enable_meci() was
    # called AND a correction is supplied for the current call/batch.
    # ------------------------------------------------------------------
    def enable_meci(self, config=None):
        \"\"\"Attach a MECI adapter to the action expert (paths a-c) and use it for path (d).\"\"\"
        cfg = config or _meci.MECIConfig(
            width=int(self.action_expert.hidden_dim),
            video_dim=int(self.video_expert.hidden_dim),
        )
        meci = _meci.MECICorrection(cfg).to(device=self.device)
        self.action_expert.attach_meci(meci)
        return meci

    @property
    def meci(self):
        return getattr(self.action_expert, "meci", None)

    def _meci_set_ctx(self, ctx) -> None:
        self.action_expert._meci_ctx = ctx

    def _meci_condition_future(self, merged_t_mod, Sv, tpf, Sd, dtpf, Sp, ptpf):
        \"\"\"MECI path (d): offset the modulation of the noisy future video/DINO/pointmap tokens.\"\"\"
        ctx = self.action_expert._meci_ctx
        if self.meci is None or ctx is None:
            return merged_t_mod
        mask = _meci.future_token_mask(Sv, tpf, Sd, dtpf, Sp, ptpf, device=merged_t_mod.device)
        return self.meci.condition_future_t_mod(ctx, merged_t_mod, mask)

    def _build_stream_t_mod(
        self, timestep_per_token: torch.Tensor, seq_len: int, B: int,
    ) -> torch.Tensor:""")
# training loss: encode correction at the start
edit(F, """    def _base_training_loss(self, sample, tiled: bool = False):
        inputs = self.build_inputs(sample, tiled=tiled)
""", """    def _base_training_loss(self, sample, tiled: bool = False):
        # MECI: correction context for this batch (None when the batch carries no correction).
        meci_ctx = _meci.correction_from_sample(self.meci, sample)
        self._meci_set_ctx(meci_ctx)
        inputs = self.build_inputs(sample, tiled=tiled)
""")
# path (d) in training (first merge site, inside _base_training_loss) and in joint inference (second)
old_merge = """        merged_tokens, merged_freqs, merged_t_mod, merged_ctx_mask = (
            self._merge_aux_into_video_stream(
                video_pre, dino_tokens, dino_t_mod, dino_freqs,
                pointmap_tokens, pt_t_mod, pt_freqs,
            )
        )
        effective_dtpf = dtpf
"""
new_merge = """        merged_tokens, merged_freqs, merged_t_mod, merged_ctx_mask = (
            self._merge_aux_into_video_stream(
                video_pre, dino_tokens, dino_t_mod, dino_freqs,
                pointmap_tokens, pt_t_mod, pt_freqs,
            )
        )
        merged_t_mod = self._meci_condition_future(merged_t_mod, Sv, tpf, Sd, dtpf, Sp, ptpf)  # MECI (d)
        effective_dtpf = dtpf
"""
edit(F, old_merge, new_merge, count=2)
# AWR weighting of the action flow-matching loss + auxiliary displacement loss
edit(F, """        wa = self.train_action_scheduler.training_weight(t_action).to(la.device, dtype=la.dtype)
        loss_action = (la * wa).mean()
""", """        wa = self.train_action_scheduler.training_weight(t_action).to(la.device, dtype=la.dtype)
        # MECI: optional per-sample AWR weights (meci_metric / meci_correction.awr_weights); 1 when absent.
        w_awr = sample.get("meci_awr_weight")
        if w_awr is not None:
            wa = wa * w_awr.to(la.device, dtype=la.dtype)
        loss_action = (la * wa).mean()
        if meci_ctx is not None:
            loss_meci_aux = self.meci.aux_loss(
                meci_ctx, sample.get("meci_delta"), sample.get("meci_effector"),
            ).mean()
        else:
            loss_meci_aux = None
""")
edit(F, """            + self.loss_lambda_pointmap * loss_pointmap
        )
        loss_dict = {""", """            + self.loss_lambda_pointmap * loss_pointmap
        )
        if loss_meci_aux is not None:
            total = total + self.meci.config.aux_weight * loss_meci_aux
        loss_dict = {""")
edit(F, """            "loss_pointmap": self.loss_lambda_pointmap * float(loss_pointmap.detach().item()),
        }
        return total, loss_dict
""", """            "loss_pointmap": self.loss_lambda_pointmap * float(loss_pointmap.detach().item()),
        }
        if loss_meci_aux is not None:
            loss_dict["loss_meci_aux"] = float(loss_meci_aux.detach().item())
        self._meci_set_ctx(None)
        return total, loss_dict
""")
# inference entry point
edit(F, """        present_dino: Optional[bool] = None,
        present_pointmap: Optional[bool] = None,
    ):
        # Temporarily override self.joint_* when runtime flags are supplied.""", """        present_dino: Optional[bool] = None,
        present_pointmap: Optional[bool] = None,
        # MECI: optional spoken correction for this call, a dict with
        # ``audio`` [T,768] (frozen Whisper-small features), optional ``audio_mask``,
        # ``active``, ``strength``, ``delta`` [6], ``effector`` and ``residual`` [H,A]
        # (normalized; see meci_metric.build_correction_inputs). None => stock behavior.
        correction: Optional[dict] = None,
    ):
        # Temporarily override self.joint_* when runtime flags are supplied.""")
edit(F, """        self._maybe_warn_unreachable_combo()
        self._maybe_log_effective_regime()
        try:
            return self._infer_action_dispatch(""", """        self._maybe_warn_unreachable_combo()
        self._maybe_log_effective_regime()
        meci_ctx = None
        if correction is not None and self.meci is not None:
            def _b(k, dtype=None):
                v = correction.get(k)
                if v is None:
                    return None
                return torch.as_tensor(v, dtype=dtype, device=self.device).unsqueeze(0)
            meci_ctx = self.meci.encode(
                _b("audio", torch.float32), _b("audio_mask", torch.bool), _b("active", torch.bool),
                _b("strength", torch.float32), _b("delta", torch.float32), _b("effector", torch.long),
            )
        self._meci_set_ctx(meci_ctx)
        try:
            out = self._infer_action_dispatch(""")
edit(F, """                return_stream_latents=return_stream_latents,
                force_joint_path=_force_joint_path,
            )
        finally:
            self.joint_video, self.joint_dino, self.joint_pointmap = _saved_joint""", """                return_stream_latents=return_stream_latents,
                force_joint_path=_force_joint_path,
            )
            if meci_ctx is not None and bool(meci_ctx.active[0]):
                # Explicit residual (normalized action space), added after sampling.
                out["action"] = _meci.apply_explicit_residual(out["action"], correction.get("residual"))
            return out
        finally:
            self._meci_set_ctx(None)
            self.joint_video, self.joint_dino, self.joint_pointmap = _saved_joint""")
print("patched OK")
