# MECI for Flex-π: spoken mid-execution corrections for a world-action model

Plugin code for *Add Mid-Execution Corrections Interface to World Action Models, as Fine-Tuning Data*
(Jessica Wu, University of Florida). Planned public home: https://github.com/jessicawu-333/meci-flexpi (not created yet).

**Status:** design + patch + CPU unit tests only. Nothing has been trained or evaluated on a robot, in
simulation, or at full model size, and no task results are claimed.

Upstream: [geyan21/flex-pi](https://github.com/geyan21/flex-pi) (MIT) pinned at
`044b8c2080a14a3d0082f81f612474643446faf3` (2026-10-28, "Flex-π: preview pretrained checkpoints with zero-shot and fine-tuning support").

| Path | What |
|---|---|
| `meci_flexpi/meci_correction.py` | Correction adapter: Whisper-feature projector, DisplacementHead, four zero-gated paths (a)–(d), AWR / flow-DPO helpers |
| `meci_flexpi/meci_metric.py` | Phrase grammar → δp; effector routing; LIBERO / RoboTwin / YAM layouts; residual in Flex-π's normalized action space |
| `patches/flexpi-meci.diff` | Patch against the pinned commit (3 files modified, 2 added; +820 / −3 lines) |
| `patches/make_patched.py` | The exact edits used to author the diff |
| `tests/` | `fetch_upstream.sh` (raw download + patch), `run_tests.sh`, tiny real-FlexPi harness, 28 tests, `test_output.txt` |
| `DESIGN.md` | Hook points with file:line, design choices, open items |

## Run the tests (CPU, ~1 min after download)
```bash
pip install "torch==2.7.1" einops imageio safetensors rich gitpython timm "torchvision==0.22.1" omegaconf hydra-core pytest
tests/fetch_upstream.sh /tmp/flexpi
FLEXPI_PRISTINE=/tmp/flexpi/pristine FLEXPI_PATCHED=/tmp/flexpi/patched tests/run_tests.sh
```
Result on the build box (torch 2.7.1, CPU): **28 passed**. Full-size adapter (action width 1024, video width 3072):
**19,662,413 parameters**, of which 1,263,681 belong to the future-stream path (d).

## Use (sketch)
```python
meci = model.enable_meci()                       # attach adapter; outputs unchanged until gates train
corr = meci_metric.build_correction_inputs("left hand more left half an inch", meci_metric.YAM,
          action_horizon=32, norm_scale=scale, anchor_rotation={"left": R_l, "right": R_r})
corr["audio"] = whisper_small_features            # [T, 768], frozen encoder, outside the graph
out = model.infer_action(..., correction=corr)    # action-only or joint mode
```
Training batches may carry `meci_audio`, `meci_audio_mask`, `meci_active`, `meci_strength`, `meci_delta`,
`meci_effector` and an optional per-sample `meci_awr_weight`.
