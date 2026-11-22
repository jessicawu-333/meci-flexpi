"""MECI for Flex-pi -- framework-agnostic (NumPy) metric-correction utilities.

Spoken mid-execution corrections are *metric end-effector nudges* that may name the effector:
    "left hand more left half an inch", "right gripper up 1 cm", "move forward 1 inch",
    "rotate gripper 10 degrees".
They are parsed into (effector e, direction, magnitude) -> a 6-D Cartesian displacement
delta_p = [dx, dy, dz, droll, dpitch, dyaw] in metres / radians in a declared frame, and then
mapped into Flex-pi's *normalized* action space for the platform's action layout.

The parser and grammar are shared with the pi0.5 version of MECI; the platform layouts below were
checked in geyan21/flex-pi @ 044b8c2080a14a3d0082f81f612474643446faf3:
  * Canonical 32-D layout (paper App. A.1; configs/data/yam.yaml norm_skip_dims_mask):
      0-2 L EE pos, 3-8 L rot6d, 9-11 R EE pos, 12-17 R rot6d, 18/19 L/R gripper, 20-31 L/R joints (6+6).
  * LIBERO (configs/data/libero.yaml): action_output_dim 32, ScatterToChannels with
      used_action_channel_ids [0,1,2,3,4,5,18] -> per-step OSC_POSE deltas [dpos(3), axis-angle(3)] in
      slots 0-5 and the gripper in slot 18; norm_default_mode z-score.
  * RoboTwin 2.0 (configs/data/robotwin.yaml): action_output_dim 14, ConcatLeftAlign, z-score; actions are
      absolute joint targets executed with task_env.take_action(action, action_type="qpos")
      (experiments/robotwin/flexpi_policy/deploy_policy.py l.763). Vector order is
      [left arm joints, left gripper, right arm joints, right gripper]
      (third_party/RoboTwin/envs/_base_task.py l.487-494); 6 joints per arm for 14-D.
  * YAM (configs/data/yam.yaml, transforms/yam_relative_action.py): 32-D, anchor-relative actions
      ("first" anchor = state at the chunk's first timestep). EE targets are body-frame relative poses
      R_base^T (p - p_base); joints are scalar deltas; grippers absolute. norm_default_mode q01/q99 with the
      rot6d slots skipped.
  * Normalization (src/flexpi/datasets/lerobot/utils/normalizer.py, SingleFieldLinearNormalizer):
      forward x_n = x * scale + offset (then clamp); backward x = (x_n - offset) / scale (no clamp).
      z-score: scale = 1 / (std + 1e-8); q01/q99: scale = 2 / (q99 - q01) (range < 1e-4 -> scale 1);
      skipped dims: scale = 1. An additive raw residual therefore maps exactly to dn = dx * scale.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import dataclasses
import math
import re

import numpy as np

# ----------------------------------------------------------------------------------------------
# 1. Units, directions, effectors
# ----------------------------------------------------------------------------------------------
INCH_TO_M = 0.0254
CM_TO_M = 0.01
MM_TO_M = 0.001
DEG_TO_RAD = math.pi / 180.0

# Declared spoken-language frame (robot base). Axis signs are PLACEHOLDERS to be calibrated per
# platform by commanding a known residual and observing the EE motion (see DESIGN.md, open decisions).
CORRECTION_FRAME = "robot_base"
DIRECTION_AXES: dict[str, tuple[int, int]] = {
    "forward": (0, +1),
    "back": (0, -1),
    "left": (1, +1),
    "right": (1, -1),
    "up": (2, +1),
    "down": (2, -1),
}
ROTATION_AXES: dict[str, tuple[int, int]] = {"roll": (3, +1), "pitch": (4, +1), "yaw": (5, +1)}
ROTATION_ALIASES = {"gripper": "yaw", "wrist": "yaw", "end effector": "yaw", "end-effector": "yaw"}

EFFECTORS = ("default", "left", "right")
EFFECTOR_TO_ID = {e: i for i, e in enumerate(EFFECTORS)}
EFFECTOR_ALIASES: tuple[tuple[str, str], ...] = (
    ("left hand", "left"),
    ("left arm", "left"),
    ("left gripper", "left"),
    ("right hand", "right"),
    ("right arm", "right"),
    ("right gripper", "right"),
)

_WORD_MAGNITUDES = {
    "a quarter": 0.25,
    "quarter": 0.25,
    "a half": 0.5,
    "half": 0.5,
    "one": 1.0,
    "two": 2.0,
    "three": 3.0,
    "four": 4.0,
    "five": 5.0,
    "ten": 10.0,
}
_UNIT_ALIASES = {
    "in": "inch",
    "inch": "inch",
    "inches": "inch",
    "cm": "cm",
    "centimeter": "cm",
    "centimeters": "cm",
    "centimetre": "cm",
    "centimetres": "cm",
    "mm": "mm",
    "millimeter": "mm",
    "millimeters": "mm",
    "millimetre": "mm",
    "millimetres": "mm",
    "m": "m",
    "meter": "m",
    "meters": "m",
    "metre": "m",
    "metres": "m",
    "deg": "deg",
    "degs": "deg",
    "degree": "deg",
    "degrees": "deg",
    "rad": "rad",
    "radian": "rad",
    "radians": "rad",
}
_UNIT_RE = (
    r"(?P<unit>inch(?:es)?|in\b|centimet(?:er|re)s?|cm\b|millimet(?:er|re)s?|mm\b|met(?:er|re)s?|m\b|"
    r"degrees?|degs?\b|radians?|rad\b)"
)


@dataclasses.dataclass
class MetricCommand:
    """A parsed metric nudge."""

    kind: str  # "translate" | "rotate"
    direction: str  # forward/back/left/right/up/down or roll/pitch/yaw
    magnitude: float  # in `unit`
    unit: str  # canonical unit
    axis: int  # index into the 6-D Cartesian delta
    sign: int
    effector: str = "default"  # "default" | "left" | "right"
    phrase: str = ""

    @property
    def effector_id(self) -> int:
        return EFFECTOR_TO_ID[self.effector]

    def delta_p(self) -> np.ndarray:
        """6-D Cartesian displacement [dx, dy, dz, droll, dpitch, dyaw] in metres / radians."""
        d = np.zeros(6, dtype=np.float32)
        d[self.axis] = self.sign * magnitude_to_si(self.magnitude, self.unit, self.kind)
        return d


def magnitude_to_si(magnitude: float, unit: str, kind: str) -> float:
    u = _UNIT_ALIASES.get(unit.lower().strip(), unit.lower().strip())
    if kind == "rotate":
        if u == "deg":
            return magnitude * DEG_TO_RAD
        if u == "rad":
            return magnitude
        raise ValueError(f"unsupported rotation unit {unit!r}")
    table = {"inch": INCH_TO_M, "cm": CM_TO_M, "mm": MM_TO_M, "m": 1.0}
    if u not in table:
        raise ValueError(f"unsupported translation unit {unit!r}")
    return magnitude * table[u]


def parse_metric_command(phrase: str) -> MetricCommand | None:
    """Rule-based parser (the ASR -> parser baseline) for templated metric nudges.

    The effector phrase ("left hand", "right gripper", ...) is matched and removed *first*, so in
    "left hand more left half an inch" the first "left" is the effector and the second the direction.
    Returns None if the phrase is not a metric nudge.
    """
    text = " ".join(phrase.lower().replace(",", " ").strip().split())
    if not text:
        return None
    effector = "default"
    for alias, canon in EFFECTOR_ALIASES:
        if re.search(rf"\b{re.escape(alias)}\b", text):
            effector = canon
            text = " ".join(re.sub(rf"\b{re.escape(alias)}\b", " ", text, count=1).split())
            break

    # magnitude + unit
    mag = unit = None
    for word, val in sorted(_WORD_MAGNITUDES.items(), key=lambda kv: -len(kv[0])):
        m = re.search(rf"\b{re.escape(word)}\b\s+(?:an?\s+)?{_UNIT_RE}", text)
        if m:
            mag, unit = val, m.group("unit")
            break
    if mag is None:
        m = re.search(rf"(?P<mag>\d+(?:\.\d+)?)\s*{_UNIT_RE}", text)
        if m:
            mag, unit = float(m.group("mag")), m.group("unit")
    if mag is None:
        return None
    unit = _UNIT_ALIASES.get(unit.strip(), unit.strip())

    if unit in ("deg", "rad") or re.search(r"\brotate\b|\bturn\b|\btwist\b", text):
        kind = "rotate"
        direction = None
        for alias, canon in ROTATION_ALIASES.items():
            if re.search(rf"\b{re.escape(alias)}\b", text):
                direction = canon
                break
        for name in ROTATION_AXES:
            if re.search(rf"\b{name}\b", text):
                direction = name
                break
        direction = direction or "yaw"
        axis, sign = ROTATION_AXES[direction]
        if re.search(r"\b(counter[- ]?clockwise|anticlockwise)\b", text):
            sign = +1
        elif re.search(r"\bclockwise\b", text):
            sign = -1
    else:
        kind = "translate"
        direction = None
        for name in ("forward", "backward", "back", "left", "right", "up", "down"):
            if re.search(rf"\b{name}\b", text):
                direction = "back" if name == "backward" else name
                break
        if direction is None:
            return None
        axis, sign = DIRECTION_AXES[direction]
    return MetricCommand(kind, direction, float(mag), unit, axis, sign, effector=effector, phrase=phrase)


# ----------------------------------------------------------------------------------------------
# 2. Platform action layouts (verified in flex-pi @ 044b8c2, see module docstring)
# ----------------------------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class ArmSlice:
    pos_start: int | None = None  # first EE-position slot (3 slots), Cartesian layouts
    rot_start: int | None = None  # first rotation slot (3 for axis-angle deltas), None = no rotation residual
    joint_start: int | None = None  # first joint slot
    num_joints: int = 0
    gripper_index: int | None = None


@dataclasses.dataclass(frozen=True)
class PlatformLayout:
    name: str
    action_dim: int  # model action width (32 canonical, 14 for RoboTwin)
    arms: dict  # effector name -> ArmSlice ("default" for single-arm platforms)
    # "cartesian": residual in EE slots; "joint": residual via DLS IK in joint slots;
    # "cartesian+joint": EE slots and (optionally, with a Jacobian) the joint-delta slots (YAM).
    space: str
    # "per_step": each action is an increment (LIBERO OSC deltas): sum over the window == delta.
    # "chunk_relative": targets relative to the chunk-anchor state (YAM): ramp to delta.
    # "absolute": absolute targets (RoboTwin qpos): ramp to delta measured from the utterance.
    semantics: str
    cartesian_scale: tuple | None = None  # physical units per unit raw action (LIBERO OSC_POSE output_max)
    body_frame: bool = False  # YAM: EE targets are expressed in the anchor body frame (R_base^T dp)


LIBERO = PlatformLayout(
    name="libero",
    action_dim=32,
    arms={"default": ArmSlice(pos_start=0, rot_start=3, gripper_index=18)},
    space="cartesian",
    semantics="per_step",
    cartesian_scale=(0.05, 0.05, 0.05, 0.5, 0.5, 0.5),  # robosuite OSC_POSE output_max (m, rad)
)
ROBOTWIN = PlatformLayout(
    name="robotwin",
    action_dim=14,
    arms={
        "left": ArmSlice(joint_start=0, num_joints=6, gripper_index=6),
        "right": ArmSlice(joint_start=7, num_joints=6, gripper_index=13),
    },
    space="joint",
    semantics="absolute",
)
YAM = PlatformLayout(
    name="yam",
    action_dim=32,
    arms={
        "left": ArmSlice(pos_start=0, joint_start=20, num_joints=6, gripper_index=18),
        "right": ArmSlice(pos_start=9, joint_start=26, num_joints=6, gripper_index=19),
    },
    space="cartesian+joint",
    semantics="chunk_relative",
    body_frame=True,
)
PLATFORMS = {p.name: p for p in (LIBERO, ROBOTWIN, YAM)}


def resolve_arm(layout: PlatformLayout, effector: str):
    """Named-effector routing. Single-arm platforms ignore the name; bimanual + unnamed -> None."""
    if effector in layout.arms:
        return layout.arms[effector]
    if "default" in layout.arms and len(layout.arms) == 1:
        return layout.arms["default"]
    return None


# ----------------------------------------------------------------------------------------------
# 3. Cartesian -> joint (first order), for joint-space slots (RoboTwin, optional on YAM)
# ----------------------------------------------------------------------------------------------
def dls_joint_delta(jacobian: np.ndarray, delta_p: np.ndarray, damping: float = 1e-3) -> np.ndarray:
    """Damped-least-squares IK step dq = J^T (J J^T + lambda^2 I)^-1 dp. `jacobian` is the (6, n) geometric
    Jacobian of the named arm at the current joint state, in the frame of delta_p. Flex-pi ships no arm
    kinematics; the caller supplies J. First order: valid for small nudges only."""
    jac = np.asarray(jacobian, dtype=np.float64)
    dp = np.asarray(delta_p, dtype=np.float64).reshape(-1)
    jjt = jac @ jac.T
    return (jac.T @ np.linalg.solve(jjt + (damping**2) * np.eye(jjt.shape[0]), dp)).astype(np.float32)


# ----------------------------------------------------------------------------------------------
# 4. Physical residual over the hold window, routed to the named effector
# ----------------------------------------------------------------------------------------------
def hold_fade_strengths(hold_steps: int = 16, fade_steps: int = 8) -> np.ndarray:
    """Per-control-step strengths of a correction: 1 during the hold, linear fade to 0."""
    s = np.ones(hold_steps, dtype=np.float32)
    if fade_steps > 0:
        f = min(fade_steps, hold_steps)
        s[-f:] = np.arange(f, 0, -1, dtype=np.float32) / float(f)
    return s


def raw_action_delta(
    layout: PlatformLayout,
    delta_p: np.ndarray,
    effector: str,
    jacobian_fn: Callable | None = None,
    anchor_rotation: dict | None = None,
) -> tuple[np.ndarray, bool]:
    """Total raw-action displacement (length layout.action_dim) for the named effector.

    Returns (delta_a, routed). routed=False when the effector cannot be resolved (bimanual + unnamed):
    delta_a is all zeros and the learned path decides. Gripper slots are never touched.
    `anchor_rotation[effector]` is the 3x3 base->EE rotation R_base of the chunk anchor (YAM body frame).
    Rotation residuals are applied only on axis-angle delta layouts (LIBERO); rot6d slots are left to the
    learned path.
    """
    out = np.zeros(layout.action_dim, dtype=np.float32)
    arm = resolve_arm(layout, effector)
    if arm is None:
        return out, False
    dp = np.asarray(delta_p, dtype=np.float64).reshape(6)
    key = effector if effector in layout.arms else "default"
    if "cartesian" in layout.space and arm.pos_start is not None:
        pos = dp[:3].copy()
        if layout.body_frame:
            if anchor_rotation is None or key not in anchor_rotation:
                raise ValueError(f"{layout.name}: body-frame targets need anchor_rotation[{key!r}] (R_base)")
            pos = np.asarray(anchor_rotation[key], dtype=np.float64).T @ pos
        if layout.cartesian_scale is not None:
            scale = np.asarray(layout.cartesian_scale, dtype=np.float64)
            out[arm.pos_start : arm.pos_start + 3] = pos / scale[:3]
            if arm.rot_start is not None:
                out[arm.rot_start : arm.rot_start + 3] = dp[3:] / scale[3:]
        else:
            out[arm.pos_start : arm.pos_start + 3] = pos
    if arm.joint_start is not None and arm.num_joints and (layout.space == "joint" or jacobian_fn is not None):
        if jacobian_fn is None:
            raise ValueError(f"{layout.name}: joint-space platform needs jacobian_fn(effector) -> (6, n_joints)")
        dq = dls_joint_delta(jacobian_fn(key), dp)
        if dq.shape[0] != arm.num_joints:
            raise ValueError(f"Jacobian has {dq.shape[0]} joints, layout expects {arm.num_joints}")
        out[arm.joint_start : arm.joint_start + arm.num_joints] = dq
    return out, True


def chunk_residual(
    layout: PlatformLayout,
    delta_a: np.ndarray,
    window_strengths: np.ndarray,
    chunk_start: int,
    action_horizon: int,
) -> np.ndarray:
    """Raw-space residual (action_horizon, action_dim) for a chunk starting `chunk_start` control steps
    after the utterance.
      per_step:       SUM over the window of r_t == delta_a
      chunk_relative: ramp of the share still to be done in this chunk (the anchor already contains the
                      motion executed in earlier chunks)
      absolute:       ramp of the cumulative share since the utterance (absolute targets carry no anchor)
    """
    w = np.asarray(window_strengths, dtype=np.float64)
    total = max(w.sum(), 1e-12)
    n_win = len(w)
    steps = np.arange(chunk_start, chunk_start + action_horizon)
    share = np.where(steps < n_win, w[np.clip(steps, 0, n_win - 1)], 0.0) / total
    if layout.semantics == "per_step":
        coeff = share
    elif layout.semantics == "chunk_relative":
        coeff = np.cumsum(share)
    elif layout.semantics == "absolute":
        before = w[: min(chunk_start, n_win)].sum() / total
        coeff = before + np.cumsum(share)
        coeff = np.where(steps < n_win, coeff, 0.0)  # released after the hold window
    else:
        raise ValueError(layout.semantics)
    return (coeff[:, None] * np.asarray(delta_a, dtype=np.float64)[None, :]).astype(np.float32)


# ----------------------------------------------------------------------------------------------
# 5. Map into Flex-pi's normalized action space (SingleFieldLinearNormalizer)
# ----------------------------------------------------------------------------------------------
def normalizer_scale(
    mode: str, *, mean=None, std=None, q01=None, q99=None, vmin=None, vmax=None, skip_mask=None
) -> np.ndarray:
    """Per-dimension `scale` of flex-pi's SingleFieldLinearNormalizer (normalizer.py l.111-169)."""
    if mode == "z-score":
        scale = 1.0 / (np.asarray(std, np.float64) + 1e-8)
    else:
        lo, hi = (q01, q99) if mode == "q01/q99" else (vmin, vmax)
        rng = np.asarray(hi, np.float64) - np.asarray(lo, np.float64)
        ignore = rng < 1e-4
        rng = np.where(ignore, 2.0, rng)
        scale = 2.0 / rng
    if skip_mask is not None:
        scale = np.where(np.asarray(skip_mask, bool), 1.0, scale)
    return scale


def normalize_residual(residual: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Additive raw residual -> additive residual in normalized space (exact: the map is affine)."""
    return (np.asarray(residual, np.float64) * np.asarray(scale, np.float64)).astype(np.float32)


def build_correction_inputs(
    phrase_or_cmd,
    layout: PlatformLayout,
    *,
    action_horizon: int,
    norm_scale: np.ndarray,
    chunk_start: int = 0,
    window_strengths: np.ndarray | None = None,
    jacobian_fn: Callable | None = None,
    anchor_rotation: dict | None = None,
) -> dict | None:
    """Policy-side helper: phrase -> the `correction` dict consumed by the patched FlexPi.infer_action.
    Whisper features (`audio`, `audio_mask`) are added separately by the frozen encoder."""
    cmd = parse_metric_command(phrase_or_cmd) if isinstance(phrase_or_cmd, str) else phrase_or_cmd
    if cmd is None:
        return None
    w = hold_fade_strengths() if window_strengths is None else np.asarray(window_strengths, np.float32)
    dp = cmd.delta_p()
    delta_a, routed = raw_action_delta(layout, dp, cmd.effector, jacobian_fn, anchor_rotation)
    r = chunk_residual(layout, delta_a, w, chunk_start, action_horizon)
    rn = normalize_residual(r, norm_scale)
    step = min(chunk_start, len(w) - 1)
    strength = float(w[step]) if chunk_start < len(w) else 0.0
    return {
        "active": np.bool_(strength > 0.0),
        "strength": np.float32(strength),
        "delta": dp.astype(np.float32),
        "effector": np.int64(cmd.effector_id),
        "residual": rn if routed else np.zeros_like(rn),
        "routed": routed,
    }


__all__ = [n for n in dir() if not n.startswith("_")]
