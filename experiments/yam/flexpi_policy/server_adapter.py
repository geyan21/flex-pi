"""BasePolicy adapter wrapping ``YamFlexPiPolicy`` for websocket serving.

Translates between the flat wire schema used by OpenPI-style bridges (one
key per per-cam stream) and the nested obs dict that
``YamFlexPiPolicy.infer_action_chunk`` consumes.

Wire schema in (from bridge, msgpack-unpacked):

    observation/image_<cam>       np.uint8  [H, W, 3]   RGB at the advertised per_cam_hw
    observation/depth_<cam>       np.uint16 [H, W]      mm, at the advertised per_cam_hw
    observation/intrinsics_<cam>  np.float32 [4]        [fx, fy, cx, cy] at the depth grid
    observation/state             np.float32 [32]       yam_eef.STATE_LAYOUT
    prompt                        str                   task instruction (optional)

Wire schema out (to bridge, before msgpack-packed):

    actions          np.float32 [action_horizon, 32]   absolute yam_eef
    policy_timing    dict[str, float]
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from experiments.yam.flexpi_policy._openpi_vendor.base_policy import BasePolicy
from experiments.yam.flexpi_policy.deploy_policy import YamFlexPiPolicy, _CAM_ORDER
from experiments.yam.flexpi_policy.prediction_recorder import (
    PredictionRecorder,
    make_session_dir,
)

logger = logging.getLogger(__name__)

# Optional post-processing modules (optional post-processing). Imports are deferred
# in type hints so callers without osqp/scipy can still import server_adapter.
if False:  # TYPE_CHECKING
    from .speed_adapter import SpeedAdapter
    from .temporal_smoother import TemporalSmoother


# Indices into the 32-D yam_eef chunk that the speed adapter's heuristic
# should consume. Limiting to the joint slice keeps ||v_step|| meaningful
# (EEF rot6d is non-Euclidean).
_ADAPTER_FEATURE_SLICE = slice(20, 32)


class YamFlexPiServerPolicy(BasePolicy):
    """Adapter so ``YamFlexPiPolicy`` plugs into ``WebsocketPolicyServer``."""

    def __init__(
        self,
        policy: YamFlexPiPolicy,
        default_prompt: str = "",
        smoother: "Optional[TemporalSmoother]" = None,
        adapter: "Optional[SpeedAdapter]" = None,
        flex_defaults: Optional[Dict[str, bool]] = None,
        record_predictions_dir: Optional[str] = None,
        record_fps: float = 7.5,
        record_frames_per_chunk: int = 0,
        record_pca_warmup_chunks: int = 3,
        record_split: str = "none",
        record_mode: str = "raw",
        record_drop_anchor: bool = True,
        record_ckpt_path: Optional[str] = None,
    ) -> None:
        self._policy = policy
        self._default_prompt = str(default_prompt or "")
        self._smoother = smoother
        self._adapter = adapter
        self._n_calls = 0
        self._flex_defaults: Dict[str, bool] = {
            k: bool(v) for k, v in (flex_defaults or {}).items() if v is not None
        }
        # ---- Prediction recording (off by default) ----
        # When ``record_predictions_dir`` is set, each WS connection becomes a
        # session under that base dir. Inside ``infer()`` we force
        # ``joint_video=joint_dino=joint_pointmap=True`` so the model returns
        # predicted video/DINO/pointmap latents alongside the action, then hand
        # them to ``PredictionRecorder``. record_mode="raw" (default) only
        # CPU-copies + enqueues them for a background dump thread — rendering
        # happens post-hoc via scripts/render_prediction_dump.py; "live"
        # renders the 6-row MP4 in-loop (VAE/DINO/PCA + encode per chunk).
        self._record_base_dir: Optional[Path] = (
            Path(record_predictions_dir).resolve() if record_predictions_dir else None
        )
        self._record_fps = float(record_fps)
        self._record_frames_per_chunk = int(record_frames_per_chunk)
        self._record_pca_warmup_chunks = int(record_pca_warmup_chunks)
        self._record_split = str(record_split)
        self._record_mode = str(record_mode)
        self._record_drop_anchor = bool(record_drop_anchor)
        self._record_ckpt_path = str(record_ckpt_path) if record_ckpt_path else None
        self._recorder: Optional[PredictionRecorder] = None

        # ---- Session timing (always on; no opt-in flag) ----
        # Starts on the FIRST successful infer() in a session (= "first
        # action the model produced"), stops when the WS client disconnects
        # (ctrl+c on the bridge). Reported via print() so the wall-clock
        # duration shows up in the server's terminal regardless of logging
        # level. ``None`` between sessions; reset by ``start_session``.
        self._session_start_t: Optional[float] = None
        self._session_remote: Optional[str] = None
        # Monotonic session token — see start_session/end_session. Guards the
        # recorder against a stale handler's late finally-block teardown.
        self._session_token: int = 0
        if self._record_base_dir is not None:
            self._record_base_dir.mkdir(parents=True, exist_ok=True)
            logger.info(
                "[server_adapter] recording predictions to %s (mode=%s, fps=%g)",
                self._record_base_dir, self._record_mode, self._record_fps,
            )

    @property
    def metadata(self) -> Dict[str, Any]:
        """Sent to the bridge on websocket connect."""
        # per_cam_hw is the checkpoint's depth grid: the bridge resizes depth
        # and rescales K onto it in lock-step, and the policy never rescales K.
        # A plain dict[str, list[int]] so it survives the msgpack round-trip.
        per_cam_hw = {cam: [int(h), int(w)] for cam, (h, w) in self._policy._depth_hw}
        return {
            "model": "yam_flexpi_unified_joint",
            "cam_order": list(_CAM_ORDER),
            "per_cam_hw": per_cam_hw,
            "action_horizon": int(self._policy.action_horizon),
            "action_dim": 32,
            "state_dim": 32,
            # Depth feeds only the present-pointmap stream; the standing default
            # is this server's flex default when set, else the model's own.
            "depth_required": bool(
                self._policy._needs_depth(self._flex_defaults.get("present_pointmap"))
            ),
            "num_inference_steps": int(self._policy.num_inference_steps),
        }

    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        """One inference. Flat obs in → action chunk out.

        Pipeline:
          1. adapter.factors(joint_slice) -> speed_factors (T,)   [if adapter]
          2. smoother.smooth(raw_chunk, factors) -> chunk (T, 32) [if smoother]

        Steps 1 and 2 are no-ops if their respective module is None.
        """
        nested = self._flat_to_nested(obs)
        instruction = str(obs.get("prompt", self._default_prompt) or "")

        flex_kw: Dict[str, bool] = nested.get("flex_kw", {}) or {}
        # Server-side defaults fill any flex key the bridge omitted.
        # Bridge-provided values still win (setdefault, not overwrite).
        for _k, _v in self._flex_defaults.items():
            flex_kw.setdefault(_k, _v)
        # If recording is active, force the joint-denoise regime so the model
        # returns video/dino/pointmap latents. The bridge-provided flex_kw
        # already biases joint_* True (see docs/YAM.md), but recording
        # makes them *required* — overwrite any False explicitly so a stale
        # bridge config can't silently disable the recorder.
        record_this_call = self._recorder is not None
        if record_this_call:
            for _k in ("joint_video", "joint_dino", "joint_pointmap"):
                flex_kw[_k] = True

        t0 = time.perf_counter()
        if record_this_call:
            out = self._policy.infer_action_chunk(
                nested, instruction, return_latents=True, **flex_kw,
            )
            action_chunk = out["action_abs"]
            pred_latents = out
        else:
            action_chunk = self._policy.infer_action_chunk(
                nested, instruction, **flex_kw,
            )
            pred_latents = None
        infer_s = time.perf_counter() - t0
        self._n_calls += 1

        # Stamp the first successful action as the session-timer start.
        # Set lazily here (not in start_session) so the user-visible elapsed
        # excludes the bridge-side handshake / metadata exchange and matches
        # "从model开始给出第一个action开始计时".
        if self._session_start_t is None:
            self._session_start_t = time.perf_counter()

        # Hand the chunk to the recorder. raw mode: CPU copies + enqueue only
        # (disk I/O on the recorder's background thread); live mode: renders
        # + writes the MP4s here, synchronously. Failures log and continue
        # rather than stalling the robot.
        if record_this_call and pred_latents is not None:
            try:
                self._recorder.append_chunk(
                    model=self._policy.model,
                    pred={
                        "video_latents": pred_latents.get("video_latents"),
                        "dino_latents": pred_latents.get("dino_latents"),
                        "pointmap_latents": pred_latents.get("pointmap_latents"),
                    },
                    present={
                        "input_image": pred_latents.get("present_input_image"),
                        "per_cam": pred_latents.get("present_per_cam"),
                        "per_cam_depth": pred_latents.get("present_per_cam_depth"),
                        "camera_intrinsics": pred_latents.get("present_camera_intrinsics"),
                    },
                    meta={
                        "t_wall": time.time(),
                        "infer_s": infer_s,
                        "n_call": self._n_calls,
                        "instruction": instruction,
                    },
                )
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "[server_adapter] recorder.append_chunk failed: %r", e,
                )

        raw_chunk = action_chunk.astype(np.float32)
        T = int(raw_chunk.shape[0])

        t1 = time.perf_counter()
        if self._adapter is not None:
            factors = self._adapter.factors(raw_chunk[:, _ADAPTER_FEATURE_SLICE])
        else:
            factors = np.ones(T, dtype=np.float64)
        if self._smoother is not None:
            chunk = self._smoother.smooth(raw_chunk, speed_factors=factors)
        else:
            chunk = raw_chunk
        post_ms = (time.perf_counter() - t1) * 1000.0

        return {
            "actions": chunk.astype(np.float32),
            "raw_actions": raw_chunk,
            "speed_factors": factors.astype(np.float32),
            "policy_timing": {
                "infer_ms": infer_s * 1000.0,
                "post_ms": post_ms,
            },
        }

    def reset(self) -> None:
        # YamFlexPiPolicy has no episode state to reset; per-prompt T5 cache
        # is invalidated whenever the prompt string changes.
        pass

    # ------------------------------------------------------------------
    # Prediction-recording session lifecycle (called by the WS server
    # wrapper in serve_yam_flexpi.py on connect / disconnect).
    # ------------------------------------------------------------------

    def is_recording(self) -> bool:
        return self._record_base_dir is not None

    def start_session(self, remote_addr: Optional[str] = None) -> int:
        """Begin a new WS-connection-scoped session; returns a session token.

        Always resets the session timer (stamped on first ``infer``). Also
        opens a fresh ``PredictionRecorder`` if recording is enabled. Closes
        any previously open recorder defensively (an abrupt disconnect on the
        prior session shouldn't contaminate this one).

        ``end_session`` takes the token back. It lets a stale handler (a dead
        bridge whose WS task only notices the drop after its keepalive
        expires, possibly ~40 s after a replacement bridge reconnected) end
        only its own session instead of tearing down the replacement's live
        recorder. One recording bridge at a time is the supported topology; a
        second concurrently-active connection still takes over the recorder
        here (defensive close below).
        """
        self._session_token += 1
        self._session_start_t = None
        self._session_remote = remote_addr
        if self._record_base_dir is None:
            return self._session_token
        if self._recorder is not None:
            logger.warning(
                "[server_adapter] start_session called with an open recorder; "
                "closing it first to avoid a leaked writer.",
            )
            self._recorder.close()
            self._recorder = None
        session_dir = make_session_dir(self._record_base_dir, remote_addr=remote_addr)
        self._recorder = PredictionRecorder(
            session_dir,
            fps=self._record_fps,
            frames_per_chunk=self._record_frames_per_chunk,
            pca_warmup_chunks=self._record_pca_warmup_chunks,
            split=self._record_split,
            mode=self._record_mode,
            drop_anchor=self._record_drop_anchor,
            ckpt_path=self._record_ckpt_path,
        )
        logger.info(
            "[server_adapter] recording session started: %s (token=%d)",
            session_dir, self._session_token,
        )
        return self._session_token

    def end_session(self, token: Optional[int] = None) -> None:
        """Finalize the recorder (if any) and echo the session duration.

        Called on WS disconnect (= ctrl+c on the bridge). The duration is
        measured from the first model action of this session to now; if
        the session ended before any action was produced, the timer is
        skipped silently.

        ``token`` (from ``start_session``) guards against the stale-handler
        race: if a newer session has started since this token was issued,
        the call is a late finally-block of a dead connection and leaves the
        newer session's timer and recorder alone. ``token=None`` ends
        whatever session is open.
        """
        if token is not None and token != self._session_token:
            logger.info(
                "[server_adapter] stale end_session (token=%d, current=%d) "
                "ignored — a newer session owns the recorder.",
                token, self._session_token,
            )
            return
        # Echo timing first so the elapsed line lands before the recorder's
        # closure log — easier to spot in the terminal scrollback.
        if self._session_start_t is not None:
            elapsed = time.perf_counter() - self._session_start_t
            remote_tag = f" remote={self._session_remote}" if self._session_remote else ""
            print(
                f"[serve_yam_flexpi] session elapsed={elapsed:.2f} s"
                f"  (first action → disconnect){remote_tag}",
                flush=True,
            )
        self._session_start_t = None
        self._session_remote = None
        if self._recorder is None:
            return
        try:
            self._recorder.close()
        finally:
            self._recorder = None

    @staticmethod
    def _flat_to_nested(obs: Dict[str, Any]) -> Dict[str, Any]:
        """Build the policy's nested obs dict from the wire-flat keys."""
        rgb: Dict[str, np.ndarray] = {}
        depth: Dict[str, np.ndarray] = {}
        K_per_cam: Dict[str, np.ndarray] = {}
        for cam in _CAM_ORDER:
            img_key = f"observation/image_{cam}"
            if img_key not in obs:
                raise ValueError(f"server_adapter: missing flat key {img_key!r} in obs")
            rgb[cam] = np.asarray(obs[img_key])

            depth_key = f"observation/depth_{cam}"
            if depth_key in obs:
                depth[cam] = np.asarray(obs[depth_key])

            K_key = f"observation/intrinsics_{cam}"
            if K_key in obs:
                fxfycxcy = np.asarray(obs[K_key], dtype=np.float32).reshape(-1)
                if fxfycxcy.size != 4:
                    raise ValueError(
                        f"server_adapter: {K_key} must be 4-vec [fx,fy,cx,cy]; "
                        f"got shape {fxfycxcy.shape}"
                    )
                fx, fy, cx, cy = fxfycxcy.tolist()
                K_per_cam[cam] = np.array(
                    [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
                    dtype=np.float32,
                )

        # Depth is all-or-nothing on the policy side ([deploy_policy.py:543-548]).
        if depth and len(depth) != len(_CAM_ORDER):
            missing = [c for c in _CAM_ORDER if c not in depth]
            raise ValueError(
                f"server_adapter: partial depth supplied; missing {missing}. "
                "Either send depth for all 3 cams or none."
            )
        depth_dict: Optional[Dict[str, np.ndarray]] = depth if depth else None

        # Intrinsics: stack in _CAM_ORDER if all 3 present (the policy restacks
        # the rows into layout slot order); else fall back to policy's _cached_K.
        K_stack: Optional[np.ndarray] = None
        if K_per_cam:
            if len(K_per_cam) != len(_CAM_ORDER):
                missing = [c for c in _CAM_ORDER if c not in K_per_cam]
                raise ValueError(
                    f"server_adapter: partial intrinsics supplied; missing {missing}. "
                    "Either send K for all 3 cams or none."
                )
            K_stack = np.stack([K_per_cam[c] for c in _CAM_ORDER], axis=0).astype(np.float32)

        state_key = "observation/state"
        if state_key not in obs:
            raise ValueError(f"server_adapter: missing {state_key!r}")
        state_32 = np.asarray(obs[state_key], dtype=np.float32)
        if state_32.shape != (32,):
            raise ValueError(f"server_adapter: state shape {state_32.shape} != (32,)")

        # Flex runtime overrides (optional). Wire-encoded as np.bool_ scalars
        # under observation/present_* and observation/joint_*. Absent → None →
        # the policy/model use trained defaults.
        flex_kw: Dict[str, Optional[bool]] = {}
        for _name in (
            "present_video", "present_dino", "present_pointmap",
            "joint_video", "joint_dino", "joint_pointmap",
        ):
            _wire_key = f"observation/{_name}"
            if _wire_key in obs:
                flex_kw[_name] = bool(obs[_wire_key])

        return {
            "rgb": rgb,
            "depth": depth_dict,
            "state_32": state_32,
            "intrinsics": K_stack,
            "flex_kw": flex_kw,
        }
