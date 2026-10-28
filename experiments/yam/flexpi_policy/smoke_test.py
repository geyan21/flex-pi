"""Offline smoke test for the YAM FlexPi deployment policy.

Loads the trained checkpoint, builds a single observation from one of three
offline sources (recorded LeRobot episode, debug .npz, or all-zero dummy),
runs ONE forward pass, and prints rich diagnostics: tensor shapes, action
ranges, EEF/grip/joint slot breakdown, NaN/Inf checks, sanity warnings.

This entrypoint deliberately does NOT connect to the real robot, ZED cameras,
or raiden. It is a standalone forward-pass validator — the bridge from this
to a real YAM runtime is described in the engineering report.

Examples::

    # Recorded source: real RGB + depth + 32D state from a LeRobot dataset.
    python -m experiments.yam.flexpi_policy.smoke_test \\
        --ckpt /path/to/runs/.../checkpoints/.../step_010000.pt \\
        --data-dir data/yam_test_2ep_360x640 \\
        --episode-idx 0 --frame-idx 0 \\
        --prompt "Place the fork on the pink plate and put the spoon into the utensil container."

    # Debug npz (you assembled rgb/depth/state/prompt offline).
    python -m experiments.yam.flexpi_policy.smoke_test \\
        --ckpt /path/to/.pt --npz /tmp/yam_obs_debug.npz

    # All-zero dummy. Useful to validate the model loads and forwards.
    python -m experiments.yam.flexpi_policy.smoke_test \\
        --ckpt /path/to/.pt --obs-source dummy \\
        --intrinsics-json data/yam_test_2ep_360x640/meta/camera_intrinsics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Path bootstrap (mirrors deploy_policy.py).
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from experiments.yam.flexpi_policy.deploy_policy import (  # noqa: E402
    YamFlexPiPolicy,
    build_policy_from_checkpoint,
    _CAM_ORDER,
)

# 32D layout slot indices (matches yam_eef.STATE_LAYOUT).
_LAYOUT = {
    "left_pos":    slice(0, 3),
    "left_rot6d":  slice(3, 9),
    "right_pos":   slice(9, 12),
    "right_rot6d": slice(12, 18),
    "left_grip":   slice(18, 19),
    "right_grip":  slice(19, 20),
    "left_joint":  slice(20, 26),
    "right_joint": slice(26, 32),
}


# ---------------------------------------------------------------------------
# Observation sources
# ---------------------------------------------------------------------------


def build_obs_dummy() -> Dict[str, Any]:
    """All-zero observation at native YAM resolution (360×640 head + wrists).

    The policy resizes RGB *and* depth to the trained per-cam grid internally,
    so the obs stays on the camera-native grid. K comes from the policy's
    ``--intrinsics-json`` cache at that same trained grid: the pointmap encoder
    unprojects at the depth's resolution with the K it is handed and never
    rescales it, so the pair must land on one grid, and the policy's depth
    regrid is what keeps them paired.
    """
    H_native, W_native = 360, 640  # matches yam_test_2ep_360x640
    rgb = {cam: np.zeros((H_native, W_native, 3), dtype=np.uint8) for cam in _CAM_ORDER}
    depth = {cam: np.zeros((H_native, W_native), dtype=np.uint16) for cam in _CAM_ORDER}
    state = np.zeros(32, dtype=np.float32)
    # Zero rot6d = degenerate rotation matrix (Gram-Schmidt produces NaN). Seed
    # rot6d slots with identity rows so Yam32DRelativeAction.backward produces
    # finite numbers even on the dummy path.
    # Identity R[:2] flattened = [1,0,0, 0,1,0].
    state[_LAYOUT["left_rot6d"]]  = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)
    state[_LAYOUT["right_rot6d"]] = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)

    return {
        "rgb": rgb,
        "depth": depth,
        "state_32": state,
        "intrinsics": None,
    }


def build_obs_npz(npz_path: Path) -> Dict[str, Any]:
    """Load observation from a debug .npz.

    Expected keys (mirrors what the openpi_bridge dumps under
    ``~/openpi_obs_debug/`` plus depth/intrinsics): ``rgb_<cam>``,
    ``depth_<cam>``, ``intrinsics_<cam>`` (3×3), ``state_32``.
    Prompt is read separately from the CLI.
    """
    data = np.load(npz_path, allow_pickle=False)
    rgb = {cam: data[f"rgb_{cam}"] for cam in _CAM_ORDER}
    depth = {cam: data[f"depth_{cam}"] for cam in _CAM_ORDER}
    intrinsics = np.stack([data[f"intrinsics_{cam}"] for cam in _CAM_ORDER], axis=0).astype(np.float32)
    state = np.asarray(data["state_32"], dtype=np.float32)
    return {"rgb": rgb, "depth": depth, "state_32": state, "intrinsics": intrinsics}


def build_obs_recorded(
    data_dir: Path,
    episode_idx: int,
    frame_idx: int,
    fps_hint: Optional[int] = None,
) -> Tuple[Dict[str, Any], str]:
    """Read one frame from a LeRobot-format YAM dataset.

    Pulls:
      - RGB cam_high / cam_left_wrist / cam_right_wrist from MP4 (decord)
      - Depth from FFV1 MKV (FlexPi's utils.depth_codec.decode_depth_mkv_at_timestamps)
      - 32D state from parquet ``observation.state[frame_idx]``
      - Camera intrinsics from ``meta/camera_intrinsics.json``
      - Prompt task from ``meta/tasks.jsonl`` keyed by parquet's ``task_index``.

    Returns ``(observation, instruction)``.
    """
    import av  # lazy: only needed for recorded mode

    def _decode_mp4_frame(mp4_path: Path, frame_idx: int) -> np.ndarray:
        """Decode a single RGB frame from an MP4 via PyAV.

        We decode forward to ``frame_idx`` instead of seeking — keyframe-only
        seeks in MP4 can land past the target. For typical YAM episodes
        (<1000 frames) this is fast enough at frame 0 / early frames; for
        late frames it stays under a second.
        """
        container = av.open(str(mp4_path))
        try:
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            i = 0
            for frame in container.decode(stream):
                if i == frame_idx:
                    rgb = frame.to_ndarray(format="rgb24")
                    return np.ascontiguousarray(rgb)
                i += 1
            raise IndexError(f"frame_idx={frame_idx} >= MP4 length {i} for {mp4_path}")
        finally:
            container.close()

    data_dir = Path(data_dir).expanduser().resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"--data-dir does not exist: {data_dir}")

    # ---- parquet → state, task_index, frame_count ----
    import pyarrow.parquet as pq
    parquet_path = data_dir / "data" / "chunk-000" / f"episode_{episode_idx:06d}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Episode parquet not found: {parquet_path}. "
            f"Check --episode-idx (got {episode_idx})."
        )
    df = pq.read_table(parquet_path).to_pandas()
    n_frames = len(df)
    if frame_idx < 0 or frame_idx >= n_frames:
        raise IndexError(f"frame_idx={frame_idx} out of range [0, {n_frames})")
    state_32 = np.asarray(df["observation.state"].iloc[frame_idx], dtype=np.float32)
    if state_32.shape != (32,):
        raise ValueError(f"Expected 32D observation.state; got shape {state_32.shape}")
    task_index = int(df["task_index"].iloc[frame_idx])
    timestamp_sec = float(df["timestamp"].iloc[frame_idx])

    # ---- tasks.jsonl → instruction ----
    tasks_path = data_dir / "meta" / "tasks.jsonl"
    instruction = "manipulate the objects on the table"  # fallback
    if tasks_path.exists():
        with open(tasks_path, "r") as f:
            for line in f:
                row = json.loads(line)
                if int(row["task_index"]) == task_index:
                    instruction = row["task"]
                    break

    # ---- info.json → fps for depth ----
    info_path = data_dir / "meta" / "info.json"
    fps = fps_hint
    if fps is None and info_path.exists():
        with open(info_path, "r") as f:
            info = json.load(f)
        fps = int(info.get("fps") or 50)
    if fps is None:
        fps = 50  # YAM datasets default

    # ---- RGB decode (PyAV) ----
    rgb: Dict[str, np.ndarray] = {}
    for cam in _CAM_ORDER:
        mp4_path = data_dir / "videos" / "chunk-000" / f"observation.images.{cam}" / f"episode_{episode_idx:06d}.mp4"
        if not mp4_path.exists():
            raise FileNotFoundError(
                f"Missing RGB MP4 for cam {cam!r}: {mp4_path}. "
                f"Datasets must be in canonical RoboTwin-rename layout."
            )
        rgb[cam] = _decode_mp4_frame(mp4_path, frame_idx)

    # ---- Depth decode (FFV1 MKV) ----
    from flexpi.datasets.lerobot.utils.depth_codec import decode_depth_mkv_at_timestamps
    depth: Dict[str, np.ndarray] = {}
    for cam in _CAM_ORDER:
        mkv_path = data_dir / "videos" / "chunk-000" / f"observation.depth_ffv1.{cam}" / f"episode_{episode_idx:06d}.mkv"
        if not mkv_path.exists():
            raise FileNotFoundError(
                f"Missing depth MKV for cam {cam!r}: {mkv_path}."
            )
        # decode_depth_mkv_at_timestamps returns int32 mm of shape [T, H, W]; we
        # cast back to uint16 to match the dataset's _build_per_cam_depth contract.
        d_int32 = decode_depth_mkv_at_timestamps(mkv_path, [timestamp_sec], fps=fps).numpy()
        depth[cam] = d_int32[0].clip(0, 65535).astype(np.uint16)

    # ---- Intrinsics ----
    # K stays with the policy: build_policy_from_checkpoint loads the dataset's
    # meta/camera_intrinsics.json at the checkpoint's depth grid, and the policy
    # resamples the storage-grid depth above onto that same grid
    # (linspace-endpoint nearest, like training) so the pair stays paired.
    obs = {"rgb": rgb, "depth": depth, "state_32": state_32, "intrinsics": None}
    return obs, instruction


def load_gt_action_chunk(
    data_dir: Path,
    episode_idx: int,
    frame_idx: int,
    action_horizon: int,
) -> np.ndarray:
    """Read the ground-truth absolute 32D action chunk from the parquet.

    Returns ``[T_gt, 32]`` where ``T_gt = min(action_horizon, n_frames - frame_idx)``.
    T_gt may be smaller than action_horizon near the end of an episode; callers
    must align comparison length accordingly. The parquet stores actions in the
    same absolute 32D layout the deploy policy emits after
    Yam32DRelativeAction.backward, so a direct slot-wise comparison is valid.
    """
    import pyarrow.parquet as pq
    parquet_path = data_dir / "data" / "chunk-000" / f"episode_{episode_idx:06d}.parquet"
    if not parquet_path.exists():
        raise FileNotFoundError(f"Episode parquet not found: {parquet_path}")
    df = pq.read_table(parquet_path).to_pandas()
    n = len(df)
    end = min(frame_idx + action_horizon, n)
    if end <= frame_idx:
        return np.zeros((0, 32), dtype=np.float32)
    return np.stack([
        np.asarray(df["action"].iloc[i], dtype=np.float32) for i in range(frame_idx, end)
    ], axis=0)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _summarize_obs(obs: Dict[str, Any], instruction: str) -> None:
    print("\n[smoke] === Observation summary ===")
    print(f"  instruction: {instruction!r}")
    for cam in _CAM_ORDER:
        rgb = obs["rgb"][cam]
        print(
            f"  rgb[{cam}]   shape={rgb.shape} dtype={rgb.dtype} "
            f"range=[{rgb.min()}, {rgb.max()}]"
        )
    if obs.get("depth") is not None:
        for cam in _CAM_ORDER:
            d = obs["depth"][cam]
            nz = (d != 0).sum()
            print(
                f"  depth[{cam}] shape={d.shape} dtype={d.dtype} "
                f"range=[{d.min()}, {d.max()}] nonzero={nz}/{d.size}"
            )
    K = obs.get("intrinsics")
    if K is not None:
        print(f"  intrinsics shape={K.shape} (3 cams in order {_CAM_ORDER})")
        for i, cam in enumerate(_CAM_ORDER):
            print(f"    K[{cam}] fx={K[i,0,0]:.2f} fy={K[i,1,1]:.2f} cx={K[i,0,2]:.2f} cy={K[i,1,2]:.2f}")
    s = obs["state_32"]
    print(f"  state_32   shape={s.shape} dtype={s.dtype}")
    for name, sl in _LAYOUT.items():
        v = s[sl]
        if v.size <= 6:
            arr_s = np.array2string(v, precision=3, suppress_small=True)
        else:
            arr_s = f"min={v.min():.3f} max={v.max():.3f}"
        print(f"    [{sl.start:2d}:{sl.stop:2d}] {name:<12} = {arr_s}")


def _summarize_action(action_abs: np.ndarray) -> None:
    print("\n[smoke] === Action summary (absolute, 32D yam_eef.STATE_LAYOUT) ===")
    n_nan = int(np.isnan(action_abs).sum())
    n_inf = int(np.isinf(action_abs).sum())
    print(f"  shape={action_abs.shape}  dtype={action_abs.dtype}  NaN={n_nan}  Inf={n_inf}")
    if n_nan or n_inf:
        print("  WARNING: NaN/Inf present in absolute action — model output likely degenerate.")
        return

    # Slot-by-slot summary, T=0 (first action) vs T=-1 (last action).
    print(f"  per-slot ranges over chunk (T=0..{action_abs.shape[0]-1}):")
    for name, sl in _LAYOUT.items():
        block = action_abs[:, sl]
        print(
            f"    [{sl.start:2d}:{sl.stop:2d}] {name:<12} "
            f"min={block.min():+.3f} max={block.max():+.3f} "
            f"std={block.std():.3f} "
            f"first={np.array2string(block[0], precision=3, suppress_small=True)} "
        )

    # Joint-block range sanity: absolute joint targets for YAM are typically
    # bounded in ~[-3.14, 3.14] rad. Anything wildly outside hints at a
    # broken relative→absolute inversion.
    joint = action_abs[:, _LAYOUT["left_joint"].start:_LAYOUT["right_joint"].stop]
    j_max_abs = float(np.abs(joint).max())
    if j_max_abs > 6.5:
        print(
            f"  WARNING: |joint target| max = {j_max_abs:.2f} rad — exceeds typical YAM range. "
            "Verify state anchor convention and Yam32DRelativeAction.backward inversion."
        )

    # Gripper range sanity: YAM grippers usually live in [0, 1] (open/closed).
    grip = action_abs[:, _LAYOUT["left_grip"].start:_LAYOUT["right_grip"].stop]
    if float(grip.min()) < -0.5 or float(grip.max()) > 1.5:
        print(
            f"  WARNING: gripper range outside [-0.5, 1.5] (got {grip.min():.2f}..{grip.max():.2f}). "
            "Possible sign / normalization mismatch."
        )


def _print_assumptions(args: argparse.Namespace) -> None:
    print("\n[smoke] === Active assumptions (explicit) ===")
    print(f"  - YAM 32D state/action layout matches yam_eef.STATE_LAYOUT (slot-for-slot identical to openpi eef32).")
    print(f"  - Training transform = Yam32DRelativeAction(anchor='first'); deploy inverts it using observation['state_32'] as anchor.")
    print(f"  - Camera order: {_CAM_ORDER} (canonical RoboTwin-rename layout).")
    print(f"  - Depth is uint16 millimetres on a per-cam grid (head 256×320 / wrists 224×224 target).")
    print(f"  - input_image composite is built from per-cam RGB on the checkpoint's composite layout (384×320 for the YAM checkpoints).")
    print(f"  - Prompt template: DEFAULT_PROMPT.format(task=...). Text encoder runs live (no cache lookup).")
    print(f"  - obs-source = {args.obs_source!r}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="YAM FlexPi smoke test: load checkpoint, run one inference pass, print diagnostics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt", required=True, help="Path to the trained checkpoint (.pt or state/step_NNNNNN/).")
    p.add_argument("--stats", default=None, help="Path to dataset_stats.json. If omitted, walked from --ckpt.")
    p.add_argument(
        "--intrinsics-json", default=None,
        help="Path to meta/camera_intrinsics.json, loaded by the policy at the "
             "checkpoint's depth grid. Required for the dummy obs source; "
             "recorded defaults to the dataset's own; npz carries its own array.",
    )
    p.add_argument(
        "--obs-source", choices=["recorded", "npz", "dummy"], default="recorded",
        help="Where the observation comes from.",
    )
    # recorded
    p.add_argument("--data-dir", default=None, help="LeRobot dataset root (recorded mode).")
    p.add_argument("--episode-idx", type=int, default=0, help="Episode index (recorded mode).")
    p.add_argument("--frame-idx", type=int, default=0, help="Frame index inside the episode (recorded mode).")
    p.add_argument("--fps", type=int, default=None, help="FPS override for depth decode (recorded mode).")
    # npz
    p.add_argument("--npz", default=None, help="Path to a debug .npz observation.")
    # prompt
    p.add_argument("--prompt", default="", help="Task instruction. For recorded mode, auto-read from tasks.jsonl when empty.")
    # inference knobs (defaults track sim_robotwin.yaml EVALUATION block)
    p.add_argument("--device", default="cuda")
    p.add_argument("--mixed-precision", default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--num-inference-steps", type=int, default=10)
    p.add_argument("--action-horizon", type=int, default=None,
                   help="Override the action horizon (default: num_frames-1 from trained config).")
    p.add_argument("--sigma-shift", type=float, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--text-cfg-scale", type=float, default=1.0)
    p.add_argument("--negative-prompt", default="")
    p.add_argument("--verbose", action="store_true", help="Print kwarg breakdown and timing.")
    p.add_argument("--save-action-npz", default=None,
                   help="If set, write the absolute action chunk to this path as .npz.")
    p.add_argument(
        "--check-against-gt", action="store_true",
        help="Recorded mode only: pull parquet['action'][frame_idx : frame_idx + action_horizon] "
             "and print MAE between the predicted absolute action chunk and ground truth, "
             "broken out by EEF pose / gripper / joint head / full 32D.",
    )
    # Perf knobs.
    p.add_argument("--offload-text-encoder", action="store_true",
                   help="Keep T5 on CPU to save ~10 GB VRAM.")
    p.add_argument("--torch-compile", action="store_true",
                   help="Compile the denoise step + capture CUDA graph (10-30 s warmup, "
                        "then ~3x speedup for subsequent calls).")
    p.add_argument("--torch-compile-mode", default="reduce-overhead",
                   choices=["default", "reduce-overhead", "max-autotune"])
    p.add_argument("--num-passes", type=int, default=1,
                   help="Run inference N times (after warmup) and report per-pass latency. "
                        "Useful for measuring torch.compile speedup.")
    return p


def main() -> int:
    args = _build_parser().parse_args()

    # Validate args by mode.
    if args.obs_source == "recorded" and args.data_dir is None:
        print("ERROR: --obs-source recorded requires --data-dir.", file=sys.stderr)
        return 2
    if args.obs_source == "npz" and args.npz is None:
        print("ERROR: --obs-source npz requires --npz.", file=sys.stderr)
        return 2

    # Pre-flight checkpoint existence (clearer than the policy's later error).
    ckpt = Path(args.ckpt).expanduser().resolve()
    if not ckpt.exists():
        print(f"ERROR: --ckpt not found: {ckpt}", file=sys.stderr)
        return 2

    print(f"[smoke] checkpoint: {ckpt}")
    _print_assumptions(args)

    # ---- Build observation FIRST (cheap, fails fast on shape/key errors) ----
    instruction = args.prompt
    if args.obs_source == "recorded":
        obs, recorded_prompt = build_obs_recorded(
            data_dir=Path(args.data_dir),
            episode_idx=args.episode_idx,
            frame_idx=args.frame_idx,
            fps_hint=args.fps,
        )
        if not instruction:
            instruction = recorded_prompt
    elif args.obs_source == "npz":
        obs = build_obs_npz(Path(args.npz))
        if not instruction:
            instruction = "manipulate the objects on the table"
    else:  # dummy
        obs = build_obs_dummy()
        if not instruction:
            instruction = "manipulate the objects on the table"

    _summarize_obs(obs, instruction)

    # ---- Build policy + load checkpoint ----
    print("\n[smoke] === Loading policy ===")
    if args.intrinsics_json:
        intrinsics_json = Path(args.intrinsics_json)
    elif args.obs_source == "recorded":
        # The dataset's own K; the policy loads it at the checkpoint's depth grid.
        intrinsics_json = Path(args.data_dir) / "meta" / "camera_intrinsics.json"
    else:
        # npz obs carry their own intrinsics array; the constructor cache is optional.
        intrinsics_json = None
    policy: YamFlexPiPolicy = build_policy_from_checkpoint(
        checkpoint_path=str(ckpt),
        dataset_stats_path=args.stats,
        intrinsics_json=str(intrinsics_json) if intrinsics_json else None,
        device=args.device,
        mixed_precision=args.mixed_precision,
        action_horizon=args.action_horizon,
        num_inference_steps=args.num_inference_steps,
        sigma_shift=args.sigma_shift,
        seed=args.seed,
        text_cfg_scale=args.text_cfg_scale,
        negative_prompt=args.negative_prompt,
        offload_text_encoder=args.offload_text_encoder,
        torch_compile=args.torch_compile,
        torch_compile_mode=args.torch_compile_mode,
    )

    # ---- Pre-inference sanity checks ----
    print("\n[smoke] === Pre-inference sanity ===")
    # State NaN/Inf
    s = obs["state_32"]
    if not np.isfinite(s).all():
        print(f"  WARNING: state_32 contains non-finite values "
              f"(NaN={int(np.isnan(s).sum())} Inf={int(np.isinf(s).sum())}). "
              "Yam32DRelativeAction.backward will propagate them.")
    # rot6d degeneracy
    for name in ("left_rot6d", "right_rot6d"):
        v = s[_LAYOUT[name]]
        a1 = v[0:3]; a2 = v[3:6]
        n1 = np.linalg.norm(a1); n2 = np.linalg.norm(a2)
        if n1 < 1e-5 or n2 < 1e-5:
            print(f"  WARNING: state_32[{name}] has near-zero row(s) "
                  f"(|a1|={n1:.2e} |a2|={n2:.2e}); Gram-Schmidt at decode will diverge.")
    # Action_state_transforms must include Yam32DRelativeAction for the 32D-rel model.
    transform_names = [type(t).__name__ for t in (policy.processor.action_state_transforms or [])]
    if "Yam32DRelativeAction" not in transform_names:
        print(
            f"  WARNING: processor.action_state_transforms does NOT include "
            f"Yam32DRelativeAction (found: {transform_names}). For the 32D-rel "
            "model this means the model output will be passed through as-is. "
            "If you trained with 32D-rel, the trained config didn't preserve the "
            "transform — verify <run_dir>/config.yaml."
        )
    else:
        print(f"  action_state_transforms = {transform_names}")
    # processor shape_meta sanity
    sm = policy.processor.shape_meta
    if int(sm["action"][0]["shape"]) != 32 or int(sm["state"][0]["shape"]) != 32:
        print(
            f"  WARNING: shape_meta declares action_dim={sm['action'][0]['shape']} "
            f"state_dim={sm['state'][0]['shape']} — expected 32 for YAM 32D-rel."
        )

    # ---- Forward pass ----
    print("\n[smoke] === Running one inference pass ===")
    import time as _time
    pass_times = []
    action_abs = policy.infer_action_chunk(obs, instruction, verbose=args.verbose)
    if args.num_passes > 1:
        # Warm up timing — first call paid the compile / KV-prefill cost above
        # (and policy._warmup paid torch.compile if --torch-compile). Now
        # measure steady-state latency across N additional passes.
        for i in range(args.num_passes - 1):
            t0 = _time.perf_counter()
            _ = policy.infer_action_chunk(obs, instruction, verbose=False)
            pass_times.append(_time.perf_counter() - t0)
        ms = [f"{t*1000:.1f}" for t in pass_times]
        avg_ms = 1000.0 * sum(pass_times) / len(pass_times)
        print(
            f"\n[smoke] === Latency over {args.num_passes - 1} additional passes ===\n"
            f"  per-pass (ms): {ms}\n"
            f"  avg: {avg_ms:.1f} ms"
        )
    _summarize_action(action_abs)

    # ---- GT comparison (recorded mode only) ----
    if args.check_against_gt:
        if args.obs_source != "recorded":
            print(
                "\n[smoke] WARN: --check-against-gt requires --obs-source recorded; "
                "skipping GT comparison."
            )
        else:
            gt_chunk = load_gt_action_chunk(
                data_dir=Path(args.data_dir),
                episode_idx=args.episode_idx,
                frame_idx=args.frame_idx,
                action_horizon=policy.action_horizon,
            )
            T_compare = min(action_abs.shape[0], gt_chunk.shape[0])
            print(f"\n[smoke] === GT comparison ===")
            print(f"  predicted: shape={action_abs.shape}")
            print(f"  gt:        shape={gt_chunk.shape} (parquet action[{args.frame_idx}:{args.frame_idx + gt_chunk.shape[0]}])")
            if T_compare == 0:
                print(
                    f"  WARN: 0 GT frames available — frame_idx={args.frame_idx} is "
                    "at or past the end of the episode."
                )
            else:
                pred = action_abs[:T_compare]
                gt = gt_chunk[:T_compare]

                def _mae_section(slc: slice, name: str) -> float:
                    block = np.abs(pred[:, slc] - gt[:, slc])
                    return float(block.mean())

                # Sections mirror yam_eef.STATE_LAYOUT chunks.
                mae_eef = _mae_section(slice(0, 18), "EEF pose")
                mae_grip = _mae_section(slice(18, 20), "gripper")
                mae_joint = _mae_section(slice(20, 32), "joint head")
                mae_all = _mae_section(slice(0, 32), "full 32D")
                print(f"  T_compare={T_compare}")
                print(f"  MAE EEF pose  (slot  0:18) : {mae_eef:.4f}")
                print(f"  MAE gripper   (slot 18:20) : {mae_grip:.4f}")
                print(f"  MAE joint head(slot 20:32) : {mae_joint:.4f}")
                print(f"  MAE full 32D  (slot  0:32) : {mae_all:.4f}")
                print(
                    "  Note: parquet 'action' stores absolute 32D actions per timestep. "
                    "The predicted chunk and GT chunk are temporally aligned to the "
                    "sample-start frame_idx. Sub-frame indexing drift (e.g. action target = "
                    "next-step state per Yam32DRelativeAction) may inflate first-frame MAE."
                )

    if args.save_action_npz is not None:
        out_path = Path(args.save_action_npz).expanduser().resolve()
        np.savez(out_path, action_abs=action_abs, state_32=obs["state_32"], instruction=instruction)
        print(f"\n[smoke] Wrote {out_path}")

    print("\n[smoke] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
