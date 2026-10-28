"""Convert raw YAM bimanual data into a LeRobot v2.1 dataset (eef32, openpi-aligned).

Output is byte-compatible with what FlexPi's RobotVideoDataset
expects via ``configs/data/yam.yaml``, AND
numerically equivalent (up to float32 noise) to a dataset built by
``yam_openpi/yam_dataset_builder/convert_yam_data_to_lerobot.py --state-layout eef32``:

  - 32D ``observation.state`` and ``action`` in yam_openpi's eef32 layout
    (matches ``src/flexpi/datasets/lerobot/utils/yam_eef.STATE_LAYOUT``).
    EEF blocks read from rd-convert's pre-computed ``lowdim.action[]`` 26D
    field (NOT recomputed via FK here — see source_reader.py for rationale).
  - 3 RGB cameras under canonical names ``cam_high``, ``cam_left_wrist``,
    ``cam_right_wrist``. Builder-stage resize done by ffmpeg with
    ``-vf scale=W:H:flags=bilinear`` in the same encode pipe (single resize
    kernel, matches openpi).
  - 3 depth videos under ``observation.depth_ffv1.<cam>`` (default codec)
    using FlexPi's ``dtype="depth_video"`` protocol. Builder-stage depth
    resize uses linspace-endpoint nearest-NN (openpi convention).
  - ``meta/camera_intrinsics.json`` rescaled with the "endpoint" convention
    ``s = (dst-1)/(src-1)`` and averaged across all episodes (matches openpi).
  - ``meta/episode_intrinsics.jsonl`` — the same K's before averaging, one
    record per episode. The loader prefers it, which keeps metric depth correct
    when a build spans rigs with different optics; the averaged file above
    stays for tools that predate it.

Example (single source):
    python scripts/yam_dataset_builder/convert.py \\
        --raw-dir /path/to/yam_raw/my_task \\
        --out-dir ./data/yam/my_task \\
        --task-name my_task \\
        --episodes 0 1 \\
        --workers 1 --overwrite

Example (merge multiple sources into one LeRobot dataset):
    python scripts/yam_dataset_builder/convert.py \\
        --raw-dir /path/to/yam_raw/my_task_1 \\
                  /path/to/yam_raw/my_task_2 \\
                  /path/to/yam_raw/my_task_3 \\
        --out-dir ./data/yam/my_task_merged \\
        --task-name my_task_merged \\
        --workers 2 --overwrite

Multi-source merge semantics:
  - Episodes are concatenated in the order the source dirs are listed and
    renumbered globally to 0..N_total-1.
  - All sources must share the same fps, camera layout, and (within drift
    tolerance) intrinsics. Task prompts from each source's metadata.json are
    deduplicated into a single tasks.jsonl.
  - --episodes is rejected when merging (ambiguous: episode 0 exists in
    every source); pre-select by listing only the source dirs you want.
  - --out-dir is required when merging (no sensible default from N names).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# Vendored LeRobot (in-tree) + the sibling source_reader. Both resolve via this
# file's location so the script works regardless of cwd or which checkout it
# lives under, and can be run directly by path (no package install, no `-m`).
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parents[1]          # scripts/yam_dataset_builder -> flex-pi
_FLEXPI_LEROBOT_ROOT = _PROJECT_ROOT / "src" / "flexpi" / "datasets" / "lerobot"
for _p in (_FLEXPI_LEROBOT_ROOT, _THIS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
from lerobot.datasets.video_utils import (  # noqa: E402
    DEPTH_VIDEO_EXT,
    encode_depth_video_frames,
)

from source_reader import (  # noqa: E402
    SourceLayout,
    discover_episodes,
    load_episode,
    validate_state32_first_episode,
)


# 32D motor names (yam_eef.STATE_LAYOUT). Names appear only in info.json
# features metadata — they don't drive training.
DEFAULT_MOTOR_NAMES_EEF32 = (
    [f"left_pos_{c}" for c in "xyz"]
    + [f"left_rot6d_{i}" for i in range(6)]
    + [f"right_pos_{c}" for c in "xyz"]
    + [f"right_rot6d_{i}" for i in range(6)]
    + ["left_gripper", "right_gripper"]
    + [f"left_joint_{i}" for i in range(6)]
    + [f"right_joint_{i}" for i in range(6)]
)
assert len(DEFAULT_MOTOR_NAMES_EEF32) == 32

CHUNK_SIZE = 1000
# RGB codec defaults match yam_openpi/yam_dataset_builder/convert_yam_data_to_lerobot.py
# (libx264 + CRF 22 + -g 1 = all I-frames). All-I-frame encoding grows file size
# ~5-8x but is the load-bearing knob for training-time random access: per-sample
# RGB fetch drops from ~1 s (with default GOP) to ~50-100 ms because the decoder
# never has to decode prior P/B frames to seek to a random index. AV1
# encoders (libsvtav1, libaom-av1) decode noticeably slower than h264 even with
# -g 1 — keep h264 unless you have a specific size-sensitive use case.
DEFAULT_RGB_CODEC = "libx264"
DEFAULT_RGB_CRF = 22
DEFAULT_RGB_GOP = 1
RGB_PIX_FMT = "yuv420p"

# Map encoder flag -> canonical codec name as reported by pyav.
_CODEC_CANONICAL = {
    "libsvtav1": "av1",
    "libaom-av1": "av1",
    "libx264": "h264",
    "libx264rgb": "h264",
    "libx265": "hevc",
    "ffv1": "ffv1",
}


# --- Helpers ----------------------------------------------------------------

def _resize_depth_nearest_endpoint(
    depth_uint16: np.ndarray, target_h: int, target_w: int
) -> np.ndarray:
    """Nearest-neighbor uint16 depth resize using openpi's linspace-endpoint convention.

    Pixel mapping: dst index ``d`` in ``[0, dst-1]`` maps to src index
    ``round(d * (src-1) / (dst-1))``, so dst endpoints (0 and dst-1) hit src
    endpoints exactly. Bit-identical to
    ``openpi.policies.depth_codec.resize_depth_nearest``. Pairs with the K
    rescale ``sx = (dst_w-1) / (src_w-1)`` applied by ``convert()`` so the
    pixel grid and K stay endpoint-aligned end-to-end.
    """
    if depth_uint16.dtype != np.uint16:
        raise TypeError(f"expected uint16, got {depth_uint16.dtype}")
    src_h, src_w = depth_uint16.shape[-2:]
    if (src_h, src_w) == (target_h, target_w):
        return depth_uint16
    ys = np.linspace(0, src_h - 1, target_h).round().astype(np.int64)
    xs = np.linspace(0, src_w - 1, target_w).round().astype(np.int64)
    return depth_uint16[..., ys[:, None], xs[None, :]]


def _ffmpeg_encode_rgb(
    frames: np.ndarray,
    out_path: Path,
    fps: int,
    rgb_codec: str,
    crf: int = DEFAULT_RGB_CRF,
    gop: int = DEFAULT_RGB_GOP,
    target_h: Optional[int] = None,
    target_w: Optional[int] = None,
) -> None:
    """Encode (N, H_src, W_src, 3) uint8 RGB frames via ffmpeg stdin.

    When ``target_h``/``target_w`` are set and differ from the source shape,
    ffmpeg rescales inside the same pipe with
    ``-vf scale={W}:{H}:flags=bilinear`` (matches yam_openpi exactly: single
    resize kernel, same kernel family as the torchvision bilinear+antialias
    resize the dataloader/deploy bridge applies downstream).

    Codec defaults: libx264 + CRF 22 + -g 1 (all keyframes) for fast
    random-access decoding at training time.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found in PATH")
    assert frames.dtype == np.uint8 and frames.ndim == 4 and frames.shape[-1] == 3
    N, H_src, W_src, _ = frames.shape
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{W_src}x{H_src}", "-framerate", str(fps),
        "-i", "-",
    ]
    if (
        target_h is not None
        and target_w is not None
        and (int(target_h), int(target_w)) != (H_src, W_src)
    ):
        # setsar=1: scale preserves the DISPLAY aspect ratio by writing a
        # compensating SAR, which would make players render the file at a size
        # that disagrees with its coded resolution and with the depth stream.
        # A no-op when the resize keeps the aspect ratio; correct when it does not.
        cmd += ["-vf", f"scale={int(target_w)}:{int(target_h)}:flags=bilinear,setsar=1"]
    cmd += [
        "-c:v", rgb_codec, "-pix_fmt", RGB_PIX_FMT,
        "-g", str(gop), "-crf", str(crf),
        str(out_path),
    ]
    proc = subprocess.run(cmd, input=np.ascontiguousarray(frames).tobytes(),
                          capture_output=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"ffmpeg RGB encode failed (rc={proc.returncode}):\n"
            f"  cmd: {' '.join(cmd)}\n"
            f"  stderr: {proc.stderr.decode(errors='replace')[:4000]}"
        )


# --- Per-episode worker (callable from ProcessPoolExecutor) -----------------

@dataclass
class EpisodeJob:
    new_ep_idx: int
    src_ep_idx: int
    ep_dir: str            # str for pickling
    prompts: list          # list[str], from metadata
    ep_task_indices: list  # list[int] (indices into global tasks.jsonl)
    frame_start: int
    out_dir: str
    depth_codec: str
    rgb_codec: str
    fps: int               # final fps written into info.json/parquet
    layout_kwargs: dict    # asdict(SourceLayout) for re-construction in workers
    target_h: Optional[int] = None  # if set, downsample RGB+depth to (target_h, target_w)
    target_w: Optional[int] = None  # both must be set or both None (native)


def _sidecar_path(out_dir: Path, new_ep_idx: int) -> Path:
    chunk_idx = new_ep_idx // CHUNK_SIZE
    return (out_dir / "data" / f"chunk-{chunk_idx:03d}"
            / f"episode_{new_ep_idx:06d}.stats.json")


def _episode_is_complete(job: EpisodeJob, cams: list[str]) -> bool:
    """True iff all expected output files exist AND the sidecar sentinel is present.

    The sidecar is written last (atomically) by ``_encode_episode``, so its
    presence guarantees parquet + RGB + depth all finished.
    """
    out_dir = Path(job.out_dir)
    new_ep_idx = job.new_ep_idx
    chunk_idx = new_ep_idx // CHUNK_SIZE
    chunk_str = f"chunk-{chunk_idx:03d}"
    ep_str = f"episode_{new_ep_idx:06d}"
    sidecar = _sidecar_path(out_dir, new_ep_idx)
    if not sidecar.exists():
        return False
    parquet = out_dir / "data" / chunk_str / f"{ep_str}.parquet"
    if not parquet.exists():
        return False
    for cam in cams:
        if not (out_dir / "videos" / chunk_str / f"observation.images.{cam}" / f"{ep_str}.mp4").exists():
            return False
    if job.depth_codec in ("ffv1", "both"):
        for cam in cams:
            if not (out_dir / "videos" / chunk_str / f"observation.depth_ffv1.{cam}" / f"{ep_str}.mkv").exists():
                return False
    if job.depth_codec in ("libx264rgb", "both"):
        for cam in cams:
            if not (out_dir / "videos" / chunk_str / f"observation.depth_x264rgb.{cam}" / f"{ep_str}.mp4").exists():
                return False
    return True


def _encode_episode(job: EpisodeJob) -> dict:
    """Per-episode encoding. Writes parquet + RGB mp4 + depth videos."""
    out_dir = Path(job.out_dir)
    layout = SourceLayout(**job.layout_kwargs)
    data = load_episode(Path(job.ep_dir), layout)
    state_dim = layout.state_dim
    cams = list(layout.camera_map.values())

    N = data["state"].shape[0]
    H_raw, W_raw = data["depth"][cams[0]].shape[1:]

    # ── Builder-stage resize (openpi-aligned) ─────────────────────────────
    # When target_h/target_w are set, ffmpeg rescales RGB inside the encode
    # pipe (-vf scale=W:H:flags=bilinear), and depth is resized in-place here
    # with the linspace-endpoint nearest-NN sampler. Both match openpi exactly.
    # K is rescaled in the aggregator (convert()) with the "endpoint"
    # convention (s = (dst-1)/(src-1)) so pixel grid + K stay endpoint-aligned.
    do_resize = (job.target_h is not None) and (job.target_w is not None)
    if do_resize:
        H, W = int(job.target_h), int(job.target_w)
        if (H, W) == (H_raw, W_raw):
            do_resize = False  # no-op resize requested; skip
    if do_resize:
        for cam in cams:
            data["depth"][cam] = _resize_depth_nearest_endpoint(data["depth"][cam], H, W)
        # RGB stays at source resolution in numpy; ffmpeg downscales below.
    else:
        H, W = H_raw, W_raw

    # One deterministic task per episode (matches openpi). The first prompt
    # listed in the episode's metadata.json drives every frame's task_index.
    # Per-frame randomization (the legacy behavior) would silently scramble
    # language conditioning the moment an episode metadata lists multiple
    # prompts. Keep this aligned with openpi which reads
    # `metadata["language"]["prompt"][0]`.
    ep_task_idx_arr = np.asarray(job.ep_task_indices, dtype=np.int64)
    per_frame_task_idx = np.full((N,), int(ep_task_idx_arr[0]), dtype=np.int64)

    chunk_idx = job.new_ep_idx // CHUNK_SIZE
    parquet_path = out_dir / "data" / f"chunk-{chunk_idx:03d}" / f"episode_{job.new_ep_idx:06d}.parquet"
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    frame_index = np.arange(N, dtype=np.int64)
    timestamp = (frame_index / float(job.fps)).astype(np.float32)
    global_index = np.arange(job.frame_start, job.frame_start + N, dtype=np.int64)

    table = pa.table({
        "observation.state": pa.array(list(data["state"]), type=pa.list_(pa.float32(), state_dim)),
        "action":            pa.array(list(data["action"]), type=pa.list_(pa.float32(), state_dim)),
        "timestamp":      pa.array(timestamp, type=pa.float32()),
        "frame_index":    pa.array(frame_index, type=pa.int64()),
        "episode_index":  pa.array(np.full((N,), job.new_ep_idx, dtype=np.int64), type=pa.int64()),
        "index":          pa.array(global_index, type=pa.int64()),
        "task_index":     pa.array(per_frame_task_idx, type=pa.int64()),
    })
    pq.write_table(table, parquet_path)

    # Per-cam ffmpeg work runs in external subprocesses, so a ThreadPoolExecutor
    # is enough to fan out — the Python threads just block on subprocess.run
    # while libx264/ffv1 do the actual work in parallel.
    def _encode_rgb_cam(cam: str) -> None:
        rgb_path = (out_dir / "videos" / f"chunk-{chunk_idx:03d}"
                    / f"observation.images.{cam}" / f"episode_{job.new_ep_idx:06d}.mp4")
        _ffmpeg_encode_rgb(
            data["rgb"][cam], rgb_path,
            fps=job.fps, rgb_codec=job.rgb_codec,
            target_h=(H if do_resize else None),
            target_w=(W if do_resize else None),
        )

    with ThreadPoolExecutor(max_workers=len(cams)) as pool:
        list(pool.map(_encode_rgb_cam, cams))

    codecs_to_emit: list[tuple[str, str]] = []
    if job.depth_codec in ("ffv1", "both"):
        codecs_to_emit.append(("ffv1", "depth_ffv1"))
    if job.depth_codec in ("libx264rgb", "both"):
        codecs_to_emit.append(("libx264rgb", "depth_x264rgb"))
    for vcodec, feat_suffix in codecs_to_emit:
        ext = DEPTH_VIDEO_EXT[vcodec]

        def _encode_depth_cam(cam: str, vcodec=vcodec, feat_suffix=feat_suffix, ext=ext) -> None:
            depth_path = (out_dir / "videos" / f"chunk-{chunk_idx:03d}"
                          / f"observation.{feat_suffix}.{cam}" / f"episode_{job.new_ep_idx:06d}{ext}")
            encode_depth_video_frames(data["depth"][cam], depth_path, fps=job.fps,
                                      vcodec=vcodec, overwrite=True)

        with ThreadPoolExecutor(max_workers=len(cams)) as pool:
            list(pool.map(_encode_depth_cam, cams))

    def _stats(x: np.ndarray) -> dict:
        return {
            "min":   x.min(axis=0).astype(float).tolist(),
            "max":   x.max(axis=0).astype(float).tolist(),
            "mean":  x.mean(axis=0).astype(float).tolist(),
            "std":   x.std(axis=0).astype(float).tolist(),
            "count": [int(N)],
        }

    result = {
        "new_ep_idx":   job.new_ep_idx,
        "src_ep_idx":   job.src_ep_idx,
        "N":            int(N),
        "prompts":      job.prompts,
        "state_stats":  _stats(data["state"]),
        "action_stats": _stats(data["action"]),
        "H":            int(H), "W": int(W),
        "H_raw":        int(H_raw), "W_raw": int(W_raw),
        # K is reported at SOURCE resolution; the aggregator rescales by
        # (W/W_raw, H/H_raw) when do_resize is active.
        "intrinsic":    {cam: data["intrinsic"][cam].tolist() for cam in cams},
    }

    # Sidecar JSON acts as the per-episode completion sentinel for --resume.
    # Written last (after parquet + RGB + depth), atomically via temp+rename:
    # if the worker crashes mid-encode, no sidecar => resume re-runs this ep.
    sidecar_path = parquet_path.with_suffix(".stats.json")
    sidecar_tmp = sidecar_path.with_suffix(".json.tmp")
    sidecar_tmp.write_text(json.dumps(result))
    sidecar_tmp.rename(sidecar_path)
    return result


# --- Main conversion --------------------------------------------------------

def convert(
    raw_dirs: list[Path],
    out_dir: Path,
    task_name: str,
    robot_type: str,
    depth_codec: str,
    fps: int | None,
    motor_names: list[str] | None,
    episodes: list[int] | None,
    layout: SourceLayout,
    overwrite: bool,
    workers: int,
    rgb_codec: str = DEFAULT_RGB_CODEC,
    target_h: Optional[int] = None,
    target_w: Optional[int] = None,
    resume: bool = False,
) -> None:
    if overwrite and resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    if (target_h is None) ^ (target_w is None):
        raise ValueError("target_h and target_w must both be set or both unset")
    if motor_names is None:
        motor_names = DEFAULT_MOTOR_NAMES_EEF32
    if len(motor_names) != layout.state_dim:
        raise ValueError(
            f"motor_names has {len(motor_names)} entries; expected {layout.state_dim}"
        )
    if len(raw_dirs) == 0:
        raise ValueError("raw_dirs must be non-empty")
    if episodes is not None and len(raw_dirs) > 1:
        raise ValueError(
            "--episodes is not supported when merging multiple --raw-dir entries "
            "(episode 0 exists in every source). Pre-select by listing only the "
            "source dirs you want, or run a single-source build first."
        )

    if out_dir.exists():
        if overwrite:
            shutil.rmtree(out_dir)
            (out_dir / "meta").mkdir(parents=True)
        elif resume:
            (out_dir / "meta").mkdir(parents=True, exist_ok=True)
        else:
            raise FileExistsError(
                f"{out_dir} exists; pass --overwrite to replace, --resume to continue"
            )
    else:
        (out_dir / "meta").mkdir(parents=True)

    # Concatenate episodes from each source in CLI order. Renumbered globally
    # downstream (new_ep_idx = 0..N_total-1). Original source dir basename for
    # each episode is retained only via the new_ep_idx -> raw_dir mapping
    # printed below; not surfaced in episodes.jsonl (intentional: don't
    # diverge from upstream LeRobot schema for a debug nicety).
    all_eps: list[Path] = []
    per_source_counts: list[tuple[str, int]] = []
    for rd in raw_dirs:
        eps_this_src = discover_episodes(rd)
        if episodes is not None:  # single-source-only path (guarded above)
            keep = set(episodes)
            eps_this_src = [p for p in eps_this_src if int(p.name) in keep]
            missing = keep - {int(p.name) for p in eps_this_src}
            if missing:
                raise ValueError(f"Requested episodes not found in {rd}: {sorted(missing)}")
        if not eps_this_src:
            raise RuntimeError(f"No episodes selected under {rd}")
        per_source_counts.append((rd.name, len(eps_this_src)))
        all_eps.extend(eps_this_src)

    total_episodes = len(all_eps)
    if len(raw_dirs) > 1:
        print(f"[sources] merging {len(raw_dirs)} raw datasets:")
        for name, n in per_source_counts:
            print(f"            {name}: {n} episodes")

    # ---- Pre-scan: per-episode prompts, frame counts, fps ------------------
    task_to_index: dict[str, int] = {}
    per_ep_prompts: list[list[str]] = []
    frame_counts: list[int] = []
    detected_fps: list[int] = []

    for ep in tqdm(all_eps, desc="scan", unit="ep"):
        meta_path = ep / layout.metadata_filename
        meta = json.loads(meta_path.read_text())
        cur = meta
        for k in layout.num_frames_path:
            cur = cur[k]
        N_raw = int(cur)
        # Builder drops the last raw frame (action[t] = state[t+1]) → N = N_raw - 1.
        if N_raw < 2:
            raise ValueError(f"{meta_path}: need >=2 raw frames for action shift; got {N_raw}")
        N = N_raw - 1
        cur = meta
        for k in layout.fps_path:
            cur = cur[k]
        ep_fps = int(cur)
        cur = meta
        for k in layout.prompt_path:
            cur = cur[k]
        prompts = list(cur)
        if not prompts:
            raise ValueError(f"{meta_path}: empty prompt list")

        for s in prompts:
            if s not in task_to_index:
                task_to_index[s] = len(task_to_index)

        per_ep_prompts.append(prompts)
        frame_counts.append(N)
        detected_fps.append(ep_fps)

    if len(set(detected_fps)) != 1:
        raise ValueError(f"Inconsistent framerate across episodes: {set(detected_fps)}")
    detected_fps_v = detected_fps[0]
    final_fps = int(fps) if fps is not None else detected_fps_v
    if fps is not None and fps != detected_fps_v:
        print(f"[warn] --fps {fps} differs from metadata framerate {detected_fps_v}; "
              f"writing {fps} into info.json (timestamps in parquet will be frame_index/{fps}).")

    total_frames = sum(frame_counts)
    print(f"[tasks] {len(task_to_index)} unique prompts across {total_episodes} episodes")
    print(f"[scan ] total frames = {total_frames} (post-action-shift), fps = {final_fps}")

    # ---- Build per-episode jobs --------------------------------------------
    layout_kwargs = asdict(layout)
    jobs: list[EpisodeJob] = []
    running = 0
    for new_ep_idx, (ep, prompts, N) in enumerate(zip(all_eps, per_ep_prompts, frame_counts)):
        ep_task_indices = [task_to_index[s] for s in prompts]
        jobs.append(EpisodeJob(
            new_ep_idx=new_ep_idx,
            src_ep_idx=int(ep.name),
            ep_dir=str(ep),
            prompts=prompts,
            ep_task_indices=ep_task_indices,
            frame_start=running,
            out_dir=str(out_dir),
            depth_codec=depth_codec,
            rgb_codec=rgb_codec,
            fps=final_fps,
            layout_kwargs=layout_kwargs,
            target_h=target_h,
            target_w=target_w,
        ))
        running += N
    assert running == total_frames

    # ---- Resume: partition jobs into already-done vs todo ------------------
    # A job is "done" iff its sidecar JSON exists AND every expected output
    # file (parquet, 3 RGB, depth-per-codec) is present. The sidecar is
    # written last in _encode_episode, so its presence is a reliable sentinel.
    cams = list(layout.camera_map.values())
    results: list[dict] = []
    todo_jobs: list[EpisodeJob] = []
    if resume:
        for j in jobs:
            if _episode_is_complete(j, cams):
                cached = json.loads(_sidecar_path(out_dir, j.new_ep_idx).read_text())
                results.append(cached)
            else:
                todo_jobs.append(j)
        print(f"[resume] {len(results)} episodes already complete; "
              f"encoding {len(todo_jobs)} new")
    else:
        todo_jobs = jobs

    # ---- Dispatch ----------------------------------------------------------
    print(f"[encode] {len(todo_jobs)} episodes, workers={workers}")
    bar_kwargs = dict(total=len(todo_jobs), desc=f"encode (w={workers})", unit="ep", dynamic_ncols=True)
    if workers <= 1:
        for j in tqdm(todo_jobs, **bar_kwargs):
            results.append(_encode_episode(j))
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex, tqdm(**bar_kwargs) as pbar:
            futures = {ex.submit(_encode_episode, j): j for j in todo_jobs}
            for fut in as_completed(futures):
                j = futures[fut]
                try:
                    results.append(fut.result())
                except Exception as e:
                    tqdm.write(f"[FAIL ep {j.new_ep_idx:06d}] {Path(j.ep_dir).name}: {e}")
                    raise
                pbar.update(1)
                pbar.set_postfix_str(f"ep {j.new_ep_idx:06d}", refresh=False)
    results.sort(key=lambda r: r["new_ep_idx"])

    # ---- Aggregate for meta writing ----------------------------------------
    episodes_records = [
        {"episode_index": r["new_ep_idx"], "tasks": r["prompts"], "length": r["N"]}
        for r in results
    ]
    episodes_stats_records = [
        {"episode_index": r["new_ep_idx"],
         "stats": {"observation.state": r["state_stats"],
                   "action":            r["action_stats"]}}
        for r in results
    ]

    # Stored output resolution (target if resize was active, else source).
    # All episodes share these since jobs use the same target_h/target_w.
    H_first, W_first = results[0]["H"], results[0]["W"]
    H_raw_first, W_raw_first = (
        results[0].get("H_raw", H_first),
        results[0].get("W_raw", W_first),
    )

    # "Endpoint" rescale convention (matches openpi
    # ``scale_intrinsics(..., convention="endpoint")``): dst pixel (0, 0) and
    # (W-1, H-1) correspond exactly to src (0, 0) and (W_raw-1, H_raw-1). Pairs
    # with the linspace-endpoint depth sampler in both this builder and the
    # dataloader, so pixel grid + K stay aligned end-to-end. When no resize is
    # active these collapse to 1.0.
    def _endpoint_scale(dst: float, src: float) -> float:
        return (dst - 1.0) / (src - 1.0) if src > 1 else 1.0

    sx = _endpoint_scale(W_first, W_raw_first)
    sy = _endpoint_scale(H_first, H_raw_first)

    # Rescale each episode's K to the output grid, then keep BOTH forms: the
    # per-episode table (meta/episode_intrinsics.jsonl, the accurate one) and
    # the across-episode mean (meta/camera_intrinsics.json, kept for datasets
    # and tools that predate the per-episode table). A large per-cell drift
    # around the mean is logged but does not abort — a build that spans a ZED
    # re-mount mid-dataset is still finishable; the user just needs to know.
    per_cam_K_stack: dict[str, list[np.ndarray]] = {cam: [] for cam in results[0]["intrinsic"]}
    per_episode_K: dict[int, dict[str, np.ndarray]] = {}
    for r in results:
        if (
            r.get("W_raw", W_raw_first) != W_raw_first
            or r.get("H_raw", H_raw_first) != H_raw_first
        ):
            print(
                f"[warn] ep {r['new_ep_idx']} source resolution "
                f"{r.get('W_raw')}x{r.get('H_raw')} differs from ep0 "
                f"{W_raw_first}x{H_raw_first}; using ep-local rescale factors"
            )
            ep_sx = _endpoint_scale(W_first, r.get("W_raw", W_raw_first))
            ep_sy = _endpoint_scale(H_first, r.get("H_raw", H_raw_first))
        else:
            ep_sx, ep_sy = sx, sy
        ep_K: dict[str, np.ndarray] = {}
        for cam, arr in r["intrinsic"].items():
            K = np.asarray(arr, dtype=np.float64).copy()
            K[0, 0] *= ep_sx  # fx
            K[0, 2] *= ep_sx  # cx
            K[1, 1] *= ep_sy  # fy
            K[1, 2] *= ep_sy  # cy
            per_cam_K_stack[cam].append(K)
            ep_K[cam] = K
        per_episode_K[int(r["new_ep_idx"])] = ep_K

    first_ep_intrinsics: dict[str, np.ndarray] = {}
    for cam, ks in per_cam_K_stack.items():
        stacked = np.stack(ks, axis=0)
        mean_K = stacked.mean(axis=0)
        first_ep_intrinsics[cam] = mean_K
        max_drift = float(np.abs(stacked - mean_K).max())
        if max_drift > 0.5:  # >0.5 px in any K cell is unusual within one rig
            print(
                f"[warn] intrinsics for {cam} drift {max_drift:.3f}px across "
                f"episodes around mean; using mean K (matches openpi)"
            )

    # ---- meta files --------------------------------------------------------
    with (out_dir / "meta" / "tasks.jsonl").open("w") as fh:
        for task_str, idx in sorted(task_to_index.items(), key=lambda kv: kv[1]):
            fh.write(json.dumps({"task_index": idx, "task": task_str}) + "\n")

    with (out_dir / "meta" / "episodes.jsonl").open("w") as fh:
        for rec in episodes_records:
            fh.write(json.dumps(rec) + "\n")

    with (out_dir / "meta" / "episodes_stats.jsonl").open("w") as fh:
        for rec in episodes_stats_records:
            fh.write(json.dumps(rec) + "\n")

    def _k_entry(K: np.ndarray) -> dict:
        return {
            "fx": float(K[0, 0]), "fy": float(K[1, 1]),
            "cx": float(K[0, 2]), "cy": float(K[1, 2]),
            "width": int(W_first), "height": int(H_first),
        }

    cam_intrinsics = {cam: _k_entry(K) for cam, K in first_ep_intrinsics.items()}
    with (out_dir / "meta" / "camera_intrinsics.json").open("w") as fh:
        json.dump(cam_intrinsics, fh, indent=2)

    # Per-episode intrinsics. Same units, same grid and same endpoint rescale as
    # camera_intrinsics.json — only the granularity differs. The loader prefers
    # this file and silently falls back to the mean when it is absent, so older
    # datasets keep working and a rebuild is not required.
    with (out_dir / "meta" / "episode_intrinsics.jsonl").open("w") as fh:
        for ep_idx in sorted(per_episode_K):
            fh.write(json.dumps({
                "episode_index": ep_idx,
                "intrinsics": {
                    cam: _k_entry(K) for cam, K in per_episode_K[ep_idx].items()
                },
            }) + "\n")
    print(f"[info] wrote per-episode intrinsics for {len(per_episode_K)} episodes")

    # ---- info.json ---------------------------------------------------------
    state_dim = layout.state_dim

    features: dict = {
        "observation.state": {"dtype": "float32", "shape": [state_dim], "names": [motor_names]},
        "action":            {"dtype": "float32", "shape": [state_dim], "names": [motor_names]},
        "timestamp":     {"dtype": "float32", "shape": [1]},
        "frame_index":   {"dtype": "int64",   "shape": [1]},
        "episode_index": {"dtype": "int64",   "shape": [1]},
        "index":         {"dtype": "int64",   "shape": [1]},
        "task_index":    {"dtype": "int64",   "shape": [1]},
    }
    rgb_codec_canonical = _CODEC_CANONICAL.get(rgb_codec, rgb_codec)
    for cam in cams:
        features[f"observation.images.{cam}"] = {
            "dtype": "video",
            "shape": [H_first, W_first, 3],
            "names": ["height", "width", "rgb"],
            "info": {
                "video.height": H_first, "video.width": W_first,
                "video.codec": rgb_codec_canonical, "video.pix_fmt": RGB_PIX_FMT,
                "video.is_depth_map": False, "video.fps": final_fps,
                "video.channels": 3, "has_audio": False,
            },
        }
    for vcodec, feat_suffix in [("ffv1", "depth_ffv1"), ("libx264rgb", "depth_x264rgb")]:
        if (vcodec == "ffv1" and depth_codec in ("ffv1", "both")) or \
           (vcodec == "libx264rgb" and depth_codec in ("libx264rgb", "both")):
            for cam in cams:
                pix_fmt = "gray16le" if vcodec == "ffv1" else "rgb24"
                ch = 1 if vcodec == "ffv1" else 3
                ext = DEPTH_VIDEO_EXT[vcodec]
                features[f"observation.{feat_suffix}.{cam}"] = {
                    # dtype = "depth_video" (not "video") so the built-in LeRobot
                    # loader skips these — RobotVideoDataset reads the
                    # `depth_encoder` + `depth_ext` fields below to construct the
                    # right path and pick the right codec.
                    "dtype": "depth_video",
                    "shape": [H_first, W_first, ch],
                    "names": ["height", "width", "depth" if ch == 1 else "rgb_packed_u16"],
                    "info": {
                        "video.height": H_first, "video.width": W_first,
                        "video.codec": _CODEC_CANONICAL[vcodec], "video.pix_fmt": pix_fmt,
                        "video.is_depth_map": True, "video.fps": final_fps,
                        "video.channels": ch, "has_audio": False,
                        "depth_unit": "mm", "depth_dtype": "uint16",
                        "depth_encoder": vcodec,
                        "depth_ext": ext,
                        "depth_packing": "rgb_hi_lo_zero" if vcodec == "libx264rgb" else "gray16le",
                    },
                }

    info = {
        "codebase_version": "v2.1",
        "robot_type": robot_type,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": len(task_to_index),
        "total_videos": total_episodes * sum(
            1 for k in features if k.startswith("observation.") and features[k]["dtype"] == "video"
        ),
        "total_chunks": (total_episodes + CHUNK_SIZE - 1) // CHUNK_SIZE,
        "chunks_size": CHUNK_SIZE,
        "fps": final_fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": features,
    }
    info["task_name"] = task_name
    info["source_dirs"] = [rd.name for rd in raw_dirs]
    # Provenance flag so a future dataloader can refuse to silently load a dir
    # built by some other (e.g. legacy FK-based) builder. openpi's eef32 build
    # is byte-equivalent to this one — both should be accepted by downstream
    # consumers that key on "*-eef32-action26-*".
    info["builder"] = "flexpi-eef32-action26-openpi-aligned"
    with (out_dir / "meta" / "info.json").open("w") as fh:
        json.dump(info, fh, indent=4)

    # Post-build validation: rot6d row convention + SE(3) round-trip on
    # episode 0. Aborts the build if state silently disagrees with the
    # rd-convert action26 it was derived from. Same gate openpi runs.
    print("[validate] running state32 vs action26 round-trip on episode 0...")
    try:
        validate_state32_first_episode(all_eps[0], layout, n_check=10)
    except Exception as e:
        raise RuntimeError(
            f"[validate] state32 round-trip failed on {all_eps[0]}: {e}. "
            f"This means the rd-convert action26 in lowdim/head/*.npz is not "
            f"a valid SE(3) — re-record the episode or skip it."
        ) from e
    print("[validate] OK")

    print(f"\n[done] wrote {total_episodes} episodes, {total_frames} frames to {out_dir}")


# --- CLI --------------------------------------------------------------------

def _parse_camera_map(s: str) -> dict:
    out = {}
    for pair in s.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise argparse.ArgumentTypeError(f"camera-map entry {pair!r} must be src=dst")
        k, v = pair.split("=", 1)
        out[k.strip()] = v.strip()
    if not out:
        raise argparse.ArgumentTypeError("camera-map must have at least one entry")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--raw-dir", type=Path, required=True, nargs="+",
                    help="One or more top-level directories, each containing per-episode subdirs "
                         "(0000/, 0001/, ...). Pass multiple to merge into a single LeRobot "
                         "dataset; episodes are concatenated in the order listed and renumbered "
                         "globally. All sources must share the same fps + camera layout.")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Output dir. Default (single source only): ./data/<raw_dir.name>. "
                         "Required when merging multiple sources.")
    ap.add_argument("--task-name", type=str, required=True,
                    help="Human-readable task name (stashed in info.json).")
    ap.add_argument("--robot-type", type=str, default="yam",
                    help="Robot label written into info.json.robot_type. Default: yam")
    ap.add_argument("--depth-codec",
                    choices=["ffv1", "libx264rgb", "both", "none"], default="ffv1",
                    help="Depth video codec(s) to emit. ffv1 = lossless 16-bit gray (.mkv). Default: ffv1")
    ap.add_argument("--rgb-codec",
                    choices=["libx264", "libx265", "libsvtav1", "libaom-av1"],
                    default=DEFAULT_RGB_CODEC,
                    help=f"RGB video encoder. Default: {DEFAULT_RGB_CODEC} (matches "
                         "yam_openpi/yam_dataset_builder/convert_yam_data_to_lerobot.py — "
                         "fast random-access decoding at training time when paired with "
                         "the all-I-frame default GOP=1). AV1 (libsvtav1, libaom-av1) "
                         "compresses harder but decodes noticeably slower; only switch if "
                         "disk size is more important than dataloader speed.")
    ap.add_argument("--fps", type=int, default=None,
                    help="Override fps written into info.json/timestamps. "
                         "Default: read from metadata.framerate (must agree across episodes).")
    ap.add_argument("--motor-names", type=str, default=None,
                    help="Comma-separated motor names for state/action. Length must equal 32. "
                         "Default: yam_eef.STATE_LAYOUT slot names.")
    ap.add_argument("--episodes", type=int, nargs="*", default=None,
                    help="Optional subset of source episode ids (default: all). Only valid "
                         "with a single --raw-dir (ambiguous when merging).")
    ap.add_argument("--workers", type=int, default=1,
                    help="Parallel episode workers (ProcessPoolExecutor). Default: 1.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Wipe out_dir if it exists.")
    ap.add_argument("--resume", action="store_true",
                    help="If out_dir exists, skip episodes whose outputs are already "
                         "complete (parquet + all RGB/depth videos + sidecar "
                         "episode_NNNNNN.stats.json) and only encode the rest. "
                         "Mutually exclusive with --overwrite.")
    # Default: downsample to 360x640 (2x downsample from YAM's native 720x1280).
    # Validated bit-exact at the dataloader output: final K identical to native
    # build, depth-valid ratio drops ~7% (boundary-only), single-worker
    # per-sample latency ~1.7x faster. Pass --no-resize to keep native
    # resolution; pass explicit --target-h/--target-w to override.
    ap.add_argument("--target-h", type=int, default=360,
                    help="Downsample every RGB+depth frame to (target-h, target-w) "
                         "BEFORE encoding. RGB uses INTER_AREA, depth uses INTER_NEAREST "
                         "(preserves uint16 mm). Camera intrinsics are rescaled by "
                         "(target_w/raw_w, target_h/raw_h) so the dataloader's downstream "
                         "K-rescale produces the identical final K at the model's per-cam "
                         "target. Default: 360 (validated 2x downsample for YAM 720x1280). "
                         "Use --no-resize to keep native resolution. No-op when target "
                         "equals source.")
    ap.add_argument("--target-w", type=int, default=640,
                    help="See --target-h. Default: 640 (matches 360 for the 16:9 YAM "
                         "camera). Must be set together with --target-h.")
    ap.add_argument("--no-resize", action="store_true",
                    help="Disable builder-stage resize; encode at native source "
                         "resolution. Equivalent to passing --target-h/--target-w that "
                         "match the source.")

    # SourceLayout overrides (most common). For deeper customization edit
    # SourceLayout defaults in source_reader.py.
    ap.add_argument("--camera-map", type=_parse_camera_map, default=None,
                    help="Override source->output camera mapping, format: "
                         "'head=cam_high,left_wrist=cam_left_wrist,...'")
    ap.add_argument("--joints-key", type=str, default=None,
                    help="NPZ key for the 14D joints+grip vector inside lowdim/<cam>/<frame>.npz "
                         f"(default: {SourceLayout.joints_npz_key!r}).")
    ap.add_argument("--depth-key", type=str, default=None,
                    help=f"NPZ key for depth array (default: {SourceLayout.depth_npz_key!r}).")
    ap.add_argument("--intrinsics-key", type=str, default=None,
                    help=f"NPZ key for the (4,) [fx,fy,cx,cy] intrinsics "
                         f"(default: {SourceLayout.intrinsics_npz_key!r}).")
    args = ap.parse_args()

    if args.out_dir is None:
        if len(args.raw_dir) == 1:
            out_dir = Path("./data") / args.raw_dir[0].name
        else:
            raise SystemExit(
                "--out-dir is required when merging multiple --raw-dir entries"
            )
    else:
        out_dir = args.out_dir

    motor_names = args.motor_names.split(",") if args.motor_names else None
    if motor_names is not None and len(motor_names) != 32:
        raise SystemExit(
            f"--motor-names has {len(motor_names)} entries; expected 32 (eef32 layout)"
        )

    layout_overrides: dict = {}
    if args.camera_map is not None:
        layout_overrides["camera_map"] = args.camera_map
    if args.joints_key is not None:
        layout_overrides["joints_npz_key"] = args.joints_key
    if args.depth_key is not None:
        layout_overrides["depth_npz_key"] = args.depth_key
    if args.intrinsics_key is not None:
        layout_overrides["intrinsics_npz_key"] = args.intrinsics_key
    layout = SourceLayout(**layout_overrides)

    # --no-resize takes precedence over --target-h/--target-w defaults.
    target_h = None if args.no_resize else args.target_h
    target_w = None if args.no_resize else args.target_w

    convert(
        raw_dirs=args.raw_dir,
        out_dir=out_dir,
        task_name=args.task_name,
        robot_type=args.robot_type,
        depth_codec=args.depth_codec,
        fps=args.fps,
        motor_names=motor_names,
        episodes=args.episodes,
        layout=layout,
        overwrite=args.overwrite,
        workers=args.workers,
        rgb_codec=args.rgb_codec,
        target_h=target_h,
        target_w=target_w,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
