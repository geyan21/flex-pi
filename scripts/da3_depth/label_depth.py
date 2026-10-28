"""Add DA3 metric depth to an RGB-only LeRobot dataset (standalone side-car pass).

Input:  a LeRobot v2.1 dataset in the canonical Flex-π layout with RGB videos
        and ``meta/camera_intrinsics.json``, but NO depth streams.
Output: the SAME dataset, plus
          videos/chunk-{c:03d}/observation.depth_ffv1.<cam>/episode_{i:06d}.mkv
          meta/info.json  →  observation.depth_ffv1.<cam> features registered
Nothing else is touched — parquet, RGB, episodes.jsonl and tasks.jsonl are
read-only here. Re-running is safe (see Resume below).

The result is byte-compatible with what scripts/yam_dataset_builder/convert.py
writes when the capture DOES have depth, so ``RobotVideoDataset`` consumes both
without knowing which is which.

WHY THIS IS STANDALONE
    Zero Flex-π imports and zero pyav: the DA3 module is defined inline below
    and all video IO goes through ffmpeg/ffprobe subprocesses. So this runs in
    its own small conda env holding only torch + depth_anything_3, with nothing
    of this repo on PYTHONPATH — which is why it can pin torch 2.10 / numpy<2
    without disturbing the ``flexpi`` env.

THE TWO THINGS THAT MUST BE RIGHT
  1. K CONVENTION. DA3METRIC's raw output is not metric; it is scaled by
     ``focal / 300`` with ``focal = (fx + fy) / 2``. K therefore only sets a
     GLOBAL SCALE on the depth — a wrong K produces a plausible-looking depth
     map that is silently the wrong size in metres, with no error anywhere.
     We rescale K from the resolution declared in camera_intrinsics.json to the
     depth grid with the ENDPOINT convention ``s = (dst-1)/(src-1)``, which is
     what ``RobotVideoDataset._load_layout_intrinsics_for_dir`` re-applies at
     load time and what the linspace-endpoint depth resampler on both sides
     assumes. The plain ratio ``dst/src`` is a different convention and
     disagrees by ~0.1-0.5%.

     When ``meta/episode_intrinsics.jsonl`` is present, each episode is scaled
     by its OWN K instead of one dataset-wide K. This matters more here than
     anywhere downstream: the focal multiplies into the millimetres we write, so
     an averaged K bakes a per-episode scale error into the files. The
     unprojection at train time reads K again, but only to turn depth into X/Y —
     it cannot undo a Z that was already stored wrong.
  2. ASPECT RATIO. Feeding DA3 a stretched frame corrupts its geometry. The
     depth grid therefore defaults to each camera's OWN native RGB resolution
     (probed per camera, so a mixed-resolution rig is handled), which also makes
     the K rescale an exact identity when the intrinsics are stated at that same
     resolution. ``--depth-hw H W`` overrides for all cams and warns if it
     changes the aspect ratio.

Camera names are DISCOVERED from videos/*/observation.images.<cam>/ intersected
with the intrinsics keys — nothing about the 3-cam canonical set is hardcoded.

RESUME / SCALE
  A camera is skipped when its output .mkv already exists (encode is atomic:
  temp file + rename, so a killed run never leaves a half-written .mkv that
  looks done). One bad episode is logged to meta/depth_failures_shard<N>.txt
  and does not kill the run. ``--num-shards N --shard-id i`` splits episodes
  interleaved across N processes (one per GPU); run ``--patch-info-only`` ONCE
  after all shards finish — concurrent info.json writers would race.

MEMORY
  Decode → DA3 → encode is streamed frame-batch by frame-batch through ffmpeg
  pipes, so peak RAM is ``batch_size`` frames regardless of episode length or
  resolution.

Usage:
    # 0. look before you leap — validates cams, K, resolutions; no GPU, no writes
    python scripts/da3_depth/label_depth.py --dataset-dir ./data/<ds> --check

    # 1. label
    python scripts/da3_depth/label_depth.py \
        --dataset-dir ./data/<ds> \
        --da3-weights /path/to/DA3METRIC-LARGE

    # sharded across 4 GPUs, then finalize
    for i in 0 1 2 3; do CUDA_VISIBLE_DEVICES=$i python scripts/da3_depth/label_depth.py \
        --dataset-dir ./data/<ds> --da3-weights ... --num-shards 4 --shard-id $i & done; wait
    python scripts/da3_depth/label_depth.py --dataset-dir ./data/<ds> --patch-info-only
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# DA3 metric-depth module, defined here rather than imported so this file
# depends only on depth_anything_3. Metric formula is the official DA3 one
# (README FAQ + alignment.py): metric_depth_m = net_output * focal / 300 with
# focal = (fx+fy)/2.
# ─────────────────────────────────────────────────────────────────────────────
_METRIC_SCALE_FACTOR = 300.0


class DA3DepthModule(nn.Module):
    PATCH_SIZE = 14
    IMAGENET_MEAN = [0.485, 0.456, 0.406]
    IMAGENET_STD = [0.229, 0.224, 0.225]

    def __init__(self, weights_dir: str):
        super().__init__()
        from depth_anything_3.api import DepthAnything3
        self.da3 = DepthAnything3.from_pretrained(weights_dir)
        self.da3.eval()
        for p in self.da3.parameters():
            p.requires_grad = False
        name = self.da3.model_name.upper()
        self.is_da3metric = "DA3METRIC" in name and "NESTED" not in name
        self.is_nested = "NESTED" in name
        self.register_buffer("img_mean", torch.tensor(self.IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("img_std", torch.tensor(self.IMAGENET_STD).view(1, 3, 1, 1))

    def _pad_to_patch(self, x):
        _, _, H, W = x.shape
        ph = (self.PATCH_SIZE - H % self.PATCH_SIZE) % self.PATCH_SIZE
        pw = (self.PATCH_SIZE - W % self.PATCH_SIZE) % self.PATCH_SIZE
        if ph or pw:
            x = nn.functional.pad(x, (0, pw, 0, ph), mode="reflect")
        return x, H, W

    @torch.no_grad()
    def forward(self, images: torch.Tensor, intrinsics: torch.Tensor) -> torch.Tensor:
        """(B,3,H,W) in [0,1] + (B,3,3) K -> (B,1,H,W) metric depth in metres."""
        x = (images - self.img_mean) / self.img_std
        x, H, W = self._pad_to_patch(x)
        # unsqueeze(1) => (B, N=1, 3, H, W): each frame is its own single-view
        # scene. Unsqueezing at dim 0 instead would make DA3 fuse the B frames
        # as one multi-view scene and collapse per-frame variation.
        raw = self.da3.forward(x.unsqueeze(1), export_feat_layers=[])
        depth = raw["depth"][:, :, :H, :W]
        if self.is_da3metric:
            focal = (intrinsics[:, 0, 0] + intrinsics[:, 1, 1]) / 2.0
            depth = depth * (focal.view(-1, 1, 1, 1) / _METRIC_SCALE_FACTOR)
        return depth


# ─────────────────────────────────────────────────────────────────────────────
# Dataset layout (canonical Flex-π / LeRobot v2.1)
# ─────────────────────────────────────────────────────────────────────────────
RGB_PREFIX = "observation.images."
DEPTH_FEAT = "observation.depth_ffv1"      # the codec the trainer defaults to
DEPTH_EXT = ".mkv"


def _rgb_dir(ds: Path, cam: str, chunk: int) -> Path:
    return ds / "videos" / f"chunk-{chunk:03d}" / f"{RGB_PREFIX}{cam}"


def _rgb_path(ds: Path, cam: str, ep: int, chunks_size: int) -> Path:
    return _rgb_dir(ds, cam, ep // chunks_size) / f"episode_{ep:06d}.mp4"


def _depth_path(ds: Path, cam: str, ep: int, chunks_size: int) -> Path:
    return (ds / "videos" / f"chunk-{ep // chunks_size:03d}"
            / f"{DEPTH_FEAT}.{cam}" / f"episode_{ep:06d}{DEPTH_EXT}")


def discover_cams(ds: Path, intr: dict, want: list[str] | None) -> list[str]:
    """Camera names = RGB video dirs on disk ∩ camera_intrinsics.json keys."""
    on_disk = sorted({
        d.name[len(RGB_PREFIX):]
        for d in ds.glob(f"videos/chunk-*/{RGB_PREFIX}*") if d.is_dir()
    })
    if not on_disk:
        raise SystemExit(f"no {RGB_PREFIX}* video dirs under {ds / 'videos'}")
    if want:
        missing = [c for c in want if c not in on_disk]
        if missing:
            raise SystemExit(f"--cams {missing} not on disk; available: {on_disk}")
        on_disk = list(want)
    no_k = [c for c in on_disk if c not in intr]
    if no_k:
        # DA3METRIC cannot produce metric depth without a focal length. Refuse
        # rather than invent one — a guessed K is a silent global scale error.
        raise SystemExit(
            f"cameras {no_k} have RGB but no entry in meta/camera_intrinsics.json "
            f"(has: {sorted(intr)}). DA3METRIC needs fx/fy; add them or pass "
            f"--cams to restrict to the cameras you do have K for."
        )
    unused = sorted(set(intr) - set(on_disk))
    if unused:
        print(f"[info] intrinsics list cameras with no RGB videos (ignored): {unused}")
    return on_disk


def episode_lengths(ds: Path) -> dict[int, int]:
    """episode_index -> declared frame count, from meta/episodes.jsonl.

    `list_episodes` parsed this file and threw `length` away. The depth pass
    then renamed its temp file into place without ever comparing what it wrote
    to what the episode is supposed to hold, so a short write -- a decoder that
    stops early, a truncated source .mp4 -- published silently and surfaced
    months later as a timestamp-alignment assertion inside a dataloader worker,
    with nothing naming the episode.
    """
    jl = ds / "meta" / "episodes.jsonl"
    if not jl.exists():
        return {}
    out: dict[int, int] = {}
    for line in jl.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if "length" in r:
            out[int(r["episode_index"])] = int(r["length"])
    return out


def list_episodes(ds: Path, cams: list[str]) -> list[int]:
    """Episode indices, from meta/episodes.jsonl when present, else from disk."""
    jl = ds / "meta" / "episodes.jsonl"
    if jl.exists():
        eps = sorted(json.loads(l)["episode_index"] for l in jl.read_text().splitlines() if l.strip())
        if eps:
            return eps
    return sorted({
        int(p.stem.split("_")[-1])
        for cam in cams for p in ds.glob(f"videos/chunk-*/{RGB_PREFIX}{cam}/episode_*.mp4")
    })


# ─────────────────────────────────────────────────────────────────────────────
# ffmpeg / ffprobe IO
# ─────────────────────────────────────────────────────────────────────────────
def probe_hw(path: Path) -> tuple[int, int]:
    out = subprocess.check_output(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
        text=True,
    ).strip().split("\n")[0]
    w, h = out.split(",")[:2]
    return int(h), int(w)


def _stderr_tail(fh, n: int = 2000) -> str:
    fh.seek(0)
    return fh.read().decode(errors="replace")[-n:]


def runs_from_source_index(src_index) -> list[tuple[int, int]]:
    """[(aligned position of the first timestep, run length)] per SOURCE packet.

    THE dedup key, in one place. A builder that resamples onto a fixed output
    rate picks the nearest SOURCE frame per output timestep, so a camera slower
    than that rate has several consecutive timesteps resolving to one source
    frame -- and because the source index is non-decreasing, every such group is
    CONTIGUOUS. That is what makes the key a run rather than a hash table: run k
    covers exactly the timesteps that share one
    `(episode, camera, source_frame_index)`, the episode and camera being fixed
    by which video is being labelled.

    Monotonicity is not assumed. A non-decreasing sequence is the invariant the
    run decomposition rests on; violate it and non-adjacent timesteps could
    share a source index, so this raises rather than silently under-deduping.

    Identity is provenance, never pixels: two source packets that happen to
    decode to the same image are two runs and get two forward passes.
    """
    src = [int(x) for x in src_index]
    for a, b in zip(src, src[1:]):
        if b < a:
            raise ValueError(
                f"frame_source_index is not non-decreasing ({a} -> {b}); "
                f"a resampler produces it that way and the run decomposition "
                f"depends on it. Refusing to dedup.")
    runs: list[tuple[int, int]] = []
    for i, v in enumerate(src):
        if i and v == src[i - 1]:
            runs[-1] = (runs[-1][0], runs[-1][1] + 1)
        else:
            runs.append((i, 1))
    return runs


def load_frame_source_index(ds: Path) -> dict[int, dict[str, list[int]]] | None:
    """meta/frame_source_index.jsonl -> {episode_index: {cam: [source idx]}}.

    Optional, and its absence is NOT an error: a dataset whose builder does not
    emit it is labelled frame by frame, exactly as this pass always did.
    Returning None rather than {} keeps "no table" distinguishable from "table
    with no rows", which would be a corrupt table.

    Format (one JSON object per line, one per episode, in episode order):

        {"episode_index": 0, "frame_source_index":
            {"<cam>": [<source frame index per output timestep>, ...]}}

    Each list is as long as the episode and non-decreasing. See the emitting
    builder for how it is produced.
    """
    f = ds / "meta" / "frame_source_index.jsonl"
    if not f.exists():
        return None
    out: dict[int, dict[str, list[int]]] = {}
    for line in f.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        out[int(r["episode_index"])] = {
            k: [int(x) for x in v] for k, v in r["frame_source_index"].items()}
    if not out:
        raise SystemExit(f"{f} exists but carries no rows; refusing to guess "
                         f"whether that means 'no dedup' or 'corrupt table'")
    return out


def stream_rgb_to_depth(
    rgb_mp4: Path, out_mkv: Path, fps: int, K: np.ndarray, grid: tuple[int, int],
    module: DA3DepthModule, batch: int, device: str,
    max_depth_mm: int | None,
    expect_frames: int | None = None,
    src_index: list[int] | None = None,
) -> tuple[int, int]:
    """Decode → resize to `grid` → DA3 → encode, streamed.

    Returns ``(aligned frames written, forward-pass images)``. Those differ only
    when ``src_index`` is given.

    `grid` is the (h, w) that K was rescaled to; RGB is resized to it BEFORE the
    DA3 forward and the depth is written at it. These have to move together —
    DA3METRIC scales its output by the focal in K, so a K for one grid applied
    to RGB at another is a silent global scale error in the output depth.
    Identity when grid == the RGB resolution (the default).

    Peak RAM is `batch` frames: ffmpeg decodes into a pipe we drain batch by
    batch, and each batch's depth is pushed straight into the ffv1 encoder's
    stdin. Encode target is a temp file renamed on success, so a killed run
    never leaves a truncated .mkv that the resume check would accept.

    ``src_index`` is this (episode, camera)'s row from
    ``meta/frame_source_index.jsonl``: which source frame each output timestep
    resolved to. With it each SOURCE frame is sent
    through DA3 once and its depth is written once per aligned timestep that
    resolved to it. Every aligned frame still gets a depth frame, in the same
    order, at the same count -- only the number of forward passes changes.

    The reuse cannot cross an episode or a camera: this function is called once
    per (episode, camera) and the run table lives inside the call. There is no
    cache to key wrongly and none to invalidate.
    """
    H, W = probe_hw(rgb_mp4)
    h_dst, w_dst = grid
    frame_bytes = H * W * 3
    out_mkv.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_mkv.with_name(out_mkv.stem + ".tmp" + DEPTH_EXT)
    K_t = torch.from_numpy(K).to(device)

    dec_err = tempfile.TemporaryFile()
    enc_err = tempfile.TemporaryFile()
    dec = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(rgb_mp4),
         "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        stdout=subprocess.PIPE, stderr=dec_err,
    )
    enc = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error",
         "-f", "rawvideo", "-pix_fmt", "gray16le", "-s", f"{w_dst}x{h_dst}",
         "-framerate", str(fps), "-i", "pipe:0",
         # -level 3 -g 1: lossless all-intra, matches encode_depth_video_frames
         # so random-access decode at training time stays cheap.
         "-c:v", "ffv1", "-level", "3", "-g", "1", str(tmp)],
        stdin=subprocess.PIPE, stderr=enc_err,
    )
    runs = runs_from_source_index(src_index) if src_index is not None else None

    def infer(arr: np.ndarray) -> np.ndarray:
        """(b,H,W,3) uint8 -> (b,h_dst,w_dst) uint16 millimetres."""
        x = torch.from_numpy(arr.copy()).to(device).permute(0, 3, 1, 2).float() / 255.0
        if (H, W) != (h_dst, w_dst):
            # bilinear + antialias is DA3's own preprocessing kernel; a
            # plain resize would hand it a differently-filtered image than
            # it was trained on.
            x = torch.nn.functional.interpolate(
                x, size=(h_dst, w_dst), mode="bilinear",
                align_corners=False, antialias=True)
        depth_m = module(x, K_t.unsqueeze(0).expand(len(arr), 3, 3).contiguous())
        depth_mm = (depth_m[:, 0] * 1000.0).round()
        if max_depth_mm is not None:
            # Beyond the cap = invalid; 0 is the sentinel the converter uses.
            depth_mm = torch.where(depth_mm > max_depth_mm,
                                   torch.zeros_like(depth_mm), depth_mm)
        return depth_mm.clamp(0, 65535).to(torch.uint16).cpu().numpy()

    n_written = 0
    n_inferred = 0
    try:
        if runs is None:
            # UNCHANGED PATH. No provenance table -> label every aligned frame,
            # in batches of `batch`, exactly as before.
            while True:
                buf = dec.stdout.read(frame_bytes * batch)
                if not buf:
                    break
                b = len(buf) // frame_bytes
                if b == 0:
                    break  # trailing partial frame: ffmpeg truncated, EOF
                arr = np.frombuffer(buf[: b * frame_bytes],
                                    dtype=np.uint8).reshape(b, H, W, 3)
                depth_mm = infer(arr)
                enc.stdin.write(np.ascontiguousarray(depth_mm, dtype="<u2").tobytes())
                n_written += b
                n_inferred += b
        else:
            # DEDUPED PATH. Read the decoded stream in aligned order; only the
            # FIRST timestep of each run is inferred, and its depth is written
            # once per timestep in the run. Peak RAM is still `batch` frames of
            # RGB plus `batch` of depth -- runs are consumed as they close, so
            # nothing accumulates with episode length.
            pend_imgs: list[np.ndarray] = []
            pend_counts: list[int] = []
            pos = 0          # aligned position of the next frame off the pipe
            run_i = 0

            def flush() -> None:
                nonlocal n_written, n_inferred
                if not pend_imgs:
                    return
                depth_mm = infer(np.stack(pend_imgs))
                n_inferred += len(pend_imgs)
                for d, c in zip(depth_mm, pend_counts):
                    # np.repeat, not a python loop of writes: one contiguous
                    # buffer per run keeps the pipe write count down.
                    enc.stdin.write(np.ascontiguousarray(
                        np.broadcast_to(d, (c,) + d.shape), dtype="<u2").tobytes())
                    n_written += c
                pend_imgs.clear()
                pend_counts.clear()

            while True:
                buf = dec.stdout.read(frame_bytes * batch)
                if not buf:
                    break
                b = len(buf) // frame_bytes
                if b == 0:
                    break
                arr = np.frombuffer(buf[: b * frame_bytes],
                                    dtype=np.uint8).reshape(b, H, W, 3)
                for j in range(b):
                    if run_i < len(runs) and runs[run_i][0] == pos:
                        pend_imgs.append(arr[j])
                        pend_counts.append(runs[run_i][1])
                        run_i += 1
                        if len(pend_imgs) == batch:
                            flush()
                    # else: this aligned frame repeats the run's first frame.
                    # It is decoded (the stream is sequential) and discarded --
                    # its depth is the run's, written when the run flushes.
                    pos += 1
            flush()
            if run_i != len(runs) or pos != len(src_index):
                raise RuntimeError(
                    f"{rgb_mp4}: frame_source_index declares {len(src_index)} "
                    f"aligned frames in {len(runs)} runs, but the video decoded "
                    f"{pos} frames covering {run_i}. Refusing to publish a "
                    f"depth stream built against the wrong provenance row.")
        dec.stdout.close()
        if dec.wait() != 0:
            raise RuntimeError(f"ffmpeg decode failed on {rgb_mp4}:\n{_stderr_tail(dec_err)}")
        enc.stdin.close()
        if enc.wait() != 0:
            raise RuntimeError(f"ffv1 encode failed for {out_mkv}:\n{_stderr_tail(enc_err)}")
        if n_written == 0:
            raise RuntimeError(f"decoded 0 frames from {rgb_mp4}")
        if expect_frames is not None and n_written != expect_frames:
            # Do NOT publish. The temp file is removed by the handler below, so
            # the episode simply stays undone and the next run redoes it -- which
            # is the whole point of writing to a temp file in the first place.
            raise RuntimeError(
                f"depth length mismatch for {out_mkv.name}: wrote {n_written} "
                f"frames, meta/episodes.jsonl declares {expect_frames}. "
                f"Refusing to publish: a short depth stream desynchronises every "
                f"frame after the gap and only surfaces as a timestamp assertion "
                f"in a dataloader worker."
            )
        tmp.rename(out_mkv)          # atomic publish
        return n_written, n_inferred
    except BaseException:
        for p in (dec, enc):
            if p.poll() is None:
                p.kill()
        tmp.unlink(missing_ok=True)
        raise
    finally:
        dec_err.close()
        enc_err.close()


# ─────────────────────────────────────────────────────────────────────────────
# Intrinsics
# ─────────────────────────────────────────────────────────────────────────────
def K_at_grid(entry: dict, h_dst: int, w_dst: int) -> np.ndarray:
    """(3,3) K rescaled from the declared resolution to the DEPTH grid (h,w).

    HALF-PIXEL, and this is the one K site in the pipeline where that is the
    correct convention. The other three sites are endpoint because they pair with the linspace-endpoint nearest DEPTH resampler.
    Here the hop being described is a different one: the RGB is resized onto the
    depth grid by ``F.interpolate(bilinear, align_corners=False,
    antialias=True)``, which is half-pixel, so K has to follow that kernel and
    not the depth one.

    It matters more than a sub-pixel argument usually does. **DA3METRIC scales
    its output by the focal in K** — measured, doubling fx doubles the returned
    depth by exactly x2.0000 (smoke test: `focal x2 -> depth x2.0000`) — so
    getting the convention wrong here is a global multiplicative error on metric
    depth, not a shift. Worst shipping hop 1920x1200 -> 224x224 is a focal ratio
    of 0.996054, i.e. -0.395% = 3.04 mm at 0.77 m.

    Both conventions are the identity when the grid equals the declared
    resolution, which is every run that does not pass ``--depth-hw``.

        half-pixel   s = dst/src            c' = (c + 0.5) * s - 0.5
        endpoint     s = (dst-1)/(src-1)    c' = c * s
    """
    fx, fy = float(entry["fx"]), float(entry["fy"])
    cx, cy = float(entry["cx"]), float(entry["cy"])
    src_w, src_h = float(entry["width"]), float(entry["height"])
    sx = w_dst / src_w if src_w > 0 else 1.0
    sy = h_dst / src_h if src_h > 0 else 1.0
    return np.array([[fx * sx, 0.0, (cx + 0.5) * sx - 0.5],
                     [0.0, fy * sy, (cy + 0.5) * sy - 0.5],
                     [0.0, 0.0, 1.0]], dtype=np.float32)


def entry_at_grid(entry: dict, h_dst: int, w_dst: int) -> dict:
    """``entry`` rewritten onto the depth grid, same half-pixel convention.

    K describes the grid its pixels live on. Once ``--depth-hw`` moves
    the depth off the native grid, an ``episode_intrinsics.jsonl`` that still
    says NATIVE makes every later stage rescale from the wrong place:
    ``resize_shard`` rescales endpoint from the declared width/height to the
    360x640 storage grid, so the depth would really have travelled
    ``native --half-pixel--> depth_grid --endpoint--> storage`` while K claimed
    one endpoint hop. That composite is wrong in a way no downstream stage can
    detect. Writing K at the depth grid makes each later hop start from the grid
    the pixels are actually on.

    ``D`` is dropped to zeros and the model set to ``pinhole``: the frames were
    already undistorted by the builder, and carrying live coefficients onto a new
    grid would invite a second undistortion.
    """
    K = K_at_grid(entry, h_dst, w_dst)
    out = dict(entry)
    out.update({
        "fx": float(K[0, 0]), "fy": float(K[1, 1]),
        "cx": float(K[0, 2]), "cy": float(K[1, 2]),
        "width": int(w_dst), "height": int(h_dst),
        "distortion_model": "pinhole",
        "D": [0.0] * len(entry.get("D", [0.0] * 5)),
        # Provenance, so a reader can tell this table was moved off the native
        # grid and by which kernel. Ignored by every current consumer.
        "grid_source": "da3_depth_grid",
        "rescale": "half_pixel",
        "native_width": int(entry["width"]),
        "native_height": int(entry["height"]),
    })
    return out


def load_per_episode_intrinsics(ds: Path) -> dict[int, dict] | None:
    """``meta/episode_intrinsics.jsonl`` → ``{episode_index: {cam: entry}}``.

    Optional, and the same file the training loader reads. Present when the
    build spans rigs with different optics, where one averaged K would put a
    per-episode error straight into the metric scale of the depth we are about
    to write — and unlike the unprojection at train time, that error is baked
    into the stored millimetres and cannot be corrected later.
    """
    path = ds / "meta" / "episode_intrinsics.jsonl"
    if not path.exists():
        return None
    table: dict[int, dict] = {}
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        ep = int(rec["episode_index"])
        if ep in table:
            raise SystemExit(f"{path}:{line_no} repeats episode_index={ep}")
        table[ep] = rec["intrinsics"]
    if not table:
        print(f"[warn] {path} is empty; using meta/camera_intrinsics.json for every episode")
        return None
    return table


# ─────────────────────────────────────────────────────────────────────────────
# VRAM budget
# ─────────────────────────────────────────────────────────────────────────────
DA3_PATCH = 14          # DA3's ViT patch size

# Calibrated on this machine, 2026-08-22, against RESERVED memory rather than
# allocated -- reserved is what actually OOMs, and it ran 1.7-2.0x allocated
# across every point measured. 11 points over three grids:
#
#     grid        tokens    batch   allocated   reserved
#     640x480      1,610       32     16.59       27.50 GiB
#     848x480      2,135       32     21.60       28.04 GiB
#     1920x1200   11,868        4     15.39       28.33 GiB
#     1920x1200   11,868        8         OOM
#
# The cost is NOT purely proportional to batch x tokens: there is a fixed floor
# (weights plus workspace) and a per-token slope that measured 411k-611k
# bytes/token depending on grid. A single-point linear fit -- which is what this
# file shipped first, from the audit's lone "11.2 GB at batch 4" figure --
# under-predicts by ~37% and would have ALLOWED the batch 8 that OOMs above.
# So: fixed floor plus the worst measured slope, which is conservative on the
# small grids and costs almost nothing there (640x480 runs 69.4 fps at batch 16
# against 70.2 at batch 32).
VRAM_FIXED_BYTES = 2.0 * 1024**3
BYTES_PER_TOKEN = 620_000
DEFAULT_VRAM_FRACTION = 0.90    # reserved may use most of the card; it IS the
                                # allocation, not a slice of it


def tokens_per_frame(h: int, w: int) -> int:
    """DA3 tokenises on a 14x14 grid; 1920x1200 -> 86 x 138 = 11,868."""
    return -(-h // DA3_PATCH) * -(-w // DA3_PATCH)


def predicted_reserved_bytes(h: int, w: int, batch: int) -> float:
    return VRAM_FIXED_BYTES + batch * tokens_per_frame(h, w) * BYTES_PER_TOKEN


def max_batch_for(h: int, w: int, budget_bytes: float) -> int:
    room = budget_bytes - VRAM_FIXED_BYTES
    if room <= 0:
        return 1
    return max(1, int(room // (tokens_per_frame(h, w) * BYTES_PER_TOKEN)))


def vram_budget_bytes(device: str, override_gb: float | None) -> float | None:
    if override_gb:
        return override_gb * 1024**3
    try:
        import torch
        if not device.startswith("cuda") or not torch.cuda.is_available():
            return None
        i = torch.cuda.current_device()
        return torch.cuda.get_device_properties(i).total_memory * DEFAULT_VRAM_FRACTION
    except Exception:
        return None


def check_batch_size(grids: dict, batch_size: int | None, device: str,
                     override_gb: float | None, force: bool) -> int:
    """Refuse a batch the largest shipping grid cannot hold.

    `plan_grids` accepted any native grid and `--batch-size` defaulted to 16, so
    a run that was fine on 640x480 OOMed the moment it reached the first
    1920x1200 episode -- hours in, on a GPU node, with the failure looking like
    a CUDA error rather than a planning mistake.

    Returns the batch size to use. ``batch_size=None`` means AUTO: take the
    largest the budget allows, rather than making every caller carry a copy of
    the VRAM-budget model. A driver whose dataset spans several native resolutions
    cannot pass one constant that fits them all -- 640x480 allows 28 and
    1920x1200 allows 3 -- and a fixed default of 16 is simply refused at the
    larger grid, which is a run that dies at the resolution it was written for.
    An EXPLICIT batch size is still
    refused rather than silently reduced: that refusal is the point of the check.
    """
    budget = vram_budget_bytes(device, override_gb)
    worst = max(grids.values(), key=lambda hw: tokens_per_frame(*hw))
    toks = tokens_per_frame(*worst)
    if budget is None:
        if batch_size is None:
            raise SystemExit(
                f"--batch-size auto needs a VRAM budget, and none is available "
                f"for device={device!r}. Pass --vram-budget-gb, or an explicit "
                f"--batch-size.")
        print(f"[warn] no VRAM budget available (device={device!r}); batch size "
              f"{batch_size} is unchecked against the largest grid "
              f"{worst[1]}x{worst[0]} ({toks:,} tokens/frame)", flush=True)
        return batch_size
    allowed = max_batch_for(*worst, budget)
    print(f"[plan] largest grid {worst[1]}x{worst[0]} = {toks:,} tokens/frame; "
          f"budget {budget / 1024**3:.1f} GiB allows batch <= {allowed} "
          f"(predicted peak at that batch: "
          f"{predicted_reserved_bytes(*worst, allowed) / 1024**3:.1f} GiB)",
          flush=True)
    if batch_size is None:
        print(f"[plan] --batch-size auto -> {allowed}", flush=True)
        return allowed
    if batch_size <= allowed:
        return batch_size
    msg = (f"--batch-size {batch_size} does not fit: the largest grid in this "
           f"dataset is {worst[1]}x{worst[0]} ({toks:,} tokens/frame), and the "
           f"{budget / 1024**3:.1f} GB budget allows at most {allowed}. "
           f"predicted peak "
           f"{predicted_reserved_bytes(*worst, batch_size) / 1024**3:.1f} GiB. "
           f"Measured on an RTX 5090 with torch 2.10.0+cu128: 1920x1200 "
           f"reserves 28.3 GiB at batch 4 and OOMs at batch 8. Pass "
           f"--batch-size {allowed} (or --vram-budget-gb / --force-batch-size "
           f"if you know better).")
    if force:
        print(f"[warn] {msg}", flush=True)
        return batch_size
    raise SystemExit(msg)


def _same_aspect(k_hw: tuple[int, int], grid: tuple[int, int]) -> str:
    """A grid near ``grid``'s width that keeps ``k_hw``'s aspect ratio."""
    w = int(grid[1])
    h = int(round(w * k_hw[0] / k_hw[1] / 2.0)) * 2  # keep it even for the codecs
    return f"{h}x{w}"


def plan_grids(
    ds: Path, cams: list[str], intr: dict, eps: list[int], chunks_size: int,
    override_hw: tuple[int, int] | None, allow_aspect_change: bool = False,
) -> dict[str, tuple[int, int]]:
    """Per-cam depth grid + the consistency warnings that matter for geometry."""
    grids: dict[str, tuple[int, int]] = {}
    for cam in cams:
        probe_src = next(
            (p for ep in eps
             for p in [_rgb_path(ds, cam, ep, chunks_size)] if p.exists()),
            None,
        )
        if probe_src is None:
            raise SystemExit(f"no RGB episode found on disk for cam {cam!r}")
        rgb_hw = probe_hw(probe_src)
        grid = override_hw or rgb_hw
        e = intr[cam]
        k_hw = (int(e["height"]), int(e["width"]))
        # A table already moved onto a depth grid declares it. Asking
        # for a DIFFERENT grid now would rescale an already-rescaled K and, since
        # DA3METRIC multiplies by the focal, silently scale the metric depth.
        if e.get("grid_source") == "da3_depth_grid" and tuple(grid) != k_hw:
            raise SystemExit(
                f"[fatal] {cam}: meta/camera_intrinsics.json is already at a DA3 "
                f"depth grid of {k_hw[0]}x{k_hw[1]} (native was "
                f"{e.get('native_height')}x{e.get('native_width')}), but this run "
                f"asks for {grid[0]}x{grid[1]}. Rescaling an already-rescaled K "
                f"would scale the metric depth, because DA3METRIC multiplies by "
                f"the focal. Re-run with --depth-hw {k_hw[0]} {k_hw[1]}, or "
                f"reconvert the shard to get native intrinsics back.")
        if rgb_hw != k_hw:
            print(f"[warn] {cam}: RGB is {rgb_hw[0]}x{rgb_hw[1]} but "
                  f"camera_intrinsics.json declares {k_hw[0]}x{k_hw[1]}; K is "
                  f"endpoint-rescaled to the depth grid, but check that the "
                  f"intrinsics really belong to this RGB stream.")
        if abs(grid[1] / grid[0] - k_hw[1] / k_hw[0]) > 1e-3:
            # This used to warn and continue. DA3's monocular geometry
            # is learned on square-pixel imagery, so a frame squashed onto a
            # foreign aspect yields depth that is geometrically wrong and NO
            # downstream K undoes it -- the same argument that makes the stage
            # order load-bearing. A warning is not enough for an error of that
            # class, so it now refuses unless the degradation is chosen
            # explicitly.
            msg = (f"{cam}: depth grid {grid[0]}x{grid[1]} (aspect "
                   f"{grid[1] / grid[0]:.4f}) has a different aspect ratio than "
                   f"the intrinsics {k_hw[0]}x{k_hw[1]} (aspect "
                   f"{k_hw[1] / k_hw[0]:.4f}). DA3 assumes square pixels; depth "
                   f"from a stretched frame is geometrically wrong and no later "
                   f"K can undo it. For {k_hw[0]}x{k_hw[1]} a same-aspect grid "
                   f"is e.g. {_same_aspect(k_hw, grid)}. "
                   f"Pass --allow-aspect-change to accept the degradation.")
            if not allow_aspect_change:
                raise SystemExit(f"[fatal] {msg}")
            print(f"[warn] DEGRADED, --allow-aspect-change: {msg}")
        grids[cam] = grid
    return grids


# ─────────────────────────────────────────────────────────────────────────────
# meta/episode_intrinsics.jsonl + meta/camera_intrinsics.json
# ─────────────────────────────────────────────────────────────────────────────
def rewrite_intrinsics_to_grid(ds: Path, cams: list[str],
                               grids: dict[str, tuple[int, int]]) -> bool:
    """Move the intrinsics tables onto the depth grid.

    A no-op — and skipped outright — whenever every camera's depth grid already
    equals its declared resolution, which is every run that does not pass
    ``--depth-hw``. When a grid HAS moved, this is not optional: see
    ``entry_at_grid``.

    Written atomically via ``.partial`` + replace, because a half-rewritten
    intrinsics table is worse than either the old one or the new one, and a
    depth run is long enough to be interrupted.
    """
    moved = {c: grids[c] for c in cams
             if (int(_declared_hw(ds, c)[0]), int(_declared_hw(ds, c)[1])) != tuple(grids[c])}
    if not moved:
        return False
    print(f"[info] depth grid differs from the declared resolution for "
          f"{sorted(moved)}; rewriting meta/episode_intrinsics.jsonl and "
          f"meta/camera_intrinsics.json onto the depth grid", flush=True)

    ei = ds / "meta" / "episode_intrinsics.jsonl"
    if ei.exists():
        tmp = ei.with_suffix(".jsonl.partial")
        n = 0
        with tmp.open("w") as fh:
            for line in ei.read_text().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                for cam, entry in rec.get("intrinsics", {}).items():
                    if cam in grids:
                        rec["intrinsics"][cam] = entry_at_grid(entry, *grids[cam])
                fh.write(json.dumps(rec) + "\n")
                n += 1
        tmp.replace(ei)
        print(f"[info] rewrote {n} episode_intrinsics rows onto the depth grid")

    ci = ds / "meta" / "camera_intrinsics.json"
    if ci.exists():
        cur = json.loads(ci.read_text())
        for cam, entry in cur.items():
            if cam in grids:
                cur[cam] = entry_at_grid(entry, *grids[cam])
        tmp = ci.with_suffix(".json.partial")
        tmp.write_text(json.dumps(cur, indent=2))
        tmp.replace(ci)
        print("[info] rewrote camera_intrinsics.json onto the depth grid")
    return True


def _declared_hw(ds: Path, cam: str) -> tuple[int, int]:
    """(h, w) camera_intrinsics.json declares for ``cam``."""
    e = json.loads((ds / "meta" / "camera_intrinsics.json").read_text())[cam]
    return int(e["height"]), int(e["width"])


# ─────────────────────────────────────────────────────────────────────────────
# meta/info.json
# ─────────────────────────────────────────────────────────────────────────────
def patch_info_json(ds: Path, cams: list[str], chunks_size: int, eps: list[int]) -> None:
    """Register observation.depth_ffv1.<cam> features, sized from the real files.

    Shapes are probed off the written .mkv rather than assumed, so this is
    correct after a sharded run and for a mixed-resolution rig. dtype is
    "depth_video" (not "video") so LeRobot's own video loader skips these and
    RobotVideoDataset picks them up via depth_encoder/depth_ext.
    """
    info_path = ds / "meta" / "info.json"
    info = json.loads(info_path.read_text())
    fps = int(info.get("fps", 30))
    feats = info.setdefault("features", {})
    for cam in cams:
        present = [p for ep in eps
                   for p in [_depth_path(ds, cam, ep, chunks_size)] if p.exists()]
        if not present:
            print(f"[warn] no depth files for {cam}; feature not registered")
            continue
        # Sample rather than probe all N — one ffprobe per episode is slow on a
        # large dataset, and a grid mismatch comes from a mis-flagged re-run
        # (e.g. part of the dataset labeled with --depth-hw), which shows up in
        # any sample. The loader itself is robust to this (it resizes to
        # shape_meta.depth and rescales K off camera_intrinsics.json, never off
        # the on-disk grid), but the shape we declare here would be a lie for
        # some episodes — worth saying out loud.
        sample = present[:: max(1, len(present) // 8)][:8] + present[-1:]
        grids = {probe_hw(p) for p in sample}
        h, w = probe_hw(present[0])
        if len(grids) > 1:
            print(f"[warn] {cam}: depth files have MIXED grids {sorted(grids)}; "
                  f"declaring {h}x{w} (from episode {present[0].stem}). Relabel "
                  f"the odd ones with a consistent --depth-hw.")
        feats[f"{DEPTH_FEAT}.{cam}"] = {
            "dtype": "depth_video",
            "shape": [h, w, 1],
            "names": ["height", "width", "depth"],
            "info": {
                "video.height": h, "video.width": w,
                "video.codec": "ffv1", "video.pix_fmt": "gray16le",
                "video.is_depth_map": True, "video.fps": fps,
                "video.channels": 1, "has_audio": False,
                "depth_unit": "mm", "depth_dtype": "uint16",
                "depth_encoder": "ffv1", "depth_ext": DEPTH_EXT,
                "depth_packing": "gray16le",
            },
        }
        print(f"[info] registered {DEPTH_FEAT}.{cam}  shape=[{h}, {w}, 1]")
    info_path.write_text(json.dumps(info, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Add DA3 metric depth to an RGB-only LeRobot dataset.")
    ap.add_argument("--dataset-dir", required=True, type=Path)
    ap.add_argument("--da3-weights", type=Path,
                    help="DA3METRIC weights dir, e.g. .../DA3METRIC-LARGE. "
                         "Required unless --check / --patch-info-only.")
    ap.add_argument("--cams", nargs="*", default=None,
                    help="Restrict to these cameras. Default: every RGB stream on disk.")
    ap.add_argument("--depth-hw", nargs=2, type=int, metavar=("H", "W"), default=None,
                    help="Force one depth grid for all cams. Default: each cam's "
                         "own native RGB resolution (K rescale becomes exact).")
    ap.add_argument("--allow-aspect-change", action="store_true",
                    help="Accept a --depth-hw whose aspect ratio differs from "
                         "the intrinsics. DA3 assumes square pixels, so this "
                         "produces geometrically wrong depth that no downstream "
                         "K can undo. Refused by default.")
    ap.add_argument("--episodes", nargs="*", type=int, default=None,
                    help="Only these episode indices (smoke tests).")
    ap.add_argument("--batch-size", default=None,
                    type=lambda v: None if str(v).lower() == "auto" else int(v),
                    help="Frames per forward pass. DEFAULT 'auto': the "
                         "largest the VRAM budget allows. It used to be a "
                         "constant 16, which the budget REFUSES at 1920x1200 "
                         "(it allows 3) -- so any run spanning native "
                         "resolutions died at the largest one, and no constant "
                         "fits both (640x480 allows 28). "
                         "An EXPLICIT value that does not fit is still refused "
                         "rather than silently reduced; that refusal is the "
                         "point of the check.")
    ap.add_argument("--vram-budget-gb", type=float, default=None,
                    help="VRAM the depth pass may use. Default: 80%% of the "
                         "device's total. Used to refuse a --batch-size the "
                         "largest grid in the dataset cannot hold.")
    ap.add_argument("--force-batch-size", action="store_true",
                    help="Run a --batch-size the budget says will not fit.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-depth-mm", type=int, default=None,
                    help="Depth above this becomes 0 (the invalid sentinel). "
                         "Default: no cap, just the uint16 ceiling.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-label episodes that already have depth files.")
    # DEFAULT OFF, decided 2026-08-27. Every aligned frame gets its own forward
    # pass, so a repeated source packet is labelled once per timestep rather
    # than once per packet. Simpler semantics -- each timestep is an
    # independent observation -- and no dependence on
    # meta/frame_source_index.jsonl being correct.
    #
    # What it costs, measured on an ABC-130k sample: 5.09% more forward passes
    # (1.72% weighted by the full corpus, -10.2% wall clock on the resolution
    # group with 12.45% repeats), AND the repeated timesteps no longer agree.
    # The dataset build re-encodes RGB lossily, so one source packet stored at two timesteps decodes to two
    # different images -- only 2.0% of repeated pairs are byte-identical, max
    # per-pixel delta 91/255 -- and DA3 gives them depths ~10 mm apart. That is
    # inside DA3's own 8-9% absolute error, which is why this is a defensible
    # choice rather than a regression.
    ap.add_argument("--dedup-source-frames", action="store_true",
                    help="Run the model once per SOURCE packet instead of once "
                         "per aligned frame, using meta/frame_source_index.jsonl. "
                         "OFF by default. Changes nothing structural (same frame "
                         "count, order, dtype, sentinel); the repeated timesteps "
                         "then carry ONE depth map instead of two computed from "
                         "two lossy re-encodings of the same source frame. "
                         "Saves ~5% of forward passes on this data.")
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--shard-id", type=int, default=0)
    ap.add_argument("--patch-info-only", action="store_true",
                    help="Only register the depth features in meta/info.json. "
                         "Run ONCE after all shards finish.")
    ap.add_argument("--check", action="store_true",
                    help="Validate the dataset and print the plan; no GPU, no writes.")
    args = ap.parse_args()

    ds = args.dataset_dir.resolve()
    info_path = ds / "meta" / "info.json"
    intr_path = ds / "meta" / "camera_intrinsics.json"
    if not info_path.exists():
        raise SystemExit(f"not a LeRobot dataset (no meta/info.json): {ds}")
    if not intr_path.exists():
        raise SystemExit(
            f"missing {intr_path}. DA3METRIC needs fx/fy per camera to produce "
            f"metric depth; there is no safe default.")
    info = json.loads(info_path.read_text())
    intr = json.loads(intr_path.read_text())
    ep_intr = load_per_episode_intrinsics(ds)
    fps = int(info.get("fps", 30))
    chunks_size = int(info.get("chunks_size", 1000))

    cams = discover_cams(ds, intr, args.cams)
    all_eps = list_episodes(ds, cams)
    # Declared lengths, so a short depth write is refused rather than
    # published. Empty when meta/episodes.jsonl is absent, in which
    # case there is nothing to check against and the write proceeds.
    ep_lengths = episode_lengths(ds)
    if not ep_lengths:
        print("[warn] no meta/episodes.jsonl lengths: depth writes cannot "
              "be length-checked", flush=True)
    if args.episodes is not None:
        missing = sorted(set(args.episodes) - set(all_eps))
        if missing:
            raise SystemExit(f"requested episodes not present: {missing}")
        all_eps = sorted(set(args.episodes))
    if not all_eps:
        raise SystemExit("no episodes found")

    # A dataset that already declares depth is either (a) not depth-free at all
    # — real sensor depth we must not clobber — or (b) a previous run of THIS
    # script that patched info.json before some episodes failed, and is now being
    # resumed. Both are handled by the per-camera file-existence skip below, so
    # this is a loud warning rather than a hard stop: without --overwrite no
    # existing depth file is ever touched.
    already = [k for k in info.get("features", {}) if k.startswith("observation.depth")]
    if already and not (args.patch_info_only or args.check):
        print(f"[warn] meta/info.json already declares depth features: {already}. "
              f"Existing depth files will be "
              f"{'OVERWRITTEN (--overwrite)' if args.overwrite else 'left untouched'}.")

    # `and not args.check`: --check's help promises "no GPU, no writes", but
    # this branch was tested FIRST and returns, so `--check --patch-info-only`
    # rewrote meta/info.json and reported success. run.sh's CHECK branch execs
    # before its PATCH_INFO_ONLY branch, so the env-var route was always safe --
    # the exposure is `CHECK=1 bash run.sh <ds> --patch-info-only`, where the
    # trailing argv is forwarded into the --check exec.
    if args.patch_info_only and not args.check:
        patch_info_json(ds, cams, chunks_size, all_eps)
        # The sharded finalizer, so it owns the same one-shot rewrite the
        # single-shard path does. Grids are re-derived from the intrinsics the
        # shards ran against; if they already moved, `moved` is empty and this
        # is idempotent.
        rewrite_intrinsics_to_grid(
            ds, cams,
            plan_grids(ds, cams, intr, all_eps, chunks_size,
                       tuple(args.depth_hw) if args.depth_hw else None,
                       args.allow_aspect_change))
        return

    grids = plan_grids(ds, cams, intr, all_eps, chunks_size,
                       tuple(args.depth_hw) if args.depth_hw else None,
                       args.allow_aspect_change)
    batch_size = check_batch_size(grids, args.batch_size, args.device,
                                  args.vram_budget_gb, args.force_batch_size)
    Ks = {c: K_at_grid(intr[c], *grids[c]) for c in cams}

    def K_for(ep: int, cam: str) -> np.ndarray:
        """The K this episode's depth is scaled by. Per-episode when available."""
        if ep_intr is None:
            return Ks[cam]
        entry = ep_intr.get(ep)
        if entry is None:
            raise SystemExit(
                f"meta/episode_intrinsics.jsonl has no entry for episode {ep}. "
                f"Rebuild the table or delete it to fall back to "
                f"meta/camera_intrinsics.json for every episode."
            )
        if cam not in entry:
            raise SystemExit(
                f"meta/episode_intrinsics.jsonl episode {ep} has no '{cam}' "
                f"(has: {sorted(entry)})"
            )
        return K_at_grid(entry[cam], *grids[cam])

    # Source-frame provenance (meta/frame_source_index.jsonl, written by the
    # dataset builder if it resamples). Absent
    # on datasets converted before it existed; then every aligned frame is
    # labelled, exactly as before.
    src_tbl = load_frame_source_index(ds) if args.dedup_source_frames else None

    def src_index_for(ep: int, cam: str) -> list[int] | None:
        """This (episode, camera)'s source-packet index per aligned timestep.

        A table that exists but has no row for an episode is an ERROR, not a
        reason to fall back: it means the table and the episodes are out of
        step, and labelling that episode undeduped would hide it.
        """
        if src_tbl is None:
            return None
        row = src_tbl.get(ep)
        if row is None:
            raise SystemExit(
                f"meta/frame_source_index.jsonl has no entry for episode {ep}. "
                f"The table is out of step with the episodes; rebuild it or "
                f"delete it to label every aligned frame.")
        if cam not in row:
            raise SystemExit(
                f"meta/frame_source_index.jsonl episode {ep} has no '{cam}' "
                f"(has: {sorted(row)})")
        return row[cam]

    print(f"[plan] {ds}")
    print(f"[plan] {len(all_eps)} episodes, fps={fps}, chunks_size={chunks_size}")
    if src_tbl is None:
        print("[plan] source-frame dedup: OFF — every aligned frame is labelled"
              + ("" if args.dedup_source_frames
                 else " (default; pass --dedup-source-frames to enable)"))
    else:
        tot = sum(len(v) for r in src_tbl.values() for v in r.values())
        uniq = sum(len(runs_from_source_index(v))
                   for r in src_tbl.values() for v in r.values())
        print(f"[plan] source-frame dedup: ON — {uniq:,} unique source frames "
              f"for {tot:,} aligned frames over {len(src_tbl)} episodes "
              f"({100 * (1 - uniq / tot) if tot else 0:.2f}% fewer forward passes)")
    if ep_intr is None:
        print("[plan] intrinsics: ONE K per camera for the whole dataset "
              "(no meta/episode_intrinsics.jsonl)")
    else:
        print(f"[plan] intrinsics: PER EPISODE, {len(ep_intr)} entries")
    for c in cams:
        h, w = grids[c]
        K = Ks[c]
        print(f"[plan] {c:>18}: depth grid {h}x{w}  K@grid fx={K[0,0]:.2f} "
              f"fy={K[1,1]:.2f} cx={K[0,2]:.2f} cy={K[1,2]:.2f}  "
              f"focal={(K[0,0]+K[1,1])/2:.2f}")
        if ep_intr is not None:
            # DA3METRIC multiplies its output by this focal, so its spread over
            # the dataset IS the spread of the metric-scale error that a single
            # averaged K would have introduced. Worth seeing before a long run.
            focals = [float((k[0, 0] + k[1, 1]) / 2)
                      for k in (K_for(e, c) for e in all_eps)]
            lo, hi, mean_f = min(focals), max(focals), sum(focals) / len(focals)
            print(f"[plan] {'':>18}  per-episode focal: min={lo:.2f} max={hi:.2f} "
                  f"mean={mean_f:.2f}  spread={100 * (hi - lo) / mean_f:.1f}% of mean")
    if args.check:
        done = sum(1 for ep in all_eps for c in cams
                   if _depth_path(ds, c, ep, chunks_size).exists())
        print(f"[check] {done}/{len(all_eps) * len(cams)} (episode, cam) pairs "
              f"already labeled. OK — nothing written.")
        return

    if args.da3_weights is None:
        raise SystemExit("--da3-weights is required (unless --check / --patch-info-only)")
    if not (0 <= args.shard_id < args.num_shards):
        raise SystemExit(f"--shard-id must be in [0, {args.num_shards})")
    # Interleaved so every shard gets a uniform mix of short and long episodes.
    ep_idxs = all_eps[args.shard_id::args.num_shards]
    tag = f"shard {args.shard_id}/{args.num_shards}"
    print(f"[{tag}] {len(ep_idxs)}/{len(all_eps)} episodes", flush=True)

    module = DA3DepthModule(str(args.da3_weights)).to(args.device).eval()
    print(f"[{tag}] DA3 loaded: is_da3metric={module.is_da3metric} "
          f"is_nested={module.is_nested}", flush=True)
    if not module.is_da3metric and not module.is_nested:
        print("[warn] weights are neither DA3METRIC nor DA3NESTED — output is "
              "relative depth, NOT metric millimetres.", flush=True)

    fail_log = ds / "meta" / f"depth_failures_shard{args.shard_id}.txt"
    done = skipped = failed = 0
    tot_aligned = tot_inferred = 0
    for n, ep in enumerate(ep_idxs):
        try:
            wrote_any = False
            for cam in cams:
                out = _depth_path(ds, cam, ep, chunks_size)
                if out.exists() and not args.overwrite:
                    continue
                src = _rgb_path(ds, cam, ep, chunks_size)
                if not src.exists():
                    raise FileNotFoundError(src)
                n_f, n_inf = stream_rgb_to_depth(
                    src, out, fps, K_for(ep, cam), grids[cam], module,
                    batch_size, args.device, args.max_depth_mm,
                    expect_frames=ep_lengths.get(ep),
                    src_index=src_index_for(ep, cam))
                wrote_any = True
                tot_aligned += n_f
                tot_inferred += n_inf
                if n == 0:
                    print(f"[{tag}] ep{ep} {cam}: {n_f} frames "
                          f"({n_inf} forward passes) -> {out.name}", flush=True)
            done += wrote_any
            skipped += (not wrote_any)
        except Exception as e:      # one bad episode must not kill a long run
            failed += 1
            with open(fail_log, "a") as fh:
                fh.write(f"ep{ep}\t{type(e).__name__}: {e}\n")
            print(f"[FAIL {tag}] ep{ep}: {type(e).__name__}: {e}", flush=True)
        if n % 50 == 0:
            print(f"[{tag}] {n}/{len(ep_idxs)} (done={done} skip={skipped} "
                  f"fail={failed})", flush=True)

    print(f"[{tag}] FINISHED: done={done} skipped={skipped} failed={failed}"
          f"{' — see ' + str(fail_log) if failed else ''}", flush=True)
    if tot_aligned:
        saved = tot_aligned - tot_inferred
        print(f"[{tag}] depth-model images: {tot_inferred:,} for {tot_aligned:,} "
              f"aligned frames"
              + (f" — {saved:,} repeated source frames not re-inferred "
                 f"({100 * saved / tot_aligned:.2f}%)" if saved else
                 " — no source frame repeated"), flush=True)
    if args.num_shards == 1:
        patch_info_json(ds, cams, chunks_size, all_eps)
        rewrite_intrinsics_to_grid(ds, cams, grids)
    else:
        print("[info] sharded run — after ALL shards, finalize once with "
              "--patch-info-only", flush=True)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
