"""Streaming recorder for the predicted observation rollout at deployment.

Follows the row order of the training-time eval video
(``Wan22Trainer._run_eval_and_log``) and adds a third column to every row:
the chunk's present observation (``gt``), or for the PCA rows its feature
norm (``gt_norm``). For a checkpoint with a pointmap stream
(``enable_pointmap``), the saved video stacks **six rows** top → bottom,
each row three ``composite_hw`` tiles side-by-side (384×320 for the 3-cam T):

  1. RGB             : pred | vae_recon | gt
  2. DINO            : pred_pca | gt_pca | gt_norm
  3. Pointmap XYZ    : pred | vae_recon | gt
  4. Pointmap depth  : pred | vae_recon | gt    (turbo)
  5. Pointmap VAE-PCA: pred_pca | gt_pca | gt_norm
  6. RGB VAE-PCA     : pred_pca | gt_pca | gt_norm

GT (and vae_recon) for each chunk = the chunk's own present input,
frozen across the chunk's predicted frame count.

**PCA color stability**: the three PCA rows (2, 5, 6) would otherwise
fit a fresh PCA basis + per-frame min/max every chunk, so the same
feature maps to different colors across chunks. To fix this, the
recorder buffers the first ``pca_warmup_chunks`` chunks (default 3),
fits ONE PCA basis + ONE global min/max per modality over those chunks,
then renders every chunk (the buffered ones and all later ones) with
those fixed stats. Colors are then consistent across the whole session.
The cost is a small bounded buffer (~3 chunks) and the first few frames
are written once the warmup window closes.

``split`` optionally writes companion MP4s next to the panel, so rows and
columns can be cut independently in an editor:

  * ``"row"``  → one file per row: ``rgb.mp4``, ``dino.mp4``,
    ``pointmap_xyz.mp4``, ``pointmap_depth.mp4``, ``pointmap_vae_pca.mp4``,
    ``rgb_vae_pca.mp4``
  * ``"tile"`` → one file per cell: ``<row>__<col>.mp4``, e.g.
    ``rgb__pred.mp4`` / ``rgb__vae_recon.mp4`` / ``rgb__gt.mp4`` and
    ``dino__pred_pca.mp4`` / ``dino__gt_pca.mp4`` / ``dino__gt_norm.mp4``

``prediction.mp4`` is always written regardless. Split files are cut from
the native-width rows (before the panel's common-width resize), so they are
never resampled, and they share the panel's fps, frame budget, and fixed
PCA stats — same frames, same colors, just separated.

**Anchor frame** (``drop_anchor``, default True): decoded frame 0 of every
chunk is the re-clamped clean reconstruction of the CURRENT real
observation — the same instant as the previous chunk's last frame, and the
one frame whose background snaps from the imagined (training-style) scene
back to reality. Dropping it leaves only the imagined frames, which tile
seamlessly across chunks: 8 frames spaced ``action_video_freq_ratio``
actions apart, i.e. exactly one chunk period at ``action_hz /
action_video_freq_ratio`` fps (30/4 = 7.5 for the YAM defaults).

**Modes** (``mode``):

  * ``"raw"`` — deployment mode. ``append_chunk`` does NO rendering: it
    CPU-copies the pred latents + present inputs, hands them to a
    background writer thread (``torch.save`` →
    ``<session>/raw_chunks/chunk_%05d.pt``), and returns.
    ``scripts/render_prediction_dump.py`` renders the dump afterwards;
    ``meta.json`` records the render settings chosen at deploy time.
  * ``"live"`` — legacy in-loop rendering (VAE decode + DINO encode + PCA
    + mp4 encode per chunk, all on the caller's thread). Also what the
    offline renderer uses internally.

The real-world observation stream at full action rate is recorded
*client-side* (the robot client's ``record_observation_dir``).
"""

from __future__ import annotations

import json
import logging
import os
import queue
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import imageio
import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


# Column labels per row, in left→right order — the names the split files get.
# PCA-based rows carry (pred_pca | gt_pca | gt_norm); the rest are the plain
# (pred | vae_recon | gt) triple built by ``_row_3col_uint8``.
_COLS_PRED_RECON_GT = ("pred", "vae_recon", "gt")
_COLS_PCA = ("pred_pca", "gt_pca", "gt_norm")
# Depth-less deploys (no depth on the wire, present_pointmap=False) get one
# combined row instead of the three pointmap rows: there is no GT depth to
# build the vae_recon/gt columns from, but the model still imagines the
# pointmap cross-modally, so the prediction alone is worth showing. See
# ``_render_pointmap_pred_only_row``.
_COLS_POINTMAP_PRED = ("xyz", "depth", "unused")
_ROW_COLS = {
    "rgb": _COLS_PRED_RECON_GT,
    "dino": _COLS_PCA,
    "pointmap_xyz": _COLS_PRED_RECON_GT,
    "pointmap_depth": _COLS_PRED_RECON_GT,
    "pointmap_vae_pca": _COLS_PCA,
    "pointmap_pred": _COLS_POINTMAP_PRED,
    "rgb_vae_pca": _COLS_PCA,
}


# ----------------------------------------------------------------------
# Frame helpers
# ----------------------------------------------------------------------

def _to_even(frame: np.ndarray) -> np.ndarray:
    """libx264 requires even H/W. Edge-pad if needed."""
    h, w = frame.shape[:2]
    pad_h = h % 2
    pad_w = w % 2
    if pad_h == 0 and pad_w == 0:
        return frame
    return np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")


def _resize_row_to_width(row_np: np.ndarray, target_w: int) -> np.ndarray:
    """Resize ``[F, H, W, 3]`` uint8 to ``target_w`` along W, bilinear, keep aspect."""
    if row_np.shape[2] == target_w:
        return row_np
    src_h, src_w = row_np.shape[1:3]
    target_h = max(int(round(src_h * target_w / src_w)), 1)
    out = np.empty((row_np.shape[0], target_h, target_w, 3), dtype=row_np.dtype)
    for t in range(row_np.shape[0]):
        out[t] = np.array(
            Image.fromarray(row_np[t]).resize((target_w, target_h), Image.BILINEAR)
        )
    return out


def _pil_frames_to_tensor01(frames: List[Image.Image]) -> torch.Tensor:
    """List[PIL RGB] → ``[3, F, H, W]`` float32 in ``[0, 1]``."""
    arrs = [np.asarray(f.convert("RGB")) for f in frames]
    arr = np.stack(arrs, axis=0).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(3, 0, 1, 2).contiguous()


def _expand_f(t: torch.Tensor, target_f: int) -> torch.Tensor:
    """Broadcast a length-1 frame axis (dim=1) to ``target_f``."""
    if t.shape[1] == target_f:
        return t
    if t.shape[1] != 1:
        idx = torch.from_numpy(
            np.clip(np.arange(target_f) * t.shape[1] // target_f, 0, t.shape[1] - 1)
        )
        return t.index_select(1, idx)
    return t.expand(-1, target_f, *([-1] * (t.ndim - 2))).contiguous()


def _row_3col_uint8(
    pred: torch.Tensor, vae_recon: torch.Tensor, gt: torch.Tensor,
) -> np.ndarray:
    """Stitch ``[pred | vae_recon | gt]`` along W per frame → ``[F, H, 3W, 3]`` uint8."""
    if not (pred.shape == vae_recon.shape == gt.shape):
        raise ValueError(
            f"_row_3col_uint8: shape mismatch  pred={tuple(pred.shape)}  "
            f"recon={tuple(vae_recon.shape)}  gt={tuple(gt.shape)}"
        )
    stitched = torch.cat([pred, vae_recon, gt], dim=-1).clamp(0.0, 1.0)
    arr = (stitched * 255.0).to(torch.uint8).cpu().numpy()
    return np.transpose(arr, (1, 2, 3, 0))


def _match_f(row_np: np.ndarray, target_f: int) -> np.ndarray:
    """Hold-last / subsample ``[F, H, W, 3]`` along F to match ``target_f``."""
    F = row_np.shape[0]
    if F == target_f:
        return row_np
    if F == 0:
        raise ValueError("_match_f: zero-frame row")
    indices = np.clip(np.arange(target_f) * F // target_f, 0, F - 1)
    return row_np[indices]


def _row_window(row_np: np.ndarray, f_dec: int, start: int, keep: int) -> np.ndarray:
    """Frames of ``row_np`` covering decoded-frame window [start, start+keep).

    Rows on the decoded grid (F == f_dec) slice directly. Rows on the coarser
    latent grid (the DINO / VAE-PCA rows: F = F_lat frames for f_dec decoded
    frames, causal ``1 + s·(F-1)`` layout with stride ``s``) map each decoded
    slot ``t`` to its causal source frame ``ceil(t / s)`` — so source frame 0
    (the clean anchor latent) leaves the window together with decoded frame 0
    when the anchor is dropped, instead of surviving as leading duplicates
    (the ``_match_f``-then-slice pitfall).
    """
    F = row_np.shape[0]
    if F == f_dec:
        return row_np[start:start + keep]
    if F <= 1:
        return _match_f(row_np, keep)
    s = max((f_dec - 1) // (F - 1), 1)
    idx = np.clip((np.arange(start, start + keep) + s - 1) // s, 0, F - 1)
    return row_np[idx]


def _unfold_dino_lat(
    lat: torch.Tensor, cam_patches: List[Tuple[int, int]], f: int,
) -> torch.Tensor:
    """Invert the encoder's lossless pixel-unshuffle fold for visualization.

    ``[B, C·f², F, N, 1]`` folded tokens → ``[B, C, F, N·f², 1]`` native-grid
    tokens. The token axis concatenates per-cam ``g_h×g_w`` grids (row-major,
    ``cam_patches`` order); channels carry ``F.pixel_unshuffle``'s ``(C, f, f)``
    sub-position layout (dino_encoder._pixel_unshuffle_patches), so
    ``F.pixel_shuffle`` recovers the true ``g·f × g·f`` DINO feature map
    exactly — e.g. 3072ch @ 7×7 → 768ch @ 14×14 per cam.
    """
    B, cf2, n_f, n_tok, _ = lat.shape
    if cf2 % (f * f) != 0:
        raise ValueError(f"_unfold_dino_lat: C={cf2} not divisible by f²={f * f}")
    if n_tok != sum(gh * gw for gh, gw in cam_patches):
        raise ValueError(
            f"_unfold_dino_lat: N={n_tok} != sum of cam grids {cam_patches}"
        )
    c_native = cf2 // (f * f)
    outs = []
    off = 0
    for gh, gw in cam_patches:
        n = gh * gw
        cam = lat[:, :, :, off:off + n, 0]                       # [B, C·f², F, n]
        off += n
        cam = cam.permute(0, 2, 1, 3).reshape(B * n_f, cf2, gh, gw)
        cam = torch.nn.functional.pixel_shuffle(cam, f)          # [B·F, C, gh·f, gw·f]
        cam = cam.reshape(B, n_f, c_native, n * f * f).permute(0, 2, 1, 3)
        outs.append(cam)
    return torch.cat(outs, dim=3).unsqueeze(-1).contiguous()


def _dino_unfold_params(model) -> Tuple[int, List[Tuple[int, int]]]:
    """(fold factor, POST-fold cam grids) from the model; (0, []) = no fold."""
    f = int(getattr(model, "_dino_pixel_unshuffle", 0) or 0)
    patches = [tuple(p) for p in getattr(model, "dino_cam_patches", []) or []]
    return (f, patches) if f > 1 and patches else (0, [])


# ----------------------------------------------------------------------
# PCA stats (fixed over the warmup window)
# ----------------------------------------------------------------------

def _cpu(t: Optional[torch.Tensor]):
    if t is None:
        return None
    return t.detach().to("cpu")


def _flatten_feat(t: torch.Tensor) -> np.ndarray:
    """Flatten a latent/feature tensor to ``[M, C]`` numpy for PCA fitting.

    Accepts ``[B,C,F,N,1]`` (DINO tokens) or ``[B,C,F,H,W]`` (VAE latents),
    or their batch-dropped forms. ``M`` = F·N or F·H·W.
    """
    if t.ndim == 5:
        t = t[0]  # drop batch → [C,F,N,1] or [C,F,H,W]
    if t.ndim == 4 and t.shape[-1] == 1:  # [C,F,N,1]
        t = t[..., 0]                      # [C,F,N]
    if t.ndim == 3:                        # [C,F,N]
        C, F, N = t.shape
        return t.permute(1, 2, 0).reshape(F * N, C).float().cpu().numpy()
    C, F, H, W = t.shape                   # [C,F,H,W]
    return t.permute(1, 2, 3, 0).reshape(F * H * W, C).float().cpu().numpy()


def _fit_pca_stats(feat_list: List[np.ndarray]) -> Optional[dict]:
    """Fit a fixed PCA basis + global min/max from a list of ``[M_i, C]`` arrays.

    Returns ``{"basis": (mean[1,C], vt[3,C]), "range": (pca_min[3], pca_max[3],
    norm_min, norm_max)}`` — the format the patched ``visualize_vae_latent`` /
    ``_compute_dino_pca_and_norm`` accept via their ``pca_basis`` + ``value_range``
    args. Returns None if no features.
    """
    feat_list = [f for f in feat_list if f is not None and f.size > 0]
    if not feat_list:
        return None
    allf = np.concatenate(feat_list, axis=0).astype(np.float32)  # [sumM, C]
    mean = allf.mean(axis=0, keepdims=True)
    _, _, vt_full = np.linalg.svd(allf - mean, full_matrices=False)
    vt = vt_full[:3]
    proj = (allf - mean) @ vt.T            # [sumM, 3]
    pca_min = proj.min(axis=0)             # [3]
    pca_max = proj.max(axis=0)
    norm = np.sqrt(np.mean(allf * allf, axis=-1))
    return {
        "basis": (mean, vt),
        "range": (pca_min, pca_max, float(norm.min()), float(norm.max())),
    }


# ----------------------------------------------------------------------
# Recorder
# ----------------------------------------------------------------------

class PredictionRecorder:
    """Writes one ``prediction.mp4`` per session with color-stable PCA rows."""

    def __init__(
        self,
        session_dir: Path,
        fps: float = 7.5,
        frames_per_chunk: int = 0,
        pca_warmup_chunks: int = 3,
        split: str = "none",
        mode: str = "live",
        drop_anchor: bool = True,
        ckpt_path: Optional[str] = None,
        write_meta: bool = True,
        policy: str = "yam",
    ) -> None:
        if split not in ("none", "row", "tile"):
            raise ValueError(f"split must be none|row|tile; got {split!r}")
        if mode not in ("live", "raw"):
            raise ValueError(f"mode must be live|raw; got {mode!r}")
        self._session_dir = Path(session_dir)
        self._session_dir.mkdir(parents=True, exist_ok=True)
        # Float so the deploy-correct rate (action_hz / action_video_freq_ratio,
        # e.g. 30/4 = 7.5) is expressible — an int cast would silently round it.
        self._fps = max(float(fps), 1e-3)
        self._mp4_path = self._session_dir / "prediction.mp4"
        self._closed = False
        self._frames_per_chunk = int(frames_per_chunk)
        self._warmup_chunks = max(1, int(pca_warmup_chunks))
        self._mode = mode
        self._drop_anchor = bool(drop_anchor)
        # Per-row / per-tile companion MP4s, opened lazily on first use (which
        # rows exist depends on the ckpt's modalities, known only once a chunk
        # has been rendered). Keyed by file stem.
        self._split = split
        self._split_writers: Dict[str, Any] = {}
        # meta.json: the render settings chosen at deploy time — the offline
        # renderer defaults to these, so a plain re-render reproduces exactly
        # what live mode would have produced. The renderer itself passes
        # write_meta=False so an in-place render can never clobber the
        # deploy-time meta.json it is reading its defaults from.
        self._write_meta_enabled = bool(write_meta)
        self._meta: Dict[str, Any] = {
            # Which deploy_policy builder the offline renderer must use to
            # reload the ckpt. Absent in dumps written before this field existed,
            # which render_prediction_dump.py reads as "yam".
            "policy": str(policy),
            "mode": mode,
            "fps": self._fps,
            "frames_per_chunk": self._frames_per_chunk,
            "drop_anchor": self._drop_anchor,
            "split": split,
            "pca_warmup_chunks": self._warmup_chunks,
            "ckpt_path": str(ckpt_path) if ckpt_path else None,
            "n_chunks": None,
        }

        if mode == "raw":
            # No render, no mp4 writers. append_chunk() enqueues CPU copies;
            # the daemon thread serializes them so disk I/O never blocks the
            # control loop. torch.save (pickle) keeps torch tensors verbatim.
            self._writer = None
            self._raw_dir = self._session_dir / "raw_chunks"
            self._raw_dir.mkdir(parents=True, exist_ok=True)
            self._raw_queue: "queue.Queue" = queue.Queue()
            self._max_qdepth = 0
            self._raw_thread = threading.Thread(
                target=self._raw_writer_loop, daemon=True, name="pred-raw-writer",
            )
            self._raw_thread.start()
            self._write_meta()
        else:
            self._writer = self._open_writer(self._mp4_path)

        # Warmup state: buffer raw per-chunk data until we've seen
        # ``_warmup_chunks`` of them, then fit fixed PCA stats and render.
        self._warmup_buf: List[Dict[str, Any]] = []
        self._stats_ready = False
        self._dino_stats: Optional[dict] = None
        self._ptvae_stats: Optional[dict] = None
        self._rgbvae_stats: Optional[dict] = None
        self._model: Any = None  # captured for close()-time flush

        self._n_chunks = 0
        self._n_frames = 0
        logger.info(
            "[PredictionRecorder] opened %s  (mode=%s, fps=%g, frames_per_chunk=%s, "
            "drop_anchor=%s, pca_warmup_chunks=%d, split=%s)",
            self._raw_dir if mode == "raw" else self._mp4_path, mode, self._fps,
            self._frames_per_chunk if self._frames_per_chunk > 0 else "all",
            self._drop_anchor, self._warmup_chunks, self._split,
        )

    def _open_writer(self, path: Path):
        return imageio.get_writer(
            str(path),
            fps=self._fps,
            codec="libx264",
            format="FFMPEG",
            pixelformat="yuv444p",
        )

    def _write_meta(self) -> None:
        if not self._write_meta_enabled:
            return
        try:
            with open(self._session_dir / "meta.json", "w") as f:
                json.dump(self._meta, f, indent=2)
        except Exception as e:  # noqa: BLE001
            logger.warning("[PredictionRecorder] meta.json write failed: %r", e)

    def _raw_writer_loop(self) -> None:
        while True:
            item = self._raw_queue.get()
            if item is None:
                return
            idx, payload = item
            # tmp + atomic rename: if the daemon thread dies mid-write (crash,
            # or close()'s join timeout followed by process exit) the final
            # chunk_*.pt name is never left truncated — a half-written file
            # would satisfy the renderer's glob and abort the whole render.
            path = self._raw_dir / f"chunk_{idx:05d}.pt"
            tmp = self._raw_dir / f".chunk_{idx:05d}.pt.tmp"
            try:
                torch.save(payload, tmp)
                os.replace(tmp, path)
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "[PredictionRecorder] raw chunk %d save failed: %r", idx, e,
                )
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass

    @property
    def path(self) -> Path:
        return self._mp4_path

    @property
    def n_chunks(self) -> int:
        return self._n_chunks

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def append_chunk(
        self,
        model,
        pred: Dict[str, Optional[torch.Tensor]],
        present: Dict[str, Any],
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        if self._closed:
            raise RuntimeError("PredictionRecorder.append_chunk after close()")

        if self._mode == "raw":
            # Dump unconditionally (even video_latents=None regimes — data
            # preservation; the offline renderer skips what it can't draw).
            # The CPU copies here are the only critical-path cost; disk I/O
            # happens on the writer thread.
            payload = {
                "pred": {
                    k: _cpu(pred.get(k))
                    for k in ("video_latents", "dino_latents", "pointmap_latents")
                },
                "present": {
                    "input_image": _cpu(present.get("input_image")),
                    "per_cam": {
                        k: _cpu(v) for k, v in (present.get("per_cam") or {}).items()
                    },
                    "per_cam_depth": {
                        k: _cpu(v)
                        for k, v in (present.get("per_cam_depth") or {}).items()
                    },
                    "camera_intrinsics": _cpu(present.get("camera_intrinsics")),
                },
                "meta": {**(meta or {}), "chunk_index": self._n_chunks},
            }
            self._raw_queue.put((self._n_chunks, payload))
            self._max_qdepth = max(self._max_qdepth, self._raw_queue.qsize())
            self._n_chunks += 1
            return

        if pred.get("video_latents") is None:
            logger.warning(
                "[PredictionRecorder] video_latents=None (no-video regime); "
                "skipping chunk."
            )
            return

        self._model = model
        cd = self._stash(pred, present)

        if self._stats_ready:
            self._render_and_write(model, cd)
            return

        # Warmup: buffer until we can fit stable PCA stats, then drain.
        self._warmup_buf.append(cd)
        if len(self._warmup_buf) >= self._warmup_chunks:
            self._finalize_stats()
            for buffered in self._warmup_buf:
                self._render_and_write(model, buffered)
            self._warmup_buf.clear()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        if self._mode == "raw":
            self._raw_queue.put(None)
            self._raw_thread.join(timeout=60.0)
            if self._raw_thread.is_alive():
                logger.warning(
                    "[PredictionRecorder] raw writer thread still alive after "
                    "60 s; trailing chunks may be missing on disk.",
                )
            self._meta["n_chunks"] = self._n_chunks
            self._meta["max_queue_depth"] = self._max_qdepth
            self._write_meta()
            logger.info(
                "[PredictionRecorder] raw dump closed %s  chunks=%d  "
                "max_queue_depth=%d  — render with "
                "scripts/render_prediction_dump.py --dump-dir %s",
                self._raw_dir, self._n_chunks, self._max_qdepth, self._session_dir,
            )
            return

        # Session ended before the warmup window filled — fit from whatever
        # we have so the short session still gets consistent colors.
        if not self._stats_ready and self._warmup_buf and self._model is not None:
            try:
                self._finalize_stats()
                for buffered in self._warmup_buf:
                    self._render_and_write(self._model, buffered)
            except Exception as e:  # noqa: BLE001
                logger.warning("[PredictionRecorder] warmup flush failed: %r", e)
            self._warmup_buf.clear()
        try:
            self._writer.close()
        except Exception as e:  # noqa: BLE001
            logger.warning("[PredictionRecorder] writer close failed: %r", e)
        for stem, writer in self._split_writers.items():
            try:
                writer.close()
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "[PredictionRecorder] split writer %s close failed: %r", stem, e,
                )
        self._meta["n_chunks"] = self._n_chunks
        self._write_meta()
        logger.info(
            "[PredictionRecorder] closed %s  chunks=%d  frames=%d  split_files=%d",
            self._mp4_path, self._n_chunks, self._n_frames, len(self._split_writers),
        )

    def __enter__(self) -> "PredictionRecorder":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Buffering + stats
    # ------------------------------------------------------------------

    @staticmethod
    def _stash(
        pred: Dict[str, Optional[torch.Tensor]],
        present: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Copy the raw tensors needed to render this chunk later (CPU)."""
        return {
            "video_lat": _cpu(pred.get("video_latents")),
            "dino_lat": _cpu(pred.get("dino_latents")),
            "pt_lat": _cpu(pred.get("pointmap_latents")),
            "input_image": _cpu(present.get("input_image")),
            "per_cam": {k: _cpu(v) for k, v in (present.get("per_cam") or {}).items()},
            "per_cam_depth": {
                k: _cpu(v) for k, v in (present.get("per_cam_depth") or {}).items()
            },
            "K": _cpu(present.get("camera_intrinsics")),
        }

    def _finalize_stats(self) -> None:
        """Fit fixed PCA stats per modality from the buffered warmup chunks.

        DINO features are unfolded to the native grid first (7×7·3072ch →
        14×14·768ch) so the basis lives in the same 768-dim space
        ``_render_dino_row`` projects at render time.
        """
        f, folded_patches = _dino_unfold_params(self._model)
        dino_feats = [
            _flatten_feat(
                _unfold_dino_lat(cd["dino_lat"], folded_patches, f)
                if f else cd["dino_lat"]
            )
            for cd in self._warmup_buf
            if cd["dino_lat"] is not None
        ]
        pt_feats = [
            _flatten_feat(cd["pt_lat"]) for cd in self._warmup_buf
            if cd["pt_lat"] is not None
        ]
        rgb_feats = [
            _flatten_feat(cd["video_lat"]) for cd in self._warmup_buf
            if cd["video_lat"] is not None
        ]
        self._dino_stats = _fit_pca_stats(dino_feats)
        self._ptvae_stats = _fit_pca_stats(pt_feats)
        self._rgbvae_stats = _fit_pca_stats(rgb_feats)
        self._stats_ready = True
        logger.info(
            "[PredictionRecorder] PCA stats fixed from %d warmup chunk(s) "
            "(dino=%s ptvae=%s rgbvae=%s)",
            len(self._warmup_buf),
            self._dino_stats is not None,
            self._ptvae_stats is not None,
            self._rgbvae_stats is not None,
        )

    # ------------------------------------------------------------------
    # Render one chunk (all 6 rows) using the fixed stats
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _render_and_write(self, model, cd: Dict[str, Any]) -> None:
        device = model.device
        dtype = model.torch_dtype
        target_hw = tuple(getattr(getattr(model, "_layout", None), "composite_hw", (384, 320)))

        # === Row 1: RGB pred | vae_recon | gt ===
        pred_video_lat = cd["video_lat"].to(device=device, dtype=dtype)
        pred_rgb = _pil_frames_to_tensor01(model._decode_latents(pred_video_lat, tiled=False))
        F_dec = pred_rgb.shape[1]

        input_image = cd["input_image"].to(device=device, dtype=dtype)
        gt_rgb_1f = ((input_image.clamp(-1.0, 1.0) + 1.0) * 0.5).unsqueeze(2)[0].cpu()
        rgb_lat_dev = model._encode_video_latents(input_image.unsqueeze(2), tiled=False)
        vae_recon_rgb_1f = _pil_frames_to_tensor01(
            model._decode_latents(rgb_lat_dev, tiled=False)[:1]
        )
        rgb_row = _row_3col_uint8(
            pred_rgb, _expand_f(vae_recon_rgb_1f, F_dec), _expand_f(gt_rgb_1f, F_dec),
        )
        named: List[Tuple[str, np.ndarray]] = [("rgb", rgb_row)]

        # === Row 2: DINO PCA (fixed stats) ===
        if cd["dino_lat"] is not None and self._dino_stats is not None:
            try:
                named.append(("dino", self._render_dino_row(model, cd, F_dec)))
            except Exception as e:  # noqa: BLE001
                logger.warning("[PredictionRecorder] DINO row failed: %r", e)

        # === Rows 3, 4, 5: Pointmap (checkpoints with a pointmap stream) ===
        enable_pt = bool(getattr(model, "enable_pointmap", False))
        if cd["pt_lat"] is not None:
            if enable_pt and not cd.get("per_cam_depth"):
                # Depth-less deploy: the three normal pointmap rows all need GT
                # depth + intrinsics for their vae_recon/gt columns. The
                # prediction itself is still there -- with present_pointmap=False
                # and cross_modal_predict_pointmap=True the model imagines the
                # future pointmap from RGB+DINO -- so render that alone.
                try:
                    named.append(
                        ("pointmap_pred", self._render_pointmap_pred_only_row(model, cd))
                    )
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "[PredictionRecorder] pointmap pred-only row failed: %r", e,
                    )
            elif enable_pt:
                try:
                    xyz_row, depth_row, ptvae_row = self._render_pointmap_rows(
                        model, cd, target_hw,
                    )
                    named.append(("pointmap_xyz", xyz_row))
                    named.append(("pointmap_depth", depth_row))
                    if ptvae_row is not None:
                        named.append(("pointmap_vae_pca", ptvae_row))
                except Exception as e:  # noqa: BLE001
                    logger.warning("[PredictionRecorder] pointmap rows failed: %r", e)
            else:
                logger.warning(
                    "[PredictionRecorder] model has no pointmap stream "
                    "(enable_pointmap is not set); pointmap rows skipped.",
                )

        # === Row 6: RGB VAE-PCA (fixed stats) ===
        if self._rgbvae_stats is not None:
            try:
                gt_rgb_lat = rgb_lat_dev[0].detach().cpu()             # [C,1,H,W]
                F_lat = pred_video_lat.shape[2]
                gt_rgb_lat_exp = gt_rgb_lat.expand(-1, F_lat, -1, -1).contiguous()
                named.append(("rgb_vae_pca", self._render_vae_pca_row(
                    gt_rgb_lat_exp, pred_video_lat[0].detach().cpu(),
                    target_hw, self._rgbvae_stats,
                )))
            except Exception as e:  # noqa: BLE001
                logger.warning("[PredictionRecorder] RGB VAE-PCA row failed: %r", e)

        # Kept window: drop the anchor (decoded frame 0 = the re-clamped clean
        # reconstruction of the current real observation — duplicates the
        # previous chunk's last instant and snaps the imagined background back
        # to reality), then keep frames_per_chunk frames (0 = all remaining).
        start = 1 if self._drop_anchor else 0
        start = min(start, max(F_dec - 1, 0))
        avail = F_dec - start
        keep = self._frames_per_chunk if self._frames_per_chunk > 0 else avail
        keep = max(1, min(keep, avail))

        # Companion per-row / per-tile MP4s are written from the NATIVE-width
        # rows, before the panel's common-width resize — a split file is never
        # resampled, only frame-matched and sliced.
        if self._split != "none":
            self._write_split(
                [(n, _row_window(a, F_dec, start, keep)) for n, a in named]
            )

        # Panel width = the widest row's native width, so no row is ever
        # downsampled at write time (uniform14: all rows 960, as before;
        # nsq576-class layouts: 3×448 = 1344, RGB tiles stored at native
        # 448×576 instead of the old 960 squeeze). All rows of a given ckpt
        # share 3×composite_w natively, so this is stable across chunks even
        # when an optional row is skipped.
        rows = [a for _, a in named]
        panel_w = max(r.shape[2] for r in rows)
        rows = [_resize_row_to_width(r, panel_w) for r in rows]
        rows = [_row_window(r, F_dec, start, keep) for r in rows]

        for k in range(keep):
            stacked = np.concatenate([row[k] for row in rows], axis=0)
            self._writer.append_data(_to_even(stacked))

        self._n_chunks += 1
        self._n_frames += keep

    # ------------------------------------------------------------------
    # Per-row / per-tile companion files
    # ------------------------------------------------------------------

    def _write_split(self, named: List[Tuple[str, np.ndarray]]) -> None:
        """Append this chunk's already-sliced kept window to the companion MP4s."""
        for row_name, arr in named:
            if self._split == "row":
                self._append_split(row_name, arr)
                continue
            cols = _ROW_COLS.get(row_name, _COLS_PRED_RECON_GT)
            w, rem = divmod(arr.shape[2], len(cols))
            if rem:
                logger.warning(
                    "[PredictionRecorder] row %r width %d not divisible by %d; "
                    "writing it whole instead of per-tile.",
                    row_name, arr.shape[2], len(cols),
                )
                self._append_split(row_name, arr)
                continue
            for i, col in enumerate(cols):
                self._append_split(f"{row_name}__{col}", arr[:, :, i * w:(i + 1) * w])

    def _append_split(self, stem: str, arr: np.ndarray) -> None:
        writer = self._split_writers.get(stem)
        if writer is None:
            writer = self._open_writer(self._session_dir / f"{stem}.mp4")
            self._split_writers[stem] = writer
            logger.info("[PredictionRecorder] split file opened: %s.mp4", stem)
        for k in range(arr.shape[0]):
            writer.append_data(_to_even(np.ascontiguousarray(arr[k])))

    # ------------------------------------------------------------------
    # PCA row renderers (call the patched leaf helpers with fixed stats)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _render_dino_row(self, model, cd: Dict[str, Any], target_f: int) -> np.ndarray:
        from flexpi.vis import (
            _arrange_layout,
            _compute_dino_pca_and_norm,
        )
        from flexpi.composite_layouts import get_layout

        pred_dino = cd["dino_lat"]               # [B,C,F,N,1]
        b, c, f_d, n, _ = pred_dino.shape

        per_cam_5d = {
            k: (v.unsqueeze(2) if v.ndim == 4 else v)
            for k, v in (cd.get("per_cam") or {}).items()
        }
        gt_dino_5d = model.dino_encoder.encode_video(
            video=None, per_cam={k: v.to(model.device) for k, v in per_cam_5d.items()},
            **model._dino_encode_kwargs(),
            temporal_stride=model.dino_temporal_stride, first_frame_only=True,
        ).cpu()

        # Undo the encoder's lossless pixel-unshuffle fold before PCA so the
        # rendered map is the NATIVE DINO grid (14×14/cam @ 768ch), not the
        # folded token grid (7×7/cam @ 3072ch) the model computes on.
        fold_f, folded_patches = _dino_unfold_params(model)
        if fold_f:
            pred_dino = _unfold_dino_lat(pred_dino, folded_patches, fold_f)
            gt_dino_5d = _unfold_dino_lat(gt_dino_5d, folded_patches, fold_f)
            cam_patches = [(h * fold_f, w * fold_f) for h, w in folded_patches]
        else:
            cam_patches = [tuple(p) for p in model.dino_cam_patches]

        gt_dino_1f = gt_dino_5d[0]               # [C,1,N,1]
        gt_features = gt_dino_1f.index_select(1, torch.zeros(f_d, dtype=torch.long))
        pred_features = pred_dino[0]             # [C,F,N,1]
        basis = self._dino_stats["basis"]
        vrange = self._dino_stats["range"]
        gt_pca, gt_norm, _ = _compute_dino_pca_and_norm(
            gt_features, cam_patches, pca_basis=basis, value_range=vrange,
        )
        pred_pca, _, _ = _compute_dino_pca_and_norm(
            pred_features, cam_patches, pca_basis=basis, value_range=vrange,
        )
        layout = get_layout(model._layout)  # pass-through for LayoutSpec
        gt_pca_t = _arrange_layout(gt_pca, is_rgb=True, layout=layout)
        gt_norm_t = _arrange_layout(gt_norm, is_rgb=False, layout=layout)
        pred_pca_t = _arrange_layout(pred_pca, is_rgb=True, layout=layout)
        return np.concatenate([pred_pca_t, gt_pca_t, gt_norm_t], axis=2)

    @torch.no_grad()
    def _render_pointmap_pred_only_row(self, model, cd: Dict[str, Any]) -> np.ndarray:
        """Imagined-pointmap row for deploys with no depth input.

        Layout ``[imagined XYZ | imagined depth | black]``. The third tile is
        blank on purpose: every row in the panel is natively ``3 x composite_w``
        and ``_render_and_write`` resizes rows to the widest one, so a 2-tile row
        would be stretched (and, since the resize keeps aspect, made taller).
        Padding to the standard width keeps this row pixel-exact.

        Needs neither the pointmap encoder nor intrinsics -- just a VAE decode of
        the predicted latents. The depth tile mirrors the ``pred`` column of the
        depth row ``_render_pointmap_rows`` builds: the same ``_depth_colormap``
        over channel 2 of the ``[0,1]`` XYZ composite, with no re-normalization
        (rescaling per chunk would break cross-chunk comparison).
        """
        from flexpi.vis import _depth_colormap

        pred_pt_lat = cd["pt_lat"].to(device=model.device, dtype=model.torch_dtype)
        xyz = _pil_frames_to_tensor01(          # [3, F, H, W] in [0,1], on CPU
            model._decode_latents(pred_pt_lat, tiled=False)
        )
        depth = _depth_colormap(                # [3, F, H, W]
            xyz[2], mode=str(getattr(model, "pointmap_depth_vis_mode", "turbo")),
        )
        stitched = torch.cat([xyz, depth, torch.zeros_like(xyz)], dim=-1).clamp(0.0, 1.0)
        arr = (stitched * 255.0).to(torch.uint8).numpy()
        return np.transpose(arr, (1, 2, 3, 0))  # [F, H, 3W, 3]

    @torch.no_grad()
    def _render_pointmap_rows(
        self, model, cd: Dict[str, Any], target_hw: tuple,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        from flexpi.vis import _depth_colormap

        device = model.device
        dtype = model.torch_dtype

        K = cd.get("K")
        if K is None:
            raise RuntimeError("pointmap row needs camera_intrinsics in present obs")
        K_t = K.to(device=device, dtype=torch.float32)
        per_cam_depth = {k: v.to(device=device) for k, v in (cd.get("per_cam_depth") or {}).items()}

        pt_composite = model.pointmap_encoder.encode_composite(
            per_cam_depth=per_cam_depth, camera_intrinsics=K_t,
            first_frame_only=True, **model._layout_kwargs(),
        )  # [1,3,1,*composite_hw] in [-1,1]
        gt_pt_1f = ((pt_composite.clamp(-1.0, 1.0) + 1.0) * 0.5)[0].cpu()
        pt_lat_dev = model._encode_video_latents(pt_composite, tiled=False)
        vae_recon_pt_1f = _pil_frames_to_tensor01(
            model._decode_latents(pt_lat_dev, tiled=False)[:1]
        )

        pred_pt_lat = cd["pt_lat"].to(device=device, dtype=dtype)
        pred_pt = _pil_frames_to_tensor01(model._decode_latents(pred_pt_lat, tiled=False))
        F_pt = pred_pt.shape[1]
        gt_pt = _expand_f(gt_pt_1f, F_pt)
        vae_recon_pt = _expand_f(vae_recon_pt_1f, F_pt)

        # Rows 3, 4: XYZ + depth (no PCA — already color-stable). Stitched here:
        # flexpi.vis's pointmap row builders take (pred, vae_recon) only, the
        # two columns of the training eval video. Depth is channel 2 of the
        # [0,1] XYZ composite, colormapped without re-normalization.
        xyz_row = _row_3col_uint8(pred_pt, vae_recon_pt, gt_pt)
        depth_vis_mode = getattr(model, "pointmap_depth_vis_mode", "turbo")
        depth_row = _row_3col_uint8(*[
            _depth_colormap(t[2].detach().cpu(), mode=depth_vis_mode)
            for t in (pred_pt, vae_recon_pt, gt_pt)
        ])

        # Row 5: pointmap VAE-PCA (fixed stats).
        ptvae_row = None
        if self._ptvae_stats is not None:
            gt_pt_lat = pt_lat_dev[0].detach().cpu()
            F_lat = pred_pt_lat.shape[2]
            gt_pt_lat_exp = gt_pt_lat.expand(-1, F_lat, -1, -1).contiguous()
            ptvae_row = self._render_vae_pca_row(
                gt_pt_lat_exp, pred_pt_lat[0].detach().cpu(),
                target_hw, self._ptvae_stats,
            )
        return xyz_row, depth_row, ptvae_row

    @staticmethod
    def _render_vae_pca_row(
        gt_latent: torch.Tensor,
        pred_latent: torch.Tensor,
        target_hw: tuple,
        stats: dict,
    ) -> np.ndarray:
        """3-col VAE-latent PCA row with the fixed (basis, range) stats."""
        from flexpi.vis import visualize_vae_latent

        th, tw = target_hw
        basis = stats["basis"]
        vrange = stats["range"]
        gt_pca, gt_norm, _ = visualize_vae_latent(
            gt_latent, target_h=th, target_w=tw, pca_basis=basis, value_range=vrange,
        )
        pred_pca, _, _ = visualize_vae_latent(
            pred_latent, target_h=th, target_w=tw, pca_basis=basis, value_range=vrange,
        )
        return np.concatenate([pred_pca, gt_pca, gt_norm], axis=2)


# ----------------------------------------------------------------------
# Session-dir helper
# ----------------------------------------------------------------------

def make_session_dir(base_dir: str | os.PathLike, remote_addr: Optional[str] = None) -> Path:
    """Create a timestamped session subdir under ``base_dir`` and return its Path."""
    import re
    from datetime import datetime
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    suffix = ""
    if remote_addr:
        cleaned = re.sub(r"[^A-Za-z0-9.]+", "_", str(remote_addr)).strip("_")
        if cleaned:
            suffix = "_" + cleaned
    out = Path(base_dir) / f"{stamp}{suffix}"
    out.mkdir(parents=True, exist_ok=True)
    return out
