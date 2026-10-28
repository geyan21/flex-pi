"""Render a raw prediction dump into the prediction MP4 set, post-deployment.

Counterpart of ``serve_yam_flexpi.py --record-predictions-mode raw`` (the
default): at deploy time the server only dumps per-chunk tensors to
``<session>/raw_chunks/chunk_%05d.pt`` on a background thread; this script
loads the checkpoint once and replays those chunks through the SAME
``PredictionRecorder`` live-render path the server would have used — so the
output (``prediction.mp4`` + optional per-row / per-tile splits) is identical
to what live mode produces, minus the control-loop latency.

Render settings default to the values recorded in ``<session>/meta.json`` at
deploy time; any CLI flag overrides its meta counterpart, so the same dump can
be re-rendered with a different split/fps/anchor choice without re-running the
robot.

Usage::

    python scripts/render_prediction_dump.py \
        --dump-dir <record-predictions-dir>/<timestamp>_<remote> \
        [--ckpt-path /path/to/step_XXXX.pt]   # default: meta.json's ckpt_path
        [--split tile] [--fps 7.5] [--keep-anchor | --drop-anchor] \
        [--frames-per-chunk N] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Path bootstrap so ``python scripts/render_prediction_dump.py`` works from any cwd.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

LOG_PREFIX = "[render_prediction_dump]"


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Offline renderer for raw prediction dumps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--dump-dir", required=True,
        help="Session dir containing raw_chunks/chunk_*.pt (+ meta.json).",
    )
    p.add_argument(
        "--ckpt-path", default=None,
        help="FlexPi checkpoint. Default: the ckpt_path recorded in meta.json.",
    )
    p.add_argument(
        "--out-dir", default=None,
        help="Where the MP4s go. Default: the dump dir itself. A fresh dir "
             "re-renders with different settings without overwriting.",
    )
    # Render settings — None = fall back to meta.json, then to the recorder
    # defaults (fps 7.5, frames 0=all, drop_anchor True, split none).
    p.add_argument("--fps", type=float, default=None)
    p.add_argument("--frames-per-chunk", type=int, default=None)
    p.add_argument("--split", default=None, choices=["none", "row", "tile"])
    p.add_argument("--pca-warmup-chunks", type=int, default=None)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--drop-anchor", dest="drop_anchor", action="store_true",
                   default=None, help="Drop each chunk's frame-0 anchor.")
    g.add_argument("--keep-anchor", dest="drop_anchor", action="store_false",
                   default=None,
                   help="Keep each chunk's frame-0 anchor (old 9-frame layout).")
    p.add_argument(
        "--policy", default=None, choices=["yam"],
        help="Which deploy_policy builder loads the ckpt. Default: the 'policy' "
             "key in meta.json, else 'yam' (dumps written before that key "
             "existed are all YAM). This tree has only the YAM builder.",
    )
    # Model loading.
    p.add_argument("--device", default="cuda")
    p.add_argument("--mixed-precision", default="bf16", choices=["no", "fp16", "bf16"])
    p.add_argument("--dataset-stats-path", default=None)
    p.add_argument("--no-offload-text-encoder", action="store_true",
                   help="Keep T5 on GPU (rendering never uses it; offload is "
                        "the default to save ~10 GB VRAM).")
    return p


def render_dump(chunk_paths, recorder, model, log_every: int = 10) -> int:
    """Replay dumped chunks through the recorder's live-render path."""
    import torch

    n = 0
    t0 = time.perf_counter()
    for path in chunk_paths:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        recorder.append_chunk(
            model=model, pred=payload["pred"], present=payload["present"],
        )
        n += 1
        if n % log_every == 0:
            dt = time.perf_counter() - t0
            print(f"{LOG_PREFIX} {n}/{len(chunk_paths)} chunks  ({dt / n * 1000:.0f} ms/chunk)")
    return n


def main() -> int:
    args = _build_parser().parse_args()

    dump_dir = Path(args.dump_dir).expanduser().resolve()
    raw_dir = dump_dir / "raw_chunks"
    chunk_paths = sorted(raw_dir.glob("chunk_*.pt"))
    if not chunk_paths:
        print(f"{LOG_PREFIX} no raw_chunks/chunk_*.pt under {dump_dir}", file=sys.stderr)
        return 1

    meta = {}
    meta_path = dump_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)

    ckpt_path = args.ckpt_path or meta.get("ckpt_path")
    if not ckpt_path:
        print(
            f"{LOG_PREFIX} --ckpt-path required (meta.json has no ckpt_path).",
            file=sys.stderr,
        )
        return 1

    # CLI > meta.json > recorder defaults.
    fps = args.fps if args.fps is not None else float(meta.get("fps", 7.5))
    frames = (
        args.frames_per_chunk if args.frames_per_chunk is not None
        else int(meta.get("frames_per_chunk", 0))
    )
    split = args.split if args.split is not None else str(meta.get("split", "none"))
    warmup = (
        args.pca_warmup_chunks if args.pca_warmup_chunks is not None
        else int(meta.get("pca_warmup_chunks", 3))
    )
    drop_anchor = (
        args.drop_anchor if args.drop_anchor is not None
        else bool(meta.get("drop_anchor", True))
    )
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else dump_dir

    print(
        f"{LOG_PREFIX} {len(chunk_paths)} chunks  ckpt={ckpt_path}\n"
        f"{LOG_PREFIX} render: fps={fps:g} frames_per_chunk={frames} "
        f"drop_anchor={drop_anchor} split={split} pca_warmup={warmup} -> {out_dir}"
    )

    from experiments.yam.flexpi_policy.prediction_recorder import PredictionRecorder

    which = args.policy or str(meta.get("policy", "yam"))
    print(f"{LOG_PREFIX} policy builder: {which}")
    if which != "yam":
        print(
            f"{LOG_PREFIX} meta.json names policy {which!r}; this tree has "
            "only the YAM deploy_policy builder.",
            file=sys.stderr,
        )
        return 1

    t0 = time.perf_counter()
    # Rendering only needs VAE decode/encode + DINO/pointmap encoders — load
    # eager: no compile and no CUDA graphs, so no boot warmup runs.
    from experiments.yam.flexpi_policy.deploy_policy import (
        build_policy_from_checkpoint,
    )
    policy = build_policy_from_checkpoint(
        ckpt_path,
        dataset_stats_path=args.dataset_stats_path,
        device=args.device,
        mixed_precision=args.mixed_precision,
        offload_text_encoder=not args.no_offload_text_encoder,
        torch_compile=False,
    )
    print(f"{LOG_PREFIX} model loaded in {time.perf_counter() - t0:.1f} s")

    recorder = PredictionRecorder(
        out_dir,
        fps=fps,
        frames_per_chunk=frames,
        pca_warmup_chunks=warmup,
        split=split,
        mode="live",
        drop_anchor=drop_anchor,
        ckpt_path=str(ckpt_path),
        policy=which,
        # Never clobber the deploy-time meta.json this render read its
        # defaults from (out_dir defaults to the dump dir itself).
        write_meta=False,
    )
    t0 = time.perf_counter()
    n = render_dump(chunk_paths, recorder, policy.model)
    recorder.close()

    outputs = sorted(p.name for p in out_dir.glob("*.mp4"))
    print(
        f"{LOG_PREFIX} done: {n} chunks in {time.perf_counter() - t0:.1f} s\n"
        f"{LOG_PREFIX} wrote {len(outputs)} files under {out_dir}:\n  " + "\n  ".join(outputs)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
