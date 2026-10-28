"""Fit frame-pooled norm stats (the pi0.5 / openpi rule) over a LeRobot dir's parquet.

Why a second rule: `get_dataset_stats` (base_lerobot_dataset.py), which the trainer runs
on a first run, takes a 1st/99th percentile PER EPISODE and then the MIN / MAX of those
over all episodes and horizon steps. Over thousands of episodes that box lands near the
min-max envelope (a 34-D bimanual rel34 corpus: 0.98 of the min-max width, median over
its position and joint dims), so its "q01/q99" is really min/max, and its mean/std weight every
episode equally regardless of length.

This follows openpi: scripts/compute_norm_stats.py feeds every training sample (a frame
plus its action chunk, AFTER the delta transform) into shared/normalize.py::RunningStats,
whose update() flattens all leading dims (`batch.reshape(-1, dim)`). Every (frame,
horizon step) target counts once; q01/q99 are the pooled 1st/99th percentiles and
mean/std are frame-weighted. openpi approximates the quantiles with a 5,000-bin
histogram; here they are exact (numpy linear interpolation, the rule torch.quantile
uses), since the whole corpus is held in memory.

The transform chain is the one training runs: the processor is instantiated from the
data preset (configs/data/<name>.yaml), windows are built with
sliding_window_with_replication (every frame is a window start; steps past the episode
end repeat the last action, like LeRobot's delta_timestamps padding that openpi's stats
also include), and the state window is the num_frames-long window the loader serves,
so a chain that reads future state frames sees them. State is pooled once per frame: the
window's first frame, as the chain emits it. The widths are the preset's shape_meta
action/state widths, i.e. the width its normalizer runs at.

Output keeps the schema of the per-episode file (global_* [D], stepwise_* [H, D] for
action and [1, D] for state), so it is a drop-in `pretrained_norm_stats`. stepwise_* are
pooled per horizon step. std is the population std (ddof=0), as in RunningStats.
Memory is ~20 bytes per (frame, step, dim) target: ~40 GB for a 1.85M-frame corpus.
Parquet is read with pyarrow, so no datasets-compatible fit view is involved.

Usage, from the repository root:
  python scripts/norm_stats/fit_pooled_stats.py --src <lerobot dir> --data-config <preset> \\
      --corpus-name <name> --out <stats.json>
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from flexpi.datasets.lerobot.base_lerobot_dataset import sliding_window_with_replication

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from norm_stats_coverage import stamp  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True, help="LeRobot dir holding data/chunk-*/episode_*.parquet")
ap.add_argument("--data-config", required=True,
                help="configs/data/<name>.yaml whose processor (transform chain) the stats are fit under")
ap.add_argument("--corpus-name", required=True, help="name stamped into `coverage`")
ap.add_argument("--out", required=True)
args = ap.parse_args()

cfg = OmegaConf.create({"data": OmegaConf.load(os.path.join(ROOT, "configs", "data", f"{args.data_config}.yaml"))})
T = int(cfg.data.train.num_frames)                         # 33 frames -> 32 action steps
H = T - 1
DA = int(cfg.data.train.shape_meta.action[0].shape)        # width the normalizer sees
DS = int(cfg.data.train.shape_meta.state[0].shape)
proc = instantiate(cfg.data.train.processor)
assert H == 32, H

files = sorted(glob.glob(os.path.join(args.src, "data", "chunk-*", "episode_*.parquet")))
assert files, f"no parquet under {args.src}/data"
lens = [pq.ParquetFile(f).metadata.num_rows for f in files]
F = int(sum(lens))
A = np.empty((F, H, DA), dtype=np.float32)                 # transformed action targets
S = np.empty((F, DS), dtype=np.float32)                    # state (proprio) at each window start
o = 0
for i, f in enumerate(files):
    t = pq.read_table(f, columns=["observation.state", "action"])
    s = torch.from_numpy(np.stack(t["observation.state"].to_numpy(zero_copy_only=False)).astype(np.float32))
    a = torch.from_numpy(np.stack(t["action"].to_numpy(zero_copy_only=False)).astype(np.float32))
    batch = {"action": {"default": sliding_window_with_replication(a, H).float()},
             "state": {"default": sliding_window_with_replication(s, T).float()}}
    batch = proc.action_state_transform(batch)
    n = len(s)
    A[o:o + n] = batch["action"]["default"].numpy()
    S[o:o + n] = batch["state"]["default"][:, 0].numpy()
    o += n
    if (i + 1) % 500 == 0:
        print(f"  {i + 1}/{len(files)} episodes, {o:,} frames", flush=True)
assert o == F


def pooled(x):  # x [M, d] -> dict of [d] (float64 accumulation, exact quantiles)
    x64 = x.astype(np.float64)
    m = x64.mean(0)
    q = np.quantile(x, [0.01, 0.99], axis=0)
    return dict(mean=m, std=np.sqrt(np.maximum(0.0, (x64 ** 2).mean(0) - m ** 2)),
                min=x.min(0).astype(np.float64), max=x.max(0).astype(np.float64), q01=q[0], q99=q[1])


def block(glob_stats, step_stats):
    out = {}
    for k in ("min", "max", "q01", "q99", "mean", "std"):
        out[f"stepwise_{k}"] = np.stack([s[k] for s in step_stats]).tolist()
        out[f"global_{k}"] = glob_stats[k].tolist()
    return out


print("pooling action targets ...", flush=True)
act_global = pooled(A.reshape(-1, DA))
act_step = [pooled(A[:, k]) for k in range(H)]
st_global = pooled(S)
chain = " -> ".join(type(t).__name__ for t in (proc.action_state_transforms or [])) or "no transform: absolute targets"
stats = {
    "state": {"default": block(st_global, [st_global])},
    "action": {"default": block(act_global, act_step)},
    "num_episodes": len(files),
    "num_transition": F,
    "fit_note": (
        "POOLED frame-weighted stats (pi0.5 / openpi RunningStats convention, exact quantiles): every "
        "(frame, horizon step) target counts once; q01/q99 are true pooled 1st/99th percentiles, mean/std "
        f"frame-weighted (ddof=0). Fit after the training chain of configs/data/{args.data_config}.yaml "
        f"({chain}), windows from sliding_window_with_replication (tail steps repeat the last action). "
        "NOT the per-episode min/max rule of get_dataset_stats. Fitter: scripts/norm_stats/fit_pooled_stats.py."
    ),
}
with open(args.out, "w") as fh:
    json.dump(stats, fh)
stamp(args.out, {args.corpus_name: len(files)})
print(f"wrote {args.out} ({os.path.getsize(args.out):,} B; {len(files):,} episodes, {F:,} frames, "
      f"{F * H:,} action targets per dim)")
