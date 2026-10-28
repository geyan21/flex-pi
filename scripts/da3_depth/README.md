# DA3 depth labeling

Turns an RGB-only LeRobot dataset into a depth-carrying one, by running
[Depth Anything 3](https://github.com/ByteDance-Seed/Depth-Anything-3) metric
depth over every RGB frame and writing the result as a depth video stream
alongside the RGB. For rigs that ship no depth sensor: build the dataset as
usual, then run this.

The output is field-identical to what `scripts/yam_dataset_builder/convert.py`
writes when the capture *did* have depth, so `RobotVideoDataset` consumes both
the same way and nothing on the training side changes.

> [!NOTE]
> This is the one component with its own conda env. The pass imports nothing
> from Flex-π, so it can pin the stack Depth Anything 3 was verified against
> (torch 2.10, numpy&nbsp;<2) without disturbing `flexpi`.

```
scripts/da3_depth/
├── setup.sh         # env + DA3 + checkpoint + verify   (run once)
├── env.sh           # written by setup.sh; gitignored, machine-local
├── run.sh           # wrapper: reads env.sh, calls label_depth.py
└── label_depth.py   # the pass
```

---

## Use

```bash
bash scripts/da3_depth/setup.sh                    # once

# validate — prints the per-camera depth grid and K. No GPU, no writes.
CHECK=1 bash scripts/da3_depth/run.sh ./data/<ds>

# label
bash scripts/da3_depth/run.sh ./data/<ds>

# sharded across 4 GPUs, then finalize ONCE (concurrent info.json writers race)
for i in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$i NUM_SHARDS=4 SHARD_ID=$i bash scripts/da3_depth/run.sh ./data/<ds> &
done; wait
PATCH_INFO_ONLY=1 bash scripts/da3_depth/run.sh ./data/<ds>
```

`setup.sh` is idempotent: it creates the `da3_depth` env, installs ffmpeg 7 and
`depth_anything_3`, downloads `DA3METRIC-LARGE` (~1.3 GB), and verifies the
checkpoint loads and produces *metric* depth — a relative-depth checkpoint fails
there rather than silently emitting unitless "millimetres". Then it writes
`env.sh`, which `run.sh` sources, so nothing downstream hardcodes a path.
Overrides: `CONDA_DIR`, `ENV_NAME`, `DA3_SRC`, `DA3_WEIGHTS`, `DA3_MODEL`,
`PY_VER`.

| Flag | |
|---|---|
| `--depth-hw H W` | force one depth grid for all cameras (moves K with it) |
| `--cams A B` | restrict to these cameras |
| `--episodes 0 1 2` | restrict to these episodes — smoke tests |
| `--max-depth-mm N` | depth above `N` becomes `0`, the invalid sentinel |
| `--overwrite` | re-label episodes that already have depth files |
| `--batch-size` | GPU-memory bound only; peak RAM is `batch_size` frames regardless of episode length |
| `--device` | default `cuda` |

---

## What it needs, what it writes

**Needs** a LeRobot v2.1 dataset with RGB videos under
`videos/chunk-*/observation.images.<cam>/` and a `meta/camera_intrinsics.json`
carrying `fx, fy, cx, cy, width, height` for every one of those cameras. Camera
names are discovered from those two, intersected — nothing about a canonical
camera set is assumed. `fps` and `chunks_size` come from `meta/info.json`.
A camera with RGB but no intrinsics is a hard error, not a guess.

**Writes** `videos/chunk-{c:03d}/observation.depth_ffv1.<cam>/episode_{i:06d}.mkv`
— FFV1 level 3, `gray16le`, uint16 **millimetres**, `0` = invalid, one keyframe
per frame, same frame count and fps as the source RGB — and registers one
`dtype: "depth_video"` feature per camera in `meta/info.json`, with shapes probed
off the files actually written.

**Never touches** parquet, RGB, `episodes.jsonl`, `tasks.jsonl`,
`episodes_stats.jsonl` or `camera_intrinsics.json`.

---

## The depth grid, and why K travels with it

> [!IMPORTANT]
> DA3METRIC's raw output is not metric — it is multiplied by `focal / 300` with
> `focal = (fx + fy) / 2`. **K therefore sets a global scale on the depth and
> nothing validates it:** a wrong K yields a perfectly plausible depth map that
> is simply the wrong size in metres, with no error anywhere.

If the dataset ships `meta/episode_intrinsics.jsonl`, each episode is scaled by
its own K, and `--check` prints the per-episode focal spread; an episode missing
from the table is an error at planning time. The focal is baked into the
millimetres written here, so a wrong K at this stage is not undone at training.

The depth grid is the resolution DA3 runs at and the depth is written at. It
defaults to **each camera's own native RGB resolution**, probed per camera —
which preserves the aspect ratio DA3's geometry depends on, and makes the K
rescale an exact identity when the intrinsics are stated at that resolution
(they are, for builder output).

K is rescaled with the endpoint convention `s = (dst-1)/(src-1)` — the same one
`RobotVideoDataset._load_layout_intrinsics_for_dir` re-applies at load time, and
the one the linspace-endpoint depth resampler assumes on both sides. The plain
ratio `dst/src` is a *different* convention and disagrees by ~0.1–0.5 %.

`--depth-hw H W` moves the grid **and** K together, and warns if the ratio you
ask for differs from the intrinsics'. Never hand-edit one without the other.
`--check` prints the resolved grid, K and focal per camera before anything runs.
Read it.

---

## Resume and failure

- A camera is **skipped when its `.mkv` already exists**. Encoding goes to a temp
  file renamed on success, so a killed run never leaves a truncated file that the
  skip would accept.
- **One bad episode does not kill the run** — it goes to
  `meta/depth_failures_shard<N>.txt` and the process exits non-zero at the end.
  Re-running retries exactly those.
- A dataset that already declares depth gets a **warning, not a hard stop**, so a
  partially-failed run stays resumable. Without `--overwrite` no existing depth
  file is ever touched — pointing this at a dataset that already has real sensor
  depth is a no-op, not a disaster.

## Is the output sane?

Against a sensor-depth twin, DA3 lands within ~20–25 % of the sensor's median and
fills 100 % of pixels (a stereo sensor typically leaves 15–27 % holes). That gap
is ordinary monocular-vs-sensor disagreement. Being off by **2×, 10× or 1000×**
is not — that is a K, grid or unit bug. Check `--check` output first.
