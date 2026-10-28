# Real-world bimanual YAM

End-to-end recipe for the real-robot setting. There is no simulator here, so
"evaluation" means deployment. Three steps:

1. [Data build](#1-data-build) — download a released dataset, or turn a raw
   capture into one
2. [Train](#2-train)
3. [Deploy server](#3-deploy-server) — serve the checkpoint to the arm

Everything runs from the **repository root**. One-time environment and weight
setup is [`docs/INSTALL.md`](INSTALL.md); the per-knob reference behind the
launcher is [`docs/TRAINING.md`](TRAINING.md).

---

## 1. Data build

The five tasks behind the paper's real-world results are released one repo per
task, ready to train on:

| Task | Dataset |
|---|---|
| Put plate on the rack | [`flex-pi/put_plate_on_the_rack`](https://huggingface.co/datasets/flex-pi/put_plate_on_the_rack) |
| Sort utensils | [`flex-pi/sort_utensils`](https://huggingface.co/datasets/flex-pi/sort_utensils) |
| Kitchen organization | [`flex-pi/kitchen_organization`](https://huggingface.co/datasets/flex-pi/kitchen_organization) |
| Soft-bag zipping | [`flex-pi/soft_bag_zipping`](https://huggingface.co/datasets/flex-pi/soft_bag_zipping) |
| Self-repair gripper | [`flex-pi/self_repair_gripper_bc`](https://huggingface.co/datasets/flex-pi/self_repair_gripper_bc) &middot; [`flex-pi/self_repair_gripper_dagger`](https://huggingface.co/datasets/flex-pi/self_repair_gripper_dagger) |

```bash
huggingface-cli download flex-pi/put_plate_on_the_rack \
  --repo-type dataset --local-dir ./data/yam/put_plate_on_the_rack
```

For your own capture, `--raw-dir` takes the per-episode directory that `rd
convert` writes — [raiden](https://github.com/TRI-ML/raiden) is the stack we
teleoperate, record and run the arm with. The builder turns it into the
canonical layout directly, no rename pass afterwards:

```bash
python scripts/yam_dataset_builder/convert.py \
    --raw-dir /path/to/yam_raw/my_task \
    --out-dir ./data/yam/my_task \
    --task-name my_task
```

`--help` documents every flag. The raw layout it reads, the LeRobot v2.1 layout it
writes, the 26D&rarr;32D action convention and the multi-source merge semantics
are all in the two module docstrings:
[`convert.py`](../scripts/yam_dataset_builder/convert.py) and
[`source_reader.py`](../scripts/yam_dataset_builder/source_reader.py).

**No depth sensor.** `scripts/da3_depth/` runs Depth Anything 3 metric depth
over an RGB-only LeRobot dataset and writes the missing
`observation.depth_ffv1.<cam>` streams, field-identical to what the builder
writes when the capture did have depth. It carries its own conda env and imports
nothing from Flex-π, so its torch pin does not disturb `flexpi`. Setup, sharding
and the K-scaling caveats are in
[`scripts/da3_depth/README.md`](../scripts/da3_depth/README.md).

**Check a dataset before booking GPUs.** The dataloader has a self-test that
assembles one sample and unprojects its depth against the intrinsics:

```bash
python -m flexpi.datasets.lerobot.robot_video_dataset \
    --data-cfg yam --task-cfg yam_unified_flex_3cam_32d_rel_1e-4 \
    --override 'data.train.dataset_dirs=[./data/yam/<your_set>]'
```

It needs the text embeddings cached first
([`TRAINING.md §1.3`](TRAINING.md#13-precompute-text-embeddings-required)) and
writes normalizer stats into `./runs`.

---

## 2. Train

```bash
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"   # Wan2.2 weights

DATASET_DIRS="[./data/<your_yam_set>]" bash scripts/train_flexpi_yam.sh
```

This launcher runs the text-embed precompute for you; extra Hydra overrides pass
straight through.

Multiple datasets are a comma-separated Hydra list literal, quoted:

```bash
DATASET_DIRS="[./data/yam/set_a,./data/yam/set_b]" bash scripts/train_flexpi_yam.sh
```

Warm-start from a checkpoint you have (optional):

```bash
PRETRAINED_CKPT=/path/to/step_040000.pt bash scripts/train_flexpi_yam.sh
```

Warm-starting from the preview checkpoints takes a 34-D preset and the
checkpoint's own stats file; [`FINETUNE.md`](FINETUNE.md) has both.

Full-state `RESUME` wins over the warm start
([`TRAINING.md §4`](TRAINING.md#4-finetune-vs-resume)); relaunching with
neither auto-resumes from the run's own output directory.

### Outputs

```
runs/yam_unified_flex_3cam_32d_rel_1e-4/<run_id>_<regime_tag>_yam_uflex/
├── config.yaml                       # the trained config — the server reads this back
├── dataset_stats.json                # normalizer statistics
├── checkpoints/weights/step_NNNNNN.pt
└── checkpoints/state/step_NNNNNN/    # DeepSpeed full state, for RESUME
```

For this preset, norm statistics are fit fresh on the first run and cached into
the run directory. Its sibling `task=yam_unified_flex_3cam_34d_rel_1e-4` trains
the same data in the 34-D action layout
([`MIXED_EMBODIMENT_34D.md`](MIXED_EMBODIMENT_34D.md)), with statistics fit the
same way.

---

## 3. Deploy server

Two processes: a **policy server** that holds the GPU and serves action chunks
over a websocket, and a **client** that owns the robot. The server is here; the
client is yours to write, against the [wire contract](#wire-contract) below.
Serving the preview checkpoints zero-shot is [`PRETRAINED.md`](PRETRAINED.md).

### What you need

One GPU — ~15 GB engine-free, ~25 GB with KV-split engines — and a checkpoint
directory holding all three files:

```
step_NNNNNN.pt        # weights
config.yaml           # trained config — the server builds the model from this
dataset_stats.json    # normalizer statistics
```

`--ckpt-path` points at the `.pt`; the other two are found up to five directories
above it, so keep the run directory intact (the preview downloads have the same
shape). Wan2.2 and DINOv3 come from [`INSTALL.md §2`](INSTALL.md#2-weights);
TensorRT only if you want the fastest path.

The environment is [`INSTALL.md`](INSTALL.md) plus its `[serve]` extra:

```bash
pip install -e '.[serve]'
export DIFFSYNTH_MODEL_BASE_PATH=/path/to/wan22_weights
```

Run from the repository root so `python -m experiments.yam.flexpi_policy.*`
resolves.

### Serving

`scripts/serve_flexpi_yam.sh` is the launcher: edit its config block and run it.
The three templates below are what it assembles — read the hardware-safety
rows in [Troubleshooting](#troubleshooting) before using them directly.

```bash
export CKPT=/path/to/run/checkpoints/weights/step_NNNNNN.pt
export PROMPT="<language instruction>"
```

Pass the bare instruction. The server wraps it in the same template training used
(`DEFAULT_PROMPT` in `src/flexpi/datasets/lerobot/robot_video_dataset.py`), so
wrapping it yourself double-wraps it.

The operating point comes from `configs/real_yam.yaml` (4 denoise steps, bf16,
T5 on CPU, compile **off**; templates B and C pass `--torch-compile` themselves,
and the launcher passes it with `TORCH_COMPILE=true`). The `--infer-*` flags are
passed explicitly, because unset means "whatever this checkpoint was trained
with".

### A · Full joint, KV-split TensorRT engines — fastest

Requires the KV-split pair baked for **this exact checkpoint** —
[`INFERENCE_OPTIMIZATION.md §2`](INFERENCE_OPTIMIZATION.md#2-building-the-engines).
`--no-torch-compile` is required: the server refuses `--glue-cache` beside
`--torch-compile`.

```bash
export SPLIT_DIR=/path/to/engines/split_024500

python scripts/serve_yam_flexpi.py \
    --ckpt-path "$CKPT" --default-prompt "$PROMPT" \
    --host 0.0.0.0 --port 8000 \
    --trt-joint-prefill-split-engine "$SPLIT_DIR/prefill_core_o3.engine" \
    --trt-joint-decode-split-engine  "$SPLIT_DIR/decode_core_o3.engine" \
    --trt-joint-free-video-blocks --glue-cache --no-torch-compile \
    --encoder-cuda-graph \
    --infer-joint-video true --infer-joint-dino true --infer-joint-pointmap true \
    --infer-present-video true --infer-present-dino true --infer-present-pointmap true
```

### B · Full joint, engine-free

```bash
python scripts/serve_yam_flexpi.py \
    --ckpt-path "$CKPT" --default-prompt "$PROMPT" \
    --host 0.0.0.0 --port 8000 \
    --torch-compile --compile-encoders \
    --infer-joint-video true --infer-joint-dino true --infer-joint-pointmap true \
    --infer-present-video true --infer-present-dino true --infer-present-pointmap true
```

To ablate, flip any of the six flags and check the regime line in the boot log.

### C · Action only — lowest latency

```bash
python scripts/serve_yam_flexpi.py \
    --ckpt-path "$CKPT" --default-prompt "$PROMPT" \
    --host 0.0.0.0 --port 8000 \
    --torch-compile --compile-encoders \
    --infer-joint-video false --infer-joint-dino false --infer-joint-pointmap false
```

If `--compile-encoders` fails to compile on your torch or driver build, swap it
for `--encoder-cuda-graph` — the same job by a different route, ~2 ms slower per
call at four steps. The two are mutually exclusive; passing both raises.

### Measured on one RTX 5090

At the shipped 4 denoise steps, on the same 3-cam 384&times;320 composite YAM
serves. Model time; the server adds observation handling and msgpack over the
websocket.

| | stack | ms/call | GPU peak |
| --- | --- | ---: | ---: |
| A | full joint, KV-split engines | **193** | 25.7 GB |
| B | full joint, engine-free | 360 | 15.8 GB |
| C | action only | **60** | 15.8 GB |

### D · 34-D checkpoints

A checkpoint trained in the 34-D action layout (`yam34`, or the preview
checkpoints) serves through the same server; the wire stays 32-D in both
directions. It runs eager (`--no-torch-compile`). The commands for the preview
checkpoints, one per regime, are [`PRETRAINED.md §4`](PRETRAINED.md#4-server-commands).
A correct boot logs `action_dim_mask: channels [26, 33] zero-filled` and
`PREWARM DONE`, after which `/healthz` answers `OK`. The regime is set on the
server and the client follows it:

| regime | server | client |
|---|---|---|
| depth in | `--infer-present-pointmap true` | sends `observation/depth_<cam>` and `observation/intrinsics_<cam>` for all three cameras |
| no depth | `--infer-present-pointmap false` | sends neither |
| full joint | all three `--infer-joint-*` `true` | sends no `joint_*` keys |

On one A100-80GB, this server returns action chunks bit-identical to the
pre-release deploy code for `flexpi-pretrain-rgb-preview` with its released
`config.yaml` (same weights, prompt, flags, seed). `--torch-compile`
and the TensorRT engines were not part of that comparison.

### Recording predictions

`--record-predictions-dir <dir>` records the model's predicted future for every
chunk, one session per websocket connection under `<dir>/<timestamp>_<remote>/`.
Recording forces the joint-denoise regime on every call.

| flag | default | effect |
|---|---|---|
| `--record-predictions-mode` | `raw` | `raw` dumps tensors off the action path, rendered later by `scripts/render_prediction_dump.py --dump-dir <session>`; `live` renders `prediction.mp4` per chunk |
| `--record-predictions-fps` | 7.5 | output frame rate (predicted frames are 4 actions apart at 30 Hz) |
| `--record-predictions-frames-per-chunk` | 0 (all) | frames kept per chunk |
| `--record-predictions-keep-anchor` | off | keep each chunk's frame 0 (the present observation) |
| `--record-predictions-split` | `none` | `row` / `tile`: one extra MP4 per row or per cell |

`scripts/test_rd_client_dummy.py --host localhost --port 8000` measures the
round-trip latency against a running server with no robot, and is the smallest
client of the wire contract below.

### Wire contract

msgpack over websocket. A request is one flat dict; `_synthesize_prewarm_obs()`
in [`scripts/serve_yam_flexpi.py`](../scripts/serve_yam_flexpi.py) builds one at
exactly the wire shapes — `observation/image_<cam>` `uint8 [H,W,3]`,
`observation/depth_<cam>` `uint16 [H,W]` in millimetres,
`observation/intrinsics_<cam>` `float32 [fx,fy,cx,cy]`, `observation/state`
`float32 [32]` laid out by
[`yam_eef.STATE_LAYOUT`](../src/flexpi/datasets/lerobot/utils/yam_eef.py), and an
optional `prompt`. Depth and intrinsics come for all three cameras or for none.
The response carries `actions`, `float32 [action_horizon, 32]` — absolute,
already de-normalized against the state you sent — beside `raw_actions` (before
the optional smoother), `speed_factors` and `policy_timing`. The contract is the
same for a 34-D checkpoint. The metadata's `depth_required` says whether the
server's regime needs depth. Each arm's pose is in that arm's own base frame with
the TCP at `grasp_site`; the gripper is a measured opening in [0, 1], 1 = open
([`PRETRAINED.md §6`](PRETRAINED.md#6-input-contract)).

`observation/state` is also the anchor the relative action is inverted against;
sending the pose of the last commanded joints (`0:18`, `20:32`) with measured
grippers (`18:20`) keeps gravity droop from compounding across chunks
([`PRETRAINED.md §7`](PRETRAINED.md#7-what-a-client-does)).

```python
from experiments.yam.flexpi_policy._openpi_vendor.websocket_client_policy import WebsocketClientPolicy
client = WebsocketClientPolicy(host="localhost", port=8000)
print(client.get_server_metadata())     # cam_order, per_cam_hw, action_horizon, depth_required
actions = client.infer(obs)["actions"]
```

---

## Troubleshooting

Environment symptoms are in [`INSTALL.md §5`](INSTALL.md#5-troubleshooting),
training-side ones — OOM included — in
[`TRAINING.md §5`](TRAINING.md#5-troubleshooting).

> A stale `$SPLIT_DIR` and a forgotten server have each driven a joint past
> 1.7 rad. Read the boot log before the robot moves.

| Symptom | Cause |
| --- | --- |
| Plausible but wrong actions on the engine path | the engine was baked for another checkpoint. `unset SPLIT_DIR CKPT PROMPT` when switching task or checkpoint — the server compares `build_meta.json`, but an engine baked before that sidecar only warns |
| Wrong actions right after connecting | the server may be a forgotten one running another task's model. Check the boot log's checkpoint path, `num_inference_steps` and `FlexPi inference regime:` line |
| Boot hangs on a joint model | `--glue-cache` with `--torch-compile`; the server rejects the pair ([`INFERENCE_OPTIMIZATION.md §3`](INFERENCE_OPTIMIZATION.md#3-what-runs-where)) |
| Boot hangs in the inductor compile on a 34-D checkpoint | serve eager (`--no-torch-compile`, the `real_yam.yaml` default) |
| `observation['depth'] is missing` on every request | the pointmap stream is present and the client sends no depth (§D's regime table) |
| `Could not locate config.yaml next to the checkpoint` / `Could not auto-locate dataset_stats.json` | no `config.yaml` / `dataset_stats.json` within five directories above the weights; `--dataset-stats-path` names the stats file directly |
| `model._target_ is '…', not a flexpi target` at boot | the `config.yaml` above the weights was saved by the pre-release codebase; the released checkpoints ship one written for this tree |
| `action_dim_mask: not armed (...)` on a 34-D checkpoint | the wrong `config.yaml` was found above the weights, the checkpoint trained without the mask, or `EVALUATION.action_dim_mask=false` in `configs/real_yam.yaml` |
| `camera_intrinsics.json missing` at launch | the dataset has no `meta/camera_intrinsics.json` — rebuild it (step 1), or write the file by hand |
| `depth not in canonical layout` | RGB/depth keys are not `cam_high` / `cam_left_wrist` / `cam_right_wrist`, or the build used `--depth-codec none` |
| `empty prompt list` / `metadata prompt path must yield list[str]` during a build | the capture's `language.prompt` is missing, empty, or not a list of strings; fix it upstream — that string is also what you send at deploy time |
| `[validate] state32 round-trip failed` | that episode's `action` field is not a valid SE(3); re-record or drop it |
| `Cannot detect model type for wan_video_vae ... File: []` | `DIFFSYNTH_MODEL_BASE_PATH` is wrong or empty — the Wan2.2 weights were not found |
| NaN actions | a zero rotation block in `observation/state` — the 6D blocks (`3:9`, `12:18`) are the first two rows of R, so identity is `[1, 0, 0, 0, 1, 0]`, not zeros |
| Shape mismatch against the engine | engine built at a different `--ctx-len`; rebuild at the trained `tokenizer_max_len + 1` |
| Robot stalls between chunks | joint denoise is ~3x action-only at 4 steps; use template A, or C if that is still too slow |
| Periodic arm sag at chunk boundaries | relative actions anchored on the measured pose instead of the last commanded one, so gravity droop compounds every chunk ([wire contract](#wire-contract)); grippers (`18:20`) stay measured either way |
| First robot frame stalls for seconds | the regime was overridden per-request, so the boot warmup compiled the wrong one. The server owns the regime — do not send `joint_*` / `present_*` from the client |
| First request after boot takes seconds | `--offload-text-encoder`: the first new prompt runs T5 on the CPU; later requests hit the cache |
| `ModuleNotFoundError: experiments.yam.flexpi_policy` | this repo's root is not on `PYTHONPATH`, or another checkout earlier on `sys.path` supplies `experiments.yam` |
