# Training a FlexPi Unified-Flex model

How to train the single checkpoint that serves every inference regime. What the
model is and why it works that way is [`OVERVIEW.md`](OVERVIEW.md); the
environment is [`INSTALL.md`](INSTALL.md); the per-benchmark recipes are
[`ROBOTWIN.md`](ROBOTWIN.md), [`LIBERO.md`](LIBERO.md) and [`YAM.md`](YAM.md).

- [1. Prerequisites](#1-prerequisites)
  - [1.1 Environment](#11-environment)
  - [1.2 Dataset layout](#12-dataset-layout-lerobot-v21-canonical-robotwin-cam-names)
  - [1.3 Precompute text embeddings](#13-precompute-text-embeddings-required)
- [2. Launching a run](#2-launching-a-run)
- [3. The flex knobs](#3-the-flex-knobs)
- [4. Finetune vs Resume](#4-finetune-vs-resume)
- [5. Troubleshooting](#5-troubleshooting)
- [6. DeepSpeed](#6-deepspeed)
- [7. LoRA fine-tuning](#7-lora-fine-tuning)

---

## How a run is configured

Everything is Hydra, and one `task=` picks the entire run. `configs/train.yaml`
is the root and selects nothing by itself:

```
task=libero_unified_flex_2cam224_32d_rotvec_1e-4
 └── configs/task/<that name>.yaml     batch size, lr, epochs, save/eval cadence
      ├── override /data: libero    →  configs/data/libero.yaml
      │                                dataset dirs, shape_meta, action processor,
      │                                text_embedding_cache_dir
      └── override /model: flexpi   →  configs/model/flexpi.yaml
                                       architecture and the flex_joint defaults
```

So *which dataset* and *which architecture* are both consequences of the task
config, and every entry point that needs them takes the same `task=` — training,
the text-embed precompute, evaluation. Anything after it on the command line is a
Hydra override and wins over all three files:

```bash
bash scripts/train_flexpi_libero.sh batch_size=4 num_epochs=10
```

The shipped task configs are one per setting (each benchmark doc names its own),
plus the 34-D presets of [`MIXED_EMBODIMENT_34D.md`](MIXED_EMBODIMENT_34D.md) and
the two preview fine-tune presets of [`FINETUNE.md`](FINETUNE.md).

---

## 1. Prerequisites

### 1.1 Environment

Finish [`INSTALL.md`](INSTALL.md) first — environment, Wan2.2 weights, DINOv3.
Everything below runs from the project root:

```bash
conda activate flexpi
cd flex-pi
```

### 1.2 Dataset layout (LeRobot v2.1, canonical RoboTwin cam names)

Every dataset directory must be a LeRobot **v2.1** dataset with **canonical camera
keys** and, for the 3D stream, **depth videos + camera intrinsics**:

```
<dataset_dir>/
├── meta/
│   ├── info.json
│   ├── episodes.jsonl
│   ├── tasks.jsonl                       # language instructions (used by text-embed precompute)
│   ├── camera_intrinsics.json            # REQUIRED for the pointmap/3D stream
│   └── episode_intrinsics.jsonl          # OPTIONAL: one K per episode (see below)
└── videos/chunk-000/
    ├── observation.images.cam_high/        episode_000000.mp4 ...
    ├── observation.images.cam_left_wrist/  ...
    ├── observation.images.cam_right_wrist/ ...
    ├── observation.depth_ffv1.cam_high/    episode_000000.mkv ...   # FFV1 gray16 depth
    ├── observation.depth_ffv1.cam_left_wrist/ ...
    └── observation.depth_ffv1.cam_right_wrist/ ...
```

`meta/info.json` must declare `"codebase_version": "v2.1"` — that is what the
vendored loader in [`src/flexpi/datasets/lerobot/`](../src/flexpi/datasets/lerobot/)
is pinned to, and what every published dataset and the YAM builder write. v2.0 still
reads, by falling back to `meta/stats.json` for the per-episode statistics; v3.0,
which reorganizes the files on disk, does not.

The loader looks for exactly those keys, and the pointmap stream needs
`meta/camera_intrinsics.json` beside them. How both are produced — the raw layout
in, the LeRobot layout out, and the intrinsics convention — is the dataset builder:
[`YAM.md` step 1](YAM.md#1-data-build).

#### Per-episode intrinsics

`camera_intrinsics.json` holds one K per camera for the whole dataset, averaged
by the builder, which mis-scales the pointmap of a capture spanning rigs with
different optics. When a dataset carries `meta/episode_intrinsics.jsonl`, the
loader uses each episode's own K instead. One JSON object per line:

```json
{"episode_index": 0, "intrinsics": {"cam_high": {"fx": 434.0, "fy": 434.0, "cx": 320.0, "cy": 240.0, "width": 640, "height": 480}, "cam_left_wrist": {...}, "cam_right_wrist": {...}}}
```

Same units, grid and endpoint rescale as `camera_intrinsics.json`, which stays
required. The YAM builder writes the file; datasets without it use the averaged K.
An episode missing from the table is an error.

> **All datasets you combine must share the same fps, camera layout, and action
> dimensionality.** This is enforced at runtime — an fps mismatch raises.

### 1.3 Precompute text embeddings (required)

T5 embeddings are cached to disk; training dies on the first batch without them.
The cache has to cover **every prompt in every dataset dir you train on** —
lookup is by prompt hash, and narrowing `TASK_NAMES` does not narrow what the
cache needs.

```bash
python scripts/precompute_text_embeds.py task=<TASK_CONFIG>

# Your own dataset, on an existing task config — pass the same dirs you will
# train on. Without this it scans the placeholder path in the data config:
python scripts/precompute_text_embeds.py \
  task=yam_unified_flex_3cam_32d_rel_1e-4 \
  data.train.dataset_dirs='[./data/dataset_a]'

# A dataset with a data config but no task config yet:
bash scripts/run_precompute_text_embeds.sh <data_config_name>
```

There is no `--dataset-dir` flag because a dataset directory is not enough to
run the pass. The prompts come from each dir's `meta/tasks.jsonl`, but the output
goes to `text_embedding_cache_dir` — a directory of its own, outside the dataset,
shared by every dir in that data config — and the filenames carry the data
config's `context_len`. Both are properties of the config, which is why `task=`
(or `data=`) is the key on both sides: it is what makes the precompute and the
training run agree on where the cache is.

A re-run only encodes what the cache is missing. `scripts/train_flexpi_yam.sh`
runs this for you; the other two launchers do not.

---

## 2. Launching a run

One launcher per setting — RoboTwin, LIBERO, real-world YAM. They are maintained
in lockstep and expose the same knobs, differing only in defaults; each one's
config block is the reference for what it sets.

```bash
export CUDA_DEVICES=0,1,2,3,4,5,6,7
export NPROC_PER_NODE=8
export DATASET_DIRS='[./data/yam/my_task]'

bash scripts/train_flexpi_yam.sh batch_size=3
```

**Not every knob reads the environment, and which ones do differs per launcher.**
A line written `VAR=value` is in-script only — exporting it does nothing. A line
written `VAR="${VAR:-value}"` honours an export. Every `FLEX_P_*` is in-script in
the LIBERO and RoboTwin launchers and read from the environment by the YAM one
(§3). Check the launcher before relying on `export`.

The path that always works is a **Hydra override on the command line** — every
launcher appends `"$@"` last, so a CLI value beats whatever the script set:

```bash
bash scripts/train_flexpi_robotwin.sh model.flex_joint.p_jv=1.0 batch_size=4
```

All three launch through `accelerate` with
`scripts/accelerate_configs/accelerate_zero1_ds.yaml`; `ACCELERATE_CONFIG` points
them at one of the others in that directory (ZeRO-2, DDP, single-GPU, ZeRO-0).

**Action normalization clamp.** In `q01/q99` mode the normalizer clamps action
targets to ±1 (±5 in `z-score`). The processor key `action_norm_clamp` sets
another bound; `.inf` (in YAML; `++data.train.processor.action_norm_clamp=inf`
on the command line) leaves targets unclamped, as frame-pooled stats are
trained. State keeps the default clamp.

**Action pad mask.** A data preset whose merger leaves channels empty passes them
to the loss as masked only with `emit_action_dim_is_pad: true` (set by `yam34`);
presets without it train on every channel
([`MIXED_EMBODIMENT_34D.md`](MIXED_EMBODIMENT_34D.md)).

---

## 3. The flex knobs

Every training step, each stream is independently sampled as **present or absent**, and
each present stream is independently sampled as **jointly fused or not**. This is what lets
one checkpoint serve all 8 regimes.

The `FLEX_P_*` knobs are plain config-block assignments in `robotwin.sh` and
`libero.sh` — edit the file; `export` does not reach them. `robotwin.sh` ships the
`0.5` defaults below; `libero.sh` ships all six at `1.0`. `yam.sh` reads each from
the environment and overrides only those that are set; unset, the task preset's
`flex_joint` value applies (the `0.5` below, except the RGB-only fine-tune preset,
whose pointmap probabilities are 0).

| Knob | Default | Meaning |
|------|---------|---------|
| `FLEX_P_PRESENT_VIDEO` | `0.5` | P(video stream provided this sample) |
| `FLEX_P_PRESENT_DINO` | `0.5` | P(dino stream provided this sample) |
| `FLEX_P_PRESENT_POINTMAP` | `0.5` | P(3D/pointmap stream provided this sample) |
| `FLEX_P_JV` / `FLEX_P_JD` / `FLEX_P_JP` | `0.5` | P(video / dino / pointmap is *jointly* fused, i.e. action attends all of its tokens) |
| `model.flex_joint.cross_modal_predict_video` / `_dino` / `_pointmap` (Hydra override, not an env var) | `true` | when a stream is **absent**, still denoise its tokens conditioned on the other streams (the "forecasting" regime) instead of fully masking it |

Tips:

- **Set a probability to `1.0` to disable that dropout** (stream always present). E.g.
  `FLEX_P_PRESENT_DINO=1.0` trains DINO as an always-on conditioning input.
- **Set to `0.0` to switch a stream fully off** for the run. For the pointmap the
  master switch is `model.enable_pointmap=false` (pair it with `data=<name>_nodepth`
  so the loader stops decoding depth nothing will read).
- `model.joint_video` / `_dino` / `_pointmap` (default `true`, Hydra override) set the
  *trained default* joint flags used when `infer_action` is called at eval **without**
  runtime overrides; per-sample sampling still randomizes every step regardless.
- The chosen probabilities are baked into the run name / `REGIME_TAG`
  (e.g. `flex_pv0.5_pd0.5_pp0.5_jv0.5_jd0.5_jp0.5`) so runs are
  self-documenting.

Example — train video always-on, dino/3d dropped 50%, no cross-modal forecasting:

```bash
# in scripts/train_flexpi_robotwin.sh
FLEX_P_PRESENT_VIDEO="1.0"
FLEX_P_PRESENT_DINO="0.5"
FLEX_P_PRESENT_POINTMAP="0.5"
```

Cross-modal forcing has no launcher variable at all — pass it as a Hydra override,
which every launcher forwards verbatim:

```bash
bash scripts/train_flexpi_robotwin.sh \
  model.flex_joint.cross_modal_predict_video=false \
  model.flex_joint.cross_modal_predict_dino=false \
  model.flex_joint.cross_modal_predict_pointmap=false
```

---

## 4. Finetune vs Resume

Mutually exclusive; `RESUME` wins if both are set.

```bash
# Weights only — fresh warmup + cosine, new output directory tagged _ft:
PRETRAINED_CKPT=/path/to/run/checkpoints/weights/step_030000.pt \
  bash scripts/train_flexpi_robotwin.sh

# Weights + optimizer + LR schedule + step counter:
RESUME=/path/to/run/state/step_030000 bash scripts/train_flexpi_robotwin.sh
```

`PRETRAINED_CKPT_STRICT_SHAPE=true` aborts on any shape mismatch; the default
`false` re-initializes whatever differs. `action_dim`, `composite_layout` and
`dino_pixel_unshuffle` all count, so a warm start has to match the donor's
geometry rather than today's default.

A weights file records `action_dim_mask_trained`, whether that run trained with
the pad mask; deploy arms the mask from it, and a payload without the key falls
back to the saved `config.yaml`'s `emit_action_dim_is_pad`. Under DeepSpeed a warm start took when
the log carries `Re-seeded DeepSpeed fp32 master params after warm-init` ([§6](#6-deepspeed)).
Warm-starting from the preview checkpoints is [`FINETUNE.md`](FINETUNE.md).

---

## 5. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `All dataset_dirs must have the same fps` | one dir has a different fps; rebuild it or drop it from `DATASET_DIRS`. |
| `camera_intrinsics.json missing` | the 3D stream needs intrinsics per dir; write them, or disable 3D with `model.enable_pointmap=false data=<benchmark>_nodepth`. |
| `samples_per_epoch must be a positive int when dataset_weights is set` | set `SAMPLES_PER_EPOCH` whenever you pass `DATASET_WEIGHTS`. |
| `dataset_weights (N) must be parallel to dataset_dirs (M)` | the weight list length must equal the number of dirs. |
| Realized mixture ≠ weights | weights set the **frame** draw ratio; check the logged `frame counts` and confirm `SAMPLES_PER_EPOCH` is large enough for the ratio to converge. |
| Strict-shape load aborts on finetune | you set `PRETRAINED_CKPT_STRICT_SHAPE=true` and `composite_layout` / `dino_pixel_unshuffle` / `action_dim` differ from the source ckpt; match them, or drop back to the default `false` to let the mismatched modules re-initialize. |
| OOM | lower `batch_size`; a larger DINO grid (`dino_pixel_unshuffle=0`) and the pointmap stream both raise memory, and `model.mot_checkpoint_mixed_attn=true` trades compute for activations. If it still OOMs at `batch_size=1`, the card is too small rather than the batch too big — a 32 GB card runs the forward/eval pass but not the backward. |
| `Missing text embedding cache: <hash>.t5_len128...pt` | re-run `precompute_text_embeds.py` with the **same** `data.train.dataset_dirs` you train on. Narrowing `TASK_NAMES` for training does **not** narrow the cache it needs — the prompt is looked up by hash, so every instruction your episodes carry must have been encoded. |
| `Checkpoint was saved with pointmap_mode=...` / `disable_video_stream=True ... refusing to load` | an ablation-only mode that no longer exists; no migration path — retrain |
| `TypeError: __init__() got an unexpected keyword argument 'dino_temporal_stride'` | you pointed a tool at a **saved run** `config.yaml`. Saved configs carry loader keys the merged `RobotVideoDataset` dropped. Eval never instantiates the dataset, so compose a fresh config instead (`task=...`). |
| `model._target_ is '…', not a flexpi target`, or `ModuleNotFoundError` on a `_target_` of a saved `config.yaml` | the `config.yaml` was saved by the pre-release codebase; the released checkpoints ship one written for this tree |
| `action_dim_is_pad width N != action head width M` | the `data.train.processor` and `model` blocks of a `config.yaml` come from different runs |

---

## 6. DeepSpeed

All three launchers run DeepSpeed ZeRO-1 through `accelerate`
(`scripts/accelerate_configs/accelerate_zero1_ds.yaml`).

- **Warm-start re-seed.** ZeRO clones its fp32 master copy inside `prepare()`, so
  weights loaded afterwards were reverted at the first optimizer step. The trainer
  now re-seeds the master after every post-`prepare()` load (`PRETRAINED_CKPT`
  and a weights-only `RESUME=<file>.pt`) and logs
  `Re-seeded DeepSpeed fp32 master params after warm-init (...)`. A canary logs
  `WARM_INIT_CANARY_FAILED` if the weights drift more than 3 % at step 1; it only
  warns. A full-state `RESUME=<state dir>` restores the master itself.
- **`max_grad_norm`.** It was inert under DeepSpeed in the initial public release.
  The trainer now passes it to DeepSpeed's `gradient_clipping`; the default is
  `null` (no clipping, which is what DeepSpeed runs always did), and any positive
  value clips on every backend. Under the DDP and single-GPU accelerate configs
  the old default of 1.0 did clip; `max_grad_norm=1.0` restores it.
- **Gradient accumulation.** Under DeepSpeed the loss is divided by
  `gradient_accumulation_steps` at the backward call, so `train/grad_norm` is no
  longer inflated by the accumulation factor.

---

## 7. LoRA fine-tuning

`lora.enabled=true` (the `lora:` block of `configs/train.yaml`) fine-tunes a
trained checkpoint through low-rank adapters instead of its MoT weights. The MoT
(both experts) is frozen; adapters (`W x + (alpha / rank) · B A x`, `B` zero at the
start) sit on the Linears `lora.target_modules` names, by default the self- and
cross-attention q, k, v, o and both FFN Linears of every block of both experts,
and the parameters `lora.full_train` names (each expert's input and output layers)
train at full rank. Outside the MoT nothing changes from a full fine-tune. At rank
16 that is 68 M trainable parameters on the RoboTwin release geometry and 81-82 M
on the YAM presets, against about 6.03 B and 6.04 B for a full fine-tune.

- **Start.** A LoRA run needs `pretrained_ckpt` or `RESUME`. The checkpoint loads
  into the model before the adapters attach and before `accelerator.prepare()`;
  a frozen tensor missing from it, unexpected, or of another shape is an error.
  The boot log reads `LoRA: loading the frozen base before attaching adapters`,
  `LoRA attached: rank=16 ...` and `LoRA: trainable ...`. A full-state `RESUME`
  refuses a `lora` block other than the run's.
- **Saves.** `checkpoints/weights/step_N.pt` holds the adapters merged into the
  base, the same payload as a full fine-tune's, which evaluation and the servers
  load unchanged. Each save logs the share of the adapters' change that the bf16
  weights keep (`change kept by the bf16 weights`), about 0.15 after a few steps
  and above 0.99 after a few hundred. `checkpoints/lora/step_N.pt` holds the
  adapters and the other trainable tensors (`flexpi.models.lora.load_adapter`
  attaches them to their base), and `checkpoints/state/step_N/` the full state.
- **Backends.** DeepSpeed ZeRO-1, the launchers' default, on one GPU or several.

**Memory and speed** on one A100-80GB, DeepSpeed ZeRO-1 in one process:

| preset | batch × accumulation | MoT checkpointing | peak allocated | s per sample |
|---|---|---|---|---|
| `robotwin_lora_release_3cam_384_3e-4` | 1 × 32 | on | 18.8 GiB | 0.97 |
| same | 4 × 8 | off | 56.4 GiB | 0.42 |
| `yam_finetune_pretrain_preview_3cam_34d_rel_1e-4` + `lora.enabled=true` | 1 × 8 | on | 17.7 GiB | 0.92 |
| `yam_finetune_pretrain_rgb_preview_3cam_34d_rel_1e-4` + `lora.enabled=true` | 1 × 8 | on | 16.1 GiB | 0.88 |

The batch-1 rows ran with the allocator capped at 22 GiB, as on a 24 GB card,
through an in-training eval and a save; `nvidia-smi` reads about 1.4 GiB more
(CUDA context).

**RoboTwin.** `robotwin_lora_release_3cam_384_3e-4` fine-tunes the released
[`flex-pi/flexpi-robotwin`](https://huggingface.co/flex-pi/flexpi-robotwin)
(downloaded to `runs/flexpi-robotwin`, as in the README) on one GPU, in that
checkpoint's geometry (`composite_layout: null`, `dino_pixel_unshuffle: 0`,
`dino_pred_x0: false`, `dino_stride_keep_far: false`); a warm start through
`robotwin_unified_flex_3cam_384_1e-4` re-initializes its DINO heads.
`TASK_NAMES` picks the tasks:

```bash
CUDA_DEVICES=0 NPROC_PER_NODE=1 TASK_NAMES=hanging_mug \
TASK_CONFIG=robotwin_lora_release_3cam_384_3e-4 bash scripts/train_flexpi_robotwin.sh
```

**YAM.** On the fine-tune presets of [`FINETUNE.md`](FINETUNE.md) the block is a
Hydra override:

```bash
CUDA_DEVICES=0 NPROC_PER_NODE=1 TASK_CONFIG=yam_finetune_pretrain_preview_3cam_34d_rel_1e-4 \
DATASET_DIRS="[./data/yam/my_task]" bash scripts/train_flexpi_yam.sh \
  lora.enabled=true batch_size=1 gradient_accumulation_steps=32 model.mot_checkpoint_mixed_attn=true
```

