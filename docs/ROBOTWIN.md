# RoboTwin 2.0 — training and evaluation

End-to-end recipe for the RoboTwin 2.0 benchmark: 50 bimanual tasks, each scored
under both a clean and a domain-randomized phase.

Everything here runs from the **repository root**. One-time
environment and weight setup is [`docs/INSTALL.md`](INSTALL.md); the
per-knob reference behind the launcher is
[`docs/TRAINING.md`](TRAINING.md).

| | |
|---|---|
| Task config | `robotwin_unified_flex_3cam_384_1e-4` |
| Cameras | 3 (head + two wrists), composited into a T at 384&times;320 |
| Action | 14-D native joint space |
| Train launcher | `scripts/train_flexpi_robotwin.sh` |
| Eval entry point | `experiments/robotwin/run_robotwin_manager.py` |

---

## 1. Data

The preprocessed set is published at
[`flex-pi/robotwin_3d`](https://huggingface.co/datasets/flex-pi/robotwin_3d):

```bash
huggingface-cli download flex-pi/robotwin_3d --repo-type dataset \
  --local-dir ./data/robotwin2.0_3d/robotwin2.0_3d
```

That is the path `configs/data/robotwin.yaml` expects, and it reads the
`x264rgb` depth tree by default.

To convert your own captures instead, the layout is in
[`docs/TRAINING.md §1.2`](TRAINING.md). The pointmap stream also needs
`meta/camera_intrinsics.json` in every dataset dir; the published set ships it.

Which tasks and episodes exist is read from
`src/flexpi/datasets/task_episode_map.json` — that file is what `TASK_NAMES=all`
resolves against, and it is the authority on the 50-task list.

---

## 2. Train

Training reads T5 text embeddings from `./data/text_embeds_cache_3d`, the path
`configs/data/robotwin.yaml` expects. Download the published cache, or encode it
yourself:

```bash
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"   # Wan2.2 weights

huggingface-cli download flex-pi/robotwin_3d_text_embeds_cache \
  --repo-type dataset --local-dir ./data/text_embeds_cache_3d

# or encode it yourself — ~1M prompts on the full set, and a re-run only fills gaps
python scripts/precompute_text_embeds.py \
  task=robotwin_unified_flex_3cam_384_1e-4

bash scripts/train_flexpi_robotwin.sh
```

Extra Hydra overrides pass straight through:
`bash scripts/train_flexpi_robotwin.sh batch_size=4 num_epochs=10`.

Training without the 3D stream takes both halves — the model flag stops the
model using depth, the data config stops the loader decoding it:

```bash
bash scripts/train_flexpi_robotwin.sh model.enable_pointmap=false data=robotwin_nodepth
```

Evaluation needs no counterpart flag; it rebuilds the model from the run's saved
`config.yaml`.

### Demonstration efficiency

The data-scarce results are the same launcher with one knob:

```bash
NUM_EPISODES_PER_TASK=50  bash scripts/train_flexpi_robotwin.sh
NUM_EPISODES_PER_TASK=100 bash scripts/train_flexpi_robotwin.sh
```

The sample is seeded, and the count is stamped into the output directory name
(`all_perTask100_epoch6`).

### LoRA fine-tuning of the released checkpoint

`robotwin_lora_release_3cam_384_3e-4` fine-tunes
[`flex-pi/flexpi-robotwin`](https://huggingface.co/flex-pi/flexpi-robotwin) with
LoRA on one GPU ([`TRAINING.md §7`](TRAINING.md#7-lora-fine-tuning)):

```bash
CUDA_DEVICES=0 NPROC_PER_NODE=1 TASK_NAMES=hanging_mug \
TASK_CONFIG=robotwin_lora_release_3cam_384_3e-4 bash scripts/train_flexpi_robotwin.sh
```

### Outputs

```
runs/robotwin_unified_flex_3cam_384_1e-4/<task_tag>/<run_id>_<regime_tag>/
├── config.yaml                       # the trained config — eval reads this back
├── dataset_stats.json                # normalizer statistics
├── checkpoints/weights/step_NNNNNN.pt
└── checkpoints/state/step_NNNNNN/    # DeepSpeed full state, for RESUME
```

---

## 3. Evaluate

### Installing the simulator

RoboTwin installs into the FlexPi environment, but never one that also carries
LIBERO — [`docs/INSTALL.md §4.1`](INSTALL.md#41-robotwin). Follow the
[official instructions](https://robotwin-platform.github.io/doc/usage/robotwin-install.html);
`assets/` and `envs/curobo` are not redistributed here.

The policy wrapper is already checked in as a relative symlink at
`third_party/RoboTwin/policy/flexpi_policy`, and `eval_robotwin_single.py`
recreates it if it goes missing.

### The full sweep

Set `CKPT` and `DATASET_STATS` in the config block at the top of the launcher,
then run it:

```bash
bash scripts/eval_flexpi_robotwin.sh
```

Every task runs twice — `demo_clean`, then `demo_randomized` — dispatched across
`NUM_GPUS` &times; `MAX_TASKS_PER_GPU` workers. The same block carries the regime
flags and the knobs a long sweep needs: `PHASES` for one phase only, `TASKS` to pin
a subset, `TASK_SHARD=N/K` to split across machines, and `RESUME_DIR` to pick up a
crashed sweep (finished `(task, phase)` results are skipped). These are plain
assignments, so edit them in the file — an env var in front of `bash` will not win.

One task on one GPU is `bash scripts/eval_flexpi_robotwin_single.sh`, the same style
of config block with the 50-task name list in it.

Both wrap Hydra entry points, if you would rather drive those directly:

```bash
python experiments/robotwin/run_robotwin_manager.py \
  task=robotwin_unified_flex_3cam_384_1e-4 \
  ckpt=runs/.../checkpoints/weights/step_NNNNNN.pt \
  EVALUATION.dataset_stats_path=runs/.../dataset_stats.json \
  MULTIRUN.num_gpus=8 MULTIRUN.max_tasks_per_gpu=2
```

### Evaluation defaults worth knowing

| Setting | Default | Note |
|---|---|---|
| `EVALUATION.eval_num_episodes` | 100 | per task, per phase |
| `MULTIRUN.phases` | `[clean, random]` | the sweep runs both phases per task. `EVALUATION.task_config` picks one (`demo_clean` / `demo_randomized`) for a single-task run |
| `EVALUATION.instruction_type` | `unseen` | held-out instructions, following Motus. Some prior work reports **seen**, which scores a point or two higher |
| `EVALUATION.replan_steps` | 32 | the whole 32-action chunk, then re-observe. Lower is more reactive |
| `EVALUATION.skip_get_obs_within_replan` | `true` | skips RGB rendering mid-chunk. Much faster, but saved videos look very low-FPS — set `false` for presentable video |
| `EVALUATION.offload_text_encoder` | `false` | `true` keeps T5 on CPU, saving ~10 GB VRAM at ~200–300 ms per call |

### Choosing the regime

Architecture is auto-loaded from the `config.yaml` saved next to the checkpoint —
geometry, DINO grid and action dim come from *training*, so `model.*` overrides on
the eval CLI are **ignored**. The regime is selected
with the `EVALUATION.infer_*` flags instead — `infer_joint_*` picks what gets
generated, `infer_present_*` picks what gets encoded as input:

```bash
# action only — the fast path
python experiments/robotwin/run_robotwin_manager.py task=... ckpt=... \
  +EVALUATION.infer_joint_video=false \
  +EVALUATION.infer_joint_dino=false \
  +EVALUATION.infer_joint_pointmap=false

# full joint generation — the accurate path
python experiments/robotwin/run_robotwin_manager.py task=... ckpt=... \
  +EVALUATION.infer_joint_video=true \
  +EVALUATION.infer_joint_dino=true \
  +EVALUATION.infer_joint_pointmap=true

# no depth sensor at deploy time: drop the pointmap input, keep predicting its
# future from RGB + DINO (needs a checkpoint trained with cross-modal forcing)
python experiments/robotwin/run_robotwin_manager.py task=... ckpt=... \
  +EVALUATION.infer_present_pointmap=false \
  +EVALUATION.infer_joint_pointmap=true
```

`EVALUATION.num_inference_steps` sets the flow-matching Euler steps, and the eval
config pins it to 4 — ~2&times; faster than 10 at close to the same success rate. The
measured latency stacks and the TensorRT builds are in
[`INFERENCE_OPTIMIZATION.md`](INFERENCE_OPTIMIZATION.md).

---

## 4. Troubleshooting

Environment symptoms are in [`INSTALL.md §5`](INSTALL.md#5-troubleshooting),
training-side ones — OOM included — in
[`TRAINING.md §5`](TRAINING.md#5-troubleshooting).

| Symptom | Cause |
|---|---|
| Segfault in `robot.set_planner()` | a user-site numpy&nbsp;2 shadowing the env — `export PYTHONNOUSERSITE=1` |
| Mesa "unknown PCI ID" warnings / Vulkan picks the iGPU | pin `VK_ICD_FILENAMES` to the NVIDIA ICD |
| `policy/flexpi_policy` missing | the relative symlink was not preserved; `eval_robotwin_single.py` recreates it |
| Saved eval videos look like a slideshow | `skip_get_obs_within_replan=true` (the default); set `false` |
| Eval ignores a `model.*` override | by design — `eval_config_source=auto` rebuilds the model from the run's own `config.yaml` |
