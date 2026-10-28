# LIBERO — training and evaluation

End-to-end recipe for the LIBERO benchmark: four suites of 10 tasks each, scored
over 50 trials per task under per-suite step budgets following Fast-WAM's setup.

Everything here runs from the **repository root**. One-time environment and
weight setup is [`docs/INSTALL.md`](INSTALL.md); the per-knob reference behind the
launcher is [`docs/TRAINING.md`](TRAINING.md).

Evaluating a released checkpoint needs §1 and §3 only.

---

## 1. Data

All four suites ship preprocessed — RGB, FFV1 depth, and the
`meta/camera_intrinsics.json` the pointmap stream needs (~12 GB):

```bash
huggingface-cli download flex-pi/libero_mujoco3.3.2_depth \
  --repo-type dataset --local-dir ./data/libero_mujoco3.3.2_depth
```

---

## 2. Train

```bash
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"   # Wan2.2 weights

# Once per machine — INSTALL.md §2.1.
python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/flexpi.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda --dtype bfloat16

# Once per dataset. This launcher does not run it for you.
python scripts/precompute_text_embeds.py \
  task=libero_unified_flex_2cam224_32d_rotvec_1e-4

bash scripts/train_flexpi_libero.sh
```

That runs 20 epochs on the `tshape_384x320` composite, the cheaper of the two
camera layouts — it trains faster and reaches comparable success. For the best
numbers, train on `tshape_libero_2cam_448x512` instead, which costs more per
sample and takes correspondingly longer. It is what the released checkpoints
used:

```bash
COMPOSITE_LAYOUT=tshape_libero_2cam_448x512 bash scripts/train_flexpi_libero.sh
```

Whichever you pick, stay on it — the layout is baked into the checkpoint, and
eval reads it back from the run's own `config.yaml`.

Hydra overrides pass straight through, so raise the epoch count with
`bash scripts/train_flexpi_libero.sh num_epochs=40`. The flexible checkpoint
(the `FLEX_P_*` knobs dropped to `0.5`) usually wants that: every sample trains
a different subset of streams, so it converges more slowly than the
fixed-regime run and keeps improving past 20 epochs.

Training without the 3D stream takes both halves — the model flag stops the
model using depth, the data config stops the loader decoding it:

```bash
bash scripts/train_flexpi_libero.sh model.enable_pointmap=false data=libero_nodepth
```

Evaluation needs no counterpart flag; it rebuilds the model from the run's saved
`config.yaml`.

`libero.yaml` does not set `emit_action_dim_is_pad`, so training averages the
action loss over all 32 channels of the 7 → 32 scatter, as for the released
checkpoints ([`MIXED_EMBODIMENT_34D.md §2`](MIXED_EMBODIMENT_34D.md#2-the-per-sample-pad-mask)).

### Outputs

```
runs/libero_unified_flex_2cam224_32d_rotvec_1e-4/<run_id>_<regime_tag>/
├── config.yaml                       # the trained config — eval reads this back
├── dataset_stats.json                # normalizer statistics
├── checkpoints/weights/step_NNNNNN.pt
└── checkpoints/state/step_NNNNNN/    # DeepSpeed full state, for RESUME
```

---

## 3. Evaluate

### The checkpoints

Both launchers take two paths and find the run's `config.yaml` beside them:

| | |
|---|---|
| `CKPT` | `<ckpt-dir>/checkpoints/weights/step_NNNNNN.pt` |
| `DATASET_STATS` | `<ckpt-dir>/dataset_stats.json` |

| Released | Layout | Trained with | Evaluate as | Download |
|---|---|---|---|---|
| dropout | 448&times;512 | stream dropout, every `p` at 0.5 | action-only **or** full joint | [Hugging Face](https://huggingface.co/flex-pi/flexpi-libero) |
| full joint | 448&times;512 | no dropout, always jointly denoised | full joint | [Hugging Face](https://huggingface.co/flex-pi/flexpi-libero-fulljoint-star) |

Architecture is auto-loaded from that saved `config.yaml` — geometry, DINO grid
and action dim come from *training*, so both take the same command and
`model.*` / `data.*` overrides on the eval CLI are **ignored**.

Evaluation rescales K by `(w − 1)/(W − 1)` and regrids depth with the
linspace-endpoint nearest sampler, as training does. The published success rates
predate this (K by `w/W`, torchvision nearest) and have not been re-measured.

### Installing LIBERO

LIBERO installs into the FlexPi environment, but never one that also carries
RoboTwin — [`INSTALL.md §4`](INSTALL.md#4-simulators-evaluation-only) covers that
and how to build a second environment if you need both. LIBERO itself is never
pip-installed; the launchers reach it through `PYTHONPATH`, which they export
themselves. The install, the `requirements.txt` warning and the binding check are
[`INSTALL.md §4.2–4.3`](INSTALL.md#42-libero) — run the check before spending
hours on a bad path.

### The full sweep

The standard protocol shards the 40 tasks round-robin across GPUs, loads the
model once per GPU, and writes `summary_4suite.{csv,json}`:

```bash
CKPT=<...>/checkpoints/weights/step_NNNNNN.pt \
DATASET_STATS=<...>/dataset_stats.json \
GPUS=0,1,2,3,4,5,6,7 \
  bash scripts/eval_flexpi_libero_4suite.sh
```

A partial sweep is reported as `INCOMPLETE` with actual/expected counts rather
than averaged silently.

One (suite, task) on one GPU — the smallest unit, and the smoke test worth
running first:

```bash
CKPT=... DATASET_STATS=... TASK_SUITE_NAME=libero_object TASK_ID=0 NUM_TRIALS=2 \
  bash scripts/eval_flexpi_libero_single.sh
```

---

## 4. Troubleshooting

Environment symptoms — torchcodec, FFmpeg, user-site shadowing — are in
[`INSTALL.md §5`](INSTALL.md#5-troubleshooting); training-side ones, OOM
included, in [`TRAINING.md §5`](TRAINING.md#5-troubleshooting).

| Symptom | Cause |
|---|---|
| `third_party/LIBERO is empty` | the submodule was never initialized — `git submodule update --init third_party/LIBERO` |
| eval refuses to start, naming mujoco | any version other than 3.3.2 — `pip install --no-deps mujoco==3.3.2` |
| `camera_intrinsics not found` | `DATA_ROOT` has no `meta/camera_intrinsics.json`; the §1 download did not complete |
| success rates far below the published numbers | the mujoco pin, or the `INFER_JOINT_*` flags against the checkpoint |
| a wall of `EGLError` **after** the result line | robosuite's EGL context at interpreter shutdown, prefixed `Exception ignored in:` — almost always a *successful* run. Judge by the exit code and `gpu*_task*_results.json` |
