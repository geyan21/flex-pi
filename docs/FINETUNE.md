# Fine-tuning the preview checkpoints

How the two preview checkpoints ([`PRETRAINED.md`](PRETRAINED.md)) warm-start a
run on a LeRobot v2.1 dataset through two task presets. Launcher knobs are
[`TRAINING.md`](TRAINING.md); the 34-D layout is
[`MIXED_EMBODIMENT_34D.md`](MIXED_EMBODIMENT_34D.md).

## 1. What this codebase provides for fine-tuning

- **DeepSpeed fp32-master re-seed.** Under ZeRO-1 (the launchers' default) a
  `PRETRAINED_CKPT` warm start of the initial public release was reverted at the
  first optimizer step; the trainer now re-seeds the master copy after the load
  ([`TRAINING.md §6`](TRAINING.md#6-deepspeed)).
- **The 34-D data path**: the `yam34` preset, the per-sample pad mask in the loss,
  and the processor's `action_norm_clamp` key.
- **LoRA fine-tuning** (`lora.enabled=true`,
  [`TRAINING.md §7`](TRAINING.md#7-lora-fine-tuning)): the MoT frozen, 81-82 M
  trainable parameters, on one 24 GB card.

## 2. The presets

| | `flexpi-pretrain-preview` | `flexpi-pretrain-rgb-preview` |
|---|---|---|
| task preset | `yam_finetune_pretrain_preview_3cam_34d_rel_1e-4` | `yam_finetune_pretrain_rgb_preview_3cam_34d_rel_1e-4` |
| modalities | RGB + DINO + depth (with intrinsics); pointmap stream trains | RGB + DINO; pointmap stream off |
| DINO temporal stride | 1 | 2, keep-far |
| norm stats (from the download) | `dataset_stats_rel32.json`: the checkpoint's statistics without channels 26 and 33 (the 7th-joint slots a YAM arm does not use), the 32-wide form `yam34` normalizes with | `dataset_stats.json`: the checkpoint's own statistics |
| action clamp | `action_norm_clamp: .inf` (as trained) | `action_norm_clamp: .inf` (as trained) |
| `pretrained_ckpt` | `./checkpoints/flexpi-pretrain-preview/checkpoints/weights/step_190000.pt` | `./checkpoints/flexpi-pretrain-rgb-preview/checkpoints/weights/model.pt` |

Both use the `yam34` data chain of `yam_unified_flex_3cam_34d_rel_1e-4`, with MoT
gradient checkpointing, strict-shape loading, batch 9 per GPU, lr 1e-4, 10 epochs, `save_every: 1000`.

## 3. Download

```bash
huggingface-cli download flex-pi/flexpi-pretrain-preview     --local-dir ./checkpoints/flexpi-pretrain-preview
huggingface-cli download flex-pi/flexpi-pretrain-rgb-preview --local-dir ./checkpoints/flexpi-pretrain-rgb-preview
```

`PRETRAINED_CKPT=<path>` points the launcher elsewhere; the environment is
[`INSTALL.md`](INSTALL.md).

## 4. Data requirements

- LeRobot v2.1 with camera keys `cam_high`, `cam_left_wrist`, `cam_right_wrist`
  ([`TRAINING.md §1.2`](TRAINING.md#12-dataset-layout-lerobot-v21-canonical-robotwin-cam-names)),
  32-D eef32 state and action in `yam_eef.STATE_LAYOUT`, 30 fps. The YAM builder
  writes this ([`YAM.md §1`](YAM.md#1-data-build)).
- `flexpi-pretrain-preview` also reads depth (`observation.depth_ffv1.<cam>`) and
  `meta/camera_intrinsics.json`; [`scripts/da3_depth/`](../scripts/da3_depth/README.md)
  adds depth to an RGB-only dataset.
- The released flex-pi YAM datasets fit as downloaded.

## 5. Norm stats

The presets read the checkpoint's own statistics from its download, which keeps
the scale the action head learned to emit. The alternative is to refit on the target data, either
per-episode by the trainer on the first run (`data.train.pretrained_norm_stats=null`)
or frame-pooled:

```bash
python scripts/norm_stats/fit_pooled_stats.py --src ./data/yam/my_task \
    --data-config yam34 --corpus-name my_task --out ./my_task_pooled_rel32.json
# then: data.train.pretrained_norm_stats=./my_task_pooled_rel32.json
```

Trade-off: keeping the preset's file preserves the targets the head was trained
against; a refit fits the new data's range but moves those targets, and the
convention matters as much as the file (a per-episode box is several times wider
than a frame-pooled one). A refit for either checkpoint stays frame-pooled with
the clamp off.

## 6. Text embeddings

`scripts/train_flexpi_yam.sh` precomputes the T5 cache before training. By hand:

```bash
python scripts/precompute_text_embeds.py \
    task=yam_finetune_pretrain_rgb_preview_3cam_34d_rel_1e-4 \
    data.train.dataset_dirs="[./data/yam/my_task]"
```

## 7. Launch

```bash
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"

# flexpi-pretrain-preview
TASK_CONFIG=yam_finetune_pretrain_preview_3cam_34d_rel_1e-4 \
DATASET_DIRS="[./data/yam/my_task]" bash scripts/train_flexpi_yam.sh

# flexpi-pretrain-rgb-preview
TASK_CONFIG=yam_finetune_pretrain_rgb_preview_3cam_34d_rel_1e-4 \
DATASET_DIRS="[./data/yam/my_task]" bash scripts/train_flexpi_yam.sh
```

`TASK_CONFIG` is required: the launcher's default is a 32-D preset whose head
does not match. The `FLEX_P_*` knobs are read from the environment; each one set
overrides the preset's `flex_joint` value, and unset ones keep it (0.5 for all six
in the base preset; the RGB preset sets both pointmap probabilities to 0, which
setting them above 0 would undo against data loaded without depth).
`model.skip_dit_load_from_pretrain=true model.action_dit_pretrained_path=null`
after the command skips loading the Wan2.2 DiT and ActionDiT init that the
checkpoint overwrites anyway.

## 8. What a correct start logs

- `Loading pretrained weights (warm-init, fresh optimizer/scheduler/step): <path>`;
- `Checkpoint step=N action_dim_mask_trained=None`;
- `Re-seeded DeepSpeed fp32 master params after warm-init (...)`;
- after step 1, `warm-init canary: weights survived the first step (drift=…)`;
- `Pointmap stream GLOBALLY OFF (...)` for `flexpi-pretrain-rgb-preview`;
- no `shape-mismatched key(s) in mot` and no `MoT keys kept at random init` line.

## 9. Hardware

Measured on one A100-80GB, single process: batch 9 with MoT checkpointing takes
51.7 GiB, without it runs out of memory. AdamW's fp32 master copy and moments come
to about 72.5 GB for the 6.04 B trainable parameters, partitioned across GPUs by
ZeRO-1, so one 80 GB card does not fit a fine-tune by that estimate. The launchers
default to 8 GPUs; the minimum count has not been measured. Each save is 12.08 GB
of weights plus about 81 GB of DeepSpeed state.

A LoRA fine-tune trains 81-82 M parameters and fits one card: with
`lora.enabled=true batch_size=1 model.mot_checkpoint_mixed_attn=true` the peak is
17.7 GiB allocated on `yam_finetune_pretrain_preview_3cam_34d_rel_1e-4` and 16.1 GiB
on `yam_finetune_pretrain_rgb_preview_3cam_34d_rel_1e-4`, about 0.9 s per sample
on an A100 ([`TRAINING.md §7`](TRAINING.md#7-lora-fine-tuning)).
Its saves are the merged 12.08 GB weights, a 160 MB adapter file and about 14 GB
of state.

## 10. Troubleshooting

| Symptom | Cause |
|---|---|
| Out of memory at batch 9 | MoT checkpointing off (the plain 34-D preset's setting), or a card under 80 GB; `batch_size=` lowers the per-GPU batch |
| `camera_intrinsics.json missing` / `depth not in canonical layout` on RGB-only data | the pointmap stream is on (the base preset, or `FLEX_P_PRESENT_POINTMAP` / `FLEX_P_JP` above 0) |
| `Checkpoint shape mismatch with strict_shape=True` | `TASK_CONFIG` is a 32-D preset, or the model block was overridden away from the checkpoint's geometry |
| A tensor shape mismatch at the normalizer | a 34-wide stats file under `yam34`, which normalizes at width 32 |
| `Missing text embedding cache: <hash>.t5_len128...pt` | the precompute ran for other dataset dirs or another task |
| Loss starts near a from-scratch run's | no `Re-seeded DeepSpeed fp32 master params` line, or an auto-resume of the run's own output directory skipped the warm start |
