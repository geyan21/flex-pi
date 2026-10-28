# Preview checkpoints: zero-shot serving

Two pretrained Flex-π checkpoints (preview) for bimanual YAM-family robots, served
without fine-tuning; a stronger pretrained model will follow. Fine-tuning is
[`FINETUNE.md`](FINETUNE.md); the server is [`YAM.md §3`](YAM.md#3-deploy-server).

## 1. The checkpoints

| | [`flex-pi/flexpi-pretrain-preview`](https://huggingface.co/flex-pi/flexpi-pretrain-preview) | [`flex-pi/flexpi-pretrain-rgb-preview`](https://huggingface.co/flex-pi/flexpi-pretrain-rgb-preview) |
|---|---|---|
| Modalities | RGB + DINO + depth (depth optional at inference) | RGB + DINO |
| Training data | ~4,600 hours: ABC-130k, MolmoAct2, AgiBot World, in-house YAM data | ~4,100 hours: ABC-130k, MolmoAct2, in-house YAM data |
| Training depth | DA3 for ABC-130k, MolmoAct2 and AgiBot World; sensor depth for YAM | none |
| Step | 190,000 | 270,000 |
| Action head | 34-D, shared by 6- and 7-joint arms; channels 26 and 33 (each arm's 7th joint) are unused on YAM, so a YAM client sends and receives 32-D | the same |
| Norm stats (`dataset_stats.json`) | frame-pooled q01/q99, targets unclamped | frame-pooled q01/q99, targets unclamped |

## 2. Where they apply

Zero-shot, both serve only a **bimanual 6-DoF YAM-family rig** with one head
camera (`cam_high`) and two wrist cameras, speaking the 32-D eef32 state (the
34-D head is invisible to the client). AgiBot (no deploy path exists here), LIBERO
and RoboTwin (other cameras and action widths) are fine-tuning targets.

## 3. Download

```bash
huggingface-cli download flex-pi/flexpi-pretrain-preview     --local-dir ./checkpoints/flexpi-pretrain-preview
huggingface-cli download flex-pi/flexpi-pretrain-rgb-preview --local-dir ./checkpoints/flexpi-pretrain-rgb-preview
```

Each repository holds the model card, `config.yaml` (written for this tree),
`dataset_stats.json` (the statistics the run trained with) and the weights, one
12.08 GB `.pt` in `checkpoints/weights/`; `flexpi-pretrain-preview` also carries
`dataset_stats_rel32.json`, the 32-wide form its fine-tune preset reads. The server finds the two sidecars by
walking up from the weights, so the layout stays as downloaded. It also needs
Wan2.2, DINOv3 ([`INSTALL.md §2`](INSTALL.md#2-weights)) and the `[serve]` extra.

## 4. Server commands

```bash
export D=./checkpoints/flexpi-pretrain-preview        # or ./checkpoints/flexpi-pretrain-rgb-preview
export CKPT=$D/checkpoints/weights/step_190000.pt # or model.pt
export PROMPT="<language instruction>"

python -u scripts/serve_yam_flexpi.py \
    --ckpt-path "$CKPT" --dataset-stats-path "$D/dataset_stats.json" \
    --default-prompt "$PROMPT" --host 0.0.0.0 --port 8000 \
    --no-torch-compile --offload-text-encoder \
    --action-horizon 32 --mixed-precision bf16 --num-inference-steps 10 \
    <regime flags>
```

Denoise steps (10 here, 4 in `configs/real_yam.yaml`) are untuned on these
checkpoints. The regime flags are explicit because an unset flag takes the
checkpoint's trained default.

**`flexpi-pretrain-rgb-preview`** (the client never sends depth):

| regime | regime flags |
|---|---|
| action only | `--infer-present-video true --infer-present-dino true --infer-present-pointmap false --infer-joint-video false --infer-joint-dino false --infer-joint-pointmap false` |
| joint video and/or DINO | as above, with `--infer-joint-video true` and/or `--infer-joint-dino true` |

`--infer-present-pointmap true` is not a trained regime (it also makes the server
require depth); video and DINO both absent is refused.

**`flexpi-pretrain-preview`**:

| regime | regime flags | client sends depth and K |
|---|---|---|
| (a) depth in, action only | all three `--infer-present-*` `true`, all three `--infer-joint-*` `false` | yes |
| (b) no depth, action only | (a) with `--infer-present-pointmap false` | no |
| (c) full joint with depth | all six `true` (also the default with no flags) | yes |
| (d) no depth, joint video and DINO | `--infer-present-pointmap false`, `--infer-joint-video true --infer-joint-dino true --infer-joint-pointmap false` | no |

**Boot lines.** `Checkpoint step=N action_dim_mask_trained=None`,
`action_dim_mask: channels [26, 33] zero-filled`, `FlexPi inference regime: ...`,
`Pointmap stream GLOBALLY OFF (...)` (RGB preview only), then `PREWARM DONE`, after
which `/healthz` answers `OK`. Metadata `depth_required` says whether depth is needed.

## 5. Prompts

The client sends the bare instruction (or relies on `--default-prompt`); the server
applies the training template, so a pre-wrapped prompt is wrapped twice. Training
instructions are imperative sentences naming the objects, as in each released
dataset's `meta/tasks.jsonl` ("Pick up the plates from the table one by one and
place each into a slot on the dish rack."); a near-miss degrades without an error.

## 6. Input contract

The flat msgpack request of [`YAM.md`, wire contract](YAM.md#wire-contract):

- **Cameras.** `observation/image_<cam>` for `cam_high`, `cam_left_wrist`,
  `cam_right_wrist`, all required, `uint8` RGB `[H, W, 3]`, resized by the server;
  training stretched 360×640 frames to 256×320 (head) and 224×224 (wrists) without
  crop or pad, so uncropped 16:9 frames match it.
- **State.** `observation/state`, `float32 [32]` in `yam_eef.STATE_LAYOUT`
  (`src/flexpi/datasets/lerobot/utils/yam_eef.py`): left position (m), left rot6d,
  right position, right rot6d, left and right gripper, six left and six right
  joints (rad). rot6d is the first two **rows** of the rotation matrix, so identity
  is `[1, 0, 0, 0, 1, 0]` (zeros give NaN actions). Each arm's pose is in that arm's
  own base frame with the TCP at `grasp_site`; the gripper is a measured opening in
  [0, 1], 1 = open.
- **Depth** (`flexpi-pretrain-preview`, regimes a and c). `observation/depth_<cam>`,
  `uint16` millimetres, all three cameras or none, on the metadata's `per_cam_hw`
  grid. Depth ≤ 0.01 m or ≥ 2.0 m is treated as invalid; in the pointmap z is
  clipped to 1.5 m and x, y to ±0.5 m.
- **Intrinsics.** `observation/intrinsics_<cam>`, `float32 [fx, fy, cx, cy]` on the
  depth grid, rescaled by the endpoint convention `(w − 1)/(W − 1)`, `(h − 1)/(H − 1)`.
- **Response.** `actions` `float32 [32, 32]`, absolute, in the state's layout and frames.

## 7. What a client does

1. **Connection:** `WebsocketClientPolicy` (`experiments/yam/flexpi_policy/_openpi_vendor/`),
   with `per_cam_hw` and `depth_required` read from the metadata.
2. **State:** forward kinematics of the last **commanded** joints (slots `0:18`,
   `20:32`) with measured grippers (`18:20`), so gravity droop does not compound
   across chunks.
3. **Depth,** when the regime takes it: metres to `uint16` mm, resampled to
   `per_cam_hw` with `round(linspace(0, src − 1, dst))` nearest sampling, K rescaled
   by the endpoint convention.
4. **Execution** at 30 Hz, reading the EEF channels `0:18` (into IK) or the joints
   `20:32` (absolute rad), plus grippers; both are supervised.
5. **Safety and prompt:** a per-step joint-delta limit; the same prompt every step.

The smallest client is [`scripts/test_rd_client_dummy.py`](../scripts/test_rd_client_dummy.py)
(zero state, so its EEF actions are NaN).

**Offline smoke test**, with no robot or server:

```bash
python -m experiments.yam.flexpi_policy.smoke_test --ckpt "$CKPT" \
    --data-dir <flex-pi YAM dataset> --check-against-gt
```

It has no `--infer-*` flags, so it runs the trained default (full joint with depth
for `flexpi-pretrain-preview`). The released flex-pi YAM datasets overlap the training
data, so this checks the pipeline, not generalization.

## 8. Troubleshooting

| Symptom | Cause |
|---|---|
| Plausible actions at the wrong scale | the stats file is not the checkpoint's own `dataset_stats.json` |
| `action_dim_mask: not armed (...)` | the `config.yaml` found above the weights is not the checkpoint's own |
| `observation['depth'] is missing` on every request | the pointmap stream is present and the client sends no depth; `flexpi-pretrain-preview` with no regime flags requires depth |
| NaN actions | a zero rot6d block in `observation/state` |
| Instruction ignored | a pre-wrapped prompt, or a near-miss of the training wording |
| Periodic sag at chunk boundaries | the state is the measured pose of a loaded arm, not the last commanded one |
