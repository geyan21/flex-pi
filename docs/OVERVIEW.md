# Flex-π — architecture and workflow

How the model is put together and how a training or evaluation run actually flows
through the code. Installation is in [INSTALL.md](INSTALL.md); the practical
training guide with every environment knob is in [TRAINING.md](TRAINING.md).

Paper: [Flex-π: A Multi-Stream World-Action Model with Compute Flexibility](https://arxiv.org/abs/2608.10860).

## 1. The idea

A world-action model predicts the future in order to act better. Nearly all of them
predict only RGB latents — trained for pixel reconstruction, carrying no explicit
signal for 3D geometry or object semantics.

The observation Flex-π is built on: the frozen Wan-2.2 VAE, trained only on RGB
images, encodes and reconstructs **3D pointmaps** almost losslessly with no
pointmap-specific training. Geometry and RGB therefore live in the *same* latent
space, and a single backbone can co-denoise both.

So Flex-π supervises three visual futures instead of one — RGB, 3D pointmaps, and
DINOv3 semantics — while paying none of the usual costs:

| Usual cost of extra supervision | Why it does not apply here |
|---|---|
| new sensors | pointmaps unproject from the depth the rig or simulator already provides, or from Depth Anything 3 on RGB when it provides none; DINO features from DINOv3 on the same image |
| new visual priors to pre-train | both encoders are off-the-shelf and frozen; the VAE is the video model's own |
| inference latency | any visual stream can be dropped at deployment while the model still keeps the benefit of having been trained on it |

## 2. Streams

Every visual modality becomes one **token stream** in a shared latent space.

| Stream | Encoder | Shape of contribution |
|---|---|---|
| RGB `z^o` | Wan-2.2 VAE (frozen) | appearance; inherits the video-pretraining prior |
| Pointmap `z^p` | the *same* Wan-2.2 VAE (frozen) | explicit 3D geometry |
| DINO `d` | DINOv3 ViT-B/16 (frozen) | object-level semantic grounding |
| Action `a` | linear projection of the action chunk | the thing we ultimately predict |
| Proprioception `s`, language `l` | state encoder, umT5 (frozen) | global conditioning, shared across streams |

Token budget for the shipped 3-camera RoboTwin/YAM layout
(`composite_layout: tshape_384x320`): 33 raw timesteps subsample at stride
4 into 9 RGB frames, which the VAE compresses 4× temporally into 3 latent frames.
A 384×320 composite gives a 24×20 latent, patchified `[1,2,2]` into **120 video
tokens per latent frame**. DINO gives every camera a uniform 14×14 grid, folded 2×2
by `dino_pixel_unshuffle: 2` into 7×7 — **147 DINO tokens per frame** at 3 cameras,
4× fewer than unfolded, with no spatial detail discarded (the fold raises each
token's feature dim 768 → 3072 instead).

## 3. Backbone

A Mixture-of-Transformers ([`src/flexpi/models/mot.py`](../src/flexpi/models/mot.py)):

- **Visual trunk** — 3072-d, 30 layers, ~5B parameters, initialized directly from
  pre-trained Wan-2.2-5B. All visual streams share it.
- **Action expert** — 1024-d, 30 layers, ~1B parameters. Separate query, key, value,
  feedforward, and normalization parameters, initialized from Wan-2.2 by
  *resampling* rather than copying.

Cross-stream attention is not applied at every depth. The **HBridge** band
(`hbridge.bottom_ratio: 0.25`, `top_ratio: 0.25` — so layers 0–6 and 23–29 of 30)
keeps each sub-stream attending only to itself, and the middle 16 layers run full
joint attention. Early encoding and late decoding stay stream-specific; fusion
happens in the trunk.

Attention is one-way: action tokens attend to the visual streams, but no visual
token ever attends to the action stream. Actions are generated *jointly* with each
future visual stream, so action generation benefits from the model's evolving
representation of the future.

## 4. The two masks

This is the mechanism that makes one checkpoint serve every regime. Two independent
per-sample binary masks over the three visual streams:

- **`m_in` — presence.** Which streams are given as *input* at time *t*. Sampled per
  stream per sample, subject to at least one visual stream remaining.
- **`m_out` — joint generation.** Which future streams the action tokens read, and
  how the future streams attend to one another.

The critical detail: **`m_out` is not a loss mask.** Every future stream is denoised
and incurs its flow-matching loss on every sample, whatever `m_out` says. `m_out`
only selects what is *mutually visible*.

Because the two masks are drawn independently, a stream dropped from the input is
still denoised at the output — the model has to synthesize that modality's future
from the streams that remain. That is **cross-modality forcing**, and it is on for
all three streams by default (`cross_modal_predict_*: true`).

It is not merely robustness insurance. Removing it *costs* 21% success on RoboTwin:
requiring each modality to be predictable from the others forces a representation in
which appearance, geometry, and semantics are mutually predictive, and that
representation is better for action prediction.

Sampling lives in
[`src/flexpi/models/helpers/flex_joint.py`](../src/flexpi/models/helpers/flex_joint.py);
defaults are in `configs/model/flexpi.yaml` under `flex_joint`.

## 5. Objective

Flow matching on the linear path, summed over the action stream and all three visual
streams:

```
L = λ_a · L_FM(a_t) + Σ_{i ∈ {o,d,p}} λ_i · L_FM(i_{t+1})
```

Per-stream weights are all 1 in every experiment reported in the paper
(`loss.lambda_*` in the model config).

On a layout whose action head is wider than the embodiment (the 34-D layout of
[MIXED_EMBODIMENT_34D.md](MIXED_EMBODIMENT_34D.md)), the padded channels are
held at exactly 0 on the DiT input and target, `L_FM(a_t)` averages over the real
channels only, and sampling re-zeroes them after every step. Dense layouts carry
no mask and are unchanged.

One exception to the parameterization: the DINO stream uses **x-prediction** — the
head predicts the clean features `d_{t+1}` rather than the velocity, since the 2×2
fold raises each token's feature dim to 3072. The output is converted analytically
back to velocity so `loss_dino` stays comparable with v-parameterized runs.

Frozen throughout: the Wan-2.2 VAE, DINOv3, and the umT5 text encoder. Trainable:
the visual trunk, the action expert, and the per-stream projectors and heads.

## 6. Training run

Per batch, in [`src/flexpi/trainer.py`](../src/flexpi/trainer.py):

1. Decode RGB and depth from the dataloader; unproject depth to a pointmap using
   per-dataset intrinsics from `meta/camera_intrinsics.json`.
2. VAE-encode the RGB composite and the pointmap composite (frozen, shared weights).
3. DINOv3-encode the RGB composite; fold 2×2.
4. Look up the T5 text embedding from the precomputed cache.
5. Sample `m_in` / `m_out` and per-stream flow-matching noise and timesteps.
6. Forward the MoT under the resulting attention mask.
7. Sum the per-stream flow-matching losses; backward.

AdamW, lr 1e-4, weight decay 0.01, cosine with warmup, bfloat16, gradient
checkpointing per block, DeepSpeed ZeRO-1 via `accelerate`.

The reported results pre-train on ~500 hours drawn from 100 tasks of AgiBot
World-Beta, then fine-tune per domain. That pre-trained checkpoint is not
released yet, so every shipped task config trains from the base WAN init;
`PRETRAINED_CKPT=<path>` opts into a warm start from any checkpoint you have.

## 7. Inference

Given a chosen input/output regime, the model runs *K* Euler steps of the
flow-matching ODE over the **active output streams only** and emits an action chunk
of length H=32, of which `EVALUATION.replan_steps` are executed before
re-planning — 32 everywhere except LIBERO, which replans every 10.

`K` is `EVALUATION.num_inference_steps`, which every eval config pins to 4. The
paper sweeps it and uses K=4: action-only success peaks there and stays within
1.0 point of the peak for every K ≥ 2.

Dropping every visual output stream gives the action-only path — the cheapest mode,
~60 ms per call on an RTX 5090, and still ahead of every baseline we measured.
Generating all three gives the most accurate mode at ~193 ms. Same checkpoint; the
choice is a runtime flag.

Regime selection from the CLI is documented in the
[README](../README.md#-inference-regimes).

## 8. Where things are

| | |
|---|---|
| Model | [`src/flexpi/models/flexpi.py`](../src/flexpi/models/flexpi.py) |
| Backbone / MoT | [`backbone.py`](../src/flexpi/models/backbone.py), [`mot.py`](../src/flexpi/models/mot.py) |
| Flex sampling | [`helpers/flex_joint.py`](../src/flexpi/models/helpers/flex_joint.py) |
| Architecture config | [`configs/model/flexpi.yaml`](../configs/model/flexpi.yaml) |
| Training entry point | [`scripts/train.py`](../scripts/train.py) |
| Evaluation | [`experiments/`](../experiments/) |
| Deployment | [`scripts/serve_yam_flexpi.py`](../scripts/serve_yam_flexpi.py) |
| 34-D mixed-embodiment layout | [`MIXED_EMBODIMENT_34D.md`](MIXED_EMBODIMENT_34D.md) |
| Preview checkpoints, zero-shot | [`PRETRAINED.md`](PRETRAINED.md) |
| Fine-tuning a preview checkpoint | [`FINETUNE.md`](FINETUNE.md) |
