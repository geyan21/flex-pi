# Inference optimization

Everything here is **training-free** — the same checkpoint, made faster.

Deployed on an RTX 5090. ms/call at 4 denoise steps:

| Stack | full joint | action only |
|---|---:|---:|
| eager PyTorch | 447 | 132 |
| **torch.compile** — the default | 360 | **60** |
| **+ TensorRT joint engine** | 230 | — |
| **+ TensorRT KV-split engines** | **193** | — |

TensorRT is for the joint path only; on the action-only path torch.compile is
already the fastest thing available.

---

## 1. Using it

Both RoboTwin eval launchers ship with the compile stack on. Nothing to do:

```bash
bash scripts/eval_flexpi_robotwin.sh          # sweep
bash scripts/eval_flexpi_robotwin_single.sh   # one task
```

To use TensorRT, build an engine ([§2](#2-building-the-engines)) and set one line
in the launcher's config block — the script switches the rest:

```bash
TRT_JOINT_ENGINE=/path/to/engines/flex_joint_core_o3.engine
# or, instead, the faster pair:
TRT_JOINT_PREFILL_SPLIT_ENGINE=/path/to/engines/split/prefill_core_o3.engine
TRT_JOINT_DECODE_SPLIT_ENGINE=/path/to/engines/split/decode_core_o3.engine
```

Set one or the other, never both. The pair is faster at every step count above
one; the single engine wins by ~5 ms at one step.

For the real robot, `scripts/serve_flexpi_yam.sh` takes `SPLIT_DIR` and makes the
same switch — see [`YAM.md`](YAM.md#serving).

### What you need to build engines

| | |
|---|---|
| Run env | torch 2.7.1+cu128, TensorRT 10.16.1.11 — the env from [`docs/INSTALL.md`](INSTALL.md) plus its `[trt]` extra (`pip install -e '.[trt]'`); the base install ships no TensorRT |
| Build env | a second env, `trt29` — torch 2.12.1+cu130 with the **same** TRT 10.16.1.11, [below](#the-build-env) |
| A checkpoint | with its `config.yaml` and `dataset_stats.json` beside it |

Two environments, because the ONNX exporter needs a newer torch than the runtime.
The TRT version must match **exactly** between them — a different patch version
deserializes to `None` and raises. Engines are also **machine-locked**: built for
one GPU arch, they will not load on another.

None of this is required to run the model. Every knob defaults to off.

### The build env

From the repository root:

```bash
conda create -n trt29 python=3.10 -y && conda activate trt29
export PYTHONNOUSERSITE=1   # or pip counts a ~/.local copy as already satisfied

# flexpi's dependencies, but not flexpi itself: this env deliberately breaks its
# torch pin, and an installed flexpi turns that into a pip conflict on every
# later call. The exporters put src/ on sys.path themselves, so imports still
# resolve.
pip install -e '.[cuda]' --extra-index-url https://download.pytorch.org/whl/cu128
pip uninstall -y flexpi

# Then the torch the exporter needs. Pin both -- unpinned, torchvision
# resolves against the newest torch and drags torch up with it.
pip install torch==2.12.1 torchvision==0.27.1 \
  --index-url https://download.pytorch.org/whl/cu130

# onnxscript is what torch.onnx.export(dynamo=True) imports at export time.
pip install tensorrt==10.16.1.11 onnx onnxruntime onnxscript
```

Worth confirming before spending ten minutes on an export:

```bash
python -c "
import torch, tensorrt, numpy, tempfile, os, sys; sys.path.insert(0, 'src')
print(torch.__version__, tensorrt.__version__, numpy.__version__)
from flexpi.runtime import create_flexpi
p = os.path.join(tempfile.mkdtemp(), 'm.onnx')
torch.onnx.export(torch.nn.Linear(4, 2).eval(), (torch.randn(1, 4),), p, dynamo=True)
print('BUILD ENV OK')"
```

---

## 2. Building the engines

First read the trained context length. **This is the most common build failure** —
it is the trained `tokenizer_max_len` **plus one** (a proprio token is appended
inside `infer_action`), not the class default of 512:

```bash
grep tokenizer_max_len <run_dir>/config.yaml     # e.g. 128  ->  --ctx-len 129
```

Then, in the `trt29` environment. Expect ~10 min of ONNX export plus ~100 s of
build for the single engine; peak disk ~46 GB.

```bash
export CKPT=/path/to/step_026304.pt
export ENGDIR=/path/to/engines
mkdir -p "$ENGDIR"

# (a) Single joint-core engine. ~12 GB artifact, weights baked in.
python scripts/inference_opt/trt_onnx_joint_engine.py \
    --ckpt "$CKPT" --workdir "$ENGDIR" --ctx-len 129 --opt-level 3

# (b) Parity check + calibration capture. Writes captured_core_io.pt.
#     Takes the UN-incremented ctx-len; it adds the +1 itself.
#     A trailing OOM here is harmless — the capture is already on disk.
python scripts/inference_opt/trt_joint_parity.py \
    --ckpt "$CKPT" --engine "$ENGDIR/flex_joint_core_o3.engine" --ctx-len 128

# (c) OPTIONAL — the faster KV-split pair. Needs (b) first.
python scripts/inference_opt/trt_onnx_joint_split_engine.py \
    --ckpt "$CKPT" --calib "$ENGDIR/captured_core_io.pt" \
    --workdir "$ENGDIR/split" --stage both
rm -f "$ENGDIR/split"/*.onnx "$ENGDIR/split"/*.onnx.data   # reclaim ~21 GB
```

Sanity bands: single-engine `merged_maxdiff` &le; ~45; split build ending
`[split] anchors=...` &rarr; `[decode] ... rel≈0.016` &rarr; `SPLIT_ENGINE_DONE`.

The split build writes `build_meta.json` beside the engines, recording the
checkpoint's name, size and step. The YAM policy server checks it against
`--ckpt-path` and refuses a mismatched pair — engine weights are baked in, and
what that has cost on real hardware is
[`YAM.md`](YAM.md#troubleshooting). An engine baked before that
sidecar existed only earns a warning, so record its checkpoint by hand.

---

## 3. What runs where

The real-robot server exposes a narrower knob set than the sim eval launchers, so
a config does not port across unchanged:

| | sim eval | policy server |
|---|---|---|
| torch.compile stack | ✓ | ✓ |
| `glue_cache` | ✓ | ✓, **but never with `--torch-compile`** — the pair hangs the boot, and the server rejects it |
| single TRT engine | ✓ | **not exposed** — the server takes the KV-split pair only |
| KV-split pair | ✓ | ✓ |

Other hard rules, all of which raise rather than silently misbehave: the single
engine and the KV-split pair are mutually exclusive; `compile_encoders` and
`encoder_cuda_graph` are mutually exclusive; `solver` must stay `euler`;
`attn_backend=flex` is rejected by any engine path; set `quantization=null` with
any engine.

On a 32 GB card either engine leaves ~5–6 GB. Enough for pure inference, tight
beside SAPIEN's Vulkan renderer, which creeps ~1 GB/episode —
`trt_joint_free_video_blocks=true` is mandatory with any engine, and without it
you get `vk ErrorDeviceLost`.
