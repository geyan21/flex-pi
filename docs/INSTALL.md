# Installation

Linux x86_64, Python 3.10. CUDA 12.8 and Intel XPU are supported for inference
and evaluation. One 32 GB CUDA GPU runs inference, evaluation and deployment;
training needs A100/H100/H200-class cards. XPU support targets Intel GPU
inference and evaluation and does not cover the CUDA-only DeepSpeed training
path.

In order:

1. **Environment** — §1, one of conda / uv / Docker.
2. **Weights** — §2.
3. **Verify** — §3. Run it; several ways of getting §1 wrong fail silently.
4. **Simulator** — §4, evaluation only.

Everything goes in one environment, simulator included. LIBERO and RoboTwin
cannot share one, so build a second if you need both. Versions are pinned in
[`pyproject.toml`](../pyproject.toml) — run the §1 commands exactly as written.

---

## 1. Environment

### 1a. conda

```bash
conda create -n flexpi python=3.10 -y
conda activate flexpi

pip install -e '.[cuda]' --extra-index-url https://download.pytorch.org/whl/cu128

# torchcodec from PyPI, not the cu128 index.
pip install --force-reinstall --no-deps \
  --index-url https://pypi.org/simple torchcodec==0.5

conda install -c conda-forge 'ffmpeg=7' -y

# Only to serve the YAM policy server.
pip install -e '.[serve]'

# Only to TRAIN. DeepSpeed shells out to `nvcc --version` at import and
# hard-fails without it; match your torch build (cu128 -> 12.8). The
# nvidia-cuda-nvcc-cu12 wheel ships headers only and does not satisfy it.
conda install -c nvidia 'cuda-nvcc=12.8' -y

# Only to serve with TensorRT engines. Pinned: an engine is tied to the
# TensorRT minor that baked it.
pip install -e '.[trt]'
```

### 1b. uv

Same environment, faster resolve. `--index-strategy unsafe-best-match` is
required — `uv` is first-index-wins, so the cu128 index otherwise shadows PyPI
and the resolve fails on an unrelated-looking pin.

```bash
uv venv --python 3.10 .venv
source .venv/bin/activate

UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu128 \
  uv pip install --index-strategy unsafe-best-match -e '.[cuda]'

uv pip install --reinstall --no-deps \
  --index-url https://pypi.org/simple torchcodec==0.5

uv pip install -e '.[serve]'     # YAM policy server
uv pip install -e '.[trt]'       # TensorRT serving
```

FFmpeg 7 and `nvcc` are not pip packages — `uv` cannot supply either. Use the
system package manager (and point `CUDA_HOME` at the toolkit) or the conda path.

### 1c. Intel XPU

Use the `xpu` extra in a separate environment from the `cuda` extra. Do not
install both extras into the same environment: PyTorch CUDA and XPU wheels are
mutually exclusive. The XPU path uses PyTorch `2.13.0+xpu` and torchvision
`0.28.0+xpu` from the PyTorch XPU wheel index.

#### conda

```bash
conda create -n flexpi-xpu python=3.10 -y
conda activate flexpi-xpu

pip install -e '.[xpu]' \
  --extra-index-url https://download.pytorch.org/whl/xpu

conda install -c conda-forge 'ffmpeg=7' -y
```

#### uv

```bash
uv venv --python 3.10 .venv-xpu
source .venv-xpu/bin/activate

UV_EXTRA_INDEX_URL=https://download.pytorch.org/whl/xpu \
  uv pip install --index-strategy unsafe-best-match \
  -e '.[xpu]'
```

FFmpeg 7 is still required by torchcodec and must be installed separately
with conda or the system package manager. XPU does not use `nvcc`,
`CUDA_HOME`, or the CUDA Docker image. The optional `[trt]` extra is CUDA-only.

Verify the selected backend before loading a model:

```bash
python -c "import torch; print(torch.__version__); print('xpu:', torch.xpu.is_available()); print('cuda:', torch.cuda.is_available())"
```

For XPU inference or evaluation, pass `--device xpu` to the supported launcher.
For example:

```bash
python scripts/inference_opt/benchmark_flex_latency.py \
  --device xpu \
  --ckpt /path/to/checkpoint
```

### 1d. Docker

Save as `Dockerfile` at the repository root:

```dockerfile
FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-devel

# FFmpeg 7 from conda-forge -- apt cannot supply it (Ubuntu LTS ships 4.4 or
# 6.1) and a container has no system copy to fall back on.
RUN apt-get update && apt-get install -y --no-install-recommends git \
      && rm -rf /var/lib/apt/lists/*
RUN conda install -y -c conda-forge 'ffmpeg=7' && conda clean -afy

WORKDIR /workspace/flex-pi
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -e '.[cuda]' \
      --extra-index-url https://download.pytorch.org/whl/cu128
RUN pip install --no-cache-dir --force-reinstall --no-deps \
      --index-url https://pypi.org/simple torchcodec==0.5

COPY . .
ENV DIFFSYNTH_MODEL_BASE_PATH=/workspace/flex-pi/checkpoints
```

```bash
docker build -t flexpi:latest .

# Mount weights and data rather than baking them in -- both are tens of GB.
docker run --gpus all -it --rm \
  -v "$PWD/checkpoints:/workspace/flex-pi/checkpoints" \
  -v "$PWD/data:/workspace/flex-pi/data" \
  flexpi:latest bash
```

#### Intel XPU

XPU containers use an Intel oneAPI base image rather than the CUDA image above.
The host must have a working Intel GPU driver and expose `/dev/dri` to the
container. This path is for XPU inference and evaluation; the CUDA-only
DeepSpeed training path is not supported in this image.

Save as `Dockerfile.xpu` at the repository root:

```dockerfile
FROM intel/oneapi-basekit:2025.3.2-0-devel-ubuntu24.04

RUN apt-get update && apt-get install -y --no-install-recommends \
  git python3 python3-pip \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/flex-pi
COPY pyproject.toml ./
COPY src ./src
RUN python3 -m pip install --no-cache-dir -e '.[xpu]' \
  --extra-index-url https://download.pytorch.org/whl/xpu
RUN python3 -m pip install --no-cache-dir --force-reinstall --no-deps \
  --index-url https://pypi.org/simple torchcodec==0.5

COPY . .
ENV DIFFSYNTH_MODEL_BASE_PATH=/workspace/flex-pi/checkpoints
ENV SYCL_CACHE_PERSISTENT=1
```

Build and run it with the Intel GPU device exposed:

```bash
docker build -f Dockerfile.xpu -t flexpi:xpu .

docker run --rm -it --ipc=host --shm-size=16g \
  --device=/dev/dri \
  --group-add "$(getent group render | cut -d: -f3)" \
  -v "$PWD/checkpoints:/workspace/flex-pi/checkpoints" \
  -v "$PWD/data:/workspace/flex-pi/data" \
  flexpi:xpu bash
```

Inside the container, verify XPU visibility before running an evaluation:

```bash
python3 -c "import torch; print(torch.__version__); print('xpu:', torch.xpu.is_available())"
```

### Environment variables

```bash
# Where Wan2.2-TI2V-5B and the ActionDiT backbone live. Every train and eval
# launch reads this -- put it in your shell profile.
export DIFFSYNTH_MODEL_BASE_PATH="$PWD/checkpoints"

# Training only. DeepSpeed reads this, not torch's bundled CUDA runtime.
export CUDA_HOME="$CONDA_PREFIX"
# conda's cuda-nvcc activate.d hook dereferences these unguarded, and every
# launcher runs `set -u`.
export NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:-}"
export NVCC_APPEND_FLAGS="${NVCC_APPEND_FLAGS:-}"
```

Optional: `DIFFSYNTH_DOWNLOAD_SOURCE=huggingface` switches the `Wan-AI/*` fetches
off their ModelScope default — the VAE and T5 are ModelScope-only, so it cannot
replace it wholesale (§2.1); `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
reduces fragmentation in the policy server; `DIFFSYNTH_SKIP_DOWNLOAD=true`
hardens a deploy machine against accidental weight fetches.

---

## 2. Weights

### 2.1 Wan2.2-TI2V-5B + the ActionDiT backbone

**Evaluation needs this too.** Even with a checkpoint supplying the video DiT,
the model still loads Wan2.2-TI2V-5B's VAE and T5 encoder, so a first eval on a
clean machine downloads ~13 GB. The ActionDiT step is training-only.

Four artifacts are fetched on the first model build and cached under
`$DIFFSYNTH_MODEL_BASE_PATH`, one directory per source repo:

```
checkpoints/
├── ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt        2.0 GB  training only
├── Wan-AI/
│   ├── Wan2.2-TI2V-5B/                                          20 GB  training only
│   │   └── diffusion_pytorch_model-0000{1,2,3}-of-00003.safetensors
│   └── Wan2.1-T2V-1.3B/google/umt5-xxl/                          5 MB  T5 tokenizer
└── DiffSynth-Studio/Wan-Series-Converted-Safetensors/
    ├── Wan2.2_VAE.safetensors                                  1.4 GB
    └── models_t5_umt5-xxl-enc-bf16.safetensors                   11 GB
```

For training, the ActionDiT step pulls all four:

```bash
mkdir -p checkpoints
export DIFFSYNTH_MODEL_BASE_PATH="$(pwd)/checkpoints"

python scripts/preprocess_action_dit_backbone.py \
  --model-config configs/model/flexpi.yaml \
  --output checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt \
  --device cuda --dtype bfloat16
```

The output filename is not arbitrary — `configs/model/flexpi.yaml` ships
`action_dit_pretrained_path` pointing at exactly that name.

`--device cuda` needs ~12 GB of VRAM. `--device cpu` works but peaks near 35 GB
of RAM — on a shared login node it is usually OOM-killed.

On an eval-only machine, or to warm the cache before going offline, fetch the
three that evaluation reads:

```bash
python - <<'EOF'
from flexpi.models.helpers.io import ModelConfig
for repo, pattern in [
    ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "Wan2.2_VAE.safetensors"),
    ("DiffSynth-Studio/Wan-Series-Converted-Safetensors", "models_t5_umt5-xxl-enc-bf16.safetensors"),
    ("Wan-AI/Wan2.1-T2V-1.3B", "google/umt5-xxl/"),
]:
    ModelConfig(model_id=repo, origin_file_pattern=pattern).download_if_necessary()
EOF
```

Downloads come from ModelScope. Leave `DIFFSYNTH_DOWNLOAD_SOURCE` unset —
`DiffSynth-Studio/Wan-Series-Converted-Safetensors` is published on ModelScope
only, so `=huggingface` fails there with a 404 on the VAE and T5.

### 2.2 Text embeddings

The T5 encoder is frozen, so prompts are encoded once and cached to disk.
Training dies on the first batch without the cache. Once per dataset, from the
repo root:

Pass the same `task=` you will train with — the cache it fills is the one that
task config names, so a cache built for another benchmark does not count:

```bash
python scripts/precompute_text_embeds.py task=<TASK_CONFIG>
```

A re-run only encodes what the cache is missing. What a task config is, plus
custom dataset dirs, multi-GPU, and datasets that have no task config yet, are in
[TRAINING.md §1.3](TRAINING.md#13-precompute-text-embeddings-required).
`scripts/train_flexpi_yam.sh` runs this for you; the other two launchers do not.

### 2.3 DINOv3

Fetched from the HuggingFace hub the first time a model is built — training, eval
and the policy server alike — so an offline machine fails there rather than at
startup. Warm it while you still have network:

```bash
export HF_HOME=/path/with/room
python -c "import timm; timm.create_model('vit_base_patch16_dinov3.lvd1689m', pretrained=True)"
```

### 2.4 Camera intrinsics (depth runs only)

Any config declaring `shape_meta.depth` needs per-dataset intrinsics in
`meta/camera_intrinsics.json`. The published datasets ship one; for your own
data, the builder that writes one is
[`YAM.md` step 1](YAM.md#1-data-build). Cameras with different
fields of view need different entries.

---

## 3. Verify

```bash
# pytest is not part of the runtime pins.
pip install pytest
pytest experiments -q

# Which FFmpeg torchcodec actually bound -- not the one on PATH, which can be a
# different install entirely. Must be major 4-7.
python -c "from torchcodec._core import get_ffmpeg_library_versions as v; \
           print(v()['ffmpeg_version'])"
```

---

## 4. Simulators (evaluation only)

Neither simulator needs an environment of its own. Both install into the §1
environment — the same one you train and deploy in — but **not both into the same
one**: `sapien` requires `opencv-python`, and installing it over the
`opencv-python-headless==4.8.1.78` that `[libero]` pins overwrites the shared
`cv2/` directory. The import silently becomes opencv 5.0.0 while `pip list` still
shows the pin.

So, for **one** benchmark, add it to the environment you already have (§4.1 or
§4.2). For **both**, build §1 a second time under another name and put the second
simulator there:

```bash
conda create -n flexpi-libero python=3.10 -y && conda activate flexpi-libero
# then §1a again, then §4.2
```

Weights and datasets are read by path, so the two environments share them; only
the installed packages differ.

### 4.1 RoboTwin

Vendored at `third_party/RoboTwin`; install per the
[official instructions](https://robotwin-platform.github.io/doc/usage/robotwin-install.html).
`sapien` and `mplib` declare no torch dependency and install straight into the
FlexPi environment — that is how the evals here run. Take RoboTwin's
`script/requirements.txt` selectively rather than wholesale, though: it hard-pins
`scipy==1.10.1` and `huggingface_hub==0.36.2` over ours.

Rendering goes through SAPIEN's Vulkan backend, so the eval host needs it:

```bash
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
```

### 4.2 LIBERO

Both commands run from the repository root:

```bash
export PYTHONNOUSERSITE=1        # a user-site copy otherwise shadows the env (§5)
git submodule update --init third_party/LIBERO

# Choose exactly one backend extra. CUDA is shown first because it is the
# default deployment target.
pip install -e '.[libero,cuda]' --extra-index-url https://download.pytorch.org/whl/cu128
# For Intel XPU environments, use this instead:
# pip install -e '.[libero,xpu]' --extra-index-url https://download.pytorch.org/whl/xpu
pip install --no-deps robosuite==1.4.0
```

> [!WARNING]
> Do **not** run `third_party/LIBERO/requirements.txt` or follow LIBERO's own
> README install — it pins python 3.8.13 and torch 1.11.0 over this environment
> and destroys it. The `[libero]` extra already covers the eval path.

Two things that are easy to get wrong:

- **LIBERO is never pip-installed** — `pip install -e third_party/LIBERO` silently
  does nothing. It resolves through `PYTHONPATH`, which the eval launchers export
  for you.
- **`robosuite` needs `--no-deps`.** Its metadata pulls `opencv-python` (which
  collides with the headless build) and an unpinned `mujoco>=2.3.0`, silently
  breaking the `mujoco==3.3.2` pin the published dataset was rendered under.

### 4.3 Verify

The same two calls every LIBERO eval entry point makes before touching a
renderer:

```bash
export PYTHONPATH="$(pwd)/third_party/LIBERO:$(pwd)/src:${PYTHONPATH:-}"
unset LIBERO_CONFIG_PATH   # let prepare_libero derive the binding

python -c "
from flexpi.utils.libero_setup import prepare_libero, assert_mujoco_pin
prepare_libero(); assert_mujoco_pin(); print('LIBERO environment OK')"
```

Expect a `[libero-bind]` line naming *this* repo's `third_party/LIBERO`, then
`LIBERO environment OK`. A bind pointing elsewhere means a stale
`~/.libero/config.yaml` is winning and you would evaluate a different scene than
you think. `assert_mujoco_pin` raises on any mujoco other than 3.3.2 — another
version renders visibly different scenes, so the policy sees out-of-distribution
images and the success rates stop being comparable, while nothing crashes.

---

## 5. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `libnppicc.so.12: cannot open shared object file` | torchcodec came from the cu128 index. Reinstall it from PyPI (§1a). Check with `python -c "import torchcodec"`, not `pip show` — the package is present, only unimportable |
| `ffmpeg -version` shows 6 or 8 | torchcodec silently falls back to pyav, which leaks memory and OOMs long runs. Install FFmpeg 7 specifically |
| `torch` installed but CUDA unavailable | the PyPI build instead of cu128 — reinstall with `--extra-index-url https://download.pytorch.org/whl/cu128` |
| `ModuleNotFoundError` on a package `pip show` reports as installed | `~/.local/lib/python3.10/site-packages` sits ahead of the env on `sys.path`, so the wrong copy imports and pip declines to reinstall. Check `python -c "import numpy; print(numpy.__file__)"`; fix with `PYTHONNOUSERSITE=1` |
| Segfault in mplib during RoboTwin eval | the above, with a user-site numpy 2 shadowing the pinned 1.26 |
| `FileNotFoundError` on `ActionDiT_..._1024hdim.pt` | §2.1 was skipped, or `--output` does not match `action_dit_pretrained_path` |
| `Cannot detect model type for wan_video_dit ... Model hash:` | a truncated ModelScope download, not a bad release. Set `DIFFSYNTH_DOWNLOAD_SOURCE=huggingface` and re-run |
| `MissingCUDAException: CUDA_HOME does not exist` | training only; it fires at `import deepspeed`, despite naming compilation. Install `cuda-nvcc` and export `CUDA_HOME` (§1a) |
| `NVCC_PREPEND_FLAGS: unbound variable` after `conda activate` | conda's `cuda-nvcc` hook under a launcher's `set -u`. Export both flags empty first |
| `ModuleNotFoundError: No module named 'osqp'` on the policy server | the `[serve]` extra was skipped. It is a module-level import, so the server cannot start even if you never enable the smoother |
| `ModuleNotFoundError: No module named 'tensorrt'` | TensorRT is imported lazily, so it fails only when you pass an engine flag. Install the `[trt]` extra |
| `HFValidationError: Repo id must be in the form ...` | a local dataset path that does not exist; the LeRobot loader falls through to treating it as a hub id. Launch from the repo root or pass absolute paths |
| `OIDN Error: out of memory` during RoboTwin eval | the SAPIEN renderer, not the policy. Pass `EVALUATION.offload_text_encoder=true` to free ~10 GB |
