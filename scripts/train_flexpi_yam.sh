#!/usr/bin/env bash
# =============================================================================
# FlexPi — real-world bimanual YAM training. 32D EEF-grouped, body-frame
# relative action. One checkpoint serves every inference regime; per-sample
# dropout over stream presence and joint flags is what makes that work.
#
#   DATASET_DIRS="[./data/yam/<your_set>]" bash scripts/train_flexpi_yam.sh
#
# Pre-flights every dataset dir (canonical RGB, plus intrinsics + depth when the
# pointmap stream trains) and caches the T5 text embeddings before touching a GPU.
#
# Details: docs/YAM.md · knob reference: docs/TRAINING.md
# =============================================================================
set -euo pipefail

export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-1800}"
export NCCL_DEBUG=WARN

# Multi-node InfiniBand pinning; a no-op on a single node.
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_IB_HCA="${NCCL_IB_HCA:-mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_7}"

CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

# ── Data ──────────────────────────────────────────────────────────────────────
TASK_CONFIG="${TASK_CONFIG:-yam_unified_flex_3cam_32d_rel_1e-4}"
NUM_EPOCHS="${NUM_EPOCHS:-10}"
# Size-agnostic mixing: DATASET_WEIGHTS are per-frame draw probabilities parallel
# to the dataset dirs, and SAMPLES_PER_EPOCH defines one epoch (required with
# weights). Both empty = a single uniform pass over the concatenated set.
DATASET_WEIGHTS="${DATASET_WEIGHTS:-}"
SAMPLES_PER_EPOCH="${SAMPLES_PER_EPOCH:-}"

DATASET_DIRS="${DATASET_DIRS:-[./data/yam/my_task]}"
VAL_SET_PROPORTION="${VAL_SET_PROPORTION:-0.0}"
MAX_STEPS="${MAX_STEPS:-}"


# ── Flex randomization ────────────────────────────────────────────────────────
# Independent Bernoulli per sample; p=1.0 disables that dropout. A knob that is
# set is passed as an override and wins over the task preset's flex_joint value;
# empty (the default) keeps the preset's value: 0.5 for all six in every preset
# except the RGB-only yam_finetune_pretrain_rgb_preview_*, whose pointmap p are 0.
FLEX_P_PRESENT_VIDEO="${FLEX_P_PRESENT_VIDEO:-}"
FLEX_P_PRESENT_DINO="${FLEX_P_PRESENT_DINO:-}"
FLEX_P_PRESENT_POINTMAP="${FLEX_P_PRESENT_POINTMAP:-}"
FLEX_P_JV="${FLEX_P_JV:-}"
FLEX_P_JD="${FLEX_P_JD:-}"
FLEX_P_JP="${FLEX_P_JP:-}"
FLEX_ARGS=()
[[ -n "${FLEX_P_PRESENT_VIDEO}" ]]    && FLEX_ARGS+=("model.flex_joint.p_present_video=${FLEX_P_PRESENT_VIDEO}")
[[ -n "${FLEX_P_PRESENT_DINO}" ]]     && FLEX_ARGS+=("model.flex_joint.p_present_dino=${FLEX_P_PRESENT_DINO}")
[[ -n "${FLEX_P_PRESENT_POINTMAP}" ]] && FLEX_ARGS+=("model.flex_joint.p_present_pointmap=${FLEX_P_PRESENT_POINTMAP}")
[[ -n "${FLEX_P_JV}" ]]               && FLEX_ARGS+=("model.flex_joint.p_jv=${FLEX_P_JV}")
[[ -n "${FLEX_P_JD}" ]]               && FLEX_ARGS+=("model.flex_joint.p_jd=${FLEX_P_JD}")
[[ -n "${FLEX_P_JP}" ]]               && FLEX_ARGS+=("model.flex_joint.p_jp=${FLEX_P_JP}")

# ── Run labeling / resume / wandb ─────────────────────────────────────────────
RUN_NAME="${RUN_NAME:-yam_uflex}"
RESUME="${RESUME:-}"

WANDB_ENABLED="${WANDB_ENABLED:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-flex-pi}"
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_GROUP="${WANDB_GROUP:-yam_flexpi}"

# =============================================================================
# Data prep: preflight, then idempotent T5 text-embed precompute
# =============================================================================

# The flex probabilities this run trains with: the task preset composed with the
# FLEX_P_* overrides and the passthrough overrides, as train.py composes it.
_FLEX_EFFECTIVE="$(python - "task=${TASK_CONFIG}" "${FLEX_ARGS[@]}" "$@" <<'PY'
import os, sys
from hydra import compose, initialize_config_dir
with initialize_config_dir(config_dir=os.path.abspath("configs"), version_base="1.3"):
    m = compose(config_name="train", overrides=sys.argv[1:]).model
fj = m.flex_joint
print(fj.p_present_video, fj.p_present_dino, fj.p_present_pointmap, fj.p_jv, fj.p_jd, fj.p_jp,
      m.get("enable_pointmap", True))
PY
)"
read -r _PV _PD _PP _JV _JD _JP _ENABLE_PM <<< "${_FLEX_EFFECTIVE}"

# The pointmap stream is off under model.enable_pointmap=false, or when it is
# never present and never joined (p_present_pointmap = p_jp = 0). Only the
# pointmap stream reads depth and intrinsics.
_POINTMAP_OFF=0
[[ "${_ENABLE_PM}" == "False" ]] && _POINTMAP_OFF=1
awk -v pp="${_PP}" -v jp="${_JP}" 'BEGIN { exit !(pp + 0 == 0 && jp + 0 == 0) }' \
  && _POINTMAP_OFF=1

# Preflight: every dataset dir must carry canonical RGB, plus intrinsics and
# canonical depth when the pointmap stream trains.
_PREFLIGHT_DIRS="$(printf '%s' "${DATASET_DIRS}" | tr -d '[]' | tr ',' ' ')"
for _d in ${_PREFLIGHT_DIRS}; do
  if [[ "${_POINTMAP_OFF}" != "1" ]]; then
    [[ -f "${_d}/meta/camera_intrinsics.json" ]] || {
      echo "ERROR: ${_d}/meta/camera_intrinsics.json missing — required by the pointmap pipeline." >&2
      exit 1; }
    ls "${_d}/videos/chunk-000/observation.depth_ffv1.cam_high/"*.mkv >/dev/null 2>&1 || {
      echo "ERROR: ${_d} depth not in canonical layout (observation.depth_ffv1.cam_high/*.mkv)." >&2; exit 1; }
  fi
  ls "${_d}/videos/chunk-000/observation.images.cam_high/"*.mp4 >/dev/null 2>&1 || {
    echo "ERROR: ${_d} RGB not in canonical layout (observation.images.cam_high/*.mp4)." >&2; exit 1; }
done
if [[ "${_POINTMAP_OFF}" == "1" ]]; then
  echo "Pre-flight OK: RGB canonical across $(echo "${_PREFLIGHT_DIRS}" | wc -w) dataset(s) (pointmap off: depth + intrinsics not checked)."
else
  echo "Pre-flight OK: depth + intrinsics canonical across $(echo "${_PREFLIGHT_DIRS}" | wc -w) dataset(s)."
fi

# Into the data config's text_embedding_cache_dir, which training reads back.
# Already-cached prompts are skipped, so a requeue is cheap.
echo "=== Precomputing YAM text embeds @ $(date) ==="
CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}" python scripts/precompute_text_embeds.py \
  task="${TASK_CONFIG}" \
  data.train.dataset_dirs="${DATASET_DIRS}"

# =============================================================================
# Build override args
# =============================================================================
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"

EXTRA_ARGS=(
  "task=${TASK_CONFIG}"
  # The flex model preset comes from the task config's defaults list.
  # ── Flex-joint per-sample randomization ──
  "model.flex_joint.enabled=true"
  "${FLEX_ARGS[@]}"
)

[[ -n "${NUM_EPOCHS}" ]]   && EXTRA_ARGS+=("num_epochs=${NUM_EPOCHS}")
# A resume never warm-starts, including from a preset's pinned pretrained_ckpt.
[[ -n "${RESUME}" ]]       && EXTRA_ARGS+=("resume=${RESUME}" "pretrained_ckpt=null")
[[ -n "${DATASET_WEIGHTS}" ]]   && EXTRA_ARGS+=("++data.train.dataset_weights=${DATASET_WEIGHTS}")
[[ -n "${SAMPLES_PER_EPOCH}" ]] && EXTRA_ARGS+=("++data.train.samples_per_epoch=${SAMPLES_PER_EPOCH}")
# ── Warm start (RESUME wins) ──────────────────────────────────────────────────
# Weights only, into a fresh run. Mutually exclusive with RESUME — a restored
# state already holds these weights, so on resume the warm start is skipped.
if [[ -n "${PRETRAINED_CKPT:-}" ]]; then
  if [[ -n "${RESUME}" ]]; then
    echo "[resume] RESUME set — skipping PRETRAINED_CKPT warm-init (resuming full state ${RESUME})."
  else
    if [[ ! -e "${PRETRAINED_CKPT}" ]]; then
      echo "Error: PRETRAINED_CKPT does not exist: ${PRETRAINED_CKPT}" >&2
      exit 1
    fi
    EXTRA_ARGS+=("pretrained_ckpt=${PRETRAINED_CKPT}")
  fi
fi
# Unset = the task preset's pretrained_ckpt_strict_shape (true in the
# yam_finetune_* presets, false from train.yaml otherwise).
[[ -n "${PRETRAINED_CKPT_STRICT_SHAPE:-}" ]] && EXTRA_ARGS+=("pretrained_ckpt_strict_shape=${PRETRAINED_CKPT_STRICT_SHAPE}")
# ── YAM-specific overrides ───────────────────────────────────────────────────
# Norm stats: the task preset's data.train.pretrained_norm_stats when it pins a
# file (the 34-D and yam_finetune_* presets do); otherwise fit on the first run
# and cached to the run dir.
EXTRA_ARGS+=("data.train.dataset_dirs=${DATASET_DIRS}")
[[ -n "${VAL_SET_PROPORTION}" ]] && EXTRA_ARGS+=("data.train.val_set_proportion=${VAL_SET_PROPORTION}")
[[ -n "${MAX_STEPS}" ]]          && EXTRA_ARGS+=("max_steps=${MAX_STEPS}")

EXTRA_ARGS+=("$@")

# ── Multi-node bookkeeping ────────────────────────────────────────────────────
NUM_MACHINES="${NNODES:-1}"
MACHINE_RANK="${NODE_RANK:-0}"
MAIN_PROCESS_IP="${MASTER_ADDR:-127.0.0.1}"
MAIN_PROCESS_PORT="${MASTER_PORT:-29501}"

is_integer() { [[ "$1" =~ ^[0-9]+$ ]]; }
if ! is_integer "${NUM_MACHINES}" || ! is_integer "${MACHINE_RANK}"; then
  echo "Error: NUM_MACHINES (${NUM_MACHINES}) and MACHINE_RANK (${MACHINE_RANK}) must be integers." >&2
  exit 1
fi

# ── Output dir / run id ───────────────────────────────────────────────────────
TASK_BASENAME="${TASK_CONFIG}"

# A 2D run arrives as a passthrough override, so the FLEX_P_* knobs above cannot
# see it directly — detect it here instead.
_POINTMAP_DISABLED=0
[[ " ${*,,} " == *"model.enable_pointmap=false "* ]] && _POINTMAP_DISABLED=1

if [[ "${_POINTMAP_DISABLED}" == "1" ]]; then
  # enable_pointmap=false forces the stream off whatever p_present_pointmap/p_jp
  # say, so tag those NA rather than print a probability that never applied.
  REGIME_TAG="flex_pv${_PV}_pd${_PD}_ppNA_jv${_JV}_jd${_JD}_jpNA_2d"
else
  REGIME_TAG="flex_pv${_PV}_pd${_PD}_pp${_PP}_jv${_JV}_jd${_JD}_jp${_JP}"
fi
[[ -n "${PRETRAINED_CKPT:-}" ]] && REGIME_TAG="${REGIME_TAG}_ft"

if [[ -z "${RUN_ID:-}" ]]; then
  if (( NUM_MACHINES <= 1 )); then
    RUN_ID="$(date +%Y-%m-%d_%H-%M-%S)"
  else
    RUN_ID_SYNC_TIMEOUT="${RUN_ID_SYNC_TIMEOUT:-180}"
    RUN_ID_SYNC_PORT="${RUN_ID_SYNC_PORT:-$((MAIN_PROCESS_PORT + 11))}"
    export RUN_ID_SYNC_HOST="${MAIN_PROCESS_IP}"
    export RUN_ID_SYNC_PORT RUN_ID_SYNC_TIMEOUT
    export RUN_ID_SYNC_MACHINE_RANK="${MACHINE_RANK}"
    export RUN_ID_SYNC_NUM_MACHINES="${NUM_MACHINES}"
    export RUN_ID_SYNC_TASK_BASENAME="${TASK_BASENAME}"
    RUN_ID="$(python - <<'PY'
import datetime, os
from datetime import timedelta
import torch.distributed as dist
host = os.environ["RUN_ID_SYNC_HOST"]
port = int(os.environ["RUN_ID_SYNC_PORT"])
timeout_s = int(os.environ["RUN_ID_SYNC_TIMEOUT"])
machine_rank = int(os.environ["RUN_ID_SYNC_MACHINE_RANK"])
num_machines = int(os.environ["RUN_ID_SYNC_NUM_MACHINES"])
task_basename = os.environ.get("RUN_ID_SYNC_TASK_BASENAME", "train")
store = dist.TCPStore(host_name=host, port=port, world_size=num_machines,
                     is_master=(machine_rank == 0), timeout=timedelta(seconds=timeout_s))
key = f"run_id::{task_basename}"
if machine_rank == 0:
    run_id = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    store.set(key, run_id)
print(store.get(key).decode("utf-8"))
PY
)"
  fi
fi

OUTPUT_DIR="./runs/${TASK_BASENAME}/${RUN_ID}_${REGIME_TAG}${RUN_NAME:+_${RUN_NAME}}"
# Auto-resume from THIS run's own output dir, so a relaunch continues the run
# matching the current config. Full-state resume wins over the warm start, whose
# weights the saved state already holds. Skipped when RESUME was given.
# `|| true` because a no-match under pipefail must not abort.
if [[ -z "${RESUME}" && -d "${OUTPUT_DIR}/checkpoints/state" ]]; then
  RESUME="$(ls -d "${OUTPUT_DIR}"/checkpoints/state/step_*/ 2>/dev/null | sort -V | tail -1 || true)"
  RESUME="${RESUME%/}"
  if [[ -n "${RESUME}" ]]; then
    _EA=(); for _a in "${EXTRA_ARGS[@]}"; do
      [[ "$_a" == pretrained_ckpt=* || "$_a" == pretrained_ckpt_strict_shape=* ]] || _EA+=("$_a")
    done
    EXTRA_ARGS=("${_EA[@]}" "resume=${RESUME}" "pretrained_ckpt=null")
    echo "[resume] auto-resume from own output dir: ${RESUME} (pretrained warm-init dropped)"
  fi
fi

echo "[launch] nproc=${NPROC_PER_NODE} machines=${NUM_MACHINES} rank=${MACHINE_RANK} run_id=${RUN_ID}"
echo "[launch] regime=${REGIME_TAG}"

accelerate launch \
  --config_file "${ACCELERATE_CONFIG:-scripts/accelerate_configs/accelerate_zero1_ds.yaml}" \
  --num_machines "${NUM_MACHINES}" \
  --machine_rank "${MACHINE_RANK}" \
  --main_process_ip "${MAIN_PROCESS_IP}" \
  --main_process_port "${MAIN_PROCESS_PORT}" \
  --num_processes "$((NPROC_PER_NODE * NUM_MACHINES))" \
  --rdzv_backend static \
  --deepspeed_multinode_launcher standard \
  scripts/train.py \
  "output_dir=${OUTPUT_DIR}" \
  "wandb.enabled=${WANDB_ENABLED}" \
  "wandb.project=${WANDB_PROJECT}" \
  "wandb.name=$(date +%m%d)_${TASK_BASENAME}_${REGIME_TAG}${RUN_NAME:+_${RUN_NAME}}" \
  "wandb.mode=${WANDB_MODE}" \
  "wandb.group=${WANDB_GROUP:-${TASK_BASENAME}}" \
  "${EXTRA_ARGS[@]}"
