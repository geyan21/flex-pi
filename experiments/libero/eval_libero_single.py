import json
import inspect
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import hydra
import numpy as np
import torch
import torchvision.transforms.functional as transforms_F
from accelerate import PartialState
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from tqdm import tqdm

# try:
#     import rootutils

#     rootutils.setup_root(__file__, indicator=".python-version", pythonpath=True)
# except ModuleNotFoundError:
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.libero.libero_utils import (
    LIBERO_ENV_RESOLUTION,
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_per_cam_depth_uint16_mm,
    invert_gripper_action,
    quat2axisangle,
    save_prediction_video,
    save_rollout_video,
)
from flexpi.datasets.lerobot.processors.flexpi_processor import FlexPiProcessor
from flexpi.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json
from flexpi.utils.pytorch_utils import set_global_seed
from flexpi.utils.libero_setup import assert_mujoco_pin, prepare_libero
from flexpi.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT
from flexpi.models.helpers.action_pad import resolve_action_dim_mask

logger = logging.getLogger(__name__)

# Idempotent; libero_utils already ran it on import above. Repeated here so the
# guarantee survives an import re-sort that moves `libero.libero` up.
prepare_libero()
# Refuse to produce numbers on a renderer other than the one that made the
# training data. See assert_mujoco_pin's docstring for why this is fatal.
assert_mujoco_pin()

from libero.libero import benchmark
from action_ensembler import ActionEnsembler

OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("max", lambda x: max(x))
OmegaConf.register_new_resolver("split", lambda s, idx: s.split("/")[int(idx)])

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _normalize_mixed_precision(mixed_precision: str) -> str:
    key = str(mixed_precision).strip().lower()
    if key not in {"no", "fp16", "bf16"}:
        raise ValueError(
            f"Unsupported mixed_precision: {mixed_precision}. "
            "Expected one of: ['no', 'fp16', 'bf16']."
        )
    return key


def _mixed_precision_to_model_dtype(mixed_precision: str) -> torch.dtype:
    precision = _normalize_mixed_precision(mixed_precision)
    if precision == "no":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    return torch.bfloat16


def _resolve_eval_device(cfg: DictConfig) -> str:
    eval_device = cfg.EVALUATION.get("device")
    if eval_device is not None:
        return str(eval_device)
    return "cuda" if torch.cuda.is_available() else "cpu"


def _resolve_dataset_stats_path(cfg: DictConfig) -> Path:
    explicit = cfg.EVALUATION.get("dataset_stats_path")
    candidates: list[Path] = []

    if explicit is not None:
        candidates.append(Path(os.path.expanduser(os.path.expandvars(str(explicit)))))

    ckpt = Path(os.path.expanduser(os.path.expandvars(str(cfg.ckpt))))
    for parent in list(ckpt.parents)[:4]:
        candidates.append(parent / "dataset_stats.json")

    seen = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if resolved.exists():
            return resolved

    msg = (
        "Failed to locate dataset_stats.json. Tried explicit "
        "EVALUATION.dataset_stats_path and checkpoint parent directories. "
        "Please pass EVALUATION.dataset_stats_path=/path/to/dataset_stats.json."
    )
    raise FileNotFoundError(msg)


def _load_model_checkpoint(model: torch.nn.Module, ckpt: str) -> None:
    model.load_checkpoint(ckpt)
    logging.info("Loaded checkpoint via model.load_checkpoint: %s", ckpt)
    return

    # deprecated legacy checkpoint loading
    payload = torch.load(ckpt, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Legacy checkpoint payload must be dict, got: {type(payload)}")

    if "mot" in payload and hasattr(model, "mot"):
        missing, unexpected = model.mot.load_state_dict(payload["mot"], strict=False)
        logging.warning(
            "Loaded fallback `mot` state_dict with strict=False. Missing=%d Unexpected=%d",
            len(missing),
            len(unexpected),
        )
        return

    state_dict = None
    for key in ("model_state_dict", "state_dict", "model"):
        value = payload.get(key)
        if isinstance(value, dict):
            state_dict = value
            break
    if state_dict is None and all(torch.is_tensor(v) for v in payload.values()):
        state_dict = payload
    if state_dict is None:
        raise ValueError(f"Cannot parse legacy checkpoint keys from: {ckpt}")

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    logging.warning(
        "Loaded fallback model state_dict with strict=False. Missing=%d Unexpected=%d",
        len(missing),
        len(unexpected),
    )


def _center_crop_resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    pil_image = Image.fromarray(image)
    src_w, src_h = pil_image.size
    scale = max(width / src_w, height / src_h)
    resized = pil_image.resize((round(src_w * scale), round(src_h * scale)), resample=Image.BILINEAR)
    rw, rh = resized.size
    left = max((rw - width) // 2, 0)
    top = max((rh - height) // 2, 0)
    cropped = resized.crop((left, top, left + width, top + height))
    return np.asarray(cropped, dtype=np.uint8)


def _build_per_cam_float_tensor(
    img_uint8: np.ndarray,
    *,
    target_hw: tuple[int, int],
    intermediate_hw: tuple[int, int] = (256, 320),
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Float32 per-cam resize chain that mirrors training EXACTLY.

    Training pipeline (RobotVideoDataset + FlexPiProcessor):
        1. ToTensor:                 uint8[H,W,3] -> float32[3,H,W] / 255
        2. Resize(intermediate_hw):  float32 bilinear+antialias to 256x320
        3. Per-cam re-resize:        float32 bilinear+antialias to target_hw
                                     (no-op for cam_high which is already 256x320)
        4. Normalize(0.5, 0.5):      (x - 0.5) / 0.5  =  2x - 1   →  [-1, 1]

    Eval previously called `_stretch_resize` (uint8 in/out) twice, so each
    bilinear hop rounded back to uint8 — accumulating up to ~7.8e-3 max abs
    diff in [-1, 1] space on the 2-hop wrist path (~10% of pixels off by
    >1/255). This helper keeps everything in float32 throughout, eliminating
    that quantisation gap. Returns [1, 3, target_h, target_w] on `device` /
    `dtype`, ready for the per_cam dict.
    """
    h_target, w_target = int(target_hw[0]), int(target_hw[1])
    h_inter,  w_inter  = int(intermediate_hw[0]), int(intermediate_hw[1])
    # Step 1: ToTensor — uint8 [H, W, 3] -> float32 [3, H, W] / 255.
    t = torch.from_numpy(np.ascontiguousarray(img_uint8)).permute(2, 0, 1).contiguous()
    assert t.dtype == torch.uint8, f"expected uint8 input, got {t.dtype}"
    t = t.to(torch.float32) / 255.0
    # Step 2: uniform processor resize to intermediate (256x320).
    if (t.shape[-2], t.shape[-1]) != (h_inter, w_inter):
        t = transforms_F.resize(
            t, size=[h_inter, w_inter],
            interpolation=transforms_F.InterpolationMode.BILINEAR,
            antialias=True,
        )
    # Step 3: per-cam re-resize to target (no-op for cam_high).
    if (h_inter, w_inter) != (h_target, w_target):
        t = transforms_F.resize(
            t, size=[h_target, w_target],
            interpolation=transforms_F.InterpolationMode.BILINEAR,
            antialias=True,
        )
    # Step 4: Normalize(mean=0.5, std=0.5) — algebraically `2x - 1`.
    t = t * 2.0 - 1.0
    # Add batch dim and move to model device/dtype.
    return t.unsqueeze(0).to(device=device, dtype=dtype)


def _stretch_resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Non-uniform resize matching the train-side per-camera concat resize:
    torchvision F.resize(size=[H, W], interpolation=BILINEAR, antialias=True).

    Used by the robotwin t-shape LEGACY eval path (and as a transitional
    primitive) so the eval input distribution tracks training. Note: input
    and output are uint8, so chaining two calls quantises between hops —
    new per_cam fast paths use `_build_per_cam_float_tensor` instead, which
    stays in float32 throughout and is bit-equivalent to training (modulo
    float-op associativity).
    """
    t = torch.from_numpy(image).permute(2, 0, 1)  # [H,W,3] uint8 -> [3,H,W]
    t = transforms_F.resize(
        t,
        size=[height, width],
        interpolation=transforms_F.InterpolationMode.BILINEAR,
        antialias=True,
    )
    return t.permute(1, 2, 0).numpy().astype(np.uint8)


def _normalize_proprio(
    proprio: np.ndarray,
    processor: FlexPiProcessor,
) -> torch.Tensor:
    state_meta = processor.shape_meta["state"]
    if len(state_meta) != 1:
        raise ValueError(
            "LIBERO eval currently expects a single merged state key in shape_meta['state']."
        )
    state_key = state_meta[0]["key"]

    state_batch = {"state": {state_key: torch.as_tensor(proprio, dtype=torch.float32).unsqueeze(0)}}
    state_batch = processor.action_state_transform(state_batch)
    state_batch = processor.normalizer.forward(state_batch)
    # Apply action_state_merger (ConcatLeftAlign) to concat + pad state to
    # state_target_dim — matches training pipeline where proprio is 64-dim
    # padded after normalization.
    state_batch = processor.action_state_merger.forward(state_batch)
    return state_batch["state"]


# Registered composite layouts handled by the T-shape / per-cam compose fast path
# below (2 real LIBERO cams + optional synthetic slot). `robotwin` and
# `tshape_robotwin_384x320_uniform` are pixel-identical (same Slot bboxes; differ only in the
# DINO RoPE grid the model resolves from its own config), so the fast path accepts
# either composite_layout with either concatenation.
# The compose body itself is fully layout-general (it uses
# `_layout.resolved_per_cam_hw(_skm)` + `compose_from_per_cam(..., slot_key_map)`),
# so extending this set is all that's needed to eval a new registered layout.
# (`horizontal` has its own dedicated path above.)
# Canonical names only -- membership is tested through `canonical_layout_name`
# below, so a pre-rename config saying "robotwin" still matches.
_TSHAPE_COMPOSITE_LAYOUTS = (
    "tshape_robotwin_384x320_uniform", "tshape_robotwin_384x320",
    "tshape_libero_2cam_448x512",
)


def _obs_to_model_input(
    obs: dict,
    cfg: DictConfig,
    processor: FlexPiProcessor,
    width: int,
    height: int,
    device: str,
    dtype: torch.dtype,
):
    imgs = get_libero_image(obs)
    image_meta = processor.shape_meta["images"]
    if len(image_meta) < int(processor.num_output_cameras):
        raise ValueError(
            f"shape_meta.images has {len(image_meta)} entries, "
            f"but num_output_cameras={processor.num_output_cameras}."
        )

    def _meta_to_hw(meta: dict, camera_idx: int) -> tuple[int, int]:
        shape = meta["shape"]
        if len(shape) != 3:
            raise ValueError(f"shape_meta.images[{camera_idx}].shape must be [C,H,W], got {shape}")
        return int(shape[1]), int(shape[2])

    concatenation = cfg.data.train.get("concat_multi_camera", "horizontal")
    # `composite_layout` is the source of truth for which compose path the model
    # was trained on. The horizontal data yaml keeps `concat_multi_camera`
    # = the T-shape for the parent-class K-bookkeeping, but the model's actual
    # composite is built by `compose_horizontal_from_per_cam`. Dispatch on the
    # model-side flag so eval mirrors training. Default T-shape preserves
    # all existing eval behavior for non-horizontal models.
    from flexpi.composite_layouts import (
        canonical_layout_name, TSHAPE_ROBOTWIN_384X320_NAME, TSHAPE_384X320_FAMILY,
    )
    composite_layout = canonical_layout_name(
        str(cfg.model.get("composite_layout", TSHAPE_ROBOTWIN_384X320_NAME)).strip().lower()
    )
    num_cameras = processor.num_output_cameras

    # ── Horizontal-layout fast path (bit-equivalent to training) ──────────────
    # Mirror the training-time per_cam construction EXACTLY:
    #   Step 1  FlexPiProcessor.train_transforms.Resize: sim 256×256 → 256×320
    #   Step 2  RobotVideoDataset per-cam re-resize:
    #             - cam_high:        256×320 (no-op, canonical)
    #             - cam_left_wrist:  256×320 → 224×224 (transforms_F.resize antialias)
    #   Step 3  Normalize to [-1, 1] (matches Normalize(mean=0.5, std=0.5))
    #   Step 4  Composite via compose_horizontal_from_per_cam — the SAME function
    #           used at training time → bit-equivalent VAE input.
    # The DINO encoder also takes the per_cam path with these tensors, matching
    # training. Skips the legacy single-shot _stretch_resize that produced a
    # different bilinear chain and a different pixel distribution.
    if num_cameras == 2 and composite_layout == "horizontal":
        tshape_resize_mode = str(cfg.EVALUATION.get("tshape_eval_resize", "stretch"))
        if tshape_resize_mode not in ("stretch", "center_crop"):
            raise ValueError(
                f"EVALUATION.tshape_eval_resize must be 'stretch' or 'center_crop', "
                f"got {tshape_resize_mode!r}"
            )
        if tshape_resize_mode == "center_crop":
            # Legacy reproducibility path — kept for users who want to compare
            # against pre-fix eval numbers. Uses uint8 hops + center crop, which
            # diverges from training's float resize chain by up to ~7.8e-3 in
            # [-1, 1] space (10% of pixels off by >1/255 on the 2-hop wrist).
            head_uniform  = _center_crop_resize(imgs["image"],       width=320, height=256)
            wrist_uniform = _center_crop_resize(imgs["wrist_image"], width=320, height=256)
            cam_high_rgb = head_uniform
            cam_left_rgb = _center_crop_resize(wrist_uniform, width=224, height=224)
            def _to_cam_tensor_norm(rgb_uint8: np.ndarray) -> torch.Tensor:
                t = torch.tensor(rgb_uint8).permute(2, 0, 1).unsqueeze(0).to(
                    device=device, dtype=dtype,
                )
                return t * (2.0 / 255.0) - 1.0
            per_cam = {
                "cam_high":        _to_cam_tensor_norm(cam_high_rgb),
                "cam_left_wrist":  _to_cam_tensor_norm(cam_left_rgb),
                "cam_right_wrist": torch.full(
                    (1, 3, 224, 224), fill_value=-1.0, device=device, dtype=dtype,
                ),
            }
        else:
            # Float32 chain — bit-equivalent to training (modulo float-op
            # associativity). One ToTensor→Resize→Resize→Normalize per cam, no
            # uint8 quantisation between hops.
            per_cam = {
                "cam_high": _build_per_cam_float_tensor(
                    imgs["image"], target_hw=(256, 320),
                    device=device, dtype=dtype,
                ),                                                                # [1, 3, 256, 320]
                "cam_left_wrist": _build_per_cam_float_tensor(
                    imgs["wrist_image"], target_hw=(224, 224),
                    device=device, dtype=dtype,
                ),                                                                # [1, 3, 224, 224]
                "cam_right_wrist": torch.full(
                    (1, 3, 224, 224), fill_value=-1.0, device=device, dtype=dtype,
                ),
            }
        # Build VAE composite via the SAME compose function the training
        # pipeline uses. compose expects 5D [B, 3, T, H, W]; squeeze T after.
        from flexpi.per_cam_compose import (
            compose_horizontal_from_per_cam,
        )
        per_cam_5d = {k: v.unsqueeze(2) for k, v in per_cam.items()}             # [1, 3, 1, H, W]
        composite_5d = compose_horizontal_from_per_cam(per_cam_5d)               # [1, 3, 1, 224, 448]
        x = composite_5d.squeeze(2)                                              # [1, 3, 224, 448]
        # Sanity-check against video_size (catches yaml drift).
        actual_h, actual_w = int(x.shape[-2]), int(x.shape[-1])
        expected_h, expected_w = int(height), int(width)
        assert actual_h == expected_h and actual_w == expected_w, (
            f"Horizontal composite size mismatch: got ({actual_h},{actual_w}), "
            f"expected ({expected_h},{expected_w}) from data.train.video_size."
        )
        proprio = _normalize_proprio(_extract_sim_state(obs), processor)
        return x, proprio, imgs, per_cam

    # ── T-shape (robotwin) fast path for 3D / unified models ──────────────────
    # Mirror RobotVideoDataset's per_cam construction EXACTLY (same
    # pattern as the horizontal path above):
    #   Step 1  FlexPiProcessor.train_transforms.Resize: sim 256×256 → 256×320
    #   Step 2  RobotVideoDataset per-cam re-resize:
    #             - cam_high:        256×320 (no-op, canonical)
    #             - cam_left_wrist:  256×320 → 224×224 (transforms_F.resize antialias)
    #             - cam_right_wrist: zeros 224×224 (synthetic)
    #   Step 3  Normalize to [-1, 1]
    #   Step 4  Composite via compose_robotwin_from_per_cam — the SAME function
    #           used at training time → bit-equivalent VAE input + DINO sees
    #           per_cam tensors at canonical sizes (no 128→224 upsample).
    # Gated on _is_3d_eval(cfg) so non-3D base FlexPi / flexpi_joint configs
    # keep the legacy numpy-rgb path (their established eval baselines stay
    # reproducible).
    #
    # tshape_robotwin_384x320_uniform is accepted alongside the asymmetric one: the two layouts'
    # PIXEL composites are bit-identical (same Slot bboxes — see
    # composite_layouts.py); they differ only in DINO patch grid / RoPE
    # regions, which the model resolves from its own config, not from the
    # compose function. Runs on the uniform grid save
    # concat_multi_camera=tshape_robotwin_384x320_uniform into config.yaml, which
    # eval_config_source=auto swaps in here. tshape_libero_2cam_448x512 also
    # routes here — see _TSHAPE_COMPOSITE_LAYOUTS; the body is layout-general.
    if (
        num_cameras == 2
        and composite_layout in _TSHAPE_COMPOSITE_LAYOUTS
        and canonical_layout_name(concatenation) in _TSHAPE_COMPOSITE_LAYOUTS
        and _is_3d_eval(cfg)
    ):
        from flexpi.per_cam_compose import compose_from_per_cam
        from flexpi.composite_layouts import get_layout
        # Resolve the trained slot assignment so a swapped composite (e.g. the
        # wrist-top LIBERO layout) is mirrored at eval. Default map (None) →
        # byte-identical to the previous cam_high-on-top behavior.
        _layout = get_layout(composite_layout)
        _skm = _resolve_model_slot_key_map(cfg)
        # Physical→canonical camera binding (fixed for LIBERO): exterior feed =
        # cam_high, wrist feed = cam_left_wrist. Any layout cam not bound here is
        # a synthetic zero cam (cam_right_wrist), matching synthetic_zero_cams.
        _phys_src = {"cam_high": imgs["image"], "cam_left_wrist": imgs["wrist_image"]}

        tshape_resize_mode = str(cfg.EVALUATION.get("tshape_eval_resize", "stretch"))
        if tshape_resize_mode not in ("stretch", "center_crop"):
            raise ValueError(
                f"EVALUATION.tshape_eval_resize must be 'stretch' or 'center_crop', "
                f"got {tshape_resize_mode!r}"
            )
        if tshape_resize_mode == "center_crop":
            if _skm is not None:
                raise ValueError(
                    "tshape_eval_resize='center_crop' is a legacy default-layout "
                    "reproduction path and does not support a swapped "
                    "composite_layout_slot_key_map; use 'stretch'."
                )
            head_uniform  = _center_crop_resize(imgs["image"],       width=320, height=256)
            wrist_uniform = _center_crop_resize(imgs["wrist_image"], width=320, height=256)
            cam_high_rgb = head_uniform
            cam_left_rgb = _center_crop_resize(wrist_uniform, width=224, height=224)
            def _to_cam_tensor_norm(rgb_uint8: np.ndarray) -> torch.Tensor:
                t = torch.tensor(rgb_uint8).permute(2, 0, 1).unsqueeze(0).to(
                    device=device, dtype=dtype,
                )
                return t * (2.0 / 255.0) - 1.0
            per_cam = {
                "cam_high":        _to_cam_tensor_norm(cam_high_rgb),
                "cam_left_wrist":  _to_cam_tensor_norm(cam_left_rgb),
                "cam_right_wrist": torch.full(
                    (1, 3, 224, 224), fill_value=-1.0, device=device, dtype=dtype,
                ),
            }
        else:
            # Float32 chain — bit-equivalent to training (modulo float-op
            # associativity). Each cam is resized to ITS SLOT's src_hw (via the
            # layout + slot_key_map), so under the wrist-top swap the wrist lands
            # at the top slot's 256×320 and the exterior at the bottom slot's
            # 224×224 — matching the dataset's _rgb_per_cam_hw. Default map →
            # cam_high 256×320 / cam_left_wrist 224×224 (unchanged).
            _per_cam_src = _layout.resolved_per_cam_hw(_skm)   # {cam_name: (h, w)}
            per_cam = {}
            for _cam_name, (_h, _w) in _per_cam_src.items():
                if _cam_name in _phys_src:
                    per_cam[_cam_name] = _build_per_cam_float_tensor(
                        _phys_src[_cam_name], target_hw=(_h, _w),
                        device=device, dtype=dtype,
                    )
                else:
                    per_cam[_cam_name] = torch.full(
                        (1, 3, _h, _w), fill_value=-1.0, device=device, dtype=dtype,
                    )
        # Build VAE composite via the SAME compose function the training pipeline
        # uses, with the trained slot_key_map so a swapped layout is honored.
        # compose expects 5D [B, 3, T, H, W]; squeeze T after.
        per_cam_5d = {k: v.unsqueeze(2) for k, v in per_cam.items()}             # [1, 3, 1, H, W]
        composite_5d = compose_from_per_cam(
            per_cam_5d, _layout, slot_key_map=_skm,
        )                                                                        # [1, 3, 1, 384, 320]
        x = composite_5d.squeeze(2)                                              # [1, 3, 384, 320]
        # Sanity-check against video_size (catches yaml drift).
        actual_h, actual_w = int(x.shape[-2]), int(x.shape[-1])
        expected_h, expected_w = int(height), int(width)
        assert actual_h == expected_h and actual_w == expected_w, (
            f"T-shape composite size mismatch: got ({actual_h},{actual_w}), "
            f"expected ({expected_h},{expected_w}) from data.train.video_size."
        )
        proprio = _normalize_proprio(_extract_sim_state(obs), processor)
        return x, proprio, imgs, per_cam

    # ── Legacy numpy-rgb paths (basic FlexPi / 1-cam / non-robotwin concat) ──
    if num_cameras == 1:
        primary_h, primary_w = _meta_to_hw(image_meta[0], camera_idx=0)
        rgb = _center_crop_resize(imgs["image"], width=primary_w, height=primary_h)
    elif num_cameras == 2:
        if canonical_layout_name(concatenation) in TSHAPE_384X320_FAMILY:
            # Match robot_video_dataset.py robotwin T-layout exactly:
            # top = image resized to 256×320; bottom = [wrist@128×160 | black@128×160].
            # Final 384×320 matches video_size. Hardcoded sizes override shape_meta
            # because robotwin concat uses fixed cam layout, not shape_meta.shape.
            #
            # Train side uses transforms_F.resize (non-uniform stretch) per camera;
            # eval should match that, not center-crop, so the input distribution
            # tracks training. `tshape_eval_resize` is a transitional knob: set
            # "center_crop" to reproduce legacy eval numbers from before this fix.
            tshape_resize_mode = str(cfg.EVALUATION.get("tshape_eval_resize", "stretch"))
            if tshape_resize_mode == "stretch":
                _resize_fn = _stretch_resize
            elif tshape_resize_mode == "center_crop":
                _resize_fn = _center_crop_resize
            else:
                raise ValueError(
                    f"EVALUATION.tshape_eval_resize must be 'stretch' or 'center_crop', "
                    f"got {tshape_resize_mode!r}"
                )
            primary = _resize_fn(imgs["image"], width=320, height=256)
            wrist = _resize_fn(imgs["wrist_image"], width=160, height=128)
            bottom = np.concatenate([wrist, np.zeros_like(wrist)], axis=1)
            rgb = np.concatenate([primary, bottom], axis=0)
        else:
            primary_h, primary_w = _meta_to_hw(image_meta[0], camera_idx=0)
            wrist_h, wrist_w = _meta_to_hw(image_meta[1], camera_idx=1)
            primary = _center_crop_resize(imgs["image"], width=primary_w, height=primary_h)
            wrist = _center_crop_resize(imgs["wrist_image"], width=wrist_w, height=wrist_h)
            if concatenation == "horizontal":
                rgb = np.concatenate([primary, wrist], axis=1)
            elif concatenation == "vertical":
                rgb = np.concatenate([primary, wrist], axis=0)
            else:
                raise ValueError(f"Invalid concat_multi_camera: {concatenation}")
    else:
        raise ValueError(f"LIBERO eval currently supports num_output_cameras in [1, 2], got {num_cameras}.")

    actual_h, actual_w = int(rgb.shape[0]), int(rgb.shape[1])
    expected_h, expected_w = int(height), int(width)
    image_shapes = [meta["shape"] for meta in image_meta]
    assert actual_h == expected_h and actual_w == expected_w, (
        "Input image size mismatch after per-camera resize + concat: "
        f"got (H,W)=({actual_h},{actual_w}), expected (H,W)=({expected_h},{expected_w}) "
        f"from data.train.video_size={[expected_h, expected_w]}; "
        f"shape_meta.images={image_shapes}, concat_multi_camera={concatenation}."
    )

    x = torch.tensor(rgb).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
    x = x * (2.0 / 255.0) - 1.0

    proprio = _normalize_proprio(_extract_sim_state(obs), processor)

    # Non-horizontal paths return per_cam=None — the model's DINO encoder will
    # take its composite-slice path (correct for T-shape and 7D-horizontal
    # configs that operate on the 384×320 / 224×448-non-3D layouts).
    return x, proprio, imgs, None


def _extract_sim_state(obs: dict) -> np.ndarray:
    """Build simulator state from current observation.

    This is used as proprio input for model inference.
    """
    state = np.concatenate(
        (
            obs["robot0_eef_pos"],
            quat2axisangle(obs["robot0_eef_quat"]),
            obs["robot0_gripper_qpos"],
        )
    ).astype(np.float32)
    return state


# ---------------------------------------------------------------------------
# FlexPi3D eval support — depth + intrinsics plumbing
# ---------------------------------------------------------------------------

def _is_3d_eval(cfg: DictConfig) -> bool:
    """Detect whether this is a FlexPi3D eval based on the model config.

    Probes ``pointmap_norm_bounds`` — the pointmap key that survived the
    knob purge, so this works for both current and pre-purge saved configs.
    """
    return cfg.model.get("pointmap_norm_bounds", None) is not None


_HEAD_KEY_ALIASES = ("cam_high", "image")
_LEFT_WRIST_KEY_ALIASES = ("cam_left_wrist", "wrist_image")


def _per_cam_depth_target_hw(cfg: DictConfig) -> dict:
    """Read per-cam depth target HW from data.train.shape_meta.depth.

    Returns a dict normalised to BOTH key styles so downstream lookups work
    regardless of whether the saved cfg uses canonical names (cam_high,
    cam_left_wrist) or legacy raw names (image, wrist_image). Each canonical
    key maps to the same (H, W) tuple as its legacy alias. Falls back to the
    default {(256,320) head, (224,224) wrist} if shape_meta.depth is absent.

    Bug fix history: prior versions returned the dict keyed only by whatever
    was in shape_meta, so when the yaml used canonical names the downstream
    `target_hw.get("image")` lookup silently fell through to the hardcoded
    default. With the libero canonical-key configs the default happened to
    match — but any future divergence (e.g. wrist depth at 256x320) would
    silently feed the wrong-sized depth tensor at eval.
    """
    out: dict = {}
    depth_meta = cfg.data.train.shape_meta.get("depth", None)
    if depth_meta is not None:
        for meta in depth_meta:
            shape = meta["shape"]   # [C, H, W]
            hw = (int(shape[1]), int(shape[2]))
            key = meta["key"]
            out[key] = hw
            # Mirror to all known aliases so canonical AND legacy lookups work.
            if key in _HEAD_KEY_ALIASES:
                for alias in _HEAD_KEY_ALIASES:
                    out.setdefault(alias, hw)
            elif key in _LEFT_WRIST_KEY_ALIASES:
                for alias in _LEFT_WRIST_KEY_ALIASES:
                    out.setdefault(alias, hw)
    # Fill in defaults for any unset alias.
    out.setdefault("cam_high",       (256, 320))
    out.setdefault("image",          (256, 320))
    out.setdefault("cam_left_wrist", (224, 224))
    out.setdefault("wrist_image",    (224, 224))
    return out


def _resolve_model_slot_key_map(cfg: DictConfig) -> Optional[dict]:
    """Resolve the model's ``composite_layout_slot_key_map`` (slot→cam) from the
    (saved) config as a plain dict, or None when unset.

    Set for the wrist-top LIBERO layout (and any non-default slot binding); eval
    mirrors it so the observation composite AND the camera-intrinsics stack order
    match the trained slot assignment. None → the layout's default_slot_key_map,
    i.e. byte-identical to the pre-swap eval behavior.
    """
    skm = cfg.model.get("composite_layout_slot_key_map", None)
    if skm is None:
        return None
    skm = OmegaConf.to_container(skm, resolve=True)
    return {str(k): str(v) for k, v in skm.items()}


def _build_K(entry: dict, target_hw: tuple) -> torch.Tensor:
    """K from a camera_intrinsics.json entry, rescaled onto ``target_hw``.

    Endpoint convention ``(target - 1) / (raw - 1)``, the one
    ``RobotVideoDataset._k_stack_from_raw`` applies at training time, paired
    with the linspace-endpoint nearest depth regrid in ``_depth_to_grid`` so K
    and depth land on one grid. A raw extent of 1 or less keeps the scale at 1.
    """
    h_t, w_t = target_hw
    raw_w, raw_h = float(entry["width"]), float(entry["height"])
    sx = (float(w_t) - 1.0) / (raw_w - 1.0) if raw_w > 1 else 1.0
    sy = (float(h_t) - 1.0) / (raw_h - 1.0) if raw_h > 1 else 1.0
    return torch.tensor(
        [[entry["fx"] * sx, 0.0, entry["cx"] * sx],
         [0.0, entry["fy"] * sy, entry["cy"] * sy],
         [0.0, 0.0, 1.0]], dtype=torch.float32,
    )


def _depth_to_grid(arr: np.ndarray, target_hw: tuple) -> torch.Tensor:
    """Depth ``[H, W]`` int32 mm on ``target_hw``, sampled the way training does.

    The linspace-endpoint nearest sampler of ``RobotVideoDataset``: rounded
    ``linspace(0, src - 1, target)`` rows and columns, gathered. That is the
    grid the endpoint K rescale in ``_build_K`` describes. int32 rather than
    uint16 because the gather has no uint16 CPU kernel; the dataset path returns
    int32 (FFV1 decoder) too, and PointmapEncoder casts to float internally.
    """
    t = torch.from_numpy(arr.astype(np.int32))
    h_t, w_t = int(target_hw[0]), int(target_hw[1])
    if tuple(t.shape) != (h_t, w_t):
        ys = torch.linspace(0, t.shape[0] - 1, h_t).round().long()
        xs = torch.linspace(0, t.shape[1] - 1, w_t).round().long()
        t = t[ys[:, None], xs[None, :]]
    return t


def _load_eval_camera_intrinsics(cfg: DictConfig) -> torch.Tensor:
    """Load the per-cam intrinsics tensor for 3D eval.

    Path is configured via ``EVALUATION.camera_intrinsics_path``. K is
    rescaled to the depth-grid layout that FlexPi3D's PointmapEncoder
    expects (head 256×320, left wrist matches the target depth shape, right
    wrist is synthetic — uses the same target as left for consistency).
    """
    path = cfg.EVALUATION.get("camera_intrinsics_path", None)
    if path is None:
        raise ValueError(
            "EVALUATION.camera_intrinsics_path is required for FlexPi3D eval. "
            "Point it at the dataset's meta/camera_intrinsics.json."
        )
    targets = _per_cam_depth_target_hw(cfg)
    head_hw = targets.get("cam_high", (256, 320))
    wrist_hw = targets.get("cam_left_wrist", (224, 224))
    # Both styles of camera_intrinsics.json are accepted:
    #   - canonical: {"cam_high": ..., "cam_left_wrist": ...} — what the published
    #     dataset ships.
    #   - legacy:    {"image": ..., "wrist_image": ...}      — older LIBERO depth
    #     dumps, kept readable so pre-canonicalisation mirrors still evaluate.
    # Falling back to the legacy keys keeps eval working on dev machines / older
    # mirrors that haven't been re-run through the canonicaliser. The dataset
    # loader (RobotVideoDataset._load_robotwin_intrinsics_for_dir) is
    # stricter — it requires whatever style shape_meta.depth uses — so any
    # train-eval drift on this would surface at training time, not eval.
    # Canonical 3-cam encoder order: cam_high → cam_left_wrist → cam_right_wrist;
    # right_wrist is synthetic for LIBERO so we reuse the left wrist's K (the
    # encoder zeros that cam's depth anyway).
    import json
    with open(str(path), "r") as f:
        raw = json.load(f)

    def _resolve_entry(aliases: tuple[str, ...]) -> dict:
        for name in aliases:
            if name in raw:
                return raw[name]
        raise KeyError(
            f"camera_intrinsics.json at {path} has none of the expected keys "
            f"{aliases}; found {list(raw)}. The published dataset ships a correct "
            "one in each suite's meta/ — re-download it rather than writing one "
            "by hand: LIBERO's two cameras have different FOVs, so intrinsics "
            "derived from a single shared FOV are wrong by 1.85x on the wrist."
        )

    K_high = _build_K(_resolve_entry(_HEAD_KEY_ALIASES),       head_hw)
    K_left = _build_K(_resolve_entry(_LEFT_WRIST_KEY_ALIASES), wrist_hw)
    K_right = K_left.clone()  # synthetic — depth is zero, K value doesn't matter
    _K_by_cam = {"cam_high": K_high, "cam_left_wrist": K_left, "cam_right_wrist": K_right}
    # The PointmapEncoder indexes camera_intrinsics BY SLOT (cam_slots() order),
    # so stack in slot order under the trained slot_key_map. Default map →
    # [cam_high, cam_left_wrist, cam_right_wrist] (unchanged); the wrist-top swap
    # → [cam_left_wrist, cam_high, cam_right_wrist]. PointmapEncoder asserts
    # K.shape[0] == len(layout.cam_slots()).
    from flexpi.composite_layouts import (
        get_layout, is_layout_name, canonical_layout_name, TSHAPE_ROBOTWIN_384X320_NAME,
    )
    _cl = canonical_layout_name(
        str(cfg.model.get("composite_layout", TSHAPE_ROBOTWIN_384X320_NAME)).strip().lower()
    )
    if is_layout_name(_cl):
        _layout = get_layout(_cl)
        _kmap = _layout.resolve_slot_key_map(_resolve_model_slot_key_map(cfg))
        return torch.stack([_K_by_cam[_kmap[s.key]] for s in _layout.cam_slots()], dim=0)
    # Legacy non-registered layouts (e.g. "horizontal"): canonical order.
    return torch.stack([K_high, K_left, K_right], dim=0)   # [3, 3, 3]


def _extract_per_cam_depth_for_model(
    obs: dict,
    env,
    target_hw: dict,
    device: str,
) -> dict:
    """Build the per_cam_depth dict that FlexPi3D.infer_action expects.

    Returns canonical-key dict
        {"cam_high": [1,1,H,W], "cam_left_wrist": [1,1,H,W], "cam_right_wrist": [1,1,H,W]}
    each tensor uint16 mm on ``device`` (depth tensors are kept as int because
    PointmapEncoder unprojects in float internally). cam_right_wrist is a
    zero tensor (LIBERO has only 2 physical cams; matches synthetic_zero_cams
    in the data config).
    """
    raw = get_libero_per_cam_depth_uint16_mm(obs, env)   # {image, wrist_image} np.uint16

    def _to_tensor(arr: np.ndarray, target_hw_tuple: tuple) -> torch.Tensor:
        return _depth_to_grid(arr, target_hw_tuple).to(device).reshape(1, 1, *target_hw_tuple)

    head_hw = target_hw.get("image", (256, 320))
    wrist_hw = target_hw.get("wrist_image", (224, 224))
    cam_high = _to_tensor(raw["image"], head_hw)
    cam_left = _to_tensor(raw["wrist_image"], wrist_hw)
    cam_right = torch.zeros_like(cam_left)               # synthetic zero — matches data config
    return {
        "cam_high":        cam_high,
        "cam_left_wrist":  cam_left,
        "cam_right_wrist": cam_right,
    }


def _denormalize_action(action: torch.Tensor, processor: FlexPiProcessor) -> np.ndarray:
    if action.ndim == 2:
        action = action.unsqueeze(0)
    if action.ndim != 3:
        raise ValueError(f"Expected action tensor [B, T, D], got {tuple(action.shape)}")

    action_meta = processor.shape_meta["action"]
    if len(action_meta) != 1:
        raise ValueError(
            "LIBERO eval currently expects a single merged action key in shape_meta['action']."
        )
    action_key = action_meta[0]["key"]
    action = action.to(dtype=torch.float32, device="cpu")

    # Inverse of training's processor chain. Model output is [B, T, action_target_dim]
    # (e.g. 30 for the lingbot-va-aligned 30D layout, 28 for arm-grouped, 7 for
    # release-native) in normalized space; we need [B, T, raw_dim] in physical units
    # for env.step.
    #
    #   merger.backward:                 [B, T, total_dim] → {action_key: [B, T, shape_dim]}
    #   normalizer.backward:             denormalize per-key
    #   action_state_transforms[reversed].backward:
    #                                    convert per-key from `shape` repr back to `raw_shape`
    #                                    (e.g. quaternion → axis_angle for the 30D quat path)
    #
    # ConcatLeftAlign.backward unconditionally processes both action and state
    # (see action_state_merger.py:31-39), so we pass a dummy state tensor to
    # satisfy the shape assertion; its denormalized output is discarded.
    merger = processor.action_state_merger
    batch = {"action": action}
    # Normalizer + merger both iterate over {"action", "state"} unconditionally, so
    # pass a dummy zero state to satisfy downstream asserts; its outputs are discarded.
    state_dim = getattr(merger, "state_target_dim", None) or getattr(merger, "total_dim", None)
    if state_dim is not None:
        batch["state"] = torch.zeros(
            action.shape[0], action.shape[1], int(state_dim),
            dtype=action.dtype, device=action.device,
        )
    batch = merger.backward(batch)
    batch = processor.normalizer.backward(batch)
    if processor.action_state_transforms is not None:
        for trans in reversed(processor.action_state_transforms):
            batch = trans.backward(batch)
    denorm = batch["action"][action_key]
    return denorm.numpy()


def _get_num_video_frames(cfg: DictConfig) -> int:
    return (int(cfg.data.train.num_frames) - 1) // int(cfg.data.train.action_video_freq_ratio) + 1


def _validate_visualize_future_video_cfg(cfg: DictConfig) -> None:
    if not bool(cfg.EVALUATION.get("visualize_future_video", False)):
        return

    action_conditioned = cfg.model.video_dit_config.get("action_conditioned", None)
    if action_conditioned is not False:
        raise ValueError(
            "EVALUATION.visualize_future_video=true requires "
            "model.video_dit_config.action_conditioned=false."
        )


def _select_predicted_future_frames(pred_video: list[Image.Image], cfg: DictConfig) -> list[Image.Image]:
    if len(pred_video) == 0:
        raise ValueError("`infer_joint` returned an empty predicted video.")

    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    action_video_freq_ratio = int(cfg.data.train.action_video_freq_ratio)
    num_future_frames = replan_steps // action_video_freq_ratio
    keep_frames = 1 + num_future_frames
    return list(pred_video[:keep_frames])


def _get_future_frame_capture_steps(cfg: DictConfig) -> list[int]:
    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    action_video_freq_ratio = int(cfg.data.train.action_video_freq_ratio)
    num_future_frames = replan_steps // action_video_freq_ratio
    return [step_idx * action_video_freq_ratio for step_idx in range(num_future_frames + 1)]


def _frame_to_rgb_array(frame: Any) -> np.ndarray:
    if isinstance(frame, dict):
        images = []
        for value in frame.values():
            value_array = np.array(value) if isinstance(value, Image.Image) else np.array(value, copy=True)
            images.append(value_array)
        return np.concatenate(images, axis=1)
    if isinstance(frame, Image.Image):
        return np.array(frame.convert("RGB"))
    return np.array(frame, copy=True)


def _compute_clip_mean_psnr(
    gt_frames: list[Any],
    pred_frames: list[Any],
    eps: float = 1e-8,
) -> Optional[float]:
    if len(gt_frames) == 0 or len(pred_frames) == 0:
        return None
    assert len(gt_frames) == len(pred_frames), (
        "GT/pred frame count mismatch for PSNR: "
        f"len(gt_frames)={len(gt_frames)} len(pred_frames)={len(pred_frames)}. "
        "This indicates temporal misalignment in future-video capture."
    )
    num_frames = len(gt_frames)

    frame_psnr_values = []
    for gt_frame, pred_frame in zip(gt_frames[:num_frames], pred_frames[:num_frames]):
        gt_image = _frame_to_rgb_array(gt_frame)
        pred_image = _frame_to_rgb_array(pred_frame)
        target_h, target_w = pred_image.shape[:2]
        if gt_image.shape[:2] != (target_h, target_w):
            gt_image = np.array(
                Image.fromarray(gt_image).resize((target_w, target_h), resample=Image.BILINEAR)
            )

        gt_f32 = gt_image.astype(np.float32)
        pred_f32 = pred_image.astype(np.float32)
        mse = float(np.mean((pred_f32 - gt_f32) ** 2))
        psnr = 10.0 * np.log10((255.0 * 255.0) / max(mse, eps))
        frame_psnr_values.append(float(psnr))

    if len(frame_psnr_values) == 0:
        return None
    return float(np.mean(frame_psnr_values))


def _predict_action_chunk(
    obs: dict,
    task_description: str,
    model: torch.nn.Module,
    processor: FlexPiProcessor,
    cfg: DictConfig,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
    env=None,
) -> tuple[np.ndarray, dict, Optional[list[Image.Image]]]:
    num_inference_steps_cfg = cfg.EVALUATION.get("num_inference_steps", None)
    if num_inference_steps_cfg is None:
        num_inference_steps = int(cfg.get("eval_num_inference_steps", 20))
    else:
        num_inference_steps = int(num_inference_steps_cfg)
    prompt_template = DEFAULT_PROMPT
    prompt = prompt_template.format(task=task_description)

    image, proprio, imgs, per_cam = _obs_to_model_input(
        obs,
        cfg=cfg,
        processor=processor,
        width=input_w,
        height=input_h,
        device=model_device,
        dtype=model.torch_dtype,
    )

    infer_kwargs = {
        "prompt": prompt,
        "input_image": image,
        "action_horizon": action_horizon,
        "negative_prompt": str(cfg.EVALUATION.get("negative_prompt", "")),
        "text_cfg_scale": float(cfg.EVALUATION.get("text_cfg_scale", 1.0)),
        "num_inference_steps": num_inference_steps,
        "proprio": proprio,
        "sigma_shift": (
            None
            if cfg.EVALUATION.get("sigma_shift") is None
            else float(cfg.EVALUATION.get("sigma_shift"))
        ),
        "seed": None if cfg.get("seed") is None else int(cfg.seed),
        "rand_device": str(cfg.EVALUATION.get("rand_device", "cpu")),
        "tiled": bool(cfg.EVALUATION.get("tiled", False)),
    }
    # Pass per_cam (built only for horizontal composite) so the model's DINO
    # encoder takes its per_cam path instead of slicing the composite.
    if per_cam is not None:
        infer_kwargs["per_cam"] = per_cam
    visualize_future_video = bool(cfg.EVALUATION.get("visualize_future_video", False))
    predicted_future_frames = None
    if visualize_future_video:
        infer_kwargs["num_video_frames"] = _get_num_video_frames(cfg)
    elif "num_video_frames" in inspect.signature(model.infer_action).parameters:
        infer_kwargs["num_video_frames"] = _get_num_video_frames(cfg)

    # DreamZero-style step-skipping: only the joint-path FlexPi variants
    # accept these kwargs. Sniff so non-joint models stay unaffected.
    if "dynamic_step_skip" in inspect.signature(model.infer_action).parameters:
        infer_kwargs["dynamic_step_skip"] = bool(cfg.EVALUATION.get("dynamic_step_skip", False))
        thr = cfg.EVALUATION.get("step_skip_thresholds", None)
        if thr is not None:
            infer_kwargs["step_skip_thresholds"] = [(float(t), int(n)) for t, n in thr]
        src = cfg.EVALUATION.get("step_skip_sim_sources", None)
        if src is not None:
            infer_kwargs["step_skip_sim_sources"] = [str(s) for s in src]
        infer_kwargs["step_skip_sim_aggregation"] = str(
            cfg.EVALUATION.get("step_skip_sim_aggregation", "mean")
        )

    # FlexPi3D: feed per-cam depth from the simulator. Camera intrinsics are
    # set once at eval startup (see eval_single_process), so we don't pass K
    # per-step — the model resolves it from its cached value.
    if _is_3d_eval(cfg):
        if env is None:
            raise RuntimeError("FlexPi3D eval requires `env` to be passed into _predict_action_chunk.")
        infer_kwargs["per_cam_depth"] = _extract_per_cam_depth_for_model(
            obs=obs,
            env=env,
            target_hw=_per_cam_depth_target_hw(cfg),
            device=model_device,
        )

    # Flex-joint per-call regime overrides (FlexPiUnifiedJoint with
    # flex_joint.enabled=True). Maps cfg.EVALUATION.infer_{present,joint}_X to
    # the model's `present_X` / `joint_X` kwargs on infer_action. Sniff the
    # signature so non-flex models (FlexPi, FlexPiLatent, ...) stay
    # unaffected. Mirrors the deploy-side forwarding in
    # experiments/robotwin/flexpi_policy/deploy_policy.py.
    _infer_sig = inspect.signature(model.infer_action).parameters
    for _eval_key, _kwarg in (
        ("infer_present_video", "present_video"),
        ("infer_present_dino", "present_dino"),
        ("infer_present_pointmap", "present_pointmap"),
        ("infer_joint_video", "joint_video"),
        ("infer_joint_dino", "joint_dino"),
        ("infer_joint_pointmap", "joint_pointmap"),
    ):
        if _kwarg in _infer_sig and cfg.EVALUATION.get(_eval_key) is not None:
            infer_kwargs[_kwarg] = bool(cfg.EVALUATION.get(_eval_key))

    with torch.no_grad():
        if visualize_future_video:
            pred = model.infer_joint(**infer_kwargs)
            predicted_future_frames = _select_predicted_future_frames(pred["video"], cfg)
        else:
            pred = model.infer_action(**infer_kwargs)
    action = pred["action"]  # [T, D]

    action = _denormalize_action(action, processor)[0]  # [T, D]

    # The dataloader flips the sign of the gripper action to align with other datasets
    # (0 = close, 1 = open), so flip it back (-1 = open, +1 = close) before executing the action
    action[..., -1] = action[..., -1] * 2 - 1
    action = invert_gripper_action(action)
    if bool(cfg.EVALUATION.get("binarize_gripper", False)):
        action[..., -1] = np.sign(action[..., -1])
    return action, imgs, predicted_future_frames


def _get_max_steps(task_suite_name: str) -> int:
    # Follows Fast-WAM's LIBERO eval setup
    suite_steps = {
        "libero_spatial": 400,
        "libero_object": 400,
        "libero_goal": 400,
        "libero_10": 700,
        "libero_90": 700,
    }
    if task_suite_name in suite_steps:
        return suite_steps[task_suite_name]
    raise ValueError(
        f"Unknown task suite: {task_suite_name}. Known: {sorted(suite_steps)}"
    )


def run_single_episode(
    env,
    initial_state,
    task_description: str,
    model: torch.nn.Module,
    processor: FlexPiProcessor,
    cfg: DictConfig,
    episode_idx: int,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
) -> tuple[bool, list, list[dict[str, Any]], Optional[float]]:
    max_steps = _get_max_steps(cfg.EVALUATION.task_suite_name)
    replan_steps = int(cfg.EVALUATION.get("replan_steps", 5))
    num_steps_wait = int(cfg.EVALUATION.get("num_steps_wait", 5))
    use_action_ensembler = bool(cfg.EVALUATION.get("use_action_ensembler", False))
    visualize_future_video = bool(cfg.EVALUATION.get("visualize_future_video", False))
    capture_steps = set(_get_future_frame_capture_steps(cfg)[1:])

    env.reset()
    obs = env.set_init_state(initial_state)
    if use_action_ensembler:
        ensembler = ActionEnsembler()
        ensembler.reset()

    replay_images = []
    predicted_future_video_clips: list[dict[str, Any]] = []
    episode_future_clip_psnr: list[float] = []
    pending_actions: list[list[float]] = []
    current_predicted_future_clip: Optional[dict[str, Any]] = None
    current_replan_step = 0
    current_replan_idx = -1

    t = 0
    done = False
    pbar = tqdm(total=max_steps + num_steps_wait, desc=f"Episode {episode_idx + 1}")
    while t < max_steps + num_steps_wait:
        pbar.update(1)
        if t < num_steps_wait:
            obs, _, done, _ = env.step(get_libero_dummy_action())
            t += 1
            continue

        if len(pending_actions) == 0:
            action_chunk, imgs, predicted_future_frames = _predict_action_chunk(
                obs=obs,
                task_description=task_description,
                model=model,
                processor=processor,
                cfg=cfg,
                action_horizon=action_horizon,
                input_w=input_w,
                input_h=input_h,
                model_device=model_device,
                env=env,
            )
            if predicted_future_frames is not None:
                current_replan_idx += 1
                current_predicted_future_clip = {
                    "replan_idx": current_replan_idx,
                    "gt_frames": [imgs.copy()],
                    "pred_frames": predicted_future_frames,
                }
            else:
                current_predicted_future_clip = None
            current_replan_step = 0
            if use_action_ensembler:
                ensembler.add_actions(action_chunk, t)
                pending_actions = [ensembler.get_action(ts).tolist() for ts in range(t, t + replan_steps)]
            else:
                pending_actions = action_chunk[:replan_steps].tolist()
            replay_images.append(imgs.copy())
        else:
            imgs = get_libero_image(obs)
            replay_images.append(imgs.copy())

        obs, _, done, _ = env.step(pending_actions.pop(0))
        if visualize_future_video and current_predicted_future_clip is not None:
            current_replan_step += 1
            if current_replan_step in capture_steps:
                current_predicted_future_clip["gt_frames"].append(get_libero_image(obs))
            if done or len(pending_actions) == 0:
                expected_frame_count = 1 + sum(
                    1 for capture_step in capture_steps if capture_step <= current_replan_step
                )
                gt_len = len(current_predicted_future_clip["gt_frames"])
                pred_len = len(current_predicted_future_clip["pred_frames"])
                assert gt_len == expected_frame_count, (
                    "GT future frames do not match expected capture count: "
                    f"gt_len={gt_len} expected={expected_frame_count} "
                    f"episode={episode_idx} replan={current_predicted_future_clip['replan_idx']} "
                    f"current_replan_step={current_replan_step} capture_steps={sorted(capture_steps)}."
                )
                assert pred_len >= expected_frame_count, (
                    "Predicted future frames shorter than expected capture count: "
                    f"pred_len={pred_len} expected={expected_frame_count} "
                    f"episode={episode_idx} replan={current_predicted_future_clip['replan_idx']}."
                )
                if pred_len != expected_frame_count:
                    logging.info(
                        "Align predicted clip length to executed steps: "
                        "episode=%s replan=%s done=%s expected=%s pred_full=%s",
                        episode_idx,
                        current_predicted_future_clip["replan_idx"],
                        done,
                        expected_frame_count,
                        pred_len,
                    )
                current_predicted_future_clip["pred_frames"] = current_predicted_future_clip["pred_frames"][
                    :expected_frame_count
                ]
                assert len(current_predicted_future_clip["gt_frames"]) == len(
                    current_predicted_future_clip["pred_frames"]
                ), (
                    "GT/pred frame count mismatch after alignment: "
                    f"len(gt_frames)={len(current_predicted_future_clip['gt_frames'])} "
                    f"len(pred_frames)={len(current_predicted_future_clip['pred_frames'])} "
                    f"episode={episode_idx} replan={current_predicted_future_clip['replan_idx']}."
                )
                clip_psnr = _compute_clip_mean_psnr(
                    current_predicted_future_clip["gt_frames"],
                    current_predicted_future_clip["pred_frames"],
                )
                if clip_psnr is not None:
                    episode_future_clip_psnr.append(clip_psnr)
                predicted_future_video_clips.append(current_predicted_future_clip)
                current_predicted_future_clip = None
        if done:
            break
        t += 1
    pbar.close()

    episode_mean_psnr = (
        float(np.mean(episode_future_clip_psnr)) if len(episode_future_clip_psnr) > 0 else None
    )
    return bool(done), replay_images, predicted_future_video_clips, episode_mean_psnr


def run_single_task(
    task,
    initial_states,
    model: torch.nn.Module,
    processor: FlexPiProcessor,
    cfg: DictConfig,
    video_dir: Path,
    predicted_video_dir: Path,
    *,
    action_horizon: int,
    input_w: int,
    input_h: int,
    model_device: str,
) -> dict:
    env, task_description = get_libero_env(
        task, LIBERO_ENV_RESOLUTION, cfg.get("seed"),
        camera_depths=_is_3d_eval(cfg),
        prompt_source=str(cfg.EVALUATION.get("prompt_source", "task.language")),
    )
    visualize_future_video = bool(cfg.EVALUATION.get("visualize_future_video", False))
    results = {
        "successes": 0,
        "failure_episodes": [],
        "success_episodes": [],
        "task_description": task_description,
    }
    if visualize_future_video:
        results["episode_future_video_psnr"] = []
        results["future_video_psnr_mean"] = None

    for trial_idx in range(int(cfg.EVALUATION.num_trials)):
        success, replay_images, predicted_future_video_clips, episode_mean_psnr = run_single_episode(
            env=env,
            initial_state=initial_states[trial_idx],
            task_description=task_description,
            model=model,
            processor=processor,
            cfg=cfg,
            episode_idx=trial_idx,
            action_horizon=action_horizon,
            input_w=input_w,
            input_h=input_h,
            model_device=model_device,
        )
        if success:
            results["successes"] += 1
            results["success_episodes"].append(trial_idx)
        else:
            results["failure_episodes"].append(trial_idx)
        if visualize_future_video:
            results["episode_future_video_psnr"].append(episode_mean_psnr)

        save_rollout_video(
            video_dir,
            replay_images,
            f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
            success=success,
            task_description=task_description,
        )
        if visualize_future_video:
            if len(predicted_future_video_clips) == 0:
                logging.warning(
                    "No predicted future frames collected for task %s trial %s.",
                    cfg.EVALUATION.task_id,
                    trial_idx,
                )
            else:
                all_gt_frames = []
                all_pred_frames = []
                for clip in predicted_future_video_clips:
                    all_gt_frames.extend(clip["gt_frames"])
                    all_pred_frames.extend(clip["pred_frames"])
                    save_prediction_video(
                        predicted_video_dir,
                        clip["gt_frames"],
                        clip["pred_frames"],
                        f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
                        clip["replan_idx"],
                        success=success,
                        task_description=task_description,
                    )
                save_prediction_video(
                    predicted_video_dir,
                    all_gt_frames,
                    all_pred_frames,
                    f"task{cfg.EVALUATION.task_id}_trial{trial_idx}",
                    "all",
                    success=success,
                    task_description=task_description,
                )

    if visualize_future_video:
        valid_episode_psnr = [x for x in results["episode_future_video_psnr"] if x is not None]
        if len(valid_episode_psnr) > 0:
            results["future_video_psnr_mean"] = float(np.mean(valid_episode_psnr))
    return results


def _find_saved_config_yaml(ckpt_path: Path) -> Optional[Path]:
    """Locate <run_dir>/config.yaml for a checkpoint, walking up parents.

    The conventional layout is `<run_dir>/checkpoints/weights/step_NNN.pt`,
    so the canonical run_dir is parents[2]. We walk a few levels in either
    direction so unconventional layouts (extra nesting, .pt at run_dir root)
    still resolve. Returns the first existing match, or None.
    """
    for parent in list(ckpt_path.parents)[:5]:
        candidate = parent / "config.yaml"
        if candidate.exists():
            return candidate
    return None


def _maybe_swap_eval_cfg_with_saved(cfg: DictConfig) -> DictConfig:
    """Optionally swap cfg.model and cfg.data with the saved phase-1 versions.

    Controlled by `cfg.eval_config_source`:
      - "auto" (default): saved if `<run_dir>/config.yaml` exists, else
        fall back to "current" with a loud warning. Aligns with the user
        intent that "eval should read config from the model's folder, not
        from the current code base", while staying safe for legacy runs
        that predate the saved-config writer.
      - "current": no-op, return cfg as-is. Eval uses the current
        Hydra-composed cfg (current codebase yamls + Hydra CLI overrides).
        Use to explicitly opt out of saved-cfg loading.
      - "saved": load <run_dir>/config.yaml from phase-1 training and swap
        ONLY cfg.model and cfg.data with the saved versions. Eval-side fields
        (cfg.ckpt, cfg.EVALUATION, cfg.gpu_id, cfg.MULTIRUN, cfg.hydra) stay
        from the current launch. A small whitelist of eval-only model fields
        (load_text_encoder, skip_dit_load_from_pretrain,
        action_dit_pretrained_path) also stays from launch — these look
        model-shaped but are runtime knobs whose train value would crash
        eval (T5 not loaded, ckpt overwritten by phase-1 path, etc).
        Phase-1 reproduction of the rest of model + data pipeline;
        CLI overrides on model.* / data.* are otherwise ignored. Hard-fails
        if the saved cfg is missing (use "auto" for graceful fallback).
      - "saved_with_overrides": same as "saved", then apply current Hydra CLI
        leaf overrides on top. Use when you want phase-1 model+data baseline
        plus targeted tweaks. Group overrides (task=..., model=..., data=...)
        are skipped — they don't apply as leaf merges.

    The run_dir is inferred from `cfg.ckpt`: a .pt file like
    `runs/.../checkpoints/weights/step_NNNNNN.pt` has the run_dir three
    parents up; `_find_saved_config_yaml` walks a few extra levels for
    unconventional layouts.
    """
    source = str(getattr(cfg, "eval_config_source", "auto")).strip().lower()
    if source == "current":
        return cfg
    if source not in ("auto", "saved", "saved_with_overrides"):
        raise ValueError(
            "eval_config_source must be one of "
            "['auto', 'current', 'saved', 'saved_with_overrides']; "
            f"got {source!r}"
        )
    if not cfg.ckpt:
        raise ValueError(
            f"eval_config_source={source!r} requires a non-empty cfg.ckpt"
        )

    ckpt_path = Path(str(cfg.ckpt))
    saved_path = _find_saved_config_yaml(ckpt_path)
    if saved_path is None:
        if source == "auto":
            logging.warning(
                "eval_config_source=auto: no saved config.yaml found near "
                "cfg.ckpt=%r (walked 5 parents). Falling back to current "
                "Hydra-composed cfg. To require saved cfg, set "
                "eval_config_source=saved explicitly.", str(cfg.ckpt),
            )
            OmegaConf.set_struct(cfg, False)
            cfg.eval_config_source = "current"
            return cfg
        raise FileNotFoundError(
            f"eval_config_source={source!r} requires a saved phase-1 "
            f"config.yaml in one of cfg.ckpt's parent directories "
            f"(searched 5 levels up from {cfg.ckpt!r}). "
            "Use eval_config_source=auto to fall back to current cfg, "
            "or eval_config_source=current to skip saved-cfg loading."
        )

    saved_cfg = OmegaConf.load(saved_path)
    OmegaConf.set_struct(cfg, False)
    OmegaConf.set_struct(saved_cfg, False)

    # Eval-only model fields that look like model knobs but are really
    # runtime-only: their training value (where T5 context is precomputed
    # and the DiT comes from a phase-1 ckpt) is wrong at eval time
    # (where prompts come from LIBERO live and we want our own ckpt). Without
    # this, infer_action → encode_prompt crashes with
    # "Prompt encoding requires loaded text encoder/tokenizer".
    _EVAL_ONLY_MODEL_FIELDS = (
        "load_text_encoder",
        "skip_dit_load_from_pretrain",
        "action_dit_pretrained_path",
    )
    launch_eval_only = {
        k: cfg.model[k]
        for k in _EVAL_ONLY_MODEL_FIELDS
        if "model" in cfg and k in cfg.model
    }

    # Swap ONLY model and data subtrees from saved. Eval-only keys (ckpt,
    # EVALUATION, gpu_id, MULTIRUN, hydra) stay from the current launch — the
    # saved cfg doesn't even have them. The eval-only model fields captured
    # above are restored after the swap.
    if "model" in saved_cfg:
        cfg.model = saved_cfg.model
    else:
        raise ValueError(
            f"saved cfg at {saved_path} has no 'model' subtree — cannot use "
            f"eval_config_source={source!r}."
        )
    for k, v in launch_eval_only.items():
        cfg.model[k] = v
    if "data" in saved_cfg:
        cfg.data = saved_cfg.data
    else:
        raise ValueError(
            f"saved cfg at {saved_path} has no 'data' subtree — cannot use "
            f"eval_config_source={source!r}."
        )

    # Persist the meta-flag so later readers see consistent state.
    cfg.eval_config_source = source

    if source == "saved_with_overrides":
        try:
            from hydra.core.hydra_config import HydraConfig
            cli_overrides = list(HydraConfig.get().overrides.task)
        except Exception as exc:
            logging.warning(
                "saved_with_overrides: could not access HydraConfig overrides "
                "(%s); falling back to plain saved-cfg load.", exc,
            )
            cli_overrides = []
        _SKIP_LEAF = ("eval_config_source",)
        _SKIP_GROUP = ("data", "model", "task")
        kept = []
        for ov in cli_overrides:
            stripped = ov.lstrip("+")
            if "=" not in stripped:
                continue
            key, _ = stripped.split("=", 1)
            key = key.strip()
            if key in _SKIP_LEAF:
                continue
            if key in _SKIP_GROUP:
                logging.warning(
                    "saved_with_overrides: skipping group override %r "
                    "(group overrides only apply at compose time).", ov,
                )
                continue
            kept.append(stripped)
        if kept:
            try:
                override_cfg = OmegaConf.from_dotlist(kept)
                cfg = OmegaConf.merge(cfg, override_cfg)
                logging.info(
                    "saved_with_overrides: applied %d override(s) on top of "
                    "saved cfg: %s", len(kept), kept,
                )
            except Exception as exc:
                raise ValueError(
                    f"saved_with_overrides: failed to apply overrides {kept!r}: {exc}"
                ) from exc

    logging.info(
        "Eval cfg source: %s. Loaded saved model+data from %s. "
        "Kept current ckpt=%s, EVALUATION, gpu_id, MULTIRUN%s.",
        source, saved_path, cfg.ckpt,
        (f", model.{{{','.join(launch_eval_only)}}}" if launch_eval_only else ""),
    )
    return cfg


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_libero.yaml")
def eval_single_process(cfg: DictConfig):
    start_time = time.time()
    partial_state = PartialState()
    partial_state.config = cfg

    if cfg.get("seed") is not None:
        set_global_seed(int(cfg.seed), get_worker_init_fn=False)

    if cfg.ckpt is None:
        raise ValueError("cfg.ckpt must not be None.")
    cfg = _maybe_swap_eval_cfg_with_saved(cfg)
    partial_state.config = cfg
    _validate_visualize_future_video_cfg(cfg)

    env_num = int(cfg.EVALUATION.get("env_num", 1))
    if env_num != 1:
        raise ValueError(
            "Only env_num=1 is supported in eval_libero_single.py. "
            "Use scripts/eval_flexpi_libero_4suite.sh for multi-GPU task parallelism."
        )

    model_device = _resolve_eval_device(cfg)
    model_dtype = _mixed_precision_to_model_dtype(cfg.get("mixed_precision", "bf16"))
    model = instantiate(cfg.model, model_dtype=model_dtype, device=model_device)
    _load_model_checkpoint(model, str(cfg.ckpt))
    model = model.to(model_device).eval()

    if hasattr(model, "prepare_for_inference"):
        _torch_compile = bool(cfg.EVALUATION.get("torch_compile", False))
        _torch_compile_mode = str(cfg.EVALUATION.get("torch_compile_mode", "reduce-overhead"))
        _quantization_raw = cfg.EVALUATION.get("quantization", None)
        _quantization = None if _quantization_raw is None or str(_quantization_raw).strip().lower() in ("", "none", "null") else str(_quantization_raw)
        model.prepare_for_inference(
            torch_compile=_torch_compile,
            torch_compile_mode=_torch_compile_mode,
            quantization=_quantization,
        )

    dataset_stats_path = _resolve_dataset_stats_path(cfg)
    dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
    processor: FlexPiProcessor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)
    logging.info("Using dataset stats: %s", dataset_stats_path)

    # Padded action-head channels (libero.yaml scatters 7 -> 32) stay at exactly
    # 0 through denoising iff training zeroed them too: a checkpoint deploys the
    # way it trained. The released LIBERO checkpoints predate the mask and run
    # unarmed; one trained on this tree carries the payload flag. Armed before
    # the first infer_action so a compiled graph captures the mask it serves with.
    # EVALUATION.action_dim_mask: null decides from the checkpoint, true/false forces.
    action_dim_mask = cfg.EVALUATION.get("action_dim_mask", None)
    pad, reason = resolve_action_dim_mask(processor, model, cfg, action_dim_mask)
    model.set_action_dim_mask(pad)
    if pad is not None:
        logging.info("action_dim_mask: channels %s zero-filled", torch.nonzero(pad).flatten().tolist())
    else:
        logging.info("action_dim_mask: not armed (%s)", reason)

    # FlexPi3D: load camera intrinsics once and cache on the model. Depth
    # comes from the LIBERO sim per step (camera_depths=True). The model's
    # _resolve_camera_intrinsics() picks up this cached K when no explicit K
    # is passed to infer_action.
    if _is_3d_eval(cfg):
        K = _load_eval_camera_intrinsics(cfg)
        model.set_camera_intrinsics(K.to(model_device))
        logging.info("FlexPi3D eval: cached camera intrinsics K=%s from %s",
                     tuple(K.shape), cfg.EVALUATION.get("camera_intrinsics_path"))

    action_horizon_cfg = cfg.EVALUATION.get("action_horizon", None)
    if action_horizon_cfg is None:
        action_horizon = int(cfg.data.train.num_frames) - 1
    else:
        action_horizon = int(action_horizon_cfg)
    if action_horizon <= 0:
        raise ValueError(f"EVALUATION.action_horizon must be positive, got {action_horizon}")

    video_size = cfg.data.train.get("video_size", [224, 224])
    if len(video_size) != 2:
        raise ValueError(f"data.train.video_size must be [H, W], got {video_size}")
    input_h = int(video_size[0])
    input_w = int(video_size[1])
    concat_multi_camera = cfg.data.train.get("concat_multi_camera", None)
    shape_meta_images = [meta["shape"] for meta in processor.shape_meta["images"]]

    local_log_dir = Path(cfg.EVALUATION.output_dir)
    local_log_dir.mkdir(parents=True, exist_ok=True)
    video_dir = local_log_dir / cfg.EVALUATION.task_suite_name / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    predicted_video_dir = local_log_dir / cfg.EVALUATION.task_suite_name / "predicted_videos"
    if bool(cfg.EVALUATION.get("visualize_future_video", False)):
        predicted_video_dir.mkdir(parents=True, exist_ok=True)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.EVALUATION.task_suite_name]()
    task = task_suite.get_task(cfg.EVALUATION.task_id)
    initial_states = task_suite.get_task_init_states(cfg.EVALUATION.task_id)
    initial_states = list(initial_states)  # ensure list for .extend() compatibility

    while len(initial_states) < int(cfg.EVALUATION.num_trials):
        initial_states.extend(initial_states[: (int(cfg.EVALUATION.num_trials) - len(initial_states))])

    results = {
        "task_suite": cfg.EVALUATION.task_suite_name,
        "task_id": cfg.EVALUATION.task_id,
        "task_description": None,
        "successes": 0,
        "total_episodes": int(cfg.EVALUATION.num_trials),
        "gpu_id": int(cfg.gpu_id),
        "success_episodes": [],
        "failure_episodes": [],
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration": 0,
    }

    logging.info("Running LIBERO evaluation with env_num=1")
    task_results = run_single_task(
        task=task,
        initial_states=initial_states,
        model=model,
        processor=processor,
        cfg=cfg,
        video_dir=video_dir,
        predicted_video_dir=predicted_video_dir,
        action_horizon=action_horizon,
        input_w=input_w,
        input_h=input_h,
        model_device=model_device,
    )
    results.update(task_results)

    results["duration"] = time.time() - start_time
    output_dir = Path(cfg.EVALUATION.output_dir) / cfg.EVALUATION.task_suite_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"gpu{cfg.gpu_id}_task{cfg.EVALUATION.task_id}_results.json"

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)

    print(
        f"Task {cfg.EVALUATION.task_id} completed: "
        f"{results['successes']}/{cfg.EVALUATION.num_trials} successes"
    )
    if results.get("future_video_psnr_mean") is not None:
        print(f"Task {cfg.EVALUATION.task_id} future-video PSNR mean: {results['future_video_psnr_mean']:.4f}")
    print(f"Time taken: {results['duration']:.2f} seconds")
    return results


if __name__ == "__main__":
    eval_single_process()
