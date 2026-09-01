"""YAM real-world FlexPi deployment policy.

Sister of ``experiments/robotwin/flexpi_policy/deploy_policy.py``.

This file targets the YAM real-world FlexPiUnifiedJoint model trained by
``scripts/train_flexpi_yam.sh``. Differences vs. the RoboTwin deploy:

- Camera keys in observations are ``cam_high / cam_left_wrist / cam_right_wrist``
  (canonical RoboTwin-rename layout — same names the training dataset on
  disk uses). The RoboTwin deploy maps those to its sim's
  ``head_camera / left_camera / right_camera``; here we pass through.
- State + action are 32D in ``yam_eef.STATE_LAYOUT``
  ``[L_pos, L_rot6d, R_pos, R_rot6d, L_grip, R_grip, L_joint(6), R_joint(6)]``.
  Identical to yam_openpi's ``eef32``.
- Training applies ``Yam32DRelativeAction(anchor="first")`` to action targets
  (state stays absolute). At inference the model emits 32D **relative** actions
  in that same layout; we invert the transform using the CURRENT 32D state as
  the anchor before producing absolute joint commands. This is the YAM analog
  of yam_openpi's server-side ``RelativeEEFActions.backward`` in
  ``yam_policy.py``.
- Camera intrinsics K come from ``meta/camera_intrinsics.json`` shipped with
  the LeRobot dataset (or from a calibration file at real-robot time). The
  RoboTwin deploy sources K live from sim every step instead.

Intentionally not implemented here:
- Real-robot adapter (FK from raiden's 7-DoF joint positions → 32D state,
  metres→mm depth cast, ZED capture). Left to a future bridge layer
  analogous to ``yam_openpi/deployment/openpi_bridge.py``.
- ``step(task_env, observation)`` loop coupling. Smoke test calls
  ``infer_action_chunk(obs, instruction)`` directly.
"""

from __future__ import annotations

import inspect
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

# Path bootstrap so ``python -m experiments.yam.flexpi_policy.smoke_test``
# (or direct execution) finds ``flexpi.*`` without relying on an installed
# editable wheel. Mirrors RoboTwin deploy.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from flexpi.datasets.lerobot.processors.flexpi_processor import FlexPiProcessor  # noqa: E402
from flexpi.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT  # noqa: E402
from flexpi.datasets.lerobot.robot_video_dataset import _PER_CAM_HW  # noqa: E402
from flexpi.datasets.lerobot.utils.normalizer import load_dataset_stats_from_json  # noqa: E402
from flexpi.per_cam_compose import compose_robotwin_from_per_cam  # noqa: E402

logger = logging.getLogger(__name__)


# Canonical YAM camera order. Matches the dataset YAML's shape_meta.images
# entries and the RGB ``_PER_CAM_HW`` table imported above.
_CAM_ORDER: Tuple[str, str, str] = ("cam_high", "cam_left_wrist", "cam_right_wrist")

# Runtime knob defaults live in configs/real_yam.yaml so the deploy, the smoke
# test and the Raiden bridge cannot drift apart. Model and processor still come
# from the checkpoint's saved config -- see the header of real_yam.yaml.
REAL_YAM_CFG = PROJECT_ROOT / "configs" / "real_yam.yaml"


def load_deploy_defaults(path: Optional[Path] = None) -> Dict[str, Any]:
    """Read the EVALUATION block of ``configs/real_yam.yaml``."""
    cfg_path = Path(path) if path is not None else REAL_YAM_CFG
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"Deploy defaults not found: {cfg_path}. It supplies the runtime "
            "knobs for YAM deploy (num_inference_steps, torch_compile, ...)."
        )
    cfg = OmegaConf.load(cfg_path)
    return OmegaConf.to_container(cfg.EVALUATION, resolve=True)


# Read once at import and used as the literal signature defaults below. This is
# what makes precedence come out right with no sentinel: an argument the caller
# actually passed shadows the default by ordinary Python rules, so
# `--bridge-kwargs num_inference_steps=4` still beats the file. Resolving the
# config INSIDE the body instead would invert that and let the file override
# the operator.
_D: Dict[str, Any] = load_deploy_defaults()


def _mixed_precision_to_dtype(mixed_precision: str) -> torch.dtype:
    key = str(mixed_precision).strip().lower()
    if key == "fp16":
        return torch.float16
    if key == "no":
        return torch.float32
    if key == "bf16":
        return torch.bfloat16
    raise ValueError(f"Unsupported mixed_precision={mixed_precision!r}; expected one of no/fp16/bf16")


def _find_trained_config(ckpt_path: Path) -> Optional[Path]:
    """Walk up to 5 parents from the checkpoint looking for ``config.yaml``.

    Training writes ``<run_dir>/config.yaml`` and the checkpoint usually lives
    at ``<run_dir>/checkpoints/.../step_XXXX.pt`` (or directly under
    ``<run_dir>/state/step_XXXXXX/``). The shallow walk covers both layouts.
    """
    for parent in list(ckpt_path.resolve().parents)[:5]:
        candidate = parent / "config.yaml"
        if candidate.is_file():
            return candidate
    return None


def _find_dataset_stats(ckpt_path: Path, override: Optional[Path]) -> Path:
    """Resolve ``dataset_stats.json``.

    Priority:
    1. Explicit override (CLI flag).
    2. Walk up from ``ckpt_path`` looking for ``dataset_stats.json`` next to
       a sibling ``config.yaml`` (training writes both into the run dir).
    """
    if override is not None:
        path = Path(override).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"--stats path does not exist: {path}")
        return path
    for parent in list(ckpt_path.resolve().parents)[:5]:
        candidate = parent / "dataset_stats.json"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Could not auto-locate dataset_stats.json from the checkpoint path. "
        "Pass --stats <path/to/dataset_stats.json> explicitly."
    )


def _load_intrinsics_for_yam(json_path: Path) -> torch.Tensor:
    """Load per-camera intrinsics rescaled to the depth grid the model trained on.

    YAM training (``yam.yaml`` shape_meta.depth) uses head 256×320,
    wrists 224×224 — the same per-cam HW table as ``_PER_CAM_HW``. The pointmap
    encoder rescales internally if depth and K disagree, but training and deploy
    agree on this grid, so the rescale here is a numerical no-op for that case.

    Returns: ``[3, 3, 3]`` float32 tensor in ``_CAM_ORDER``.
    """
    if not json_path.exists():
        raise FileNotFoundError(f"camera_intrinsics.json not found: {json_path}")
    with open(json_path, "r") as f:
        raw = json.load(f)
    Ks = []
    for cam_name, (h_cam, w_cam) in _PER_CAM_HW:
        if cam_name not in raw:
            raise KeyError(
                f"{json_path} missing camera '{cam_name}' (found: {list(raw)})"
            )
        entry = raw[cam_name]
        sx = float(w_cam) / float(entry["width"])
        sy = float(h_cam) / float(entry["height"])
        K = torch.tensor(
            [
                [float(entry["fx"]) * sx, 0.0,                       float(entry["cx"]) * sx],
                [0.0,                     float(entry["fy"]) * sy,   float(entry["cy"]) * sy],
                [0.0,                     0.0,                       1.0],
            ],
            dtype=torch.float32,
        )
        Ks.append(K)
    return torch.stack(Ks, dim=0)


def _resize_rgb_antialiased(rgb_uint8: np.ndarray, hw: Tuple[int, int]) -> torch.Tensor:
    """Bilinear antialiased resize matching the dataset's torchvision call.

    Same helper RoboTwin deploy uses — single source of truth so train/deploy
    don't drift on the resampling kernel.
    """
    import torchvision.transforms.functional as TF
    t = torch.from_numpy(rgb_uint8.astype(np.uint8)).permute(2, 0, 1).float() / 255.0
    return TF.resize(
        t, size=list(hw),
        interpolation=TF.InterpolationMode.BILINEAR, antialias=True,
    )


class YamFlexPiPolicy:
    """FlexPi deployment policy for the YAM real-world 32D-rel model.

    Construction mirrors ``WorldActionRobotWinPolicy``: instantiate model from
    ``cfg.model``, load checkpoint via ``model.load_checkpoint``, instantiate
    the FlexPiProcessor and pre-load its normalizer from ``dataset_stats.json``.

    Observations consumed by ``infer_action_chunk`` are dicts with these keys
    (offline-friendly contract; the real-robot bridge converts into this same
    shape):

      - ``rgb`` : ``{cam_name: np.uint8 [H_native, W_native, 3]}``  (RGB order)
      - ``depth``: ``{cam_name: np.uint16 [H, W]}`` in millimetres, optional.
      - ``state_32``: ``np.float32 [32]`` per ``yam_eef.STATE_LAYOUT``.
      - ``intrinsics``: ``np.float32 [3, 3, 3]`` per-cam K at the depth grid
        in ``_CAM_ORDER``, optional. When None the policy uses the K cached at
        construction time (from ``meta/camera_intrinsics.json``).

    ``infer_action_chunk`` returns a ``np.float32 [T_act, 32]`` numpy array of
    **absolute** 32D actions (Yam32DRelativeAction inverted using ``state_32``
    as the anchor).
    """

    def __init__(
        self,
        model_cfg: DictConfig,
        processor_cfg: DictConfig,
        checkpoint_path: str,
        dataset_stats_path: Path,
        device: str,
        model_dtype: torch.dtype,
        intrinsics_K: Optional[torch.Tensor],
        action_horizon: int,
        num_inference_steps: int,
        num_video_frames: int,
        sigma_shift: Optional[float] = None,
        seed: Optional[int] = None,
        text_cfg_scale: float = 1.0,
        negative_prompt: str = "",
        rand_device: str = "cpu",
        tiled: bool = False,
        offload_text_encoder: bool = True,
        torch_compile: bool = True,
        torch_compile_mode: str = "reduce-overhead",
        torch_compile_scope: str = "loop",
        quantization: Optional[str] = None,
        attn_backend: str = "auto",
        trt_joint_free_video_blocks: bool = False,
        trt_joint_prefill_split_engine_path: Optional[str] = None,
        trt_joint_decode_split_engine_path: Optional[str] = None,
        glue_cache: bool = False,
        encoder_cuda_graph: bool = False,
        compile_encoders: bool = False,
        dynamic_step_skip: bool = False,
        # Flex regime for the boot warmup (keys: joint_*/present_*). The
        # warmup compiles whatever regime it runs — pass the regime this boot
        # will actually serve so the compile cost is paid at startup, not on
        # the first robot frame. None = trained-config default.
        warmup_flex: Optional[Dict[str, bool]] = None,
    ) -> None:
        # Text encoder load: usually live (default). offload_text_encoder=True
        # keeps T5 on CPU end-to-end (saves ~10 GB VRAM) — mirrors RoboTwin
        # deploy. Either way the text encoder is needed; YAM training doesn't
        # rely on the offline embed cache at infer time, so the live encoder
        # encodes prompts on the fly.
        model_cfg_copy = OmegaConf.create(OmegaConf.to_container(model_cfg, resolve=True))
        if offload_text_encoder:
            model_cfg_copy.load_text_encoder = False
        else:
            model_cfg_copy.load_text_encoder = True

        self.model = instantiate(model_cfg_copy, model_dtype=model_dtype, device=device)
        self.model.load_checkpoint(checkpoint_path)
        if offload_text_encoder:
            # Load T5 to CPU separately (mirrors RoboTwin deploy_policy.py).
            from flexpi.models.helpers.loader import load_text_encoder_to_device
            te, tok = load_text_encoder_to_device(
                device="cpu",
                torch_dtype=model_dtype,
                model_id=model_cfg_copy.get("model_id", "Wan-AI/Wan2.2-TI2V-5B"),
                tokenizer_model_id=model_cfg_copy.get("tokenizer_model_id", "Wan-AI/Wan2.1-T2V-1.3B"),
                tokenizer_max_len=int(model_cfg_copy.get("tokenizer_max_len", 512)),
            )
            self.model.text_encoder = te
            self.model.tokenizer = tok
            self.model.offload_text_encoder = True
            print("[YAM-deploy] Text encoder loaded on CPU (saves ~10GB VRAM).")
        self.model = self.model.to(device).eval()
        if hasattr(self.model, "prepare_for_inference"):
            self.model.prepare_for_inference(
                torch_compile=bool(torch_compile),
                torch_compile_mode=str(torch_compile_mode),
                quantization=quantization,
                torch_compile_scope=torch_compile_scope,
                attn_backend=attn_backend,
                trt_joint_free_video_blocks=trt_joint_free_video_blocks,
                trt_joint_prefill_split_engine_path=trt_joint_prefill_split_engine_path,
                trt_joint_decode_split_engine_path=trt_joint_decode_split_engine_path,
                glue_cache=glue_cache,
                encoder_cuda_graph=encoder_cuda_graph,
                compile_encoders=compile_encoders,
            )

        self.processor: FlexPiProcessor = instantiate(processor_cfg).eval()
        dataset_stats = load_dataset_stats_from_json(str(dataset_stats_path))
        self.processor.set_normalizer_from_stats(dataset_stats)

        self.action_horizon = int(action_horizon)
        self.num_inference_steps = int(num_inference_steps)
        self.num_video_frames = int(num_video_frames)
        self.sigma_shift = sigma_shift
        self.seed = seed
        self.text_cfg_scale = float(text_cfg_scale)
        self.negative_prompt = str(negative_prompt)
        self.rand_device = str(rand_device)
        self.tiled = bool(tiled)
        self.dynamic_step_skip = bool(dynamic_step_skip)

        if intrinsics_K is not None:
            self._cached_K = intrinsics_K.to(device=self.model.device)
        else:
            self._cached_K = None

        # Cache the resolved infer_action signature once. The unified-joint
        # model's ``infer_action`` takes ``per_cam``, ``per_cam_depth``,
        # ``num_video_frames``, plus the step-skip knobs — we forward whatever
        # the signature declares so this file stays compatible with neighboring
        # FlexPi model variants without changes.
        self._infer_sig = inspect.signature(self.model.infer_action).parameters

        self._cached_prompt: Optional[str] = None
        self._cached_context: Optional[torch.Tensor] = None
        self._cached_context_mask: Optional[torch.Tensor] = None

        logger.info(
            "YamFlexPiPolicy ready | ckpt=%s | stats=%s | horizon=%d | inference_steps=%d",
            checkpoint_path, dataset_stats_path, self.action_horizon, self.num_inference_steps,
        )

        # When torch.compile is on, the first real ``infer_action`` call eats
        # the 10-30 s compile + CUDA-Graph capture cost. Trigger it here on
        # dummy inputs so deploy time is paid up-front rather than on the
        # first robot step. Mirrors RoboTwin deploy_policy._warmup().
        # Same reasoning for the other lazily-initialized acceleration paths:
        # encoder CUDA-graph capture and TRT engine first-call (deserialize +
        # shape guard) plus the glue-cache prime.
        self._warmup_flex: Dict[str, bool] = {
            k: bool(v) for k, v in (warmup_flex or {}).items() if v is not None
        }
        if (
            bool(torch_compile)
            or bool(encoder_cuda_graph)
            or trt_joint_decode_split_engine_path is not None
        ):
            self._warmup()

    def _warmup(self) -> None:
        """Run one dummy ``infer_action`` to trigger torch.compile + CUDA-Graph capture.

        Shapes match what ``infer_action_chunk`` will feed at real inference:
        composite ``input_image`` ``[1, 3, 384, 320]``, per-cam RGB at the YAM
        ``_PER_CAM_HW`` sizes, per-cam depth on the same grids in ``uint16``,
        and a 32D proprio. Identity intrinsics avoid hitting the pointmap
        encoder's K-rescale path during warmup.
        """
        warmup_t0 = time.perf_counter()
        device = self.model.device
        dtype = self.model.torch_dtype

        # Dummy prompt — cleared after warmup so the first real prompt re-encodes.
        dummy_prompt = DEFAULT_PROMPT.format(task="warmup")
        with torch.no_grad():
            dummy_context, dummy_context_mask = self.model.encode_prompt(dummy_prompt)

        proprio_dim = getattr(self.model, "proprio_dim", 32)
        dummy_proprio = torch.zeros((1, int(proprio_dim)), dtype=torch.float32)
        dummy_image = torch.zeros((1, 3, 384, 320), device=device, dtype=dtype)

        warmup_kwargs: Dict[str, Any] = {
            "prompt": None,
            "context": dummy_context,
            "context_mask": dummy_context_mask,
            "input_image": dummy_image,
            "action_horizon": self.action_horizon,
            "proprio": dummy_proprio,
            "negative_prompt": self.negative_prompt,
            "text_cfg_scale": self.text_cfg_scale,
            "num_inference_steps": self.num_inference_steps,
            "sigma_shift": self.sigma_shift,
            "seed": 0,
            "rand_device": self.rand_device,
            "tiled": self.tiled,
        }
        if "num_video_frames" in self._infer_sig:
            warmup_kwargs["num_video_frames"] = int(self.num_video_frames)
        if "return_stream_latents" in self._infer_sig:
            warmup_kwargs["return_stream_latents"] = False
        for _k, _v in self._warmup_flex.items():
            if _k in self._infer_sig:
                warmup_kwargs[_k] = _v
        if "per_cam" in self._infer_sig:
            warmup_kwargs["per_cam"] = {
                cam_name: torch.zeros((1, 3, h, w), device=device, dtype=dtype)
                for cam_name, (h, w) in _PER_CAM_HW
            }
        if "per_cam_depth" in self._infer_sig:
            warmup_kwargs["per_cam_depth"] = {
                cam_name: torch.zeros((1, 1, h, w), device=device, dtype=torch.uint16)
                for cam_name, (h, w) in _PER_CAM_HW
            }
            if hasattr(self.model, "set_camera_intrinsics"):
                K_id = (
                    torch.eye(3, device=device)
                    .unsqueeze(0)
                    .expand(len(_PER_CAM_HW), -1, -1)
                    .contiguous()
                )
                self.model.set_camera_intrinsics(K_id)

        with torch.no_grad():
            self.model.infer_action(**warmup_kwargs)
        if device.type == "cuda":
            torch.cuda.synchronize()

        # Clear the cached context so the first real prompt re-encodes
        # (prompt text differs by task).
        self._cached_prompt = None
        self._cached_context = None
        self._cached_context_mask = None

        print(
            f"[YAM-deploy] torch.compile warmup done in "
            f"{time.perf_counter() - warmup_t0:.2f} s"
        )

    # ------------------------------------------------------------------
    # Observation tensors
    # ------------------------------------------------------------------

    def _build_per_cam_rgb(self, rgb_by_cam: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Per-camera RGB at per-cam target sizes (head 256×320, wrists 224×224).

        Matches RobotVideoDataset → exactly one antialiased bilinear
        resize from native, then linear normalize to ``[-1, 1]``.
        """
        out: Dict[str, torch.Tensor] = {}
        for cam_name, (h_cam, w_cam) in _PER_CAM_HW:
            if cam_name not in rgb_by_cam:
                raise KeyError(
                    f"observation['rgb'] is missing camera '{cam_name}'; "
                    f"got keys={list(rgb_by_cam)}. Expected {_CAM_ORDER}."
                )
            rgb = rgb_by_cam[cam_name]
            if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
                raise ValueError(
                    f"observation['rgb'][{cam_name}] must be uint8 [H, W, 3], "
                    f"got dtype={rgb.dtype} shape={rgb.shape}"
                )
            t = _resize_rgb_antialiased(rgb, (h_cam, w_cam))  # [3, h, w] in [0, 1]
            t = (t * 2.0 - 1.0).unsqueeze(0).to(
                device=self.model.device, dtype=self.model.torch_dtype
            )
            out[cam_name] = t  # [1, 3, h, w]
        return out

    def _build_per_cam_depth(self, depth_by_cam: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Depth tensors at sim-native resolution. Pass-through uint16 mm.

        Same contract RoboTwin deploy uses: ``[1, 1, H, W]`` uint16 on the
        model's device; the encoder does the ``.float()/1000`` cast itself.
        """
        out: Dict[str, torch.Tensor] = {}
        for cam_name in _CAM_ORDER:
            if cam_name not in depth_by_cam:
                raise KeyError(
                    f"observation['depth'] is missing camera '{cam_name}'; "
                    f"got keys={list(depth_by_cam)}. Expected {_CAM_ORDER}."
                )
            depth_np = np.asarray(depth_by_cam[cam_name])
            if depth_np.ndim == 3 and depth_np.shape[0] == 1:
                depth_np = depth_np[0]
            if depth_np.ndim != 2:
                raise ValueError(
                    f"observation['depth'][{cam_name}] must be [H, W] or [1, H, W]; "
                    f"got shape {depth_np.shape}"
                )
            depth = torch.from_numpy(depth_np).contiguous()
            if depth.dtype != torch.uint16:
                depth = depth.to(torch.int32).clamp_(0, 65535).to(torch.uint16)
            out[cam_name] = depth.unsqueeze(0).unsqueeze(0).to(device=self.model.device)
        return out

    def _compose_input_image(self, per_cam: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Derive the [1, 3, 384, 320] composite from the per-cam tensors.

        Same helper RobotVideoDataset uses internally — single
        compose pipeline shared by train and deploy.
        """
        per_cam_5d = {k: v.unsqueeze(2) for k, v in per_cam.items()}  # [B, 3, T=1, H, W]
        composite = compose_robotwin_from_per_cam(per_cam_5d)  # [1, 3, 1, 384, 320]
        return composite.squeeze(2)  # [1, 3, 384, 320]

    # ------------------------------------------------------------------
    # State + action plumbing
    # ------------------------------------------------------------------

    def _normalize_state(self, state_32: np.ndarray) -> torch.Tensor:
        """Apply the same forward pipeline the dataloader applies to state.

        Note Yam32DRelativeAction leaves state untouched (only action is
        transformed there), so the relevant step is only the normalizer.
        We still route through ``action_state_transform`` to mirror the
        RoboTwin deploy exactly — it would be a no-op for state, but the
        shape-assertions inside catch wrong-dim bugs early.
        """
        if state_32.shape != (32,):
            raise ValueError(f"state_32 must have shape (32,); got {state_32.shape}")
        state_meta = self.processor.shape_meta["state"]
        if len(state_meta) != 1:
            raise ValueError("Expected exactly one state key in shape_meta.")
        state_key = state_meta[0]["key"]

        batch = {"state": {state_key: torch.as_tensor(state_32, dtype=torch.float32).unsqueeze(0)}}
        batch = self.processor.action_state_transform(batch)
        batch = self.processor.normalizer.forward(batch)
        # [1, 32]; the action_dit expects [B, proprio_dim]. Mirrors RoboTwin path.
        return batch["state"][state_key]

    def _denormalize_action(
        self,
        action_normalized: torch.Tensor,
        anchor_state_32: np.ndarray,
    ) -> np.ndarray:
        """Invert normalize + Yam32DRelativeAction.

        Steps:
          1. ``normalizer.backward`` on the per-key action  → relative action
             in (mostly) original units. For YAM this is the 32D layout with
             EEF blocks in body-frame relative form and the joint block in
             scalar-delta form.
          2. Apply each ``action_state_transforms`` in *reverse* order with
             ``backward=True``. Yam32DRelativeAction reads ``state[..., 0, :]``
             as the anchor — we feed ``anchor_state_32`` (T_obs=1) as that
             anchor, mirroring yam_openpi's server-side
             ``RelativeEEFActions.backward`` which uses ``data["state"]``.

        Returns: ``np.float32 [T_act, 32]`` absolute action chunk.
        """
        if action_normalized.ndim == 2:
            action_normalized = action_normalized.unsqueeze(0)
        if action_normalized.ndim != 3:
            raise ValueError(f"Expected action [B, T, D]; got {tuple(action_normalized.shape)}")

        action_meta = self.processor.shape_meta["action"]
        if len(action_meta) != 1:
            raise ValueError("Expected exactly one action key in shape_meta.")
        action_key = action_meta[0]["key"]

        # Step 1: per-key denormalize. action_state_merger is identity scatter
        # (ScatterToChannels(total_dim=32)) for YAM, so we don't need to call
        # action_state_merger.backward — the model already emits 32D in the
        # final scatter layout.
        normalizer = self.processor.normalizer.normalizers["action"][action_key]
        denorm = normalizer.backward(action_normalized.to(dtype=torch.float32, device="cpu"))

        # Step 2: invert action_state_transforms in reverse order. For YAM the
        # only transform is Yam32DRelativeAction; in the general case we walk
        # the list to stay generic. Anchor state is the CURRENT obs (T_obs=1).
        transforms = self.processor.action_state_transforms or []
        if len(transforms) > 0:
            state_meta = self.processor.shape_meta["state"]
            state_key = state_meta[0]["key"]
            anchor = torch.as_tensor(anchor_state_32, dtype=torch.float32)  # [32]
            # [B=1, T_obs=1, 32] — Yam32DRelativeAction reads state[..., 0, :].
            anchor_state = anchor.unsqueeze(0).unsqueeze(0)
            batch = {
                "action": {action_key: denorm},                # [B=1, T_act, 32]
                "state":  {state_key:  anchor_state},          # [B=1, T_obs=1, 32]
            }
            for trans in reversed(transforms):
                batch = trans.backward(batch)
            denorm = batch["action"][action_key]

        return denorm.squeeze(0).numpy()  # [T_act, 32]

    # ------------------------------------------------------------------
    # Inference entrypoint
    # ------------------------------------------------------------------

    def infer_action_chunk(
        self,
        observation: Dict[str, Any],
        instruction: str,
        *,
        # Flex runtime overrides (forwarded only if the model's infer_action
        # accepts them — sig-gated below). None = no override → model uses
        # trained default.
        joint_video: Optional[bool] = None,
        joint_dino: Optional[bool] = None,
        joint_pointmap: Optional[bool] = None,
        present_video: Optional[bool] = None,
        present_dino: Optional[bool] = None,
        present_pointmap: Optional[bool] = None,
        verbose: bool = False,
        # When True, return a dict with the full model output (action +
        # video/dino/pointmap latents from the joint-denoise path) instead
        # of just the absolute action ndarray. Caller is responsible for
        # forcing the joint regime (joint_video/dino/pointmap=True) — if
        # not, the model may still return action-only and the latent keys
        # will be missing.
        return_latents: bool = False,
    ):
        """Run a single forward pass and return ``[T_act, 32]`` absolute actions.

        ``observation`` keys are documented on the class docstring. Missing
        ``depth`` is supported only if the model's ``infer_action`` signature
        does NOT declare ``per_cam_depth`` — for the YAM unified-joint model it does,
        so depth is required.

        When ``return_latents=True`` the return shape changes from
        ``np.ndarray`` to a dict with keys ``action_abs`` (the usual ndarray)
        plus ``video_latents`` / ``dino_latents`` / ``pointmap_latents``
        (raw torch tensors as the model returned them, or None when the
        regime didn't produce them).
        """
        # ---- RGB & composite ----
        per_cam_rgb = self._build_per_cam_rgb(observation["rgb"])
        input_image = self._compose_input_image(per_cam_rgb)

        # ---- Depth & intrinsics (required for the YAM unified-joint model) ----
        per_cam_depth: Optional[Dict[str, torch.Tensor]] = None
        K_tensor: Optional[torch.Tensor] = None
        if "per_cam_depth" in self._infer_sig:
            if "depth" not in observation or observation["depth"] is None:
                raise ValueError(
                    "Model signature declares `per_cam_depth` but observation['depth'] is missing. "
                    "The YAM unified-joint model requires per-camera depth at deploy."
                )
            per_cam_depth = self._build_per_cam_depth(observation["depth"])
            K = observation.get("intrinsics")
            if K is not None:
                K_tensor = torch.as_tensor(K, dtype=torch.float32, device=self.model.device)
            elif self._cached_K is not None:
                K_tensor = self._cached_K
            else:
                raise ValueError(
                    "per_cam_depth provided but no camera intrinsics available — "
                    "supply observation['intrinsics'] [3,3,3] or pass --intrinsics-json at construction."
                )
            self.model.set_camera_intrinsics(K_tensor)

        # ---- Proprio ----
        state_32 = np.asarray(observation["state_32"], dtype=np.float32)
        proprio = self._normalize_state(state_32)

        # ---- Prompt ----
        prompt = DEFAULT_PROMPT.format(task=instruction)
        if self._cached_prompt != prompt:
            self._cached_context, self._cached_context_mask = self.model.encode_prompt(prompt)
            self._cached_prompt = prompt

        # ---- Build kwargs, forwarding only those the signature accepts ----
        infer_kwargs: Dict[str, Any] = {
            "prompt": None,
            "context": self._cached_context,
            "context_mask": self._cached_context_mask,
            "input_image": input_image,
            "action_horizon": self.action_horizon,
            "proprio": proprio,
            "negative_prompt": self.negative_prompt,
            "text_cfg_scale": self.text_cfg_scale,
            "num_inference_steps": self.num_inference_steps,
            "sigma_shift": self.sigma_shift,
            "seed": self.seed,
            "rand_device": self.rand_device,
            "tiled": self.tiled,
        }
        if "num_video_frames" in self._infer_sig:
            infer_kwargs["num_video_frames"] = int(self.num_video_frames)
        if "return_stream_latents" in self._infer_sig:
            # The bridge reads only out["action"]; each stream latent costs a
            # blocking pageable D2H copy (~25 ms/call at full joint
            # generation). Only the recorder path (return_latents=True) needs
            # them. Mirrors experiments/robotwin/flexpi_policy/deploy_policy.py.
            infer_kwargs["return_stream_latents"] = bool(return_latents)
        # Models trained with per-cam RGB (RobotVideoDataset → the YAM
        # unified-joint model falls in this bucket) accept per_cam alongside
        # the composite.
        if "per_cam" in self._infer_sig:
            infer_kwargs["per_cam"] = per_cam_rgb
        if per_cam_depth is not None:
            infer_kwargs["per_cam_depth"] = per_cam_depth
        if "dynamic_step_skip" in self._infer_sig:
            infer_kwargs["dynamic_step_skip"] = self.dynamic_step_skip
        for _k, _v in (
            ("joint_video", joint_video),
            ("joint_dino", joint_dino),
            ("joint_pointmap", joint_pointmap),
            ("present_video", present_video),
            ("present_dino", present_dino),
            ("present_pointmap", present_pointmap),
        ):
            if _v is not None and _k in self._infer_sig:
                infer_kwargs[_k] = bool(_v)

        if verbose:
            print(f"[YAM-deploy] infer_kwargs keys: {sorted(infer_kwargs)}")
            print(
                f"[YAM-deploy] input_image={tuple(input_image.shape)} "
                f"per_cam_high={tuple(per_cam_rgb['cam_high'].shape)} "
                f"per_cam_lw={tuple(per_cam_rgb['cam_left_wrist'].shape)} "
                f"per_cam_rw={tuple(per_cam_rgb['cam_right_wrist'].shape)} "
                f"proprio={tuple(proprio.shape)} "
                f"prompt={prompt!r}"
            )
            if per_cam_depth is not None:
                d = per_cam_depth["cam_high"]
                # torch.min/max are not implemented for uint16 on CUDA; cast
                # to int32 just for the diagnostic print (no effect on the
                # tensor that flows into the model — that stays uint16).
                d_i32 = d.to(torch.int32)
                print(
                    f"[YAM-deploy] per_cam_depth.cam_high={tuple(d.shape)} dtype={d.dtype}  "
                    f"range=[{d_i32.min().item()}, {d_i32.max().item()}]"
                )

        t0 = time.perf_counter()
        with torch.no_grad():
            pred = self.model.infer_action(**infer_kwargs)
        if self.model.device.type == "cuda":
            torch.cuda.synchronize()
        infer_s = time.perf_counter() - t0

        action_norm_rel = pred["action"]  # [T_act, 32] normalized + relative
        action_abs = self._denormalize_action(action_norm_rel, anchor_state_32=state_32)
        # Pure model latency (cuda-synced above) — always printed so the server
        # terminal shows per-chunk inference time without --verbose.
        print(f"[YAM-deploy] model.infer_action wall={infer_s*1000:.1f} ms", flush=True)
        if verbose:
            print(
                f"[YAM-deploy] pred.action shape={tuple(action_norm_rel.shape)}  "
                f"absolute action shape={action_abs.shape}"
            )
        if return_latents:
            return {
                "action_abs": action_abs,
                "video_latents": pred.get("video_latents"),
                "dino_latents": pred.get("dino_latents"),
                "pointmap_latents": pred.get("pointmap_latents"),
                # Present obs used to produce this chunk's prediction;
                # forwarded so the recorder can derive per-modality GT
                # via the model's own encoders (matches training-eval).
                "present_input_image": input_image,
                "present_per_cam": per_cam_rgb,
                "present_per_cam_depth": per_cam_depth,
                "present_camera_intrinsics": K_tensor if per_cam_depth is not None else None,
            }
        return action_abs


# ---------------------------------------------------------------------------
# Factory: build policy from a checkpoint, auto-loading the trained config.
# ---------------------------------------------------------------------------


def build_policy_from_checkpoint(
    checkpoint_path: str,
    *,
    dataset_stats_path: Optional[str] = None,
    intrinsics_json: Optional[str] = None,
    # Defaults come from configs/real_yam.yaml (see _D above). Passing any of
    # them explicitly overrides the file, per normal Python argument rules.
    device: str = _D["device"],
    mixed_precision: str = _D["mixed_precision"],
    action_horizon: Optional[int] = _D["action_horizon"],
    num_inference_steps: int = _D["num_inference_steps"],
    sigma_shift: Optional[float] = _D["sigma_shift"],
    seed: Optional[int] = _D["seed"],
    text_cfg_scale: float = _D["text_cfg_scale"],
    negative_prompt: str = _D["negative_prompt"],
    rand_device: str = _D["rand_device"],
    tiled: bool = _D["tiled"],
    offload_text_encoder: bool = _D["offload_text_encoder"],
    torch_compile: bool = _D["torch_compile"],
    torch_compile_mode: str = _D["torch_compile_mode"],
    quantization: Optional[str] = _D["quantization"],
    dynamic_step_skip: bool = _D["dynamic_step_skip"],
    # Acceleration / engine knobs. Not in real_yam.yaml: serve_yam_flexpi.py
    # passes every one of them explicitly, and the two engine paths are
    # machine-specific -- a published config is the wrong place for them.
    torch_compile_scope: str = "loop",
    attn_backend: str = "auto",
    trt_joint_free_video_blocks: bool = False,
    trt_joint_prefill_split_engine_path: Optional[str] = None,
    trt_joint_decode_split_engine_path: Optional[str] = None,
    glue_cache: bool = False,
    encoder_cuda_graph: bool = False,
    compile_encoders: bool = False,
    warmup_flex: Optional[Dict[str, bool]] = None,
) -> YamFlexPiPolicy:
    """Build a YamFlexPiPolicy by loading the trained ``config.yaml``.

    Mirrors the RoboTwin deploy's behavior: the trained config (saved next to
    the checkpoint by ``runtime.run_training``) is the source of truth for the
    model architecture and processor — eval-side preset configs (sim yamls)
    would risk drifting (wrong joint flags, layer counts, …).

    For YAM this also gives us the right ``action_state_transforms`` block
    (``Yam32DRelativeAction(anchor="first")``) without needing to compose
    a sim yaml.
    """
    ckpt = Path(checkpoint_path).expanduser().resolve()
    if not ckpt.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    trained_cfg_path = _find_trained_config(ckpt)
    if trained_cfg_path is None:
        raise FileNotFoundError(
            "Could not locate config.yaml next to the checkpoint. "
            "FlexPi deploys depend on the training-time config (model + processor) "
            "to avoid eval/train drift."
        )
    print(f"[YAM-deploy] Loaded trained config: {trained_cfg_path}")
    trained_cfg = OmegaConf.load(trained_cfg_path)

    # Neutralize the training-time DiT bootstrap loads. The trained checkpoint
    # is the source of truth for *all* DiT parameters (video + action) — the
    # `from_wan22_pretrained` path's ActionDiT initializer and the WAN DiT
    # pretrained load would just get overwritten by `model.load_checkpoint`.
    # Skipping them (a) avoids ~2 GB of wasted I/O for the ActionDiT init,
    # (b) makes deploy robust to the training-time relative path
    # ``action_dit_pretrained_path: checkpoints/...`` not resolving from the
    # deploy CWD, and (c) does NOT affect VAE / text encoder loads — those
    # come from ``redirect_common_files`` / ``load_text_encoder`` and stay on.
    trained_cfg.model.action_dit_pretrained_path = None
    trained_cfg.model.skip_dit_load_from_pretrain = True
    print(
        "[YAM-deploy] Overriding model.action_dit_pretrained_path=None and "
        "model.skip_dit_load_from_pretrain=True (load_checkpoint provides both)."
    )

    print(
        f"[YAM-deploy] model._target_={trained_cfg.model.get('_target_', '?')}  "
        f"action_dim={trained_cfg.data.train.processor.action_output_dim}  "
        f"proprio_dim={trained_cfg.data.train.processor.proprio_output_dim}"
    )

    # Resolve device + dtype.
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[YAM-deploy] CUDA unavailable; falling back to CPU.")
        device = "cpu"
    model_dtype = _mixed_precision_to_dtype(mixed_precision)

    # Resolve stats path.
    stats_path = _find_dataset_stats(
        ckpt, override=Path(dataset_stats_path) if dataset_stats_path else None
    )
    print(f"[YAM-deploy] dataset_stats.json: {stats_path}")

    # Resolve intrinsics path. Optional at construction — observation can
    # carry K itself, but if neither is provided we'll raise later.
    K = None
    if intrinsics_json is not None:
        K = _load_intrinsics_for_yam(Path(intrinsics_json).expanduser())
        print(
            f"[YAM-deploy] intrinsics_json: {intrinsics_json}  K_high.fx={K[0,0,0].item():.2f} "
            f"K_high.fy={K[0,1,1].item():.2f}"
        )

    # Resolve runtime knobs. Defaults are aligned with sim_robotwin.yaml
    # EVALUATION block (RoboTwin parity).
    num_frames = int(trained_cfg.data.train.num_frames)
    action_video_freq_ratio = int(trained_cfg.data.train.action_video_freq_ratio)
    horizon = action_horizon if action_horizon is not None else (num_frames - 1)
    n_video_frames = (num_frames - 1) // action_video_freq_ratio + 1
    print(
        f"[YAM-deploy] num_frames={num_frames}  action_video_freq_ratio={action_video_freq_ratio}  "
        f"-> action_horizon={horizon}  num_video_frames={n_video_frames}"
    )

    return YamFlexPiPolicy(
        model_cfg=trained_cfg.model,
        processor_cfg=trained_cfg.data.train.processor,
        checkpoint_path=str(ckpt),
        dataset_stats_path=stats_path,
        device=device,
        model_dtype=model_dtype,
        intrinsics_K=K,
        action_horizon=horizon,
        num_inference_steps=num_inference_steps,
        num_video_frames=n_video_frames,
        sigma_shift=sigma_shift,
        seed=seed,
        text_cfg_scale=text_cfg_scale,
        negative_prompt=negative_prompt,
        rand_device=rand_device,
        tiled=tiled,
        offload_text_encoder=offload_text_encoder,
        torch_compile=torch_compile,
        torch_compile_mode=torch_compile_mode,
        torch_compile_scope=torch_compile_scope,
        quantization=quantization,
        attn_backend=attn_backend,
        trt_joint_free_video_blocks=trt_joint_free_video_blocks,
        trt_joint_prefill_split_engine_path=trt_joint_prefill_split_engine_path,
        trt_joint_decode_split_engine_path=trt_joint_decode_split_engine_path,
        glue_cache=glue_cache,
        encoder_cuda_graph=encoder_cuda_graph,
        compile_encoders=compile_encoders,
        dynamic_step_skip=dynamic_step_skip,
        warmup_flex=warmup_flex,
    )
