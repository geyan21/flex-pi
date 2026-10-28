"""Utils for evaluating policies in LIBERO simulation environments."""

import math
import time
import pathlib

import imageio
from PIL import Image, ImageDraw
import numpy as np
from flexpi.utils.libero_setup import prepare_libero

# Pin the LIBERO data dirs to whichever package is on sys.path, and make its
# numpy-pickled init states loadable under torch>=2.6. MUST run before
# `import libero.libero` — see flexpi/utils/libero_setup.py for why.
prepare_libero()

from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv, SubprocVectorEnv
import libero.libero.envs.bddl_utils as BDDLUtils
from flexpi.utils.video_io import save_mp4

DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
LIBERO_ENV_RESOLUTION = 256  # eval render size; the training corpus is 512x512 (configs/data/libero.yaml raw_shape)


def _resolve_bddl_for_language(task_bddl_file):
    """Resolve the actual BDDL file path for reading the :language field.

    LIBERO-Plus tasks with _view_..._initstate_... suffixes (Camera, Robot Init,
    Sensor Noise categories) have NO separate BDDL file — they reuse the base BDDL.
    The env_wrapper.py strips the suffix at runtime, but we need the base path to
    read the :language field before env creation.
    See: https://github.com/sylvestf/LIBERO-plus/issues/48
    """
    s = str(task_bddl_file)
    if "_view_" in s and "_initstate_" in s:
        base = s.split("_view_")[0] + ".bddl"
        return base
    return s


def get_libero_env(task, resolution, seed, env_num=1, camera_depths=False,
                   prompt_source: str = "task.language"):
    """Initializes and returns the LIBERO environment, along with the task description.

    Args:
        camera_depths: when True, the env will populate
            ``obs[<cam>_depth]`` (normalized z-buffer in [0, 1]) at every step.
            Used by FlexPi3D eval to feed per-cam depth into the model.
        prompt_source: which source to use for the returned ``task_description``.
            ``"task.language"`` (default): libero benchmark's filename-derived
                prompt (``grab_language_from_filename``). Matches the LIBERO
                training set exactly (tasks.jsonl is built from filenames).
                Use this for the standard 4 suites (libero_spatial /
                libero_object / libero_goal / libero_10).
            ``"bddl_language"``: parse the BDDL ``:language`` field. Required
                for LIBERO-PRO _task perturbed suites (instructions vary per
                perturbation file → filename gives the base task, only BDDL
                holds the perturbation-specific prompt) and LIBERO-Plus
                _view_..._initstate_... suites (filename has perturbation
                suffix that would pollute task.language → strip via
                _resolve_bddl_for_language and read base BDDL :language).

                Be aware: even in stock LIBERO the base 4-suite BDDL
                ``:language`` fields carry wording that differs from the
                filename (e.g. libero_goal task=0 BDDL says ``"Open the
                middle layer of the drawer"`` while the filename — and hence
                the training set — says ``"open the middle drawer of the
                cabinet"``). Do NOT pass ``"bddl_language"`` for standard
                4-suite eval.
    """
    if prompt_source not in ("task.language", "bddl_language"):
        raise ValueError(
            f"prompt_source must be 'task.language' or 'bddl_language', "
            f"got {prompt_source!r}"
        )
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    if prompt_source == "task.language":
        task_description = task.language
    else:
        bddl_for_language = _resolve_bddl_for_language(task_bddl_file)
        parsed = BDDLUtils.robosuite_parse_problem(bddl_for_language)
        li = parsed["language_instruction"]
        task_description = " ".join(li) if isinstance(li, list) else li
    env_args = {
        "bddl_file_name": str(task_bddl_file),
        "camera_heights": resolution,
        "camera_widths": resolution,
        "camera_depths": camera_depths,
        # hard_reset=True (robosuite's own default) tears down + rebuilds
        # MjSim + MjRenderContextOffscreen on every env.reset(). That is what
        # the original (pre-release) eval ran, and it is load-bearing for score
        # parity: a soft reset (False) carries mjData internals (solver
        # warm-start) across trials, flipping borderline episodes — the
        # 2026-08-18 old-vs-new A/B only reproduces the reference numbers
        # per-task exactly with True (+num_steps_wait=30, see
        # sim_libero.yaml). Known caveat kept for the record: each hard
        # reset leaks an EGL FBO/texture handle, and historically a
        # fast-inference variant (~13 it/s) could hit the driver's handle
        # limit and abort in robosuite read_pixels; at current eval
        # cadences (action-only and joint, 50 resets/task) the 2026-08-18
        # 4-suite runs completed clean. If an EGL abort ever resurfaces,
        # slow the reset cadence rather than flipping this back — False
        # changes the numbers.
        "hard_reset": True,
    }
    if env_num > 1:
        env = SubprocVectorEnv([lambda: OffScreenRenderEnv(**env_args) for _ in range(env_num)])
    else:
        env = OffScreenRenderEnv(**env_args)
    env.seed(
        seed
    )  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


# One-time-per-process latch so the raw normalized-depth range is logged once per
# camera (see get_libero_per_cam_depth_uint16_mm), not every step.
_DEPTH_RANGE_LOGGED = set()


def get_libero_per_cam_depth_uint16_mm(obs, env):
    """Extract LIBERO depth as uint16 mm, aligned with disk-encoded RGB.

    This is the depth-encoding contract the training data was built under, and
    eval must reproduce it exactly or the pointmap stream sees a different
    distribution than it was trained on:
      1. ``camera_utils.get_real_depth_map(env.sim, depth_norm)`` → metric metres.
      2. multiply by 1000, clip to [0, 65535], cast uint16.
      3. apply 180° rotation ``[::-1, ::-1]`` so depth aligns with RGB
         (``get_libero_image`` does the same flip on RGB).

    Returns:
        ``{"image": np.uint16(H, W), "wrist_image": np.uint16(H, W)}`` —
        keyed by the disk RGB key names (``image`` = agentview,
        ``wrist_image`` = eye_in_hand). Caller is responsible for any
        further per-cam canonicalization / resizing.

    Requires the env to have been created with ``camera_depths=True``.
    """
    from robosuite.utils import camera_utils

    def _convert(depth_norm, cam):
        arr = np.asarray(depth_norm)
        if cam not in _DEPTH_RANGE_LOGGED:
            _DEPTH_RANGE_LOGGED.add(cam)
            print(f"[depth-diag] {cam}: raw normalized depth min={float(arr.min()):.6f} "
                  f"max={float(arr.max()):.6f} (get_real_depth_map requires [0,1]). "
                  f"Tiny overshoot => benign far-plane fp; large => mujoco/robosuite "
                  f"version drift vs the mujoco 3.3.2 that rendered training depth.",
                  flush=True)
        # Clip the normalized z-buffer into robosuite's expected [0,1] before the
        # metric conversion. The training data was generated under mujoco 3.3.2,
        # where the buffer was already in range; a newer runtime can emit a hair
        # outside it and trip get_real_depth_map's assert. The clip is a no-op for
        # in-range values — the printed range tells you if the overshoot is benign.
        arr = np.clip(arr, 0.0, 1.0)
        depth_m = camera_utils.get_real_depth_map(env.sim, arr)  # (H, W, 1) float
        depth_mm = np.clip(depth_m[..., 0] * 1000.0, 0.0, 65535.0).astype(np.uint16)
        return np.ascontiguousarray(depth_mm[::-1, ::-1])

    return {
        "image": _convert(obs["agentview_depth"], "agentview"),
        "wrist_image": _convert(obs["robot0_eye_in_hand_depth"], "eye_in_hand"),
    }

def get_libero_dummy_action():
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]

def get_libero_image(obs):
    """Extracts image from observations and preprocesses it."""
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    # IMPORTANT: rotate 180 degrees to match train preprocessing

    # [yc] wrist image
    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    # IMPORTANT: rotate 180 degrees to match train preprocessing

    return {
        "image": img,
        "wrist_image": wrist_img
    }

def save_rollout_video(rollout_dir, rollout_images, idx, success, task_description, log_file=None, fps=24):
    """Saves an MP4 replay of an episode."""
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=fps)
    for img in rollout_images:
        if isinstance(img, dict):
            image = []
            for key, value in img.items():
                value_array = np.array(value) if isinstance(value, Image.Image) else value.copy()
                pil_img = Image.fromarray(value_array)
                draw = ImageDraw.Draw(pil_img)
                draw.text((10, 10), f"{key}", fill=(255, 255, 255))
                image.append(np.array(pil_img))
            frame = np.concatenate(image, axis=1)
        elif isinstance(img, Image.Image):
            frame = np.array(img.convert("RGB"))
        else:
            frame = np.array(img)
        video_writer.append_data(frame)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path


def save_prediction_video(
    rollout_dir,
    gt_frames,
    pred_frames,
    idx,
    replan_idx,
    success,
    task_description,
    log_file=None,
    fps=8,
):
    """Saves an MP4 comparison of ground-truth and predicted future frames for one replanning clip."""
    num_frames = min(len(gt_frames), len(pred_frames))
    if num_frames <= 0:
        raise ValueError("Cannot save prediction video with empty GT/pred frame lists.")

    stitched_frames = []
    for gt_frame, pred_frame in zip(gt_frames[:num_frames], pred_frames[:num_frames]):
        if isinstance(gt_frame, dict):
            gt_images = []
            for value in gt_frame.values():
                value_array = np.array(value) if isinstance(value, Image.Image) else value.copy()
                gt_images.append(value_array)
            gt_image = np.concatenate(gt_images, axis=1)
        elif isinstance(gt_frame, Image.Image):
            gt_image = np.array(gt_frame.convert("RGB"))
        else:
            gt_image = np.array(gt_frame)

        if isinstance(pred_frame, Image.Image):
            pred_image = np.array(pred_frame.convert("RGB"))
        else:
            pred_image = np.array(pred_frame)

        target_h, target_w = pred_image.shape[:2]
        if gt_image.shape[:2] != (target_h, target_w):
            gt_image = np.array(
                Image.fromarray(gt_image).resize((target_w, target_h), resample=Image.BILINEAR)
            )

        gt_pil = Image.fromarray(gt_image)
        ImageDraw.Draw(gt_pil).text((10, 10), "gt", fill=(255, 255, 255))
        pred_pil = Image.fromarray(pred_image)
        ImageDraw.Draw(pred_pil).text((10, 10), "pred", fill=(255, 255, 255))
        stitched_frames.append(
            Image.fromarray(np.concatenate([np.array(pred_pil), np.array(gt_pil)], axis=0))
        )

    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    try:
        replan_tag = f"{int(replan_idx):04d}"
    except (TypeError, ValueError):
        replan_tag = str(replan_idx)
    mp4_path = (
        f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}"
        f"--task={processed_task_description}--replan={replan_tag}--gt-pred.mp4"
    )
    save_mp4(stitched_frames, mp4_path, fps=fps)
    print(f"Saved predicted future comparison MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved predicted future comparison MP4 at path {mp4_path}\n")
    return mp4_path

def binarize_gripper_open(open_val: np.ndarray | float) -> np.ndarray:
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    bin_val = (v > 0.5)
    return np.asarray(bin_val, dtype=np.float32)


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den

def invert_gripper_action(action):
    """
    Flips the sign of the gripper action (last dimension of action vector).
    This is necessary for some environments where -1 = open, +1 = close, since
    the RLDS dataloader aligns gripper actions such that 0 = close, 1 = open.
    """
    action[..., -1] = action[..., -1] * -1.0
    return action
