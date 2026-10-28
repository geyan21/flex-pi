"""Whether a checkpoint's padded action channels were masked in training, and
which channels those are.

The rule: a checkpoint trained WITH pad masking (padded action channels zeroed
on the DiT input and target, loss over the real channels) is deployed WITH the
mask armed; one trained WITHOUT it (every released RoboTwin / LIBERO / YAM
checkpoint: pre-port code never masked) is deployed WITHOUT it. Arming the
mask on an unmasked checkpoint zeroes channels the model learned to fill;
leaving it off on a masked checkpoint feeds N(0, 1) where training fed 0.

The checkpoint carries the fact. ``FlexPi._base_training_loss`` records that a
mask with any True entry reached the loss, ``save_checkpoint`` writes it as
``payload['action_dim_mask_trained']`` and ``load_checkpoint`` exposes it as
``model.action_dim_mask_trained`` (None when the payload predates the key).

For a checkpoint without the key the saved training config is the fallback:
``data.train.emit_action_dim_is_pad`` when present (the streaming dataset's
knob for emitting the per-sample mask; the pre-release streaming-loader runs'
configs carry it as ``true`` and trained masked); else a streaming ``data.train._target_``
(``AnyDataStreamingDataset`` / ``DroidAnyDataStreamingDataset``) counts as
True, because that loader defaulted the knob to True; else False, which is
what the pre-port map-style configs read, and they trained unmasked.

The one case the fallback misreads: a pre-flag map-style checkpoint from the
pre-release streaming-loader tree whose processor pads (a scatter with unused channels). It
trained masked, its config has neither the key nor a streaming target, so it
reads False; deploying it armed takes ``EVALUATION.action_dim_mask=true``.
"""
from __future__ import annotations

from typing import Optional

import torch

from flexpi.datasets.lerobot.transforms.misc import StaticActionDimPad
from flexpi.datasets.lerobot.transforms.mixed_embodiment_align import (
    _PAD_SLOTS,
    AlignTo34DByWidth,
)

# Loaders whose ``emit_action_dim_is_pad`` defaulted to True; a saved config
# naming one without the key trained masked. Both class names share the suffix.
_STREAMING_TARGET_SUFFIX = "AnyDataStreamingDataset"


def derive_action_dim_pad(processor) -> Optional[torch.Tensor]:
    """Bool ``[total_dim]``, True where the processor leaves the action head empty.

    Four sources:

    1. A ``ConcatLeftAlign`` merger (no ``total_dim``) pads everything past its
       concatenated channels up to ``action_target_dim`` (None when that is
       unset: its ``_pad`` then pads nothing). It replaces any per-sample mask,
       so nothing from the transform chain is added.
    2. An ``AlignTo34DByWidth(backward_mode='yam32')`` in the transform chain
       marks slots 26/33 (the AgiBot-only 7th joints) and replaces any earlier
       mark, as its 32-wide branch does for the 32-wide sample deploy serves.
    3. A ``StaticActionDimPad`` in the chain adds its ``pad_ids``.
    4. A scatter merger's unused ids (an explicit ``used_action_channel_ids``),
       OR-ed last, as ``ScatterToChannels.forward`` does.

    Sources 2 and 3 are replayed in chain order, the way training writes the
    per-sample mask. None when nothing is padded.
    """
    merger = processor.action_state_merger
    if not hasattr(merger, "total_dim"):
        target = merger.action_target_dim
        if target is None:
            return None
        in_dim = sum(int(m["shape"]) for m in merger.action_meta)
        pad = torch.zeros(int(target), dtype=torch.bool)
        pad[in_dim:] = True
        return pad if bool(pad.any()) else None
    total_dim = int(merger.total_dim)
    pad = torch.zeros(total_dim, dtype=torch.bool)
    for trans in processor.action_state_transforms or []:
        if isinstance(trans, AlignTo34DByWidth) and trans.backward_mode == "yam32":
            pad = torch.zeros(total_dim, dtype=torch.bool)
            pad[list(_PAD_SLOTS)] = True
        elif isinstance(trans, StaticActionDimPad):
            pad[trans.pad_ids] = True
    used = merger.used_action_channel_ids
    if used is not None:
        unused = torch.ones(total_dim, dtype=torch.bool)
        unused[list(used)] = False
        pad = pad | unused
    return pad if bool(pad.any()) else None


def trained_with_action_dim_mask(model, trained_cfg, override: Optional[bool] = None) -> bool:
    """Whether the checkpoint loaded into ``model`` trained with the pad mask.

    ``override`` (True/False) wins; else ``model.action_dim_mask_trained`` when
    the checkpoint carried it; else the saved config's
    ``data.train.emit_action_dim_is_pad`` (dict or DictConfig) when present;
    else True iff ``data.train._target_`` is a streaming dataset; else False.
    """
    if override is not None:
        return bool(override)
    if model.action_dim_mask_trained is not None:
        return bool(model.action_dim_mask_trained)
    node = trained_cfg
    for key in ("data", "train"):
        node = node.get(key) if node is not None and hasattr(node, "get") else None
    if node is None or not hasattr(node, "get"):
        return False
    if "emit_action_dim_is_pad" in node:
        # An explicit key wins, null included: the loader read it as bool(None).
        return bool(node.get("emit_action_dim_is_pad"))
    target = node.get("_target_")
    return isinstance(target, str) and target.endswith(_STREAMING_TARGET_SUFFIX)


def resolve_action_dim_mask(
    processor,
    model,
    trained_cfg,
    override: Optional[bool] = None,
) -> tuple[Optional[torch.Tensor], str]:
    """The pad mask to arm, or ``None`` with the reason it stays off.

    A checkpoint deploys the way it trained: padded action-head channels are
    zero-filled through denoising only when training zeroed them on the DiT
    input and target (``trained_with_action_dim_mask``), and only when the
    processor pads any channel at all (``derive_action_dim_pad``). ``override``
    is ``EVALUATION.action_dim_mask``: ``None`` decides from the checkpoint,
    ``True``/``False`` forces.
    """
    pad = derive_action_dim_pad(processor)
    if pad is None or not bool(pad.any()):
        return None, "no padded channels"
    if override is False:
        return None, "EVALUATION.action_dim_mask=false"
    if not trained_with_action_dim_mask(model, trained_cfg, override):
        return None, "checkpoint trained without pad masking"
    return pad, ""
