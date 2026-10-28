from typing import List

import torch


class StaticActionDimPad:
    """Marks a fixed set of action-head channels as padding on every sample.

    ``forward`` sets ``batch["action_dim_is_pad"]`` ([total_dim] bool, True =
    padded), OR-ed with any mask an earlier transform set. ``ScatterToChannels``
    ORs it into its own static mask, so the listed channels are dropped from
    the action loss, zeroed on the DiT input and target
    (``_mask_padded_action_dims``) and, through ``set_action_dim_mask``, at
    sampling: from the val batch in the trainer, from
    ``derive_action_dim_pad`` at deploy. The values in those channels are still
    produced and normalized; nothing supervises them.

    ``pad_ids`` index the action head (the merger's output). ``ConcatLeftAlign``
    replaces the mask instead of OR-ing it, so under that merger this transform
    has no effect. ``AlignTo34DByWidth``'s 32-wide branch replaces it too, so an
    instance listed before the router has no effect on 32-wide samples.

    A joint-only arm of a 34-D bimanual corpus pads the EEF pos/rot6d slots
    0-17 and the absent 7th joints 26/33 of the 34-D layout, supervising the
    grippers 18/19 and the joints 20-25, 27-32. Action only: proprio is
    untouched.
    ``backward`` is the identity.

    A checkpoint whose processor pads the EEF channels 0-17 has no supervised
    EEF output. With the mask armed, those channels decode to the normalizer's
    centre, which on the identity-mapped rot6d slots is a zero rot6d that
    Gram-Schmidt turns into NaN. The YAM ``eef_ik`` action source reads those
    channels; ``model_joint`` reads only the gripper and joint channels. The
    action source is a client setting; nothing selects it from the checkpoint's
    padded channels.
    """

    def __init__(self, pad_ids: List[int], total_dim: int):
        self.total_dim = int(total_dim)
        self.pad_ids = sorted(int(i) for i in pad_ids)
        assert len(set(self.pad_ids)) == len(self.pad_ids), "duplicate pad_ids"
        assert 0 <= self.pad_ids[0] and self.pad_ids[-1] < self.total_dim, \
            f"pad_ids must lie in [0, {self.total_dim})"
        assert len(self.pad_ids) < self.total_dim, "cannot pad every channel"

    def forward(self, batch):
        if "action" in batch:
            mask = torch.zeros(self.total_dim, dtype=torch.bool)
            mask[self.pad_ids] = True
            prior = batch.get("action_dim_is_pad", None)
            batch["action_dim_is_pad"] = mask if prior is None else (mask | prior.to(dtype=torch.bool))
        return batch

    def backward(self, batch):
        return batch
