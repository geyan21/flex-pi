"""Per-sample alignment of mixed-embodiment corpora into the shared 34D layout.

THE PROBLEM THIS SOLVES (YAM + MolmoAct2 + AgiBot co-training): the corpora do
not share action SEMANTICS on disk —

  * YAM / MolmoAct2 shards: 32-wide eef32 vectors, action already
    lookahead-aligned (action[t] = state[t+1]-style future pose). Feeding them
    through AgiBotAbsoluteLookahead would double-shift the target.
  * AgiBot 500hr shards: 34-wide vectors where action is a CURRENT-STATE
    recording on 32 of 34 dims. Without AgiBotAbsoluteLookahead the model
    trains on zero-motion targets and never errors — the silent-failure trap.

One processor has ONE transform chain, so the chain's first step must route
per sample. The discriminator is the sample's own width: 34 -> AgiBot handling,
32 -> yam/mm2 handling. Width is intrinsic to the data (no metadata plumbing,
no shard-path parsing) and a wrong-width sample fails loudly.

  34-wide: AgiBotAbsoluteLookahead(keep_all_joints=True)   -> 34D, all real
  32-wide: zero-scatter into 34D                            -> 34D, slots 26/33
           (each arm's absent 7th joint) marked pad via a per-sample
           action/state_dim_is_pad that ScatterToChannels ORs into its own
           static mask (see action_state_merger.forward).

Chained BEFORE Yam32DRelativeAction(anchor="first", state_dim=34): the rel
math is slot-local, so the padded slots come out as 0-0=0 deltas for 32-wide
samples and real deltas for AgiBot.

NORMALIZATION runs after the chain, so the stats file must be 34-wide REL
stats pooled over every corpus in the mix (the eef32 stats embedded at the 32
real slots plus the AgiBot rel34 stats). Pad slots normalize through AgiBot's
bounds; the values there are loss-masked for 32-wide samples and
constant-per-corpus on the proprio input.

backward() (eval/deploy inverse) cannot know the target embodiment from a
model output, so it is explicit: backward_mode="yam32" (default) gathers the
32 real slots back into the eef32 layout — the YAM deployment path;
backward_mode="agibot34" applies AgiBotAbsoluteLookahead.backward. It is a
deploy-config setting; training never calls backward.
"""
from __future__ import annotations

from typing import Dict

import torch

from .absolute_pose_lookahead import AgiBotAbsoluteLookahead

_PAD_SLOTS = (26, 33)
_KEEP_IDS = list(range(26)) + list(range(27, 33))  # 32 entries: eef32 -> 34D


class AlignTo34DByWidth:
    def __init__(self, backward_mode: str = "yam32") -> None:
        if backward_mode not in ("yam32", "agibot34"):
            raise ValueError(f"backward_mode must be 'yam32' or 'agibot34', got {backward_mode!r}")
        self.backward_mode = backward_mode
        self._agibot = AgiBotAbsoluteLookahead(keep_all_joints=True)

    def forward(self, batch: Dict) -> Dict:
        width = batch["state"]["default"].shape[-1]
        if width == 34:
            # AgiBot: current-state recording -> future-pose lookahead, 34D kept.
            return self._agibot.forward(batch)
        if width == 32:
            # yam/mm2 eef32: already lookahead-aligned; scatter into 34D and
            # mark the two absent 7th-joint slots pad (per-sample mask, OR-ed
            # in by ScatterToChannels so it survives the identity merger).
            keep = torch.tensor(_KEEP_IDS, dtype=torch.long)
            for cat in ("state", "action"):
                if cat not in batch:
                    continue
                x = batch[cat]["default"]
                y = x.new_zeros(*x.shape[:-1], 34)
                y[..., keep] = x
                batch[cat]["default"] = y
            mask = torch.zeros(34, dtype=torch.bool)
            mask[list(_PAD_SLOTS)] = True
            batch["action_dim_is_pad"] = mask.clone()
            batch["state_dim_is_pad"] = mask.clone()
            return batch
        raise ValueError(
            f"AlignTo34DByWidth: sample width {width} is neither 32 (yam/mm2 "
            f"eef32) nor 34 (AgiBot keep-all-joints) — unknown corpus in the mix?"
        )

    def backward(self, batch: Dict) -> Dict:
        if self.backward_mode == "agibot34":
            return self._agibot.backward(batch)
        # yam32: gather the 32 real slots back into the eef32 layout.
        keep = torch.tensor(_KEEP_IDS, dtype=torch.long)
        for cat in ("action", "state"):
            if cat in batch:
                x = batch[cat]
                if isinstance(x, dict):
                    x["default"] = x["default"][..., keep]
                else:
                    batch[cat] = x[..., keep]
        return batch
