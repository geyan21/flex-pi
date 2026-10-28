# Mixed-embodiment training in the 34-D layout

One 34-wide action head serves 6-joint-per-arm robots (YAM and the other eef32
corpora) and 7-joint-per-arm robots (AgiBot), so a checkpoint trained on one
warm-starts the other. Both preview checkpoints ([`PRETRAINED.md`](PRETRAINED.md))
use this head.

## 1. The layout

| 34-D slots | content | from a 32-D eef32 sample |
|---|---|---|
| `0:3`, `3:9` | left position (m), left rot6d | `0:9` |
| `9:12`, `12:18` | right position, right rot6d | `9:18` |
| `18`, `19` | left, right gripper | `18:20` |
| `20:27` | left joints × 7 | `20:26` → `20:26`; slot **26** is pad |
| `27:34` | right joints × 7 | `26:32` → `27:33`; slot **33** is pad |

A 32-D dim `d` lands at `d` for `d < 26` and at `d + 1` for `d ≥ 26`. Slots 26 and
33, each arm's seventh joint, are filled only by 7-joint data.

## 2. The per-sample pad mask

The merger (`ScatterToChannels`) emits `action_dim_is_pad` / `proprio_dim_is_pad`,
True on the channels the embodiment does not fill. The map-style loader forwards
them to the model only when the data preset sets `emit_action_dim_is_pad: true`
(opt-in; `yam34.yaml` sets it). With the mask, the model:

- multiplies the noisy action input and the flow-matching target by `~mask`;
- averages the action loss over the real channels only;
- zeroes the padded channels at every action-latent init and after every
  denoising step, so they stay exactly 0 at sampling.

Presets that leave the key off (LIBERO, RoboTwin, the 32-D YAM preset) train on
every action channel, as their released checkpoints did. A checkpoint records
whether it trained masked in its payload (`action_dim_mask_trained`); a payload
without the key falls back to the saved `config.yaml`'s `emit_action_dim_is_pad`.

## 3. Presets

| | data | chain | normalizer width | stats |
|---|---|---|---|---|
| `configs/data/yam34.yaml`, task `yam_unified_flex_3cam_34d_rel_1e-4` | 32-D eef32 LeRobot dirs | `Yam32DRelativeAction(anchor=first)` → normalizer → `ScatterToChannels(total_dim=34, used_*_channel_ids)` | 32 | fit from the training data on the first run |
| `configs/data/agibot34.yaml`, task `agibot_unified_flex_3cam_34d_1e-4` | 34-D AgiBot LeRobot dirs | `AgiBotAbsoluteLookahead(keep_all_joints=true)` → `Yam32DRelativeAction(state_dim=34)` → normalizer → identity merger | 34 | fit from the training data on the first run |

The normalizer's width is the width of the chain's output, so `yam34` takes a
32-wide stats file and `agibot34` a 34-wide one; the other width is a shape
mismatch at the normalizer. `AgiBotAbsoluteLookahead` turns AgiBot's on-disk
action (current state plus a gripper close command) into the lookahead target the
eef32 corpora already carry; on eef32 data it would shift the target twice. The presets ship no
data: `agibot34` reads a local AgiBot World conversion, which is gated under
AgiBot's CC BY-NC-SA 4.0 terms. Both presets share the head geometry
(`action_encoder (1024, 34)`, `head (34, 1024)`, `proprio_encoder (4096, 34)`), so
a checkpoint of one warm-starts the other with `PRETRAINED_CKPT=`.

## 4. Mapping another embodiment in

- **Another action width.** `ScatterToChannels(total_dim=34,
  used_action_channel_ids=[...], used_state_channel_ids=[...])` places each
  incoming channel at an explicit 34-D slot; every slot not listed is pad.
- **Masking channels the data does carry.** `StaticActionDimPad(pad_ids,
  total_dim)` (`transforms/misc.py`) marks fixed channels as pad, OR-ed with the
  merger's mask; for example, a joint-only run pads the EEF slots `0:18`. Such a
  checkpoint's EEF channels `0:18` are unsupervised (NaN on rot6d once armed); its
  client reads the joints `20:32` and grippers.
- A new preset that pads channels sets `emit_action_dim_is_pad: true`, or the pads
  are trained as real zeros.

## 5. Deploy

At load the policy derives the pad channels from the processor chain and arms the
mask only when the checkpoint trained with it, logging
`action_dim_mask: channels [26, 33] zero-filled` (or `not armed (<reason>)`). The
wire stays 32-D in both directions: the server widens the incoming state to 34 and
gathers the 34-D chunk back to 32 before inverting the relative action.
