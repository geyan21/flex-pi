"""Absolute-future-pose transforms for AgiBot pretrain and LIBERO finetune.

Both transforms implement the same supervision pattern:
    action_target[t] := layout_shuffle(state[t+1])      # absolute future pose
    state[t]         := layout_shuffle(state[t])         # absolute current pose
    grippers in action come from disk action's binary command, NOT state lookahead

This is the only viable absolute target for AgiBot: its raw disk action is
bit-exact equal to state on the arm and joint dims (verified on the raw h5 and
the LeRobot parquet), so without the lookahead the model trains on zero-motion
targets and nothing raises. For LIBERO, lookahead is mathematically equivalent
to "state[t] + disk_delta[t]" but more robust (uses actual achieved state, not
commanded delta).

Why absolute (and not delta) for rot6d:
    rot6d on delta concentrates near identity [1,0,0,0,1,0] with std=O(0.003)
    -> any normalizer choice fails (z-score+clamp clips 30-40%; identity
    passthrough lets N(0,1) noise dominate signal at SNR=0.006).
    rot6d on absolute covers SO(3) with std=O(1) -> identity passthrough
    works as UMI guideline prescribes.

Rot6d convention:
    AgiBot's disk format stores 6D rotation as the first 2 COLUMNS of R
    flattened (Zhou et al. 2019). FlexPi's `matrix_to_rotation_6d` uses the
    first 2 ROWS of R (the UMI/Drake convention). Decoding column-based 6D with
    the row-based decoder `rotation_6d_to_matrix` returns R^T, not R, so the
    layout shuffle applies `col6d_to_row6d` to slots [3:9] and [12:18] of the
    AgiBot disk vector (see utils/rotation.py). Without it AgiBot pretrain
    would learn rot6d in column convention while the LIBERO / YAM finetunes
    use row convention, breaking cross-embodiment rotation transfer.
"""

from typing import Dict

import torch

from ..utils.rotation import col6d_to_row6d


class AgiBotAbsoluteLookahead:
    """AgiBot disk 34D native -> FlexPi 32D (or 34D) layout, absolute lookahead.

    Disk layout (AgiBot native, see modality.json):
        [0:3]    left_pos
        [3:9]    left_ori_6d         (already 6D rotation, passthrough)
        [9:12]   right_pos
        [12:18]  right_ori_6d
        [18]     left_grip
        [19]     right_grip
        [20:27]  left_joint   (7 DoF — modality.json declares left_joint=[20,27))
        [27:34]  right_joint  (7 DoF — modality.json declares right_joint=[27,34))

    Output layout (FlexPi 32D, EEF-grouped + grip-grouped + joint-grouped):
        [0:3]    L_pos               <- disk[0:3]
        [3:9]    L_rot6d             <- disk[3:9] (col->row)
        [9:12]   R_pos               <- disk[9:12]
        [12:18]  R_rot6d             <- disk[12:18] (col->row)
        [18]     L_grip              <- disk[18]
        [19]     R_grip              <- disk[19]
        [20:26]  L_joint  (6 DoF)    <- disk[20:26]   (drops disk[26], see below)
        [26:32]  R_joint  (6 DoF)    <- disk[27:33]   (drops disk[33], see below)

    Output layout with keep_all_joints=True (34D — no joint is discarded):
        [0:20]   identical to the 32D layout above
        [20:27]  L_joint  (7 DoF)    <- disk[20:27]
        [27:34]  R_joint  (7 DoF)    <- disk[27:34]

    On the dropped slots: disk[26]/disk[33] are NOT a trailing zero pad on the
    500hr corpus — meta/modality.json declares 7 DoF per arm, and both dims are
    non-constant in 99/99 task dirs (per-task range 1.28-6.14 rad; mean std
    0.478 vs 0.400 for the twelve joint dims that are kept). They are each
    arm's 7th joint. keep_all_joints=True keeps them; the 32D default is
    retained byte-for-byte because every existing AgiBot checkpoint and the
    cross-embodiment YAM finetune layout are 32D.

    State output:
        For each t, output[t] = layout_shuffle(state_disk[t]). Absolute current pose.

    Action output (state lookahead, gripper from disk action):
        For each t in [0, T_act):
            output[t][L_arm | R_arm | joint slots] = layout_shuffle(state_disk[t+1])
            output[t][L_grip=18] = action_disk[t][18]   (binary command)
            output[t][R_grip=19] = action_disk[t][19]   (binary command)
        Boundary: t == T_obs-1 -> use state_disk[T_obs-1] (stay-still).

    Backward (eval / deploy inverse): defined ONLY for keep_all_joints=True,
    where the layout shuffle is a bijection. The 32D variant raises — it
    dropped two joints, so no inverse exists. See `backward`.
    """

    def __init__(self, keep_all_joints: bool = False,
                 gripper_command_is_close: bool = True) -> None:
        # False (default) -> 32D output, each arm's 7th joint discarded (the
        # layout every existing AgiBot checkpoint and the YAM rel finetune use).
        # True -> 34D output, all 14 joint DoF kept. Pair it with
        # Yam32DRelativeAction(state_dim=34) and 34-wide norm stats / masks.
        self.keep_all_joints = bool(keep_all_joints)
        self.out_dim = 34 if self.keep_all_joints else 32
        # AgiBot's disk action gripper is a CLOSE command, running opposite to
        # its own state and to every YAM-family corpus; True emits target
        # openness instead, so slots 18/19 mean the same thing across a mixed
        # pool. See _encode_action_lookahead for the measurement. False restores
        # the convention earlier AgiBot checkpoints were trained on.
        self.gripper_command_is_close = bool(gripper_command_is_close)

    def set_shape_meta(self, shape_meta) -> None:
        pass

    def forward(self, batch: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Dict[str, torch.Tensor]]:
        if "state" not in batch:
            raise ValueError("AgiBotAbsoluteLookahead expects 'state' in batch")
        state_raw = batch["state"]["default"]  # [..., T_obs, 34]
        assert state_raw.shape[-1] == 34, f"state last dim must be 34, got {state_raw.shape}"

        if "action" in batch:
            action_raw = batch["action"]["default"]  # [..., T_act, 34]
            assert action_raw.shape[-1] == 34, f"action last dim must be 34, got {action_raw.shape}"
            T_act = action_raw.shape[-2]
            T_obs = state_raw.shape[-2]
            state_next = self._select_state_next(state_raw, T_act, T_obs)
            batch["action"]["default"] = self._encode_action_lookahead(state_next, action_raw)

        batch["state"]["default"] = self._reshuffle(state_raw)
        return batch

    def backward(self, batch):
        """FlexPi layout -> AgiBot disk layout (the deploy direction).

        Only defined for keep_all_joints=True. The 32D layout discards disk[26]
        and disk[33], so its output cannot be turned back into a 14-joint AgiBot
        command — the inverse does not exist, which is exactly why 32D is
        pretrain-only and 34D is deployable.

        Inverse of `_reshuffle_to_flexpi34`: undo the joint grouping and re-encode
        the two rotation blocks in AgiBot's column convention. `col6d_to_row6d` is
        its own inverse on a valid 6D rotation encoding (it is
        "decode-as-rows -> transpose -> encode-as-rows"; applying it twice returns
        the original, verified to 6.7e-16 in float64 on exact rotations and to
        float32 epsilon on real AgiBot disk vectors).

        Acts on `action` only — proprio is an input, never a model output, and the
        sibling Yam32DRelativeAction.backward has the same contract.

        Under `gripper_command_is_close` (the default) this also inverts slots
        18:20 back to a close command, undoing the flip `_encode_action_lookahead`
        applies. The reshuffle alone is NOT the inverse of forward.
        """
        if not self.keep_all_joints:
            raise NotImplementedError(
                "AgiBotAbsoluteLookahead.backward requires keep_all_joints=True. The 32D "
                "layout drops disk[26]/disk[33] (each arm's 7th joint), so the AgiBot disk "
                "vector cannot be reconstructed from it. Train with keep_all_joints=True "
                "(the 34D layout) to deploy on AgiBot."
            )
        if "action" in batch:
            a = self._reshuffle_from_flexpi34(batch["action"]["default"])
            if self.gripper_command_is_close:
                # Undo the forward flip. `_encode_action_lookahead` emits
                # `1 - command` on 18:20, and `_reshuffle_from_flexpi34` is a
                # pure layout permutation, so the sense has to be inverted
                # here — symmetrically to where forward applies it. Without
                # this the round trip returns the complement (error 1.0) and a
                # deployed policy predicting "close" sends close_command=0,
                # inverting every grasp.
                a[..., 18:20] = 1.0 - a[..., 18:20]
            batch["action"]["default"] = a
        return batch

    def _reshuffle_from_flexpi34(self, a: torch.Tensor) -> torch.Tensor:
        """FlexPi 34D layout [..., 34] -> AgiBot native disk layout [..., 34].

        A PURE LAYOUT PERMUTATION. The gripper un-flip that makes
        forward/backward an inverse lives in `backward()`, not here: `1 - x` is
        its own inverse, so applying it in both places would flip twice and
        silently restore the inverted-gripper deploy bug with every value still
        inside [0, 1].
        """
        out = torch.empty_like(a)
        out[..., 0:3]    = a[..., 0:3]                          # left_pos
        out[..., 3:9]    = col6d_to_row6d(a[..., 3:9])         # left_ori  (row->col)
        out[..., 9:12]   = a[..., 9:12]                         # right_pos
        out[..., 12:18]  = col6d_to_row6d(a[..., 12:18])       # right_ori (row->col)
        out[..., 18:20]  = a[..., 18:20]                        # left_grip, right_grip (un-flip is in backward)
        out[..., 20:27]  = a[..., 20:27]                        # left_joint  (7 DoF)
        out[..., 27:34]  = a[..., 27:34]                        # right_joint (7 DoF)
        return out

    @staticmethod
    def _select_state_next(state_raw: torch.Tensor, T_act: int, T_obs: int) -> torch.Tensor:
        """Pick state[t+1] for action target lookahead.

        Two modes:
        - Training (T_obs >= T_act + 1): real lookahead state[..., 1:T_act+1, :].
        - Stats-calc (T_obs == 1): the dataset's get_dataset_stats path passes
          state with T_obs=1 (one obs per episode-frame) while action runs
          through sliding_window_with_replication to T_act. State is broadcast
          across T_act — the marginal value distribution is equivalent to true
          lookahead because state[t+1] and state[t] have the same marginals
          across an episode trajectory, and only the per-dim distribution
          matters for normalizer fitting.
        """
        if T_obs >= T_act + 1:
            return state_raw[..., 1:T_act + 1, :]
        if T_obs == 1:
            # repeat rather than expand: downstream code may write in place.
            return state_raw.repeat(*([1] * (state_raw.ndim - 2)), T_act, 1)
        raise ValueError(
            f"unexpected T_obs={T_obs} for T_act={T_act}; "
            f"expected T_obs >= T_act+1 (training) or T_obs == 1 (stats calc)"
        )

    def _reshuffle(self, s: torch.Tensor) -> torch.Tensor:
        """Layout shuffle into whichever output width this instance was built for."""
        return (self._reshuffle_to_flexpi34(s) if self.keep_all_joints
                else self._reshuffle_to_flexpi32(s))

    @staticmethod
    def _reshuffle_to_flexpi34(s: torch.Tensor) -> torch.Tensor:
        """AgiBot native [..., 34] -> FlexPi 34D layout, no joint discarded.

        Identical to `_reshuffle_to_flexpi32` on slots [0:20] (same col->row 6D
        rotation fix); the joint blocks stay 7 DoF wide, so disk[26] and
        disk[33] — each arm's 7th joint — land at out[26] and out[33].
        """
        out = torch.empty(s.shape[:-1] + (34,), dtype=s.dtype, device=s.device)
        out[..., 0:3]    = s[..., 0:3]                          # L_pos
        out[..., 3:9]    = col6d_to_row6d(s[..., 3:9])         # L_rot6d (col->row fix)
        out[..., 9:12]   = s[..., 9:12]                         # R_pos
        out[..., 12:18]  = col6d_to_row6d(s[..., 12:18])       # R_rot6d (col->row fix)
        out[..., 18:19]  = s[..., 18:19]                        # L_grip
        out[..., 19:20]  = s[..., 19:20]                        # R_grip
        out[..., 20:27]  = s[..., 20:27]                        # L_joint (7 DoF)
        out[..., 27:34]  = s[..., 27:34]                        # R_joint (7 DoF)
        return out

    @staticmethod
    def _reshuffle_to_flexpi32(s: torch.Tensor) -> torch.Tensor:
        """Pure layout shuffle, AgiBot native [..., 34] -> FlexPi 32D layout.

        Drops each arm's 7th joint (disk[26], disk[33]) to reach the 32D
        cross-embodiment layout. Those slots are NOT zero pad — see the class
        docstring; keep_all_joints=True retains them.

        Applies the col->row 6D rotation convention fix on slots [3:9] and
        [12:18]. AgiBot disk encodes [c0;c1] of R; FlexPi expects [r0;r1].
        See the module docstring and `col6d_to_row6d` for the math.
        """
        out = torch.empty(s.shape[:-1] + (32,), dtype=s.dtype, device=s.device)
        out[..., 0:3]    = s[..., 0:3]                          # L_pos
        out[..., 3:9]    = col6d_to_row6d(s[..., 3:9])         # L_rot6d (col->row fix)
        out[..., 9:12]   = s[..., 9:12]                         # R_pos
        out[..., 12:18]  = col6d_to_row6d(s[..., 12:18])       # R_rot6d (col->row fix)
        out[..., 18:19]  = s[..., 18:19]                        # L_grip
        out[..., 19:20]  = s[..., 19:20]                        # R_grip
        out[..., 20:26]  = s[..., 20:26]                        # L_joint (6 DoF; drop disk[26])
        out[..., 26:32]  = s[..., 27:33]                        # R_joint (6 DoF; drop disk[33])
        return out

    def _encode_action_lookahead(self, state_next: torch.Tensor, action_raw: torch.Tensor) -> torch.Tensor:
        """Action target = layout_shuffle(state_next), grippers from action_raw.

        AgiBot's disk action gripper is a CLOSE COMMAND, not a target width, and
        it runs opposite to both AgiBot's own state and every YAM-family corpus.
        Measured on 12 episodes whose gripper actually moves (an idle arm shows
        state constant 1.0 / action constant 0.0 and proves nothing):
        corr(state, action) is -0.47..-0.88 in every case, state is continuous
        with 1.000 = fully open, and action is effectively {0, 1} with 1 = close.
        Corpus-wide, mean|1-a-s| = 0.111 against mean|a-s| = 0.888.

        YAM's action gripper is the same quantity as its state (1 = open;
        state[0] = 0.995 open, median 0.02 closed). So pooling AgiBot with YAM
        without this flip puts opposite senses on action slots 18/19 within one
        batch, and nothing downstream would raise.

        `1 - command` converts close-command to target-openness, matching YAM.
        The shape still differs -- AgiBot's is near-binary where YAM's is smooth
        -- so this aligns the SENSE, not the distribution.

        The default is True because that is what the data IS: AgiBot's disk
        command genuinely is a close command. False reproduces the unflipped
        convention earlier AgiBot checkpoints' gripper heads were trained on
        and is the wrong setting for any new run.
        """
        out = self._reshuffle(state_next)
        # Binary 0/1 is cleaner supervision than state's continuous width, which
        # lags through gripper dynamics -- so the command is still the source.
        grip = action_raw[..., 18:20]
        if self.gripper_command_is_close:
            grip = 1.0 - grip
        out[..., 18:20] = grip
        return out


class LiberoAbsoluteLookahead:
    """LIBERO disk 7D action / 8D state -> 7D absolute action / 7D state.

    Disk layouts:
        action[t]: [Δx, Δy, Δz, Δrx, Δry, Δrz, gripper]            (7D, axis_angle delta)
        state[t]:  [x, y, z, rx, ry, rz, qpos_L, qpos_R]            (8D, absolute axis_angle + 2 finger qpos)

    Output (both 7D, fed downstream to PoseRotationTransform(axis_angle_to_rotation_6d)
    which converts axis_angle -> rot6d, producing 10D for both):
        state_out[t]:  [x, y, z, rx, ry, rz, qpos_L - qpos_R]        (7D, absolute pos+rot, gripper width)
        action_out[t]: [state[t+1].x, .y, .z, .rx, .ry, .rz, action[t].gripper]
                       (7D, absolute future pose + binary gripper command)

    Why qpos_L - qpos_R as a single scalar:
        - LIBERO single-arm Franka has a parallel-jaw gripper. qpos_L and qpos_R
          are the two finger displacements (positive vs negative sign).
        - qpos_L stays positive ~[0.034, 0.040], qpos_R stays negative ~[-0.040, -0.022]
          (verified empirically on libero_object episode_000000).
        - qpos_L - qpos_R is a single positive scalar = total opening width.
          qpos_R has larger std (0.006) than qpos_L (0.002), so collapsing to a
          single scalar via difference preserves the high-variance signal.
        - This matches AgiBot L_grip (FlexPi 32D output slot 18) which is also
          a single scalar (state width or action binary command). LIBERO's
          R_grip slot (19) stays unused (zero) for single-arm.
        - Drops asymmetric grasp info (qpos_L + qpos_R deviates from 0 when
          holding an object), but visual frames recover that signal.

    Backward (eval inverse): identity no-op. The model's 7D output is already
    in absolute world-frame coords (post PoseRotationTransform.backward
    converts rot6d back to axis_angle). The eval driver does the absolute->delta
    conversion separately (it needs the simulator's current state, which is
    not available to dataloader transforms).

    Lookahead boundary: t == T_obs - 1 -> use state[T_obs-1] (stay-still).
    """

    def __init__(self) -> None:
        pass

    def set_shape_meta(self, shape_meta) -> None:
        pass

    def forward(self, batch: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, Dict[str, torch.Tensor]]:
        if "state" not in batch:
            raise ValueError("LiberoAbsoluteLookahead expects 'state' in batch")
        state_raw = batch["state"]["default"]  # [..., T_obs, 8]
        assert state_raw.shape[-1] == 8, f"LIBERO state last dim must be 8, got {state_raw.shape}"

        if "action" in batch:
            action_raw = batch["action"]["default"]  # [..., T_act, 7]
            assert action_raw.shape[-1] == 7, f"LIBERO action last dim must be 7, got {action_raw.shape}"
            T_act = action_raw.shape[-2]
            T_obs = state_raw.shape[-2]
            state_next = AgiBotAbsoluteLookahead._select_state_next(state_raw, T_act, T_obs)
            # action_out = [state_next.pos, state_next.axis_angle, action_raw.gripper]
            action_out = torch.cat(
                [
                    state_next[..., 0:3],          # absolute future pos
                    state_next[..., 3:6],          # absolute future axis_angle
                    action_raw[..., 6:7],          # binary gripper command from disk action
                ],
                dim=-1,
            )
            batch["action"]["default"] = action_out

        # state_out = [state.pos, state.axis_angle, qpos_L - qpos_R]
        gripper_width = state_raw[..., 6:7] - state_raw[..., 7:8]
        state_out = torch.cat(
            [
                state_raw[..., 0:3],               # absolute pos
                state_raw[..., 3:6],               # absolute axis_angle
                gripper_width,                     # collapsed single-scalar gripper width
            ],
            dim=-1,
        )
        batch["state"]["default"] = state_out
        return batch

    def backward(self, batch):
        # Identity no-op. Model output stays in absolute world-frame coords.
        # The absolute->delta conversion (needed for LIBERO controller) is done
        # in the eval driver after PoseRotationTransform.backward, because it
        # requires the simulator's current state which is not in `batch`.
        return batch
