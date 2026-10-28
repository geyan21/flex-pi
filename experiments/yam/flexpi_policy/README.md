# YAM deployment code

The server side of a real-robot deployment: everything between a trained YAM
checkpoint and an action chunk on the wire.
[`docs/YAM.md §3`](../../../docs/YAM.md#3-deploy-server) is the guide — the
environment, the three serving configurations, the msgpack wire contract, and
the rules that have caused emergency stops on real hardware. Nothing here
repeats it.

The robot-side client is yours to write. This repo ships no robot stack, so
what a client does with the 32D chunk — inverse kinematics, motor ordering,
gripper conventions — belongs to whatever drives your arm.

---

## 1. Layout

| | |
|---|---|
| `deploy_policy.py` | `YamFlexPiPolicy` + `build_policy_from_checkpoint`: model load, observation tensors, denormalization, `Yam32DRelativeAction` inversion. Returns an **absolute** 32D chunk laid out by `yam_eef.STATE_LAYOUT`. |
| `server_adapter.py` | wraps the policy as a `BasePolicy` for the websocket server |
| `temporal_smoother.py`, `speed_adapter.py` | optional strategies — a trajectory smoother (OSQP QP over inverse step-durations) and a per-step `speed_factor` map. `server_smoother.py` / `server_speed.py` expose them to the server. |
| `prediction_recorder.py` | the server's prediction recorder (`--record-predictions-*`), driven by `server_adapter.py`; `scripts/render_prediction_dump.py` renders its raw dumps ([`docs/YAM.md`](../../../docs/YAM.md#recording-predictions)) |
| `client_rtc_step_broker.py` | step-indexed RTC stitching, for a client that overlaps chunks |
| `_openpi_vendor/` | vendored openpi websocket client/server and msgpack-numpy |

Runtime knobs default from
[`configs/real_yam.yaml`](../../../configs/real_yam.yaml) rather than from code,
so the server and the offline tools cannot drift apart. Explicit arguments win,
and anything left `null` is derived from the checkpoint's saved config.

## 2. Offline checks

None needs a robot, and `--help` documents every flag.

```bash
# One inference against a recorded frame, compared to ground truth.
python -m experiments.yam.flexpi_policy.smoke_test \
    --ckpt <ckpt.pt> --data-dir <dataset> --check-against-gt --verbose

# One RTC stitching cycle on recorded data.
python -m experiments.yam.flexpi_policy.simulate_rtc_stitching \
    --ckpt <ckpt.pt> --data-dir <dataset>

# Round-trip latency against a running server.
python scripts/test_rd_client_dummy.py --host localhost --port 8000 --n 15

# A raw prediction dump of a recorded session, rendered to MP4.
python scripts/render_prediction_dump.py --dump-dir <record-dir>/<session>
```

`smoke_test.py --num-passes N` measures steady-state latency; for per-stack
figures and the TensorRT engines see
[`docs/INFERENCE_OPTIMIZATION.md`](../../../docs/INFERENCE_OPTIMIZATION.md).
`pytest experiments -q` covers the broker, the smoother and the speed adapter.

## 3. Rules

- **The trained config is the source of truth**, never the sim YAMLs — Hydra
  drift silently corrupts the joint flags and scheduler shifts. That is also why
  `build_policy_from_checkpoint` drops `action_dit_pretrained_path` and sets
  `skip_dit_load_from_pretrain=True`: `load_checkpoint` supplies both DiT weight
  sets, so loading them first is pure I/O — and on an inference-only machine it
  would make deployment depend on the Wan2.2 base shards.
- **Actions leave absolute.** The relative-action inversion against the current
  robot state happens inside the policy, so a client receives absolute targets,
  each arm's pose in that arm's own base frame as the state was sent, and must
  not invert anything again.
- **The prompt is not optional in practice**: with an empty one the policy is
  unconditioned and success collapses silently.
