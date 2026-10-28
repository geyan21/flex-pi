"""Loopback latency probe for a running YAM FlexPi websocket server — no robot.

Sends a zero-filled dummy obs (the wire shapes the server advertises in its connect
metadata) every ``--interval`` seconds and reports round-trip latency (encoders +
joint denoise + msgpack + localhost socket = the real per-call deploy latency).

Camera shapes come from the SERVER's metadata (``per_cam_hw`` / ``cam_order``), so no
checkpoint/config parsing is needed. The client is imported through the
``experiments.yam.flexpi_policy`` package, whose ``__init__`` loads ``deploy_policy``
and with it torch, hydra and flexpi: the probe runs in the server's environment
(base install plus the ``[serve]`` extra), not in a bare websockets + msgpack venv.

The state is all zeros, so its rotation blocks are degenerate and the EEF dims of
the returned actions are NaN: the probe measures latency, not behavior.

Usage (against a live server):
    python scripts/test_rd_client_dummy.py --host localhost --port 8000 \
        --n 15 --interval 1.0
"""

import argparse
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (PROJECT_ROOT, PROJECT_ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from experiments.yam.flexpi_policy._openpi_vendor.websocket_client_policy import (  # noqa: E402
    WebsocketClientPolicy,
)


def synth_obs(per_cam_hw, prompt):
    """Zero obs at the server's advertised wire shapes."""
    obs = {}
    for cam, (h, w) in per_cam_hw:
        obs[f"observation/image_{cam}"] = np.zeros((h, w, 3), dtype=np.uint8)
        obs[f"observation/depth_{cam}"] = np.zeros((h, w), dtype=np.uint16)
        obs[f"observation/intrinsics_{cam}"] = np.array(
            [200.0, 200.0, w / 2.0, h / 2.0], dtype=np.float32
        )
    obs["observation/state"] = np.zeros(32, dtype=np.float32)
    obs["prompt"] = prompt
    return obs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="localhost")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--interval", type=float, default=1.0, help="seconds between sends")
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--prompt", default="test dummy obs")
    args = ap.parse_args()

    client = WebsocketClientPolicy(host=args.host, port=args.port)
    md = client.get_server_metadata()
    cam_order = md.get("cam_order") or list((md.get("per_cam_hw") or {}).keys())
    hw = md.get("per_cam_hw") or {}
    if not cam_order or not hw:
        raise SystemExit(
            f"server metadata has no per_cam_hw/cam_order: {md} — is this a YAM "
            "FlexPi server?"
        )
    per_cam_hw = [(cam, tuple(hw[cam])) for cam in cam_order]
    print(f"[client] connected to {args.host}:{args.port}; per_cam_hw = {per_cam_hw}")

    lat = []
    for i in range(args.n):
        obs = synth_obs(per_cam_hw, args.prompt)
        t0 = time.perf_counter()
        out = client.infer(obs)
        dt = (time.perf_counter() - t0) * 1e3
        lat.append(dt)
        akey = next((k for k in out if "action" in k.lower()), None)
        ashape = getattr(out.get(akey), "shape", "?") if akey else list(out.keys())
        print(f"[{i + 1:2d}/{args.n}] round-trip {dt:8.1f} ms | {akey}={ashape}")
        time.sleep(max(0.0, args.interval - (time.perf_counter() - t0)))

    warm = lat[1:] if len(lat) > 1 else lat  # drop first (cold) call
    print(
        f"\n[client] round-trip latency (ms): mean {st.mean(warm):.1f} | "
        f"median {st.median(warm):.1f} | min {min(warm):.1f} | max {max(warm):.1f} "
        f"| n={len(warm)} (first call {lat[0]:.1f} ms shown separately)"
    )


if __name__ == "__main__":
    main()
