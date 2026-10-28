"""Unit tests for RTCStepBroker — see client_rtc_step_broker.py."""
from __future__ import annotations

import importlib.util
import sys
from concurrent.futures import Future
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pytest

# Import the broker module DIRECTLY by path, so this test never depends on the
# `experiments.yam.flexpi_policy` package __init__. Note that pytest imports the
# parent package during collection regardless of this, so what actually keeps
# the test runnable in any python env with numpy + pytest is that the __init__
# resolves its heavy re-exports lazily.
_BROKER_PATH = (
    Path(__file__).resolve().parent / "client_rtc_step_broker.py"
)
_spec = importlib.util.spec_from_file_location(
    "_client_rtc_step_broker_test_target", _BROKER_PATH
)
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)
RTCStepBroker = _module.RTCStepBroker


# ---------------------------------------------------------------- test doubles


class _FakeWsPolicy:
    """Stand-in for ``WebsocketClientPolicy``. Returns deterministic chunks.

    Each call returns a chunk built by ``chunk_fn(call_idx, H, D)`` (defaults
    to ``[call_idx*100 + row, ...]`` so per-call chunks are easy to identify).
    """

    def __init__(
        self,
        action_horizon: int,
        action_dim: int,
        chunk_fn: Optional[Callable[[int, int, int], np.ndarray]] = None,
    ) -> None:
        self.H = int(action_horizon)
        self.D = int(action_dim)
        self.call_count = 0
        self.reset_calls = 0
        self.last_obs: Optional[Dict[str, Any]] = None
        self._chunk_fn = chunk_fn or self._default_chunk

    @staticmethod
    def _default_chunk(call_idx: int, H: int, D: int) -> np.ndarray:
        # Rows are c*100 + row_idx, broadcast to D cols.
        base = call_idx * 100 + np.arange(H, dtype=np.float32)
        return np.broadcast_to(base[:, None], (H, D)).astype(np.float32).copy()

    def infer(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        self.last_obs = obs
        chunk = self._chunk_fn(self.call_count, self.H, self.D)
        self.call_count += 1
        return {"actions": chunk}

    def reset(self) -> None:
        self.reset_calls += 1


class _DeferredExecutor:
    """Executor that doesn't run tasks until ``complete()`` is called.

    Useful for asserting that the broker's queue mutations happen exactly when
    the future is observed-done, not when it was submitted.
    """

    def __init__(self) -> None:
        self._queue: List[Tuple[Future, Callable, tuple, dict]] = []
        self.submit_count = 0

    def submit(self, fn: Callable, *args, **kwargs) -> Future:
        fut: Future = Future()
        self._queue.append((fut, fn, args, kwargs))
        self.submit_count += 1
        return fut

    def complete_next(self) -> None:
        if not self._queue:
            raise RuntimeError("no pending tasks to complete")
        fut, fn, args, kwargs = self._queue.pop(0)
        try:
            fut.set_result(fn(*args, **kwargs))
        except Exception as e:  # noqa: BLE001
            fut.set_exception(e)

    @property
    def pending(self) -> int:
        return len(self._queue)

    def shutdown(self, **_) -> None:  # noqa: D401
        self._queue.clear()


# --------------------------------------------------------------------- helpers


def _make_broker(
    ws: _FakeWsPolicy,
    *,
    merge_steps: int = 4,
    replan_steps: int = 8,
    executor: Optional[_DeferredExecutor] = None,
) -> Tuple[RTCStepBroker, _DeferredExecutor]:
    broker = RTCStepBroker(
        ws_policy=ws,
        action_horizon=ws.H,
        action_dim=ws.D,
        merge_steps=merge_steps,
        replan_steps=replan_steps,
    )
    # Swap out the real executor for a deferred one so tests are deterministic.
    deferred = executor or _DeferredExecutor()
    broker._executor.shutdown(wait=False)  # noqa: SLF001
    broker._executor = deferred  # type: ignore[assignment]  # noqa: SLF001
    return broker, deferred


def _make_obs(D: int) -> Dict[str, np.ndarray]:
    return {"observation/state": np.zeros(D, dtype=np.float32)}


# ============================================================== tests


def test_cold_start_seeds_queue_via_sync_fallback() -> None:
    H, D = 32, 32
    ws = _FakeWsPolicy(H, D)
    broker, deferred = _make_broker(ws)

    out = broker.infer(_make_obs(D))

    assert ws.call_count == 1, "cold start should sync-call the WS policy once"
    assert deferred.submit_count == 0, "no async submission on cold start"
    assert out["actions"].shape == (D,)
    expected_row0 = ws._default_chunk(0, H, D)[0]
    np.testing.assert_allclose(out["actions"], expected_row0)

    stats = broker.stats
    assert stats["cold_starts"] == 1
    assert stats["drain_recoveries"] == 0
    assert stats["submits"] == 1
    assert stats["merges"] == 1
    assert stats["queue_len"] == H - 1  # popped row 0
    assert broker._step == 1  # noqa: SLF001


def test_steady_state_no_new_submits_before_replan_steps() -> None:
    H, D = 32, 32
    ws = _FakeWsPolicy(H, D)
    broker, deferred = _make_broker(ws, replan_steps=8)

    # 1 cold-start call + 7 more = 8 pops total. With replan_steps=8, on the
    # 8th call calls_since_last_submit becomes 8 (>= replan_steps), but the
    # async submit decision happens at the START of the call — so call #8
    # does NOT trigger a submit (calls_since_last_submit is still 7 at entry).
    # Call #9 is the one that triggers.
    for _ in range(8):
        broker.infer(_make_obs(D))

    assert ws.call_count == 1, "no async fires before reaching replan_steps"
    assert deferred.submit_count == 0
    assert broker.stats["submits"] == 1


def test_replan_triggers_async_submit_at_interval() -> None:
    H, D = 32, 32
    ws = _FakeWsPolicy(H, D)
    broker, deferred = _make_broker(ws, replan_steps=8)

    # 1 cold start + 8 pops = 9 calls. At the 9th, calls_since_last_submit
    # is 8 at entry → trigger submit.
    for _ in range(9):
        broker.infer(_make_obs(D))

    assert deferred.submit_count == 1, "async submit at call #9 (replan_steps after cold)"
    assert ws.call_count == 1, "fake WS not yet called for the async submission"
    assert broker._inflight is not None  # noqa: SLF001
    assert broker._inflight_obs_send_step == 8  # noqa: SLF001  step at submission


def test_merge_aligns_by_obs_send_step() -> None:
    """The canonical scenario from the design discussion.

    H=32, replan_steps=8, merge_steps=4. Submit fires at step 8. Advance 4
    more steps before completing the future → offset=4. Verify regions 1/2/3.
    """
    H, D = 32, 32
    ws = _FakeWsPolicy(H, D)
    broker, deferred = _make_broker(ws, merge_steps=4, replan_steps=8)

    # Cold start + 8 pops: 9 calls (final triggers async submit).
    for _ in range(9):
        broker.infer(_make_obs(D))
    assert deferred.submit_count == 1
    # Step is now 9 (incremented at the end of the 9th call).
    assert broker._step == 9  # noqa: SLF001
    # Submission was recorded at step 8.
    assert broker._inflight_obs_send_step == 8  # noqa: SLF001

    # Snapshot the queue BEFORE merge — those are old[9 .. 32] rows from the
    # cold-start chunk (call_idx=0): values 9..31 in row 0 col 0.
    old_q = [row.copy() for row in broker._queue]  # noqa: SLF001
    assert len(old_q) == H - 9

    # Advance 3 more pops (no submit), bringing step to 12.
    for _ in range(3):
        broker.infer(_make_obs(D))
    assert broker._step == 12  # noqa: SLF001
    assert ws.call_count == 1, "still no async-completed merge"

    # Now complete the future. The "new" chunk is call_idx=1 (cold was 0).
    # When the broker pumps, step=12 ⇒ offset = 12 - 8 = 4.
    deferred.complete_next()
    broker.infer(_make_obs(D))  # one more call pumps the inflight + pops
    # After pump+pop, step=13.

    # At pump time: step was 12, obs_send_step=8 → offset=4.
    # useful_new = new1[4:32], M = 28.
    # old queue length before merge: snapshot after 3 more pops past old_q ⇒
    #   we popped 3 of old_q's entries between the snapshot and the merge.
    #   Original old_q had 23 entries. After 3 pops, 20 entries remained.
    # So L=20, M=28, overlap=20, blend_n=min(4, 20)=4.
    stats = broker.stats
    assert stats["last_offset"] == 4
    assert stats["last_blend_n"] == 4

    # Verify region 1 (cosine blend) on row 0 (col 0) of the queue right
    # after the merge. Recover queue state by re-snapshotting now.
    # By this point we've ALSO popped one more row (the broker.infer at
    # pump-time), so the queue is one shorter than after-merge. We need to
    # reconstruct expected by working from the chunks directly.
    #
    # At merge time:
    #   old queue had rows = cold_chunk[12..32]  (col 0 values: 12, 13, ...31)
    #     -> these are at queue indices 0..19 (L=20)
    #   useful_new = new1[4..32]  (col 0 values: 104, 105, ..., 131)
    #   blend_n=4 → indices 0..3 blended, indices 4..19 overwritten, append 20..27
    #
    # We have already popped one MORE row (index 0 of post-merge queue) on
    # the last broker.infer(). So queue[0] now = post-merge index 1.

    expected_post_merge = []
    cold_chunk = ws._chunk_fn(0, H, D)
    new1_chunk = ws._chunk_fn(1, H, D)
    L = 20
    M = 28
    overlap = min(L, M)
    blend_n = min(4, overlap)
    for i in range(blend_n):
        w = 0.5 - 0.5 * np.cos(np.pi * (i + 1) / (blend_n + 1))
        old_row = cold_chunk[12 + i]  # old queue's i-th
        new_row = new1_chunk[4 + i]
        expected_post_merge.append((1.0 - w) * old_row + w * new_row)
    for i in range(blend_n, overlap):
        expected_post_merge.append(new1_chunk[4 + i])
    for i in range(overlap, M):
        expected_post_merge.append(new1_chunk[4 + i])

    # broker._queue is now post-merge minus 1 popped row.
    queue_snapshot = [row.copy() for row in broker._queue]  # noqa: SLF001
    assert len(queue_snapshot) == len(expected_post_merge) - 1
    for q_row, e_row in zip(queue_snapshot, expected_post_merge[1:]):
        np.testing.assert_allclose(q_row, e_row, rtol=1e-5)


def test_stale_chunk_dropped_when_offset_geq_horizon() -> None:
    """If a slow inflight returns after ``offset >= H`` steps, drop it.

    Set this up via direct state mutation rather than trying to drain the
    queue past the inflight — in real use the broker blocks on
    ``Future.result()`` in that drain path, but the test's
    ``DeferredExecutor`` never auto-completes (that's its purpose).
    """
    H, D = 8, 4
    ws = _FakeWsPolicy(H, D)
    broker, deferred = _make_broker(ws, merge_steps=2, replan_steps=4)

    # Cold start (queue gets H entries, pop 1 → queue has H-1; step=1).
    broker.infer(_make_obs(D))
    assert len(broker._queue) == H - 1  # noqa: SLF001

    # Manually submit an async at step=1 (records obs_send_step=1).
    broker._submit_async(_make_obs(D))  # noqa: SLF001
    assert broker._inflight is not None  # noqa: SLF001
    assert broker._inflight_obs_send_step == 1  # noqa: SLF001

    # Fast-forward step to simulate "we've been popping for a long time;
    # the inflight is taking forever". When the inflight finally pumps,
    # offset = self._step - obs_send_step = (1 + H) - 1 = H → stale.
    queue_before = [row.copy() for row in broker._queue]  # noqa: SLF001
    broker._step = 1 + H  # noqa: SLF001  # offset will be H

    deferred.complete_next()
    broker._pump_inflight()  # noqa: SLF001

    assert broker.stats["stale_drops"] == 1
    assert broker.stats["last_offset"] == H
    assert broker._inflight is None  # noqa: SLF001
    queue_after = [row.copy() for row in broker._queue]  # noqa: SLF001
    # Nothing was merged: queue is unchanged.
    assert len(queue_after) == len(queue_before)
    for a, b in zip(queue_after, queue_before):
        np.testing.assert_array_equal(a, b)


def test_queue_drain_falls_back_to_sync() -> None:
    H, D = 16, 4
    ws = _FakeWsPolicy(H, D)
    # replan_steps higher than horizon ⇒ queue drains before async fires.
    broker, deferred = _make_broker(ws, merge_steps=2, replan_steps=100)

    # Cold start fills queue with H rows; pop H total.
    for _ in range(H):
        broker.infer(_make_obs(D))
    assert deferred.submit_count == 0  # never triggers async (replan too high)
    assert ws.call_count == 1  # only the cold-start

    # Queue should be empty now; next call triggers sync fallback.
    broker.infer(_make_obs(D))
    assert ws.call_count == 2, "queue drain triggers a second sync infer"
    # No inflight ever existed (replan_steps too high), so both syncs are
    # cold_starts; no drain_recoveries.
    assert broker.stats["cold_starts"] == 2
    assert broker.stats["drain_recoveries"] == 0


def test_reset_clears_queue_inflight_and_step() -> None:
    H, D = 32, 32
    ws = _FakeWsPolicy(H, D)
    broker, deferred = _make_broker(ws, replan_steps=8)

    # Warm up + trigger an async (still in-flight).
    for _ in range(9):
        broker.infer(_make_obs(D))
    assert deferred.submit_count == 1
    assert broker._inflight is not None  # noqa: SLF001
    assert len(broker._queue) > 0  # noqa: SLF001
    assert broker._step == 9  # noqa: SLF001

    broker.reset()

    assert broker._step == 0  # noqa: SLF001
    assert len(broker._queue) == 0  # noqa: SLF001
    assert broker._inflight is None  # noqa: SLF001
    assert broker._inflight_obs_send_step is None  # noqa: SLF001
    assert broker._calls_since_last_submit == 0  # noqa: SLF001
    assert broker.stats["submits"] == 0
    assert broker.stats["cold_starts"] == 0
    assert broker.stats["drain_recoveries"] == 0
    assert broker.stats["merges"] == 0
    assert ws.reset_calls == 1


def test_merge_steps_zero_is_hard_cut() -> None:
    """With merge_steps=0, blend_n=0; entire overlap is hard-overwritten."""
    H, D = 32, 32
    ws = _FakeWsPolicy(H, D)
    broker, deferred = _make_broker(ws, merge_steps=0, replan_steps=8)

    for _ in range(9):
        broker.infer(_make_obs(D))
    # advance 3 more pops, then complete + pump
    for _ in range(3):
        broker.infer(_make_obs(D))
    deferred.complete_next()
    broker.infer(_make_obs(D))

    stats = broker.stats
    assert stats["last_blend_n"] == 0
    assert stats["last_offset"] == 4

    # All of the overlap (L=20 rows) should be exact new1[4..24] — no blend.
    new1_chunk = ws._chunk_fn(1, H, D)
    # broker queue's first 19 rows = new1[5..24] (we popped one after merge).
    queue_snapshot = [row.copy() for row in broker._queue]  # noqa: SLF001
    for i in range(19):  # 20 in overlap minus 1 popped
        np.testing.assert_allclose(queue_snapshot[i], new1_chunk[5 + i])


def test_chunk_with_wrong_shape_raises() -> None:
    H, D = 8, 4
    ws = _FakeWsPolicy(H, D, chunk_fn=lambda c, H, D: np.zeros((H + 1, D), dtype=np.float32))
    broker, _ = _make_broker(ws)
    with pytest.raises(ValueError, match=r"action_horizon"):
        broker.infer(_make_obs(D))


def test_constructor_validation() -> None:
    ws = _FakeWsPolicy(8, 4)
    with pytest.raises(ValueError, match=r"action_horizon"):
        RTCStepBroker(ws, action_horizon=0, action_dim=4)
    with pytest.raises(ValueError, match=r"action_dim"):
        RTCStepBroker(ws, action_horizon=8, action_dim=0)
    with pytest.raises(ValueError, match=r"merge_steps"):
        RTCStepBroker(ws, action_horizon=8, action_dim=4, merge_steps=-1)
    with pytest.raises(ValueError, match=r"replan_steps"):
        RTCStepBroker(ws, action_horizon=8, action_dim=4, replan_steps=0)
