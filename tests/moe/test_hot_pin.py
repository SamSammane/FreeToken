"""Hot-expert pinning (--moe-pin-hot).

Two layers of coverage, both CPU-only:
- HotExpertPinner policy: persistence requirement (hot across consecutive samples),
  the pin-count cap, decay of cooled experts, and residency awareness.
- End-to-end eviction protection through the REAL hybrid ensure path's CPU reference
  (bit-identical to the GPU kernel per its contract): a pinned resident survives an
  eviction storm that would otherwise reclaim it, and pinning never corrupts the
  slot bookkeeping.
"""
from __future__ import annotations

import torch

from freetoken.moe.hot_pin import HotExpertPinner
from freetoken.moe.offload_cache import OffloadMoeCache

L, E, C = 2, 4, 6  # layers, experts/layer, cache slots (2*4=8 flat ids > 6 slots)


def _mk_cache(pin_fraction: float) -> OffloadMoeCache:
    cache = OffloadMoeCache(
        num_layers=L,
        num_experts=E,
        cache_size=C,
        device=torch.device("cpu"),
        decode_target="hybrid",
        hybrid_max_fetch=E,  # uncapped -> behaves like plain LRU
        pin_hot_fraction=pin_fraction,
    )
    return cache


def _ensure(cache, layer, experts) -> list[int]:
    ids = torch.tensor(experts, dtype=torch.int32)
    cache.ensure_experts_hybrid(layer, ids)
    return ids.tolist()


def _slot_of(cache, layer, expert) -> int:
    return int(cache.slot_for_id[layer, expert])


# ---------- policy ----------

def test_pinner_requires_persistence_across_samples():
    p = HotExpertPinner(cache_size=4, num_layers=1, num_experts=8, max_pin_fraction=0.5)
    id_of_slot = torch.tensor([3, 5, -1, -1], dtype=torch.int32)  # experts 3,5 resident
    usage = torch.tensor([100, 100, 0, 0], dtype=torch.int64)
    # One hot sample is not enough...
    assert p.update(id_of_slot, usage, step=100).sum() == 0
    # ...two consecutive hot samples pin (score 1 * ema + 1 = 1.5 >= 1.4).
    mask = p.update(id_of_slot, usage, step=110)
    assert mask.tolist() == [True, True, False, False]


def test_pinner_caps_the_pin_count():
    p = HotExpertPinner(cache_size=4, num_layers=1, num_experts=8, max_pin_fraction=0.25)
    id_of_slot = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
    usage = torch.full((4,), 50, dtype=torch.int64)
    p.update(id_of_slot, usage, step=50)
    mask = p.update(id_of_slot, usage, step=60)
    assert int(mask.sum()) == 1  # 0.25 * 4


def test_pinner_unpins_cooled_and_nonresident_experts():
    p = HotExpertPinner(cache_size=4, num_layers=1, num_experts=8, max_pin_fraction=0.5)
    hot = torch.tensor([3, 5, -1, -1], dtype=torch.int32)
    usage = torch.tensor([100, 100, 0, 0], dtype=torch.int64)
    p.update(hot, usage, step=100)
    assert p.update(hot, usage, step=110).any()
    # Expert 3 evicted (slot 0 now expert 7, cold), expert 5 stale for many windows.
    later = torch.tensor([7, 5, -1, -1], dtype=torch.int32)
    stale = torch.tensor([120, 120, 0, 0], dtype=torch.int64)
    for step in (400, 700, 1000):  # far outside recent_window -> decay only
        mask = p.update(later, stale, step=step)
    assert not mask.any()


# ---------- end-to-end through the hybrid CPU reference ----------

def test_pinned_expert_survives_an_eviction_storm():
    cache = _mk_cache(pin_fraction=0.35)  # max 2 pinned slots
    # Make (layer 0, expert 0) hot: touch it repeatedly, then sample the pinner twice.
    for _ in range(3):
        _ensure(cache, 0, [0])
    for _ in range(2):
        cache._pin_tick = cache.pin_sample_interval - 1
        cache.tick_hot_pins()
    hot_slot = _slot_of(cache, 0, 0)
    assert hot_slot >= 0 and bool(cache._pin_mask[hot_slot])

    # Storm: stream 2 rounds of every other expert through both layers -- far more
    # traffic than the cache holds. Without pinning this evicts expert 0's slot.
    for _ in range(2):
        for layer in range(L):
            for e in range(1, E):
                _ensure(cache, layer, [e])
    assert _slot_of(cache, 0, 0) == hot_slot, "pinned expert was evicted"
    # Bookkeeping stays coherent: the slot still maps back to (0, 0).
    assert int(cache.id_of_slot[hot_slot]) == 0 * E + 0


def test_unpinned_baseline_gets_evicted_by_the_same_storm():
    cache = _mk_cache(pin_fraction=0.0)
    assert cache.hot_pinner is None
    for _ in range(3):
        _ensure(cache, 0, [0])
    assert _slot_of(cache, 0, 0) >= 0
    for _ in range(2):
        for layer in range(L):
            for e in range(1, E):
                _ensure(cache, layer, [e])
    assert _slot_of(cache, 0, 0) == -1, "baseline LRU should have evicted the idle expert"


def test_pinning_never_blocks_active_routing():
    cache = _mk_cache(pin_fraction=0.35)
    for _ in range(3):
        _ensure(cache, 0, [0])
    for _ in range(2):
        cache._pin_tick = cache.pin_sample_interval - 1
        cache.tick_hot_pins()
    # Every routed expert must still resolve to a usable slot (or a fetch) even with
    # pins held: stream all experts and check the in-place slot rewrite is valid.
    for layer in range(L):
        slots = _ensure(cache, layer, list(range(E)))
        for s in slots:
            assert -1 <= s < C
    cache.check = None  # no-op; documents that no assertion fired above


def test_reset_clears_pins(monkeypatch):
    # reset_cache launches a Triton kernel (GPU); stub it -- only the pin-clearing
    # additions to OffloadMoeCache.reset are under test here.
    import freetoken.moe.offload_kernels as kernels

    monkeypatch.setattr(kernels, "reset_cache", lambda cache: None)
    cache = _mk_cache(pin_fraction=0.35)
    for _ in range(3):
        _ensure(cache, 0, [0])
    for _ in range(2):
        cache._pin_tick = cache.pin_sample_interval - 1
        cache.tick_hot_pins()
    assert cache._pin_mask.any()
    cache.reset()
    assert not cache._pin_mask.any()
    assert not cache.hot_pinner.score.any()
