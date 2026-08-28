"""The CacheManager free list is a fixed-capacity buffer with a watermark: frees write in
place, allocations slice off the top, and neither reallocates. These tests pin the invariants
the rest of the scheduler relies on: conservation (free + allocated == num_pages), no page
handed out twice, allocated tensors surviving later frees, page alignment, the lazy-free
region, eviction refill, and wholesale replacement (rebuild / direct assignment).
CPU, real CacheManager, no engine."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.core import Req, SamplingParams
from freetoken.scheduler.cache import CacheManager


def _pend(ids):
    t = torch.tensor(ids, dtype=torch.int32)
    return SimpleNamespace(input_ids=t, input_len=len(ids), mm_embeds=None)


def _admit(cm, table_idx, ids):
    handle = cm.match_req(_pend(ids)).cuda_handle
    req = Req(input_ids=torch.tensor(ids, dtype=torch.int32), table_idx=table_idx,
              cached_len=0, output_len=0, uid=table_idx, sampling_params=SamplingParams(),
              cache_handle=handle)
    req.device_len = len(ids)
    cm.lock(handle)
    cm.allocate_paged([req])
    req.cached_len = len(ids)
    return req


@pytest.mark.parametrize("page_size", [1, 4])
def test_alloc_free_conserves_pages_and_never_double_allocates(page_size):
    num_pages = 16
    page_table = torch.zeros(4, num_pages * page_size, dtype=torch.int32)
    cm = CacheManager(num_pages, page_size, page_table, "naive")

    a = cm._allocate(5)
    b = cm._allocate(3)
    assert len(cm.free_slots) == num_pages - 8
    got = set(a.tolist()) | set(b.tolist())
    assert len(got) == 8, "a page was handed out twice"
    assert got.isdisjoint(set(cm.free_slots.tolist()))
    assert all(p % page_size == 0 for p in got)

    cm._free(cm._page_to_token(a))
    cm._free(cm._page_to_token(b))
    assert len(cm.free_slots) == num_pages
    assert set(cm.free_slots.tolist()) == {i * page_size for i in range(num_pages)}
    cm.check_integrity()


def test_allocated_tensor_survives_later_frees():
    page_table = torch.zeros(4, 8, dtype=torch.int32)
    cm = CacheManager(8, 1, page_table, "naive")
    a = cm._allocate(4)
    snapshot = a.tolist()
    # Freeing other pages (and even re-allocating) must not mutate `a` under the caller.
    b = cm._allocate(2)
    cm._free(b)
    cm._free(torch.tensor([], dtype=torch.int32))
    assert a.tolist() == snapshot


def test_lazy_free_region_defers_then_returns_pages():
    page_table = torch.zeros(4, 8, dtype=torch.int32)
    cm = CacheManager(8, 1, page_table, "naive")
    a = cm._allocate(3)
    before = len(cm.free_slots)
    with cm.lazy_free_region():
        cm._free(a)
        assert len(cm.free_slots) == before, "free must be deferred inside the region"
    assert len(cm.free_slots) == before + 3
    cm.check_integrity()


def test_allocate_beyond_free_evicts_from_the_tree():
    num_pages = 8
    page_table = torch.zeros(4, num_pages, dtype=torch.int32)
    cm = CacheManager(num_pages, 1, page_table, "radix")
    ids = list(range(100, 100 + num_pages))
    req = _admit(cm, 0, ids)
    # Finished commit donates the pages to the tree (evictable), free list is empty.
    cm.cache_req(req, finished=True)
    assert len(cm.free_slots) < num_pages
    # Allocating more than the free list holds must evict tree pages to satisfy it.
    a = cm._allocate(num_pages)
    assert a.numel() == num_pages
    assert len(set(a.tolist())) == num_pages
    cm._free(a)
    cm.check_integrity()


def test_wholesale_replacement_supports_self_aliasing_views():
    page_table = torch.zeros(4, 8, dtype=torch.int32)
    cm = CacheManager(8, 1, page_table, "naive")
    keep = cm.free_slots[:3].tolist()
    cm.free_slots = cm.free_slots[:3]  # aliases the internal buffer
    assert cm.free_slots.tolist() == keep


def test_rebuild_resets_the_free_list():
    page_table = torch.zeros(4, 8, dtype=torch.int32)
    cm = CacheManager(8, 1, page_table, "naive")
    cm._allocate(5)
    new_table = torch.zeros(4, 40, dtype=torch.int32)
    cm.rebuild(20, new_table)
    assert cm.free_slots.tolist() == [i * 1 for i in range(20)]
    cm.check_integrity()
