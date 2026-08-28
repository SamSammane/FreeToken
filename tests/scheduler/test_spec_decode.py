"""Speculative decoding round trip through the real scheduler pieces, CPU only.

Drives the unbound Scheduler methods (_try_speculate, _drain_spec_batch,
_free_req_resources) against real CPU-built managers, simulating only the engine
forward (complete_one + fabricated per-position argmax). Pins:
- decode->verify batch conversion (drafts appended to ids and token_pool, phase flip),
- fallbacks (non-greedy, no match) that leave the batch a plain decode,
- accept/rollback bookkeeping: lengths, token_pool fix-up, page conservation,
- termination inside a round (EOS on the bonus token) freeing everything, and
- the page-free arithmetic of CacheManager.rollback_req at page_size > 1.
"""
from __future__ import annotations

from types import SimpleNamespace

import torch

from freetoken.core import Batch, Req, SamplingParams
from freetoken.scheduler.cache import CacheManager
from freetoken.scheduler.decode import DecodeManager
from freetoken.scheduler.scheduler import Scheduler
from freetoken.scheduler.table import TableManager
from freetoken.spec import NgramDrafter
from freetoken.utils import div_ceil

MAX_RUNNING = 4
WIDTH = 64
NUM_PAGES = 64
EOS = 99
# Prompt whose tail (after tok0=12) recurs: [10,11,12] at position 0 -> draft [13,20,21...]
PROMPT = [10, 11, 12, 13, 20, 21, 10, 11]
TOK0 = 12


class _Sched:
    """Stub carrying the real unbound Scheduler methods over CPU managers."""

    _try_speculate = Scheduler._try_speculate
    _drain_spec_batch = Scheduler._drain_spec_batch
    _free_req_resources = Scheduler._free_req_resources
    _match_stop_str = Scheduler._match_stop_str

    def __init__(self, page_size: int = 1, spec_tokens: int = 4):
        pt = torch.zeros((MAX_RUNNING + 1, WIDTH), dtype=torch.int32)
        self.cache_manager = CacheManager(NUM_PAGES, page_size, pt, "radix")
        self.table_manager = TableManager(MAX_RUNNING, pt)
        self.decode_manager = DecodeManager(page_size)
        self.token_pool = self.table_manager.token_pool
        self.device = torch.device("cpu")
        self.eos_token_ids = {EOS}
        self.finished_reqs = set()
        self.spec_drafter = NgramDrafter(max_tokens=spec_tokens)
        self.tokenizer = None  # stop-strings unused in these tests


def _admit_decoding_req(s: _Sched, output_len: int = 16, uid: int = 1) -> Req:
    """A request in steady decode state: prompt KV computed, one generated token."""
    prompt = torch.tensor(PROMPT, dtype=torch.int32)
    handle = s.cache_manager.match_req(
        SimpleNamespace(input_ids=prompt, input_len=len(PROMPT), mm_embeds=None)
    ).cuda_handle
    table_idx = s.table_manager.allocate()
    req = Req(input_ids=prompt.clone(), table_idx=table_idx, cached_len=0,
              output_len=output_len, uid=uid, sampling_params=SamplingParams(),
              cache_handle=handle)
    s.cache_manager.lock(handle)
    s.cache_manager.allocate_paged([req])          # pages for the prompt
    req.complete_one()                             # prefill forward happened
    req.append_host(torch.tensor([TOK0], dtype=torch.int32))
    s.token_pool[table_idx, : len(PROMPT)] = prompt
    s.token_pool[table_idx, len(PROMPT)] = TOK0
    s.decode_manager.filter_reqs([req])
    return req


def _run_round(s: _Sched, batch: Batch, targets_per_req: dict[int, list[int]]):
    """Simulate _prepare_batch's allocation + the engine forward, then drain."""
    s.cache_manager.allocate_paged(batch.reqs)
    for req in batch.reqs:
        req.complete_one()
    rows = []
    for req, draft in zip(batch.reqs, batch.spec_drafts):
        t = targets_per_req[req.uid]
        assert len(t) == len(draft) + 1
        rows.extend(t)
    reply, finished = [], set()
    with s.cache_manager.lazy_free_region():
        s._drain_spec_batch(batch, torch.tensor(rows, dtype=torch.int32), reply, finished)
    s.finished_reqs = finished
    return reply, finished


def _free_pages(s: _Sched) -> int:
    return len(s.cache_manager.free_slots)


def test_try_speculate_converts_a_greedy_decode_batch():
    s = _Sched()
    req = _admit_decoding_req(s)
    batch = s.decode_manager.schedule_next_batch()
    s._try_speculate(batch)
    assert batch.spec_verify and batch.is_prefill
    draft = batch.spec_drafts[0]
    assert draft == [13, 20, 21, 10]  # continuation after the matched [10,11,12]
    L = len(PROMPT) + 1
    assert req.device_len == L + len(draft)
    assert req.input_ids.tolist()[-len(draft):] == draft
    assert s.token_pool[req.table_idx, L : L + len(draft)].tolist() == draft
    assert req.extend_len == len(draft) + 1  # last committed token + drafts


def test_non_greedy_batch_stays_plain_decode():
    s = _Sched()
    req = _admit_decoding_req(s)
    req.sampling_params.temperature = 0.7
    batch = s.decode_manager.schedule_next_batch()
    s._try_speculate(batch)
    assert not batch.spec_verify and batch.is_decode
    assert req.device_len == len(PROMPT) + 1


def test_no_draftable_ngram_stays_plain_decode():
    s = _Sched()
    req = _admit_decoding_req(s)
    # replace the history with non-repeating tokens
    fresh = torch.arange(40, 40 + len(PROMPT) + 1, dtype=torch.int32)
    req._ids_buf[: len(fresh)] = fresh
    batch = s.decode_manager.schedule_next_batch()
    s._try_speculate(batch)
    assert not batch.spec_verify and batch.is_decode


def test_full_accept_commits_draft_plus_bonus():
    s = _Sched()
    req = _admit_decoding_req(s)
    batch = s.decode_manager.schedule_next_batch()
    s._try_speculate(batch)
    draft = batch.spec_drafts[0]
    bonus = 55
    reply, finished = _run_round(s, batch, {req.uid: draft + [bonus]})

    L = len(PROMPT) + 1
    want = PROMPT + [TOK0] + draft + [bonus]
    assert req.input_ids.tolist() == want
    assert req.device_len == L + len(draft) + 1
    assert req.cached_len == req.device_len - 1
    assert [m.next_token for m in reply] == draft + [bonus]
    assert not any(m.finished for m in reply)
    assert req in s.decode_manager.running_reqs and not finished
    # token_pool carries the bonus at its position for the next round's extend read
    assert int(s.token_pool[req.table_idx, req.device_len - 1]) == bonus
    # page conservation: exactly ceil(cached_len) pages held, nothing leaked
    held = div_ceil(req.cached_len, s.cache_manager.page_size)
    assert _free_pages(s) == NUM_PAGES - held


def test_rejection_rolls_back_the_draft_tail():
    s = _Sched()
    req = _admit_decoding_req(s)
    batch = s.decode_manager.schedule_next_batch()
    s._try_speculate(batch)
    draft = batch.spec_drafts[0]
    assert len(draft) >= 2
    # first draft token accepted, second rejected: model says 77 there
    targets = [draft[0], 77] + [0] * (len(draft) - 1)
    reply, _ = _run_round(s, batch, {req.uid: targets})

    want = PROMPT + [TOK0, draft[0], 77]
    assert req.input_ids.tolist() == want
    assert [m.next_token for m in reply] == [draft[0], 77]
    assert req.device_len == len(want) and req.cached_len == len(want) - 1
    assert int(s.token_pool[req.table_idx, req.device_len - 1]) == 77
    held = div_ceil(req.cached_len, s.cache_manager.page_size)
    assert _free_pages(s) == NUM_PAGES - held  # rejected tail pages returned
    assert req in s.decode_manager.running_reqs


def test_eos_inside_a_round_finishes_and_frees_everything():
    s = _Sched()
    req = _admit_decoding_req(s)
    batch = s.decode_manager.schedule_next_batch()
    s._try_speculate(batch)
    draft = batch.spec_drafts[0]
    # accept the whole draft, bonus token is EOS
    reply, finished = _run_round(s, batch, {req.uid: draft + [EOS]})

    assert reply[-1].finished and reply[-1].finish_reason == "stop"
    assert req in finished and req not in s.decode_manager.running_reqs
    assert req.table_idx == -1  # resources freed exactly once
    # every page is free or tree-owned; the manager's own invariant must hold
    s.cache_manager.check_integrity()


def test_output_budget_caps_committed_tokens():
    s = _Sched(spec_tokens=4)
    # room for exactly 2 more committed tokens after tok0
    req = _admit_decoding_req(s, output_len=3)
    batch = s.decode_manager.schedule_next_batch()
    s._try_speculate(batch)
    draft = batch.spec_drafts[0]
    assert len(draft) == 1  # capped by max_device_len - device_len - 1
    reply, finished = _run_round(s, batch, {req.uid: draft + [56]})
    assert [m.next_token for m in reply] == draft + [56]
    assert reply[-1].finished and reply[-1].finish_reason == "length"
    assert req in finished
    s.cache_manager.check_integrity()


def test_rollback_req_frees_whole_pages_only():
    s = _Sched(page_size=4)
    req = _admit_decoding_req(s)           # prompt 8 tokens -> 2 pages; cached_len 8
    req.device_len = 16                    # 7 draft tokens appended (simulated)
    s.cache_manager.allocate_paged([req])  # verify-round pages for tokens [8, 16)
    before = _free_pages(s)
    req.cached_len = 9                     # committed KV after the round
    s.cache_manager.rollback_req(req, alloc_len=16)
    # keep ceil(9/4)=3 pages, allocated ceil(16/4)=4 -> exactly one whole page returned
    assert _free_pages(s) == before + 1
