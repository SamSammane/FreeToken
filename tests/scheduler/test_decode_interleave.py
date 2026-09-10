"""--decode-interleave: after N consecutive prefill batches scheduled while decodes are
waiting, the scheduler must run one decode batch before the next prefill chunk, so a long
chunked prefill cannot stall every running request's next token. Drives the real (unbound)
``Scheduler._schedule_next_batch`` against CPU-built managers, stubbing only batch
preparation. interleave=0 must reproduce the historical strict prefill priority."""
from __future__ import annotations

from types import SimpleNamespace

import torch

from freetoken.core import Req, SamplingParams
from freetoken.scheduler.cache import CacheManager
from freetoken.scheduler.decode import DecodeManager
from freetoken.scheduler.prefill import PrefillManager
from freetoken.scheduler.scheduler import Scheduler
from freetoken.scheduler.table import TableManager
from freetoken.scheduler.utils import PendingReq

CHUNK = 8
MAX_RUNNING = 4
WIDTH = 128


def _stub_scheduler(decode_interleave: int, n_chunks: int, with_decode: bool):
    pt = torch.zeros((MAX_RUNNING + 1, WIDTH), dtype=torch.int32, device="cpu")
    cm = CacheManager(num_pages=WIDTH, page_size=1, page_table=pt, type="radix")
    tm = TableManager(max_running_reqs=MAX_RUNNING, page_table=pt)
    dm = DecodeManager(page_size=1)
    pm = PrefillManager(cm, tm, dm)
    pm.pending_list = [
        PendingReq(uid=1, input_ids=torch.arange(100, 100 + n_chunks * CHUNK, dtype=torch.int32),
                   sampling_params=SamplingParams(max_tokens=CHUNK), mm_embeds=None)
    ]
    if with_decode:
        # A running request mid-generation; _prepare_batch is stubbed, so it only needs
        # to satisfy the decode manager (can_decode == remain_len > 0).
        handle = cm.match_req(SimpleNamespace(
            input_ids=torch.tensor([7], dtype=torch.int32), input_len=1, mm_embeds=None,
        )).cuda_handle
        running = Req(input_ids=torch.tensor([7], dtype=torch.int32), table_idx=MAX_RUNNING,
                      cached_len=0, output_len=8, uid=99,
                      sampling_params=SamplingParams(), cache_handle=handle)
        dm.filter_reqs([running])
        assert dm.runnable
    stub = SimpleNamespace(
        prefill_manager=pm,
        decode_manager=dm,
        prefill_budget=CHUNK,
        config=SimpleNamespace(decode_interleave=decode_interleave),
        spec_drafter=None,
        _consecutive_prefills=0,
        _prepare_batch=lambda batch: batch,  # pass the batch through as the "forward input"
        _report_prompt_admissions=lambda batch: None,
    )
    return stub


def _phases(stub, n: int) -> list[str]:
    out = []
    for _ in range(n):
        batch = Scheduler._schedule_next_batch(stub)
        assert batch is not None
        out.append(batch.phase)
    return out


def test_interleave_0_keeps_strict_prefill_priority():
    stub = _stub_scheduler(decode_interleave=0, n_chunks=4, with_decode=True)
    assert _phases(stub, 4) == ["prefill"] * 4


def test_interleave_1_alternates_prefill_and_decode():
    stub = _stub_scheduler(decode_interleave=1, n_chunks=4, with_decode=True)
    assert _phases(stub, 6) == [
        "prefill", "decode", "prefill", "decode", "prefill", "decode",
    ]


def test_interleave_2_runs_a_decode_every_third_batch():
    stub = _stub_scheduler(decode_interleave=2, n_chunks=4, with_decode=True)
    assert _phases(stub, 6) == [
        "prefill", "prefill", "decode", "prefill", "prefill", "decode",
    ]


def test_no_waiting_decode_accrues_no_debt():
    # With nothing decoding, interleave must not delay prefill chunks.
    stub = _stub_scheduler(decode_interleave=1, n_chunks=4, with_decode=False)
    assert _phases(stub, 4) == ["prefill"] * 4
    assert stub._consecutive_prefills == 0


def test_decode_resets_the_debt_counter():
    stub = _stub_scheduler(decode_interleave=3, n_chunks=8, with_decode=True)
    phases = _phases(stub, 8)
    # Never more than 3 consecutive prefills while a decode is waiting.
    run = 0
    for p in phases:
        run = run + 1 if p == "prefill" else 0
        assert run <= 3
    assert "decode" in phases
