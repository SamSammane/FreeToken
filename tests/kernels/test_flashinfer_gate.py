"""use_flashinfer_kernels: flashinfer may only be IMPORTED when CUDA is usable.

Recent flashinfer queries torch.cuda.get_device_properties at import time and raises
on CPU-only torch, so every kernel-selection site (norm, rope, activation, sampling)
gates on this probe and falls back to the triton/torch implementations. These tests
poison sys.modules['flashinfer'] so any gated-off import attempt fails loudly, then
construct the layers on this CUDA-less box.
"""
from __future__ import annotations

import sys

import pytest
import torch

from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.kernel import backend

if try_get_tp_info() is None:
    set_tp_info(rank=0, size=1)

pytestmark = pytest.mark.skipif(
    torch.cuda.is_available(), reason="exercises the CPU-only (CUDA-less) gate"
)


@pytest.fixture
def poisoned_flashinfer(monkeypatch):
    # "Installed" as far as the probe can tell, but any actual import blows up --
    # exactly what a flashinfer with import-time CUDA calls does on CPU-only torch.
    monkeypatch.setattr(backend, "is_flashinfer_installed", lambda: True)
    monkeypatch.setitem(sys.modules, "flashinfer", None)  # -> ImportError on import
    monkeypatch.setitem(sys.modules, "flashinfer.sampling", None)


def test_probe_is_false_without_cuda(poisoned_flashinfer):
    assert not backend.use_flashinfer_kernels()


def test_norm_and_rope_construct_without_importing_flashinfer(poisoned_flashinfer):
    from freetoken.layers.norm import RMSNorm, RMSNormFused
    from freetoken.layers.rotary import get_rope

    RMSNorm(64, eps=1e-6)
    RMSNormFused(64, eps=1e-6)
    get_rope(head_dim=64, rotary_dim=64, max_position=128, base=10000.0, rope_scaling=None)
