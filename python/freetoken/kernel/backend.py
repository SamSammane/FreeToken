"""Availability probes for the optional native kernel packages.

When flashinfer / sgl_kernel are installed the call-sites use their fused CUDA
ops; otherwise they fall back to the pure-Triton kernels in
``freetoken.kernel.triton``. ``find_spec`` only checks that the package is
importable (no import side effects), and the result is cached.
"""
from __future__ import annotations

import functools
import importlib.util


def _importable(name: str) -> bool:
    # find_spec normally returns None when a package is absent, but it can raise
    # (broken parent package, or a meta_path finder that blocks the name); treat
    # any failure as "not available" so callers cleanly fall back to triton.
    try:
        return importlib.util.find_spec(name) is not None
    except Exception:
        return False


@functools.cache
def is_flashinfer_installed() -> bool:
    return _importable("flashinfer")


def use_flashinfer_kernels() -> bool:
    """Gate for kernel-selection sites that actually IMPORT flashinfer (norm, rope,
    activation, sampling): installed AND CUDA usable. Recent flashinfer queries device
    properties at import time and raises on CPU-only torch, so on a GPU-less box the
    triton/torch fallbacks must win even when the package is present. Capability
    probing (backend auto-selection) keeps using is_flashinfer_installed -- it never
    imports the package. Not cached: torch.cuda.is_available() is."""
    import torch

    return is_flashinfer_installed() and torch.cuda.is_available()


@functools.cache
def is_sgl_kernel_installed() -> bool:
    return _importable("sgl_kernel")


@functools.cache
def is_triton_kernels_installed() -> bool:
    """OpenAI's ``triton_kernels`` (the fused MoE router used by ``moe.fused.fused_topk``).

    Distinct from the ``triton`` runtime we always depend on: it ships with the Triton
    source tree and has no Windows wheel. It is also not one of the six ops
    ``freetoken.kernel.triton`` reimplements, so its call-site carries its own fallback.
    """
    return _importable("triton_kernels")


@functools.cache
def driver_cuda_version() -> int | None:
    """Max CUDA version the installed NVIDIA driver supports (``13000`` == CUDA 13.0),
    or None if undetermined. Driver-JIT kernels (PTX compiled at runtime, e.g.
    flashinfer's CuTe-DSL paths) are gated by this, not by any package's build-time
    toolkit version. Resolved through the ``_pinned_tensor`` extension's link-time
    cudart, so it works wherever the extension builds (including Windows) -- no dlopen
    by soname."""
    try:
        from freetoken.kernel.pinned import _load_pinned_extension

        version = int(_load_pinned_extension().driver_cuda_version())
    except Exception:
        return None
    return version or None  # 0 == no driver installed
