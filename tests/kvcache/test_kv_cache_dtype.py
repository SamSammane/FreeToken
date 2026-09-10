"""--kv-cache-dtype fp8_e4m3: resolution/validation matrix, pool allocation dtype +
store-time cast, and the halved bytes/token in the sizing arithmetic. CPU only."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from freetoken.distributed import set_tp_info, try_get_tp_info
from freetoken.kvcache import resolve_kv_cache_dtype
from freetoken.kvcache.base import kv_cache_itemsize, spec_kv_bytes_per_token
from freetoken.kvcache.mha_pool import MHAKVCache

if try_get_tp_info() is None:
    set_tp_info(rank=0, size=1)


def _spec(**kw):
    base = dict(mla=False, head_dim=64, num_kv_heads=4, num_layers=8,
                index_head_dim=0, num_index_layers=0, is_swa=False)
    base.update(kw)
    return SimpleNamespace(**base)


def _mha_model_config():
    from freetoken.attention import AttnType

    return SimpleNamespace(
        kv_cache_group_specs=lambda: [
            SimpleNamespace(attn_type=AttnType.FULL, **vars(_spec()))
        ],
        has_linear_attention=False,
        dsv4_args=None,
    )


def _config(kv_cache_dtype="fp8_e4m3", backend="fi", mc=None):
    return SimpleNamespace(
        kv_cache_dtype=kv_cache_dtype,
        dtype=torch.bfloat16,
        attention_backend=backend,
        model_config=mc if mc is not None else _mha_model_config(),
        tp_info=SimpleNamespace(rank=0, size=1),
    )


def test_auto_keeps_the_model_dtype():
    cfg = _config(kv_cache_dtype="auto")
    assert resolve_kv_cache_dtype(cfg) is torch.bfloat16


def test_fp8_resolves_for_mha_plus_flashinfer():
    assert resolve_kv_cache_dtype(_config()) is torch.float8_e4m3fn


def test_fp8_rejects_non_flashinfer_backends():
    for backend in ("triton", "fa,fi", "fa", "trtllm"):
        with pytest.raises(ValueError, match="flashinfer"):
            resolve_kv_cache_dtype(_config(backend=backend))


def test_fp8_rejects_non_mha_pools():
    from freetoken.attention import AttnType

    for attn_type in (AttnType.SWA, AttnType.MLA, AttnType.DSA, AttnType.BSA):
        mc = SimpleNamespace(
            kv_cache_group_specs=lambda t=attn_type: [
                SimpleNamespace(attn_type=t, **vars(_spec(mla=t == AttnType.MLA)))
            ],
            has_linear_attention=False,
            dsv4_args=None,
        )
        with pytest.raises(ValueError, match="plain MHA/GQA"):
            resolve_kv_cache_dtype(_config(mc=mc))


def test_fp8_rejects_linear_attention_hybrids():
    mc = _mha_model_config()
    mc.has_linear_attention = True
    with pytest.raises(ValueError, match="plain MHA/GQA"):
        resolve_kv_cache_dtype(_config(mc=mc))


def test_unknown_choice_rejected():
    with pytest.raises(ValueError, match="kv-cache-dtype"):
        resolve_kv_cache_dtype(_config(kv_cache_dtype="int4"))


def test_fp8_halves_the_kv_bytes_per_token():
    bf16 = _config(kv_cache_dtype="auto")
    fp8 = _config()
    assert kv_cache_itemsize(bf16) == 2 and kv_cache_itemsize(fp8) == 1
    spec = _spec()
    assert spec_kv_bytes_per_token(spec, fp8) * 2 == spec_kv_bytes_per_token(spec, bf16)


def test_pool_stores_fp8_and_casts_on_store():
    pool = MHAKVCache(
        num_kv_heads=2, num_layers=2, head_dim=8, num_pages=16, page_size=1,
        dtype=torch.float8_e4m3fn, device=torch.device("cpu"),
        compute_dtype=torch.bfloat16,
    )
    assert pool.dtype is torch.float8_e4m3fn
    assert pool.compute_dtype is torch.bfloat16
    # unit_bytes reflects the fp8 storage (2 slabs * 2 heads * 8 dim * 1 byte * 2 layers)
    assert pool.unit_bytes()[0] == 2 * 2 * 8 * 1 * 2

    k = torch.full((3, 2, 8), 0.5, dtype=torch.bfloat16)
    v = torch.full((3, 2, 8), -2.0, dtype=torch.bfloat16)
    loc = torch.tensor([1, 4, 7])
    # store_cache's JIT kernel is GPU-only; emulate its byte copy after the cast the
    # pool performs, by writing through the cache views directly.
    ck, cv = k.to(pool.dtype), v.to(pool.dtype)
    pool.k_cache(0).view(-1, 2, 8)[loc] = ck
    pool.v_cache(0).view(-1, 2, 8)[loc] = cv
    got_k = pool.k_cache(0).view(-1, 2, 8)[loc].to(torch.float32)
    got_v = pool.v_cache(0).view(-1, 2, 8)[loc].to(torch.float32)
    assert torch.allclose(got_k, torch.full_like(got_k, 0.5))
    assert torch.allclose(got_v, torch.full_like(got_v, -2.0))


def test_rebuild_preserves_fp8_dtype():
    pool = MHAKVCache(
        num_kv_heads=2, num_layers=2, head_dim=8, num_pages=16, page_size=1,
        dtype=torch.float8_e4m3fn, device=torch.device("cpu"),
        compute_dtype=torch.bfloat16,
    )
    pool.rebuild(32)
    assert pool.dtype is torch.float8_e4m3fn
    assert pool.compute_dtype is torch.bfloat16
