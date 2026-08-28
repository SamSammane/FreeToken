# Performance features

The engine's fast paths (overlap scheduling, CUDA graphs, chunked prefill, radix
prefix cache, double-buffered prefill streaming, the hybrid CPU/GPU MoE split) are on
by default. This page covers the opt-in features, when to reach for each, and how to
validate them on real hardware.

## Quick wins first

Before any flag below: install the `accel` extra (FlashInfer is the fast attention
path on consumer GPUs), run `ft bench bw` once per machine so `--moe-backend auto`
can upgrade to `hybrid`, and pre-convert checkpoints with `ft checkpoint`.

## Speculative decoding (`--speculative ngram`)

```bash
ft serve --model ... --speculative ngram --speculative-tokens 4
```

Prompt-lookup speculation: each request's last n-gram is matched against its own
prompt + generation history and the tokens that followed become the draft; one extend
forward verifies the whole draft and always commits at least what plain decode would.
Output is **bit-identical** to non-speculative greedy decoding.

- Best on coding-agent workloads (heavy copying from context) with MoE offload, where
  each accepted token amortizes the round's PCIe expert traffic.
- Applies to greedy requests on plain radix/naive KV caches (no SWA / GDN-hybrid /
  DSV4 models) with GPU MoE decode; unsupported combinations log a warning and run
  unspeculated. Implies non-overlap scheduling.

## Hot-expert pinning (`--moe-pin-hot 0.3`)

Protects up to the given fraction of the offload expert cache from LRU eviction,
tracking which experts stay hot across decode windows. Guards the recurring experts
against one-off eviction storms (topic shifts, prefill materializes). Pure
eviction-order bias: residency bookkeeping and both ensure kernels are untouched, so
the worst case degrades to plain LRU. Start at `0.25–0.35` and compare steady-state
decode miss rates (`--moe-collect-stats`).

## FP8 KV cache (`--kv-cache-dtype fp8_e4m3`)

Halves KV bytes per token (static scale 1.0, saturating cast at store). The freed
bytes become ~2x KV pages for the same budget, or more expert cache under
`--moe-cache-auto`. Plain MHA/GQA models with the `fi` attention backend only;
validated with a clear error at startup. Expect a small, usually negligible quality
cost — verify on your eval set before adopting.

## Decode/prefill interleave (`--decode-interleave 1`)

Bounds how long a long chunked prefill (or a queue of prefills) can stall every
running request's next token: after N consecutive prefill batches scheduled while
decodes wait, one decode batch runs. `1` alternates chunk/decode — recommended for
interactive multi-request serving; `0` (default) keeps strict prefill priority for
maximum prefill throughput.

## Validating on real hardware

Per the contributing bar, performance claims need A/B numbers — same model, prompt
and settings on `main` and on the feature branch:

1. **Correctness gates**: `pytest tests/ -m "not slow"` on the GPU box (the GPU-gated
   tests self-enable); one `e2e/test_aime.py` run per model family you serve.
2. **Speculative decoding**: greedy request, a long agentic transcript as the prompt;
   compare tokens/s and confirm byte-identical output with `--speculative none` vs
   `ngram`. Watch the acceptance-sensitive knobs: `--speculative-tokens 2..8`.
3. **Hot-expert pinning**: `--moe-collect-stats`; compare decode miss/active ratios
   and tokens/s over a >5-minute mixed workload, pinning off vs `0.25/0.35`.
4. **FP8 KV**: tokens/s + max context reached, plus a task-quality spot check
   (AIME/e2e) against bf16 KV.
5. **Interleave**: measure per-token latency percentiles of a running decode while a
   long prompt prefills concurrently, `0` vs `1`.

## Deferred by design: cross-layer expert prefetch

Prefetching layer k+1's experts while layer k computes was evaluated and deliberately
NOT implemented blind. The miss fetch is already a device-side gather kernel on the
compute stream, so same-stream prefetch buys nothing; a second stream requires making
in-flight warm copies un-evictable to the concurrently running LRU kernels, and the
only race-free barrier placement erases the overlap window. A correct version needs
dual-stream capture with per-layer ready events and an eviction guard inside the
ensure kernels — build and validate it with a GPU in hand, starting from the
`expert_recency` tracking the hybrid path already maintains.
