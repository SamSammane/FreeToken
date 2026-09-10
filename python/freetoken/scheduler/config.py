from __future__ import annotations

from dataclasses import dataclass, field

from freetoken.engine import EngineConfig


def _get_pid_suffix() -> str:
    import os

    return f".pid={os.getpid()}"


@dataclass(frozen=True)
class SchedulerConfig(EngineConfig):
    max_extend_tokens: int = 8192
    cache_type: str = "radix"
    offline_mode: bool = False
    decode_log_interval: int = 40
    special_token_ckpt: bool = False
    # After this many consecutive prefill batches scheduled while decode requests are
    # waiting, run one decode batch before the next prefill (chunk). Bounds how long a
    # long / queued chunked prefill can stall running decodes. 0 = strict prefill
    # priority (the historical behavior).
    decode_interleave: int = 0
    # Speculative decoding: "none" or "ngram" (prompt lookup; see freetoken/spec/ngram.py).
    # Greedy requests only, plain radix/naive caches only (no SWA / hybrid-GDN / DSV4
    # window pools), and implies non-overlap scheduling; unsupported combinations
    # disable it with a warning. Output is bit-identical to plain greedy decoding.
    speculative: str = "none"
    # Max draft tokens proposed per request per verify round (--speculative ngram).
    speculative_tokens: int = 4
    # N-gram match lengths the drafter tries, longest first.
    speculative_min_match: int = 2
    speculative_max_match: int = 4

    # networking config
    _unique_suffix: str = field(default_factory=_get_pid_suffix)

    @property
    def zmq_backend_addr(self) -> str:
        return "ipc:///tmp/freetoken_0" + self._unique_suffix

    @property
    def zmq_detokenizer_addr(self) -> str:
        return "ipc:///tmp/freetoken_1" + self._unique_suffix

    @property
    def zmq_scheduler_broadcast_addr(self) -> str:
        return "ipc:///tmp/freetoken_2" + self._unique_suffix

    @property
    def max_forward_len(self) -> int:
        return self.max_extend_tokens

    @property
    def backend_create_detokenizer_link(self) -> bool:
        return True
