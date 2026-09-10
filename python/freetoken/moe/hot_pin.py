"""Hot-expert pinning for the offload MoE slot cache (--moe-pin-hot).

Expert activation under real decode workloads is heavily skewed: a small set of experts
per layer serves most tokens. Plain timestamp-LRU keeps them resident in the steady
state, but a burst of one-off experts (a topic shift, a rare-expert stretch, a prefill
materialize) can evict them, and each re-fetch is a full expert row over PCIe.

Pinning protects the persistently hot residents WITHOUT touching either ensure kernel:
the eviction victim in both the flashlib slot-cache kernel and the hybrid kernel is
``argmin(usage)``, so a slot whose ``usage`` is refreshed to the current step can never
be chosen while any colder slot exists. The cache captures one fixed-shape refresh op
per decode forward (``OffloadMoeCache._refresh_pinned``) that rewrites
``usage[pinned] = step`` from a persistent boolean mask; this module is the host-side
policy that decides the mask's content, off the hot path, every ``sample_interval``
decode steps.

Policy: each sample reads ``id_of_slot``/``usage`` back and scores every resident
expert that was touched within the last ``recent_window`` steps; scores decay by
``ema`` per sample, so an expert must stay hot across consecutive samples (score >=
``min_score``) to be pinned. At most ``max_pin_fraction * cache_size`` slots are ever
pinned, so the LRU always keeps working room. Pinning is purely advisory: it only
biases eviction order, never residency bookkeeping, so a stale mask (e.g. right after
a prefill materialize remapped slots) degrades to plain LRU, not to corruption.
"""
from __future__ import annotations

import torch


class HotExpertPinner:
    def __init__(
        self,
        cache_size: int,
        num_layers: int,
        num_experts: int,
        max_pin_fraction: float,
        recent_window: int = 64,
        ema: float = 0.5,
        min_score: float = 1.4,
    ):
        assert 0.0 < max_pin_fraction <= 0.9, max_pin_fraction
        self.cache_size = cache_size
        self.max_pins = max(1, int(max_pin_fraction * cache_size))
        self.recent_window = recent_window
        self.ema = ema
        # An expert scores +1 per sample it was recently active in; with ema 0.5 a score
        # >= 1.4 means "hot in this sample AND at least ~the previous one".
        self.min_score = min_score
        # Host-side hotness score per flat expert id (layer * num_experts + expert).
        self.score = torch.zeros(num_layers * num_experts, dtype=torch.float32)

    def update(
        self, id_of_slot: torch.Tensor, usage: torch.Tensor, step: int
    ) -> torch.Tensor:
        """One sample: fold recent activity into the scores and return the new pin mask.

        ``id_of_slot`` / ``usage`` are host (CPU) snapshots of the cache's bookkeeping;
        returns a host bool mask over slots (True = protect from eviction).
        """
        resident = id_of_slot >= 0
        recent = resident & (usage >= step - self.recent_window)
        self.score *= self.ema
        self.score[id_of_slot[recent].long()] += 1.0

        # Score per *slot* through the current residency map; non-resident slots and
        # residents below the persistence threshold never pin.
        slot_score = torch.where(
            resident,
            self.score[id_of_slot.clamp(min=0).long()],
            torch.zeros((), dtype=torch.float32),
        )
        slot_score = torch.where(slot_score >= self.min_score, slot_score, torch.zeros(()))
        mask = torch.zeros(self.cache_size, dtype=torch.bool)
        eligible = int((slot_score > 0).sum())
        k = min(self.max_pins, eligible)
        if k > 0:
            mask[torch.topk(slot_score, k).indices] = True
        return mask

    def reset(self) -> None:
        self.score.zero_()
