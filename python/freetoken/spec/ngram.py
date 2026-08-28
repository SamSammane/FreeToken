"""N-gram (prompt-lookup) speculative decoding (--speculative ngram).

Drafting needs no draft model: the last ``m`` generated tokens are matched against the
request's own history (prompt + generation, one CPU tensor), and the tokens that
followed the most recent earlier occurrence are proposed as the draft. Coding-agent
workloads copy heavily from context (paths, identifiers, quoted code), so acceptance
is high exactly where FreeToken runs; every accepted token amortizes the expert
traffic of one MoE forward across more output.

Verification is greedy-exact: the engine runs one extend forward over
[last committed token, draft...] and returns the argmax at every position; a draft
token is accepted while it equals the model's argmax, and the first mismatch (or the
position after the last accepted draft) contributes the model's own token, so a round
always commits at least the one token plain decode would have. Output is therefore
bit-identical to non-speculative greedy decoding.
"""
from __future__ import annotations

from typing import List, Sequence

import torch


class NgramDrafter:
    """Propose draft tokens by prompt lookup over the request's token history."""

    def __init__(self, max_tokens: int = 4, min_match: int = 2, max_match: int = 4):
        assert 1 <= min_match <= max_match, (min_match, max_match)
        assert max_tokens >= 1, max_tokens
        self.max_tokens = max_tokens
        self.min_match = min_match
        self.max_match = max_match

    def propose(self, ids: torch.Tensor, max_tokens: int | None = None) -> List[int]:
        """Draft up to ``max_tokens`` continuation tokens for ``ids`` (1-D CPU tensor,
        prompt + generated so far). Longest match wins; among equal-length matches the
        most recent occurrence wins. Returns [] when no history n-gram recurs."""
        budget = self.max_tokens if max_tokens is None else min(max_tokens, self.max_tokens)
        n = int(ids.numel())
        if budget <= 0 or n < self.min_match + 1:
            return []
        flat = ids.view(-1)
        for m in range(min(self.max_match, n - 1), self.min_match - 1, -1):
            pattern = flat[n - m :]
            # Windows over ids[:-1]: candidate matches end strictly before the final
            # position, so the trailing pattern can never match itself.
            windows = flat[: n - 1].unfold(0, m, 1)
            hits = (windows == pattern).all(dim=1).nonzero(as_tuple=False)
            if hits.numel() == 0:
                continue
            start = int(hits[-1]) + m  # continuation begins after the matched n-gram
            draft = flat[start : min(start + budget, n)]
            if draft.numel():
                return draft.tolist()
        return []


def verify_greedy(draft: Sequence[int], target: Sequence[int]) -> List[int]:
    """Commit tokens from a greedy verify round.

    ``target`` holds the model's argmax at each verified position: ``target[i]`` is
    the model's next token after committing ``draft[:i]``; ``len(target)`` must be
    ``len(draft) + 1``. Every round commits ``target[i]`` up to and including the
    first position where the draft diverges (or the bonus position after a fully
    accepted draft) -- at least one token, matching plain greedy decode exactly.
    """
    assert len(target) == len(draft) + 1, (len(draft), len(target))
    committed: List[int] = []
    for i, tok in enumerate(target):
        committed.append(int(tok))
        if i == len(draft) or int(draft[i]) != int(tok):
            break
    return committed
