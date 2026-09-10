"""NgramDrafter proposal rules and greedy verification commits."""
from __future__ import annotations

import torch

from freetoken.spec import NgramDrafter, verify_greedy


def _ids(*toks: int) -> torch.Tensor:
    return torch.tensor(toks, dtype=torch.int32)


def test_proposes_continuation_of_a_repeated_ngram():
    d = NgramDrafter(max_tokens=3, min_match=2, max_match=3)
    # tail [1,2] recurred earlier, followed by [3,4,5]
    assert d.propose(_ids(1, 2, 3, 4, 5, 9, 1, 2)) == [3, 4, 5]


def test_no_match_proposes_nothing():
    d = NgramDrafter(max_tokens=3, min_match=2, max_match=3)
    assert d.propose(_ids(1, 2, 3, 4, 5, 6, 7)) == []
    assert d.propose(_ids(1, 2)) == []  # too short to have history


def test_longest_match_wins_over_shorter():
    d = NgramDrafter(max_tokens=2, min_match=2, max_match=3)
    # tail [7,1,2]: 3-gram occurs at position 2 (-> continuation [30, ...]);
    # the shorter 2-gram [1,2] also occurs at 0 (-> [7, ...]). Longest must win.
    ids = _ids(1, 2, 7, 1, 2, 30, 31, 9, 7, 1, 2)
    assert d.propose(ids) == [30, 31]


def test_most_recent_occurrence_wins():
    d = NgramDrafter(max_tokens=1, min_match=2, max_match=2)
    # [1,2] occurs twice; the later occurrence is followed by 50, the earlier by 40.
    assert d.propose(_ids(1, 2, 40, 1, 2, 50, 9, 1, 2)) == [50]


def test_budget_caps_the_draft():
    d = NgramDrafter(max_tokens=8, min_match=2, max_match=2)
    ids = _ids(1, 2, 3, 4, 5, 6, 9, 1, 2)
    assert d.propose(ids, max_tokens=2) == [3, 4]
    assert d.propose(ids, max_tokens=0) == []
    # continuation is clipped at the end of history
    assert d.propose(_ids(1, 2, 3, 1, 2)) == [3, 1, 2][: d.max_tokens]


def test_trailing_pattern_never_matches_itself():
    d = NgramDrafter(max_tokens=3, min_match=2, max_match=4)
    # the only occurrence of the tail 2-gram is the tail itself
    assert d.propose(_ids(9, 8, 1, 2)) == []


def test_verify_full_accept_commits_draft_plus_bonus():
    assert verify_greedy([5, 6, 7], [5, 6, 7, 8]) == [5, 6, 7, 8]


def test_verify_partial_accept_stops_at_first_mismatch():
    # target[1] != draft[1]: commit the accepted 5 and the model's own 9.
    assert verify_greedy([5, 6, 7], [5, 9, 42, 43]) == [5, 9]


def test_verify_immediate_mismatch_still_commits_one_token():
    assert verify_greedy([5, 6], [7, 1, 2]) == [7]


def test_verify_empty_draft_is_plain_decode():
    assert verify_greedy([], [3]) == [3]
