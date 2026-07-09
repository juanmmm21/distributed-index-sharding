from __future__ import annotations

import pytest

from distributed_index_sharding.merge import merge_shard_hits, merge_shard_outcomes
from distributed_index_sharding.models import ShardHit, ShardOutcome


def test_merge_two_disjoint_sorted_lists() -> None:
    shard_a = [ShardHit(doc_id=0, score=9.0), ShardHit(doc_id=3, score=4.0)]
    shard_b = [ShardHit(doc_id=1, score=7.0), ShardHit(doc_id=4, score=1.0)]

    merged = merge_shard_hits([shard_a, shard_b], top_k=10)

    assert merged == [
        ShardHit(doc_id=0, score=9.0),
        ShardHit(doc_id=1, score=7.0),
        ShardHit(doc_id=3, score=4.0),
        ShardHit(doc_id=4, score=1.0),
    ]


def test_merge_truncates_to_top_k() -> None:
    shard_a = [ShardHit(doc_id=0, score=9.0), ShardHit(doc_id=3, score=4.0)]
    shard_b = [ShardHit(doc_id=1, score=7.0), ShardHit(doc_id=4, score=1.0)]

    merged = merge_shard_hits([shard_a, shard_b], top_k=2)

    assert merged == [ShardHit(doc_id=0, score=9.0), ShardHit(doc_id=1, score=7.0)]


def test_merge_tie_break_is_ascending_doc_id() -> None:
    shard_a = [ShardHit(doc_id=5, score=3.0)]
    shard_b = [ShardHit(doc_id=2, score=3.0)]

    merged = merge_shard_hits([shard_a, shard_b], top_k=10)

    assert merged == [ShardHit(doc_id=2, score=3.0), ShardHit(doc_id=5, score=3.0)]


def test_merge_handles_empty_shard_list() -> None:
    assert merge_shard_hits([[], []], top_k=5) == []


def test_merge_rejects_non_positive_top_k() -> None:
    with pytest.raises(ValueError, match="top_k"):
        merge_shard_hits([[]], top_k=0)


def test_merge_shard_outcomes_ignores_unhealthy_shards() -> None:
    outcomes = [
        ShardOutcome(shard_id=0, status="ok", hits=(ShardHit(doc_id=0, score=5.0),)),
        ShardOutcome(shard_id=1, status="timeout", hits=(), error_message="sin respuesta"),
        ShardOutcome(shard_id=2, status="ok", hits=(ShardHit(doc_id=2, score=8.0),)),
    ]

    merged = merge_shard_outcomes(outcomes, top_k=10)

    assert merged == [ShardHit(doc_id=2, score=8.0), ShardHit(doc_id=0, score=5.0)]


def test_merge_shard_outcomes_all_failed_returns_empty() -> None:
    outcomes = [
        ShardOutcome(shard_id=0, status="timeout", error_message="sin respuesta"),
        ShardOutcome(shard_id=1, status="error", error_message="conexión rechazada"),
    ]

    assert merge_shard_outcomes(outcomes, top_k=10) == []
