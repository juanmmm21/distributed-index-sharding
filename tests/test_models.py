from __future__ import annotations

import pytest

from distributed_index_sharding.models import (
    QueryFilter,
    ShardHit,
    ShardSearchRequest,
    ShardSearchResponse,
    ShardTarget,
)


def test_shard_target_base_url() -> None:
    target = ShardTarget(shard_id=0, host="127.0.0.1", port=9300)
    assert target.base_url == "http://127.0.0.1:9300"


def test_shard_search_request_round_trip() -> None:
    request = ShardSearchRequest(
        query_terms=("python", "tutorial"),
        excluded_terms=("deprecated",),
        filters=(QueryFilter(field="site", value="example.com"),),
        top_k=5,
    )
    restored = ShardSearchRequest.from_json_dict(request.to_json_dict())
    assert restored == request


def test_shard_search_request_defaults() -> None:
    request = ShardSearchRequest.from_json_dict({"query_terms": ["python"], "top_k": 3})
    assert request.excluded_terms == ()
    assert request.filters == ()


def test_shard_search_request_rejects_non_positive_top_k() -> None:
    with pytest.raises(ValueError, match="top_k"):
        ShardSearchRequest(query_terms=("python",), top_k=0)


def test_shard_hit_round_trip() -> None:
    hit = ShardHit(doc_id=4, score=8.5)
    assert ShardHit.from_json_dict(hit.to_json_dict()) == hit


def test_shard_search_response_round_trip() -> None:
    response = ShardSearchResponse(shard_id=1, hits=(ShardHit(doc_id=0, score=1.0),))
    restored = ShardSearchResponse.from_json_dict(response.to_json_dict())
    assert restored == response
