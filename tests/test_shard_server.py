from __future__ import annotations

from pathlib import Path

from aiohttp.test_utils import TestClient, TestServer

from distributed_index_sharding.shard_server import create_app


async def test_health_endpoint(tiny_index_dir: Path) -> None:
    app = create_app(shard_id=0, shard_dir=tiny_index_dir)
    async with TestClient(TestServer(app)) as client:
        response = await client.get("/health")
        assert response.status == 200
        assert await response.json() == {"shard_id": 0, "status": "ok"}


async def test_search_endpoint_returns_scored_hits(tiny_index_dir: Path) -> None:
    app = create_app(shard_id=0, shard_dir=tiny_index_dir)
    async with TestClient(TestServer(app)) as client:
        response = await client.post("/search", json={"query_terms": ["python"], "top_k": 10})
        assert response.status == 200
        payload = await response.json()
        assert payload["shard_id"] == 0
        assert {hit["doc_id"] for hit in payload["hits"]} == {0, 1, 3}


async def test_search_endpoint_applies_excluded_terms(tiny_index_dir: Path) -> None:
    app = create_app(shard_id=0, shard_dir=tiny_index_dir)
    async with TestClient(TestServer(app)) as client:
        response = await client.post(
            "/search",
            json={"query_terms": ["python"], "excluded_terms": ["tutorial"], "top_k": 10},
        )
        payload = await response.json()
        # doc_id=0 ("Python Tutorial for Beginners") tiene "python" y "tutorial" a
        # la vez, así que lo excluye "tutorial"; los docs 1 y 3 no mencionan "tutorial".
        assert {hit["doc_id"] for hit in payload["hits"]} == {1, 3}


async def test_search_endpoint_applies_site_filter(tiny_index_dir: Path) -> None:
    app = create_app(shard_id=0, shard_dir=tiny_index_dir)
    async with TestClient(TestServer(app)) as client:
        response = await client.post(
            "/search",
            json={
                "query_terms": ["python"],
                "filters": [{"field": "site", "value": "nonexistent.example"}],
                "top_k": 10,
            },
        )
        payload = await response.json()
        assert payload["hits"] == []


async def test_search_endpoint_rejects_malformed_body(tiny_index_dir: Path) -> None:
    app = create_app(shard_id=0, shard_dir=tiny_index_dir)
    async with TestClient(TestServer(app)) as client:
        response = await client.post("/search", json={"not_query_terms": True})
        assert response.status == 400
