from __future__ import annotations

from dataclasses import dataclass, field

from distributed_index_sharding.coordinator import SearchCoordinator
from distributed_index_sharding.models import (
    ShardHit,
    ShardOutcome,
    ShardSearchRequest,
    ShardTarget,
)


@dataclass
class _FakeShardTransport:
    """Doble en memoria de `protocols.ShardTransport`: cada shard responde con
    un `ShardOutcome` precocinado, sin abrir ningún socket, para poder
    testear el fan-out y la degradación de forma determinista y rápida."""

    outcomes_by_shard_id: dict[int, ShardOutcome]
    calls: list[tuple[int, float]] = field(default_factory=list)

    async def search(
        self, target: ShardTarget, request: ShardSearchRequest, timeout_seconds: float
    ) -> ShardOutcome:
        self.calls.append((target.shard_id, timeout_seconds))
        return self.outcomes_by_shard_id[target.shard_id]


def _targets(*shard_ids: int) -> list[ShardTarget]:
    return [ShardTarget(shard_id=i, host="127.0.0.1", port=9300 + i) for i in shard_ids]


async def test_coordinator_merges_healthy_shards() -> None:
    transport = _FakeShardTransport(
        outcomes_by_shard_id={
            0: ShardOutcome(shard_id=0, status="ok", hits=(ShardHit(doc_id=0, score=9.0),)),
            1: ShardOutcome(shard_id=1, status="ok", hits=(ShardHit(doc_id=1, score=7.0),)),
        }
    )
    coordinator = SearchCoordinator(_targets(0, 1), transport=transport)

    result = await coordinator.search(ShardSearchRequest(query_terms=("python",), top_k=10))

    assert result.merged == (ShardHit(doc_id=0, score=9.0), ShardHit(doc_id=1, score=7.0))
    assert result.healthy_shard_ids == (0, 1)
    assert result.failed_shard_ids == ()


async def test_coordinator_degrades_on_shard_timeout_without_raising() -> None:
    transport = _FakeShardTransport(
        outcomes_by_shard_id={
            0: ShardOutcome(shard_id=0, status="ok", hits=(ShardHit(doc_id=0, score=9.0),)),
            1: ShardOutcome(shard_id=1, status="timeout", error_message="sin respuesta en 2.0s"),
        }
    )
    coordinator = SearchCoordinator(_targets(0, 1), transport=transport)

    result = await coordinator.search(ShardSearchRequest(query_terms=("python",), top_k=10))

    # La búsqueda entera no lanza, y el shard sano sigue apareciendo en el resultado.
    assert result.merged == (ShardHit(doc_id=0, score=9.0),)
    assert result.healthy_shard_ids == (0,)
    assert result.failed_shard_ids == (1,)


async def test_coordinator_all_shards_down_returns_empty_not_an_exception() -> None:
    transport = _FakeShardTransport(
        outcomes_by_shard_id={
            0: ShardOutcome(shard_id=0, status="error", error_message="conexión rechazada"),
            1: ShardOutcome(shard_id=1, status="timeout", error_message="sin respuesta"),
        }
    )
    coordinator = SearchCoordinator(_targets(0, 1), transport=transport)

    result = await coordinator.search(ShardSearchRequest(query_terms=("python",), top_k=10))

    assert result.merged == ()
    assert result.failed_shard_ids == (0, 1)


async def test_coordinator_propagates_timeout_seconds_to_transport() -> None:
    transport = _FakeShardTransport(
        outcomes_by_shard_id={0: ShardOutcome(shard_id=0, status="ok", hits=())}
    )
    coordinator = SearchCoordinator(_targets(0), transport=transport, timeout_seconds=0.5)

    await coordinator.search(ShardSearchRequest(query_terms=("python",), top_k=10))

    assert transport.calls == [(0, 0.5)]


async def test_coordinator_close_is_a_no_op_without_owned_session() -> None:
    transport = _FakeShardTransport(outcomes_by_shard_id={})
    coordinator = SearchCoordinator(_targets(), transport=transport)
    await coordinator.close()  # no debe lanzar aunque no haya sesión HTTP propia
