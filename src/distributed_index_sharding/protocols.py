"""Interfaz que desacopla el fan-out (`coordinator.py`) de cómo se habla
físicamente con un shard, para poder testear la degradación ante timeout/error
y la lógica de merge con un transporte falso en memoria, sin levantar sockets
reales — el mismo desacoplamiento que `bm25_ranking_engine.protocols.IndexReader`
aplica entre el scorer y la lectura física del índice.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from distributed_index_sharding.models import ShardOutcome, ShardSearchRequest, ShardTarget


@runtime_checkable
class ShardTransport(Protocol):
    """Envía una petición de búsqueda a un shard concreto y nunca lanza: toda
    condición de fallo (timeout, conexión rechazada, respuesta malformada) se
    traduce a un `ShardOutcome` con `status` distinto de `"ok"`."""

    async def search(
        self, target: ShardTarget, request: ShardSearchRequest, timeout_seconds: float
    ) -> ShardOutcome: ...
