"""Fan-out asíncrono de una búsqueda a todos los shards de un clúster,
seguido de la fusión (`merge.py`) de las respuestas sanas.

`asyncio.gather` (sin `return_exceptions=True`) es seguro aquí porque
`protocols.ShardTransport.search` tiene la garantía documentada de no lanzar
nunca — cualquier fallo de un shard concreto ya llega como un `ShardOutcome`
con `status` distinto de `"ok"`, no como una excepción que `gather` tendría
que propagar o silenciar.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from types import TracebackType
from typing import Final

import aiohttp

from distributed_index_sharding.merge import merge_shard_outcomes
from distributed_index_sharding.models import FanOutResult, ShardSearchRequest, ShardTarget
from distributed_index_sharding.protocols import ShardTransport
from distributed_index_sharding.shard_client import HttpShardTransport

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_SECONDS: Final[float] = 2.0


class SearchCoordinator:
    """Punto de entrada programático del fan-out: gestor de contexto asíncrono
    que cierra su propia `aiohttp.ClientSession` al salir (`../CLAUDE.md`,
    sección 2.E) — salvo que se le inyecte un `transport` propio (tests con
    un doble en memoria), en cuyo caso no hay sesión de red que gestionar."""

    def __init__(
        self,
        targets: Sequence[ShardTarget],
        *,
        transport: ShardTransport | None = None,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self._targets = tuple(targets)
        self._timeout_seconds = timeout_seconds
        if transport is not None:
            self._transport: ShardTransport = transport
            self._owned_session: aiohttp.ClientSession | None = None
        else:
            session = aiohttp.ClientSession()
            self._transport = HttpShardTransport(session=session)
            self._owned_session = session

    async def __aenter__(self) -> SearchCoordinator:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()

    async def close(self) -> None:
        if self._owned_session is not None:
            await self._owned_session.close()

    async def search(self, request: ShardSearchRequest) -> FanOutResult:
        outcomes = await asyncio.gather(
            *(
                self._transport.search(target, request, self._timeout_seconds)
                for target in self._targets
            )
        )
        for outcome in outcomes:
            if outcome.status != "ok":
                logger.warning(
                    "shard %s degradado (%s): %s",
                    outcome.shard_id,
                    outcome.status,
                    outcome.error_message,
                )
        merged = merge_shard_outcomes(outcomes, top_k=request.top_k)
        return FanOutResult(outcomes=tuple(outcomes), merged=tuple(merged))
