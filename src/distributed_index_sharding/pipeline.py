"""Punto de entrada programático único para "buscar contra un clúster ya
particionado": ata `cluster.py` (subprocesos de shard) + `coordinator.py`
(fan-out + merge) + `query_translation.py` (traducción de `ParsedQuery`)
detrás de una sola clase, el mismo papel que
`bm25_ranking_engine.pipeline.BM25RankingPipeline` cumple atando lector +
scorer + tokenizador en ese proyecto. El particionado en sí
(`partitioning.partition_index`) es una operación offline independiente, no
parte de este ciclo de vida — se ejecuta una vez, antes de arrancar ningún
shard.
"""

from __future__ import annotations

from pathlib import Path
from types import TracebackType
from typing import Any

from distributed_index_sharding.cluster import LocalShardCluster
from distributed_index_sharding.coordinator import SearchCoordinator
from distributed_index_sharding.models import FanOutResult, ShardTarget
from distributed_index_sharding.query_translation import from_parsed_query_dict, from_raw_text

_DEFAULT_TIMEOUT_SECONDS = 2.0


class DistributedSearchPipeline:
    """Gestor de contexto asíncrono: levanta un clúster local de shards a
    partir de un directorio ya particionado y expone búsqueda distribuida de
    principio a fin. `close`/`__aexit__` cierran el coordinador (su sesión
    HTTP) y detienen todos los subprocesos de shard, en ese orden."""

    def __init__(self, cluster: LocalShardCluster, coordinator: SearchCoordinator) -> None:
        self._cluster = cluster
        self._coordinator = coordinator

    @classmethod
    async def start(
        cls,
        partitioned_root: Path,
        *,
        host: str = "127.0.0.1",
        base_port: int = 9300,
        timeout_seconds: float = _DEFAULT_TIMEOUT_SECONDS,
    ) -> DistributedSearchPipeline:
        cluster = await LocalShardCluster.start(partitioned_root, host=host, base_port=base_port)
        coordinator = SearchCoordinator(cluster.targets, timeout_seconds=timeout_seconds)
        return cls(cluster, coordinator)

    @property
    def targets(self) -> tuple[ShardTarget, ...]:
        return self._cluster.targets

    async def __aenter__(self) -> DistributedSearchPipeline:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.close()

    async def close(self) -> None:
        await self._coordinator.close()
        await self._cluster.shutdown()

    async def search_parsed_query(
        self, parsed_query: dict[str, Any], *, top_k: int = 10
    ) -> FanOutResult:
        """Contrato principal: recibe `ParsedQuery.to_json_dict()` de
        `query-parser-autocomplete` (ver `query_translation.py`)."""
        request = from_parsed_query_dict(parsed_query, top_k=top_k)
        return await self._coordinator.search(request)

    async def search_text(self, text: str, *, top_k: int = 10) -> FanOutResult:
        """Conveniencia standalone: texto en crudo sin operadores, tokenizado
        internamente (ver `query_translation.from_raw_text`)."""
        request = from_raw_text(text, top_k=top_k)
        return await self._coordinator.search(request)
