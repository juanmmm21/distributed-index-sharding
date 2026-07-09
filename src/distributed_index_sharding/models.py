"""Tipos de datos del dominio para distributed-index-sharding.

`ShardSearchRequest`/`ShardHit` son el contrato JSON que viaja por HTTP entre
el coordinador y cada shard (ver `shard_server.py`/`shard_client.py`) — no se
reutiliza `bm25_ranking_engine.models.ScoredDocument` para ese contrato porque
cruza un límite de proceso real (cada shard es un proceso HTTP independiente,
ver `../AGENTS.md`), así que necesita una forma serializable propia, igual que
`bm25-ranking-engine` define su propia copia mínima de los conceptos de
`inverted-index-builder` en vez de importar su paquete.

`QueryFilter` es una copia mínima y propia de la forma de
`query_parser_autocomplete.models.Filter` (mismo `{field, value}`), por el
mismo motivo: la integración entre subproyectos es siempre vía el JSON
documentado, nunca un import cruzado del paquete Python de otro repo
(`../AGENTS.md`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ShardStatus = Literal["ok", "timeout", "error"]


@dataclass(frozen=True, slots=True)
class ShardTarget:
    """Dirección de red de un shard: quién es y dónde escucha."""

    shard_id: int
    host: str
    port: int

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass(frozen=True, slots=True)
class QueryFilter:
    """Copia mínima de `query_parser_autocomplete.models.Filter`: un par
    `field`/`value` sin semántica propia hasta que un shard decide cómo
    resolverlo (ver `shard_server.py`, sección de filtros soportados)."""

    field: str
    value: str

    def to_json_dict(self) -> dict[str, str]:
        return {"field": self.field, "value": self.value}

    @staticmethod
    def from_json_dict(data: dict[str, Any]) -> QueryFilter:
        return QueryFilter(field=str(data["field"]), value=str(data["value"]))


@dataclass(frozen=True, slots=True)
class ShardSearchRequest:
    """Cuerpo JSON de `POST /search` contra un shard.

    `excluded_terms` y `filters` viajan hasta el shard (en vez de resolverse
    solo en el coordinador) para que el post-filtro se aplique donde vive el
    índice, sin tener que devolver primero el resultado sin filtrar por la
    red — ver `shard_server.py`.
    """

    query_terms: tuple[str, ...]
    excluded_terms: tuple[str, ...] = field(default_factory=tuple)
    filters: tuple[QueryFilter, ...] = field(default_factory=tuple)
    top_k: int = 10

    def __post_init__(self) -> None:
        if self.top_k <= 0:
            raise ValueError(f"top_k debe ser positivo, recibido {self.top_k}")

    def to_json_dict(self) -> dict[str, Any]:
        return {
            "query_terms": list(self.query_terms),
            "excluded_terms": list(self.excluded_terms),
            "filters": [f.to_json_dict() for f in self.filters],
            "top_k": self.top_k,
        }

    @staticmethod
    def from_json_dict(data: dict[str, Any]) -> ShardSearchRequest:
        return ShardSearchRequest(
            query_terms=tuple(str(t) for t in data["query_terms"]),
            excluded_terms=tuple(str(t) for t in data.get("excluded_terms", [])),
            filters=tuple(QueryFilter.from_json_dict(f) for f in data.get("filters", [])),
            top_k=int(data.get("top_k", 10)),
        )


@dataclass(frozen=True, slots=True)
class ShardHit:
    """Un documento candidato y su score, ya resuelto por un shard concreto."""

    doc_id: int
    score: float

    def to_json_dict(self) -> dict[str, Any]:
        return {"doc_id": self.doc_id, "score": self.score}

    @staticmethod
    def from_json_dict(data: dict[str, Any]) -> ShardHit:
        return ShardHit(doc_id=int(data["doc_id"]), score=float(data["score"]))


@dataclass(frozen=True, slots=True)
class ShardSearchResponse:
    """Cuerpo JSON de la respuesta de `POST /search` de un shard sano."""

    shard_id: int
    hits: tuple[ShardHit, ...]

    def to_json_dict(self) -> dict[str, Any]:
        return {"shard_id": self.shard_id, "hits": [h.to_json_dict() for h in self.hits]}

    @staticmethod
    def from_json_dict(data: dict[str, Any]) -> ShardSearchResponse:
        return ShardSearchResponse(
            shard_id=int(data["shard_id"]),
            hits=tuple(ShardHit.from_json_dict(h) for h in data["hits"]),
        )


@dataclass(frozen=True, slots=True)
class ShardOutcome:
    """Resultado del fan-out contra un shard concreto, sano o no.

    `status` distingue explícitamente un shard que respondió (`ok`) de uno
    que no lo hizo a tiempo (`timeout`) o que falló por cualquier otra razón
    de red/protocolo (`error`) — la degradación explícita que exige
    `../AGENTS.md`: ninguno de los dos últimos casos lanza una excepción que
    tumbe la búsqueda completa, ambos se representan como datos.
    """

    shard_id: int
    status: ShardStatus
    hits: tuple[ShardHit, ...] = field(default_factory=tuple)
    error_message: str | None = None
    latency_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class FanOutResult:
    """Resultado completo de una búsqueda distribuida: qué respondió cada
    shard y la lista ya fusionada."""

    outcomes: tuple[ShardOutcome, ...]
    merged: tuple[ShardHit, ...]

    @property
    def healthy_shard_ids(self) -> tuple[int, ...]:
        return tuple(o.shard_id for o in self.outcomes if o.status == "ok")

    @property
    def failed_shard_ids(self) -> tuple[int, ...]:
        return tuple(o.shard_id for o in self.outcomes if o.status != "ok")
