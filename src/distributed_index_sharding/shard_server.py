"""Servidor HTTP de un shard: envuelve una `BM25RankingPipeline` propia
(sobre el directorio de ese shard, ver `partitioning.py`) detrás de
`POST /search` y `GET /health`.

Cada shard es un proceso `aiohttp` independiente que carga su propio índice
una vez al arrancar y lo mantiene abierto para toda su vida — "cada shard es
una instancia independiente de `bm25-ranking-engine` operando sobre una
partición del índice" (`../AGENTS.md`). El cierre explícito del lector del
índice al terminar el proceso (`../CLAUDE.md`, sección 2.E) se hace en el hook
`on_cleanup` de `aiohttp`, no confiando en que el proceso simplemente termine.

## Sobre-pedido (`_OVERFETCH_FACTOR`) para el post-filtro

`excluded_terms`/`filters` se aplican *después* de que `BM25RankingPipeline`
ya eligió su top-K por score — así que si varios de los mejores candidatos
quedan excluidos, el shard podría devolver menos de `top_k` resultados aunque
existan más candidatos válidos más abajo en su ranking. Pedir
`top_k * _OVERFETCH_FACTOR` candidatos a BM25 antes de filtrar mitiga esto sin
tener que rehacer la búsqueda con una ventana creciente; sigue siendo una
mitigación, no una garantía — un caso patológico donde más del
`_OVERFETCH_FACTOR`-ésimo de los mejores candidatos estén excluidos seguiría
devolviendo menos de `top_k` resultados de ese shard. Documentado en el
README, sección Troubleshooting.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from aiohttp import web
from bm25_ranking_engine.pipeline import BM25RankingPipeline

from distributed_index_sharding.models import ShardHit, ShardSearchRequest, ShardSearchResponse
from distributed_index_sharding.post_filtering import ExclusionIndex, is_excluded, matches_filters

logger = logging.getLogger(__name__)

_OVERFETCH_FACTOR: Final[int] = 3


@dataclass(slots=True)
class _ShardState:
    shard_id: int
    pipeline: BM25RankingPipeline
    exclusion_index: ExclusionIndex


_STATE_KEY: Final[web.AppKey[_ShardState]] = web.AppKey("shard_state", _ShardState)


async def _handle_search(request: web.Request) -> web.Response:
    state = request.app[_STATE_KEY]
    started_at = time.monotonic()

    try:
        payload = await request.json()
        search_request = ShardSearchRequest.from_json_dict(payload)
    except (ValueError, KeyError, TypeError) as exc:
        return web.json_response({"error": f"petición malformada: {exc}"}, status=400)

    candidate_pool_size = search_request.top_k * _OVERFETCH_FACTOR
    scored_documents = state.pipeline.search_tokens(
        search_request.query_terms, top_k=candidate_pool_size
    )

    hits: list[ShardHit] = []
    for scored in scored_documents:
        if len(hits) >= search_request.top_k:
            break
        if is_excluded(state.exclusion_index, scored.doc_id, search_request.excluded_terms):
            continue
        document = state.pipeline.document(scored.doc_id)
        if document is not None and not matches_filters(document, search_request.filters):
            continue
        hits.append(ShardHit(doc_id=scored.doc_id, score=scored.score))

    elapsed_ms = (time.monotonic() - started_at) * 1000
    logger.info(
        "shard %s: %d términos de query, %d hits devueltos en %.2f ms",
        state.shard_id,
        len(search_request.query_terms),
        len(hits),
        elapsed_ms,
    )
    response = ShardSearchResponse(shard_id=state.shard_id, hits=tuple(hits))
    return web.json_response(response.to_json_dict())


async def _handle_health(request: web.Request) -> web.Response:
    state = request.app[_STATE_KEY]
    return web.json_response({"shard_id": state.shard_id, "status": "ok"})


def create_app(shard_id: int, shard_dir: Path) -> web.Application:
    """Construye la `web.Application` de un shard sin arrancar el servidor
    (`run_shard_server` la arranca; los tests la ejecutan con
    `aiohttp.test_utils` directamente sobre esta función)."""
    app = web.Application()
    app.router.add_post("/search", _handle_search)
    app.router.add_get("/health", _handle_health)

    async def _on_startup(app: web.Application) -> None:
        pipeline = BM25RankingPipeline(shard_dir)
        exclusion_index = ExclusionIndex.build(shard_dir)
        app[_STATE_KEY] = _ShardState(
            shard_id=shard_id, pipeline=pipeline, exclusion_index=exclusion_index
        )
        logger.info("shard %s: índice cargado desde %s", shard_id, shard_dir)

    async def _on_cleanup(app: web.Application) -> None:
        state = app[_STATE_KEY]
        state.pipeline.close()
        logger.info("shard %s: lector del índice cerrado", shard_id)

    app.on_startup.append(_on_startup)
    app.on_cleanup.append(_on_cleanup)
    return app


def run_shard_server(shard_id: int, shard_dir: Path, host: str, port: int) -> None:
    """Punto de entrada bloqueante para un proceso de shard (invocado por
    `__main__.py serve-shard` y por `cluster.py` al lanzar subprocesos)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    app = create_app(shard_id, shard_dir)
    web.run_app(app, host=host, port=port)
