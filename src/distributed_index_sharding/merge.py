"""Fusión (*k-way merge*) de las listas de `ShardHit` ya ordenadas por score
descendente que devuelve cada shard sano.

## Por qué `heapq.merge`, no ordenar todo desde cero

Cada shard ya devuelve su top-K ordenado por score descendente
(`bm25_ranking_engine.scoring.BM25Scorer` lo garantiza, ties por `doc_id`
ascendente — ver su README). Concatenar las `k` listas de `n` elementos cada
una y volver a ordenar el resultado completo costaría `O(kn log(kn))`,
tirando a la basura el orden que cada shard ya calculó. `heapq.merge` fusiona
`k` iterables ya ordenados en `O(kn log k)`, manteniendo un heap de tamaño `k`
(uno por shard) en vez de materializar y reordenar `kn` elementos — la
definición exacta de "merge eficiente de listas ya ordenadas" que pide
`../AGENTS.md`. La librería estándar ya expone el argumento `reverse=True`
(desde Python 3.5) para fusionar iterables descendentes sin tener que invertir
cada lista de shard a mano primero.

## El tie-break de `doc_id` ascendente sobrevive al merge

`heapq.merge` con `key=` no compara elementos por identidad, compara por la
clave — así que basta con que la clave codifique tanto "score descendente"
como "doc_id ascendente en caso de empate" en un único valor comparable:
`(score, -doc_id)`, fusionado con `reverse=True`, ordena primero por `score`
descendente y, a igualdad de `score`, por `-doc_id` descendente (`doc_id`
ascendente) — el mismo criterio de determinismo que ya aplica cada shard
individualmente, preservado a través del merge.
"""

from __future__ import annotations

import heapq
import itertools
from collections.abc import Iterable

from distributed_index_sharding.models import ShardHit, ShardOutcome


def _merge_key(hit: ShardHit) -> tuple[float, int]:
    return (hit.score, -hit.doc_id)


def merge_shard_hits(
    shard_hit_lists: Iterable[Iterable[ShardHit]], *, top_k: int
) -> list[ShardHit]:
    """Fusiona listas de `ShardHit` ya ordenadas por score descendente
    (tie-break `doc_id` ascendente) en una única lista global, truncada a
    `top_k`. Perezoso: `heapq.merge` es un generador, así que `top_k`
    resultados nunca materializan más de eso en memoria."""
    if top_k <= 0:
        raise ValueError(f"top_k debe ser positivo, recibido {top_k}")
    merged = heapq.merge(*shard_hit_lists, key=_merge_key, reverse=True)
    return list(itertools.islice(merged, top_k))


def merge_shard_outcomes(outcomes: Iterable[ShardOutcome], *, top_k: int) -> list[ShardHit]:
    """Conveniencia sobre `merge_shard_hits`: ignora los shards que no
    respondieron con éxito (`status != "ok"`) — la degradación explícita ante
    shards caídos o con timeout vive en `coordinator.py`, este merge solo
    fusiona lo que sí llegó."""
    healthy_hit_lists = (outcome.hits for outcome in outcomes if outcome.status == "ok")
    return merge_shard_hits(healthy_hit_lists, top_k=top_k)
