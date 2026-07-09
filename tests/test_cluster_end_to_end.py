"""Prueba de extremo a extremo con procesos reales: particiona el índice de
ejemplo en 3 shards, arranca 3 subprocesos `serve-shard` de verdad (no un
doble en memoria), lanza una query real por HTTP contra los tres, y mata uno
de los procesos a mitad de sesión para comprobar que el coordinador degrada
(resultados parciales) en vez de lanzar — la combinación exacta que exige el
checklist de cierre de `../distributed-index-sharding/AGENTS.md`.
"""

from __future__ import annotations

import random
from pathlib import Path

from distributed_index_sharding.cluster import LocalShardCluster
from distributed_index_sharding.coordinator import SearchCoordinator
from distributed_index_sharding.models import ShardSearchRequest
from distributed_index_sharding.partitioning import partition_index


async def test_cluster_search_and_degradation_on_killed_shard(
    tiny_index_dir: Path, tmp_path: Path
) -> None:
    partition_index(tiny_index_dir, tmp_path, num_shards=3)
    # Puerto base aleatorio en vez de uno fijo: evita colisiones si varios
    # runs de este test (u otro proceso local) usan el mismo rango a la vez.
    base_port = random.randint(20000, 60000)

    cluster = await LocalShardCluster.start(tmp_path, base_port=base_port)
    try:
        async with SearchCoordinator(cluster.targets, timeout_seconds=5.0) as coordinator:
            healthy_result = await coordinator.search(
                ShardSearchRequest(query_terms=("python",), top_k=10)
            )
            # "python" aparece globalmente en doc_ids 0, 1, 3 (ver samples/tiny_example).
            assert {hit.doc_id for hit in healthy_result.merged} == {0, 1, 3}
            assert healthy_result.failed_shard_ids == ()
            assert sorted(healthy_result.healthy_shard_ids) == [0, 1, 2]

            # shard-0 (doc_id % 3 == 0) posee los doc_ids 0 y 3, ambos con "python".
            killed_shard_id = cluster.targets[0].shard_id
            await cluster.kill(killed_shard_id)

            degraded_result = await coordinator.search(
                ShardSearchRequest(query_terms=("python",), top_k=10)
            )
            assert degraded_result.failed_shard_ids == (killed_shard_id,)
            # El shard caído no tumba la búsqueda: los otros dos siguen respondiendo.
            assert len(degraded_result.healthy_shard_ids) == 2
            # Solo doc_id=1 (en shard-1) sigue apareciendo: 0 y 3 vivían en el shard muerto.
            assert {hit.doc_id for hit in degraded_result.merged} == {1}
    finally:
        await cluster.shutdown()
