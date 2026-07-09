"""Simulación local de un clúster de shards: un subproceso `serve-shard` por
cada directorio de `cluster_manifest.json` (`partitioning.py`), cada uno en su
propio puerto de `localhost`, sin depender de infraestructura de despliegue
real — la "simulación local de múltiples shards (varios procesos o puertos)"
que exige `../AGENTS.md`.

`LocalShardCluster` es un gestor de contexto asíncrono: garantiza terminar
todos los subprocesos (`SIGTERM`, con `SIGKILL` de respaldo si alguno no
responde a tiempo) tanto al salir normalmente como si el arranque falla a
mitad de camino — el cierre explícito de recursos que exige `../CLAUDE.md`,
sección 2.E, aplicado a procesos en vez de a descriptores de archivo.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from pathlib import Path
from types import TracebackType
from typing import Final

import aiohttp

from distributed_index_sharding.models import ShardTarget
from distributed_index_sharding.partitioning import read_cluster_manifest

logger = logging.getLogger(__name__)

_HEALTH_CHECK_INTERVAL_SECONDS: Final[float] = 0.1
_HEALTH_CHECK_TIMEOUT_SECONDS: Final[float] = 10.0
_GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS: Final[float] = 5.0


class LocalShardCluster:
    """Ver el docstring del módulo. Constrúyase con `LocalShardCluster.start`,
    nunca con `__init__` directamente (necesita `await` para arrancar los
    subprocesos y esperar a que respondan)."""

    def __init__(
        self, targets: tuple[ShardTarget, ...], processes: tuple[asyncio.subprocess.Process, ...]
    ) -> None:
        self._targets = targets
        self._processes = processes

    @property
    def targets(self) -> tuple[ShardTarget, ...]:
        return self._targets

    @classmethod
    async def start(
        cls, partitioned_root: Path, *, host: str = "127.0.0.1", base_port: int = 9300
    ) -> LocalShardCluster:
        manifest = read_cluster_manifest(partitioned_root)
        targets = tuple(
            ShardTarget(shard_id=shard_id, host=host, port=base_port + shard_id)
            for shard_id in range(manifest.num_shards)
        )

        processes: list[asyncio.subprocess.Process] = []
        try:
            for shard_id, target in enumerate(targets):
                shard_dir = partitioned_root / manifest.shard_dir_names[shard_id]
                process = await asyncio.create_subprocess_exec(
                    sys.executable,
                    "-m",
                    "distributed_index_sharding",
                    "serve-shard",
                    str(shard_dir),
                    "--shard-id",
                    str(shard_id),
                    "--host",
                    host,
                    "--port",
                    str(target.port),
                )
                processes.append(process)
            await cls._wait_until_healthy(targets)
        except Exception:
            for process in processes:
                if process.returncode is None:
                    process.terminate()
            raise

        logger.info("clúster local levantado: %d shards en %s", len(targets), host)
        return cls(targets, tuple(processes))

    @staticmethod
    async def _wait_until_healthy(targets: tuple[ShardTarget, ...]) -> None:
        deadline = time.monotonic() + _HEALTH_CHECK_TIMEOUT_SECONDS
        async with aiohttp.ClientSession() as session:
            for target in targets:
                while True:
                    if time.monotonic() > deadline:
                        raise TimeoutError(
                            f"shard {target.shard_id} ({target.base_url}) no respondió "
                            f"a /health en {_HEALTH_CHECK_TIMEOUT_SECONDS}s"
                        )
                    try:
                        async with session.get(
                            f"{target.base_url}/health",
                            timeout=aiohttp.ClientTimeout(total=1.0),
                        ) as response:
                            if response.status == 200:
                                break
                    except aiohttp.ClientError:
                        pass
                    await asyncio.sleep(_HEALTH_CHECK_INTERVAL_SECONDS)

    async def kill(self, shard_id: int) -> None:
        """Mata sin gracia (`SIGKILL`) el proceso de un shard concreto — para
        poder probar la tolerancia del coordinador a un shard caído en mitad
        de una sesión (`tests/test_cluster_end_to_end.py`) sin esperar a un
        timeout real de red, y para reproducir manualmente el mismo escenario
        durante el smoke test de cierre del proyecto."""
        for target, process in zip(self._targets, self._processes, strict=True):
            if target.shard_id == shard_id:
                process.kill()
                await process.wait()
                return
        raise ValueError(f"no existe shard_id={shard_id} en este clúster")

    async def shutdown(self) -> None:
        for process in self._processes:
            if process.returncode is None:
                process.terminate()
        for process in self._processes:
            try:
                await asyncio.wait_for(process.wait(), timeout=_GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS)
            except TimeoutError:
                logger.warning(
                    "proceso de shard pid=%s no terminó a tiempo, forzando kill", process.pid
                )
                process.kill()
                await process.wait()
        logger.info("clúster local detenido")

    async def __aenter__(self) -> LocalShardCluster:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.shutdown()
