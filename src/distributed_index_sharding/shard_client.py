"""Implementación HTTP real de `protocols.ShardTransport`.

Nunca deja escapar una excepción de red: toda condición de fallo (timeout,
conexión rechazada, host caído, respuesta HTTP distinta de 200, cuerpo JSON
malformado) se traduce a un `ShardOutcome` con `status` `"timeout"` o
`"error"` — la tolerancia explícita a shards caídos que exige `../AGENTS.md`
("degradación explícita — resultados parciales, documentado — nunca una
excepción que tumbe la búsqueda completa") vive aquí, en el único punto del
sistema donde de verdad puede fallar la red.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import aiohttp

from distributed_index_sharding.models import (
    ShardOutcome,
    ShardSearchRequest,
    ShardSearchResponse,
    ShardTarget,
)


@dataclass(slots=True)
class HttpShardTransport:
    """Envía `ShardSearchRequest` como JSON a `POST {target.base_url}/search`
    reutilizando una única `aiohttp.ClientSession` compartida entre shards —
    abrir una sesión nueva por petición desperdiciaría el *connection
    pooling* que `aiohttp` ya ofrece para fan-out concurrente."""

    session: aiohttp.ClientSession

    async def search(
        self, target: ShardTarget, request: ShardSearchRequest, timeout_seconds: float
    ) -> ShardOutcome:
        started_at = time.monotonic()
        url = f"{target.base_url}/search"
        try:
            async with self.session.post(
                url,
                json=request.to_json_dict(),
                timeout=aiohttp.ClientTimeout(total=timeout_seconds),
            ) as response:
                if response.status != 200:
                    body_text = await response.text()
                    return ShardOutcome(
                        shard_id=target.shard_id,
                        status="error",
                        error_message=f"HTTP {response.status}: {body_text}",
                        latency_ms=self._elapsed_ms(started_at),
                    )
                payload = await response.json()
        except TimeoutError:
            return ShardOutcome(
                shard_id=target.shard_id,
                status="timeout",
                error_message=f"sin respuesta en {timeout_seconds:.1f}s",
                latency_ms=self._elapsed_ms(started_at),
            )
        except aiohttp.ClientError as exc:
            return ShardOutcome(
                shard_id=target.shard_id,
                status="error",
                error_message=str(exc),
                latency_ms=self._elapsed_ms(started_at),
            )

        try:
            parsed_response = ShardSearchResponse.from_json_dict(payload)
        except (KeyError, TypeError, ValueError) as exc:
            return ShardOutcome(
                shard_id=target.shard_id,
                status="error",
                error_message=f"respuesta malformada: {exc}",
                latency_ms=self._elapsed_ms(started_at),
            )

        return ShardOutcome(
            shard_id=target.shard_id,
            status="ok",
            hits=parsed_response.hits,
            latency_ms=self._elapsed_ms(started_at),
        )

    @staticmethod
    def _elapsed_ms(started_at: float) -> float:
        return (time.monotonic() - started_at) * 1000
