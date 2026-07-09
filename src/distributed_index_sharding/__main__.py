"""CLI de `distributed-index-sharding`: `partition` (índice completo -> N
shards en disco), `serve-shard` (arranca el servidor HTTP de un shard,
bloqueante), `simulate` (levanta un clúster local de varios procesos, lanza
una query de una vez y lo detiene — para probar sin infraestructura real) y
`search` (lanza una query contra un clúster ya en marcha, arrancado a mano con
`serve-shard` o con `simulate --keep-running`)."""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from distributed_index_sharding.coordinator import SearchCoordinator
from distributed_index_sharding.models import FanOutResult, ShardTarget
from distributed_index_sharding.partitioning import partition_index
from distributed_index_sharding.pipeline import DistributedSearchPipeline
from distributed_index_sharding.query_translation import from_parsed_query_dict, from_raw_text
from distributed_index_sharding.shard_server import run_shard_server


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="distributed-index-sharding",
        description="Particiona un índice invertido en shards y sirve búsqueda distribuida.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    partition_parser = subparsers.add_parser(
        "partition", help="Particiona un índice (plano o comprimido) en N shards por documento"
    )
    partition_parser.add_argument(
        "source_index", type=Path, help="Índice de inverted-index-builder o index-compression-codec"
    )
    partition_parser.add_argument(
        "output_root", type=Path, help="Directorio donde escribir los shards"
    )
    partition_parser.add_argument("--num-shards", type=int, default=3)

    serve_parser = subparsers.add_parser(
        "serve-shard", help="Arranca el servidor HTTP de un único shard (bloqueante)"
    )
    serve_parser.add_argument("shard_dir", type=Path, help="Directorio de índice de este shard")
    serve_parser.add_argument("--shard-id", type=int, required=True)
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, required=True)

    simulate_parser = subparsers.add_parser(
        "simulate",
        help="Levanta un clúster local (varios procesos), lanza una query y lo detiene",
    )
    simulate_parser.add_argument("partitioned_root", type=Path, help="Salida de 'partition'")
    simulate_parser.add_argument("--host", default="127.0.0.1")
    simulate_parser.add_argument("--base-port", type=int, default=9300)
    simulate_parser.add_argument("--top-k", type=int, default=10)
    simulate_parser.add_argument("--timeout-seconds", type=float, default=2.0)
    simulate_parser.add_argument("--text", help="Query en crudo, sin operadores")
    simulate_parser.add_argument(
        "--parsed-query-file", type=Path, help="JSON de ParsedQuery (query-parser-autocomplete)"
    )
    simulate_parser.add_argument(
        "--keep-running",
        action="store_true",
        help="No lanzar ninguna query: dejar el clúster arriba hasta Ctrl+C, para probar 'search' "
        "a mano contra sus puertos o simular un shard caído matando su proceso",
    )

    search_parser = subparsers.add_parser(
        "search", help="Lanza una query contra un clúster ya en marcha (arrancado aparte)"
    )
    search_parser.add_argument(
        "--shard",
        action="append",
        required=True,
        dest="shards",
        metavar="SHARD_ID:HOST:PORT",
        help="Repetible: un shard del clúster, p. ej. --shard 0:127.0.0.1:9300",
    )
    search_parser.add_argument("--top-k", type=int, default=10)
    search_parser.add_argument("--timeout-seconds", type=float, default=2.0)
    search_parser.add_argument("--text", help="Query en crudo, sin operadores")
    search_parser.add_argument(
        "--parsed-query-file", type=Path, help="JSON de ParsedQuery (query-parser-autocomplete)"
    )

    return parser


def _run_partition(args: argparse.Namespace) -> int:
    source_index: Path = args.source_index
    if not source_index.exists():
        print(f"Error: no existe source_index={source_index}", file=sys.stderr)
        return 2

    manifest = partition_index(source_index, args.output_root, args.num_shards)
    print(
        f"Índice particionado en {manifest.num_shards} shards bajo {args.output_root}: "
        f"{', '.join(manifest.shard_dir_names)}"
    )
    return 0


def _run_serve_shard(args: argparse.Namespace) -> int:
    shard_dir: Path = args.shard_dir
    if not shard_dir.exists():
        print(f"Error: no existe shard_dir={shard_dir}", file=sys.stderr)
        return 2

    run_shard_server(args.shard_id, shard_dir, args.host, args.port)
    return 0


def _parse_shard_target(raw: str) -> ShardTarget:
    parts = raw.split(":")
    if len(parts) != 3:
        raise ValueError(f"formato esperado SHARD_ID:HOST:PORT, recibido '{raw}'")
    shard_id_text, host, port_text = parts
    return ShardTarget(shard_id=int(shard_id_text), host=host, port=int(port_text))


def _build_query_request_kwargs(args: argparse.Namespace) -> tuple[str, str | dict[str, object]]:
    """Valida que se ha pasado exactamente una fuente de query y la lee. Devuelve
    `("text", texto)` o `("parsed", parsed_query_dict)`."""
    if bool(args.text) == bool(args.parsed_query_file):
        raise ValueError("pasa exactamente uno de --text o --parsed-query-file")
    if args.text:
        return "text", args.text
    parsed_query_file: Path = args.parsed_query_file
    if not parsed_query_file.exists():
        raise ValueError(f"no existe parsed_query_file={parsed_query_file}")
    return "parsed", json.loads(parsed_query_file.read_text(encoding="utf-8"))


def _print_fan_out_result(result: FanOutResult) -> None:
    for outcome in result.outcomes:
        status_note = "" if outcome.status == "ok" else f" ({outcome.error_message})"
        print(
            f"  shard {outcome.shard_id}: {outcome.status}{status_note} "
            f"[{outcome.latency_ms:.1f} ms, {len(outcome.hits)} hits]",
            file=sys.stderr,
        )
    if not result.merged:
        print("Sin resultados")
        return
    for position, hit in enumerate(result.merged, start=1):
        print(f"{position}. doc_id={hit.doc_id} score={hit.score:.4f}")


def _run_simulate(args: argparse.Namespace) -> int:
    partitioned_root: Path = args.partitioned_root
    if not partitioned_root.exists():
        print(f"Error: no existe partitioned_root={partitioned_root}", file=sys.stderr)
        return 2

    if not args.keep_running:
        try:
            kind, payload = _build_query_request_kwargs(args)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 2
    else:
        kind, payload = "text", ""

    async def _run() -> int:
        pipeline = await DistributedSearchPipeline.start(
            partitioned_root,
            host=args.host,
            base_port=args.base_port,
            timeout_seconds=args.timeout_seconds,
        )
        try:
            targets_description = ", ".join(t.base_url for t in pipeline.targets)
            print(f"Clúster local levantado: {targets_description}", file=sys.stderr)

            if args.keep_running:
                print("Ctrl+C para detener el clúster.", file=sys.stderr)
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.Event().wait()
                return 0

            if kind == "text":
                assert isinstance(payload, str)
                result = await pipeline.search_text(payload, top_k=args.top_k)
            else:
                assert isinstance(payload, dict)
                result = await pipeline.search_parsed_query(payload, top_k=args.top_k)
            _print_fan_out_result(result)
            return 0
        finally:
            await pipeline.close()

    try:
        return asyncio.run(_run())
    except KeyboardInterrupt:
        return 0


def _run_search(args: argparse.Namespace) -> int:
    try:
        targets = [_parse_shard_target(raw) for raw in args.shards]
        kind, payload = _build_query_request_kwargs(args)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    async def _run() -> FanOutResult:
        async with SearchCoordinator(targets, timeout_seconds=args.timeout_seconds) as coordinator:
            if kind == "text":
                assert isinstance(payload, str)
                request = from_raw_text(payload, top_k=args.top_k)
            else:
                assert isinstance(payload, dict)
                request = from_parsed_query_dict(payload, top_k=args.top_k)
            return await coordinator.search(request)

    result = asyncio.run(_run())
    _print_fan_out_result(result)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "partition":
        return _run_partition(args)
    if args.command == "serve-shard":
        return _run_serve_shard(args)
    if args.command == "simulate":
        return _run_simulate(args)
    if args.command == "search":
        return _run_search(args)

    parser.error(f"Comando desconocido: {args.command}")
    return 2  # inalcanzable: parser.error() termina el proceso con sys.exit()


if __name__ == "__main__":
    sys.exit(main())
