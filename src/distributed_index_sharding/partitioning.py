"""Particionado document-based del índice invertido en N shards.

## Estrategia elegida: por documento, no por rango de términos

Cada shard recibe un subconjunto disjunto de documentos (`doc_id % num_shards`)
y construye, sobre ese subconjunto, un índice invertido self-contained con sus
propias estadísticas (`document_frequency` por término, `average_document_length`,
`total_documents`) recalculadas localmente. Esto es lo que hace que el
contrato de esta pieza del sistema — "cada shard devuelve una lista ya
puntuada y ordenada, el coordinador solo hace *k-way merge*" (`../AGENTS.md`)
— sea correcto: el score BM25 que un shard calcula para un documento es ya su
score final, porque el shard tiene toda la información (todas las apariciones
de todos los términos de la query) que existe sobre ese documento. Ningún
documento vive partido entre dos shards.

**Alternativa descartada — por rango de términos**: cada shard tendría *todos*
los documentos pero solo una porción del vocabulario (p. ej. términos A-M en
un shard, N-Z en otro). Esto preservaría un IDF exactamente global (cada shard
que posee un término ya tiene su `document_frequency` real sobre todo el
corpus), pero rompe la precondición del merge: si los términos de una query
caen en shards distintos, cada shard solo vería una parte de los términos de
esa query y devolvería una contribución *parcial* al score de un mismo
documento — el coordinador tendría que sumar contribuciones parciales por
`doc_id` a través de shards antes de poder ordenar nada, un problema de
agregación distribuida, no la fusión de listas ya ordenadas que pide la
Definition of Done de este proyecto. Además, cualquier documento que use
vocabulario disperso alfabéticamente tocaría casi todos los shards al
indexarlo, perdiendo el beneficio de partición de disco/memoria que el
sharding existe para dar en primer lugar.

**Coste aceptado del particionado por documento — IDF local**: `document_frequency`
y `average_document_length` pasan a ser estadísticas *del shard*, no del
corpus completo. El score BM25 que un shard calcula para un documento es una
aproximación razonable del que obtendría un índice sin particionar, no un
valor idéntico bit a bit — el mismo trade-off que aceptan por defecto sistemas
reales de búsqueda distribuida a gran escala (Elasticsearch, SolrCloud) en
lugar de pagar una ronda de red adicional para acordar estadísticas de término
globales antes de puntuar. Se documenta aquí explícitamente, no se finge
resuelto.

## Por qué esta pieza reutiliza `bm25_ranking_engine.index_reader`

El índice de origen puede venir en el formato sin comprimir de
`inverted-index-builder` o en el comprimido de `index-compression-codec`
(binario, con delta encoding + varint/bitpacking, ver ese proyecto). Volver a
implementar el decodificador binario aquí solo para poder particionarlo
duplicaría exactamente lo que `bm25-ranking-engine` ya reimplementa de forma
independiente (`compressed_postings_codec.py`) — y esta pieza del ecosistema
ya depende de `bm25-ranking-engine` como paquete real (no solo por contrato en
disco) para el scoring por shard, la excepción explícita documentada en
`../PROGRESS.md` a la regla general de "nunca imports cruzados". Reutilizar su
lector aquí, para el mismo formato, es la misma dependencia ya aceptada, no
una nueva.

## Por qué cada shard se escribe en formato *sin comprimir*

Cada shard de salida es un directorio en el formato de `inverted-index-builder`
(`manifest.json` + `documents.jsonl` + `postings.jsonl` + `stats.json`), no en
el binario de `index-compression-codec`. Reescribir bytes comprimidos por
shard exigiría reimplementar también el *codificador* (no solo el decodificador
ya reutilizado) — trabajo que no compra nada aquí, porque particionar por
documento ya obliga a tocar y filtrar cada posting de todas formas, así que no
hay acceso aleatorio que preservar en esta etapa. Si se necesita el beneficio
de espacio en disco de la compresión en producción, cada directorio de shard
ya resultante es, él mismo, una entrada válida para
`index-compression-codec compress` como paso posterior y opcional — documentado
en el README, no reimplementado aquí. `shard_server.py` sirve cualquiera de
los dos formatos sin cambios, porque delega la apertura del índice en
`bm25_ranking_engine.pipeline.BM25RankingPipeline`, que ya autodetecta ambos.

## `doc_id` global preservado

El `doc_id` de cada documento se conserva literal en el shard al que cae
(`doc_id % num_shards`), nunca se renumera localmente. Renumerar obligaría a
mantener una tabla de traducción `(shard_id, local_id) -> doc_id` global para
que el resultado final pudiera correlacionarse con el corpus original —
complejidad que preservar el ID evita por completo, a cambio de que el array
`document_lengths` de cada shard (indexado directamente por `doc_id`, ver
`bm25_ranking_engine.index_reader`) quede disperso en vez de compacto. Para el
tamaño de corpus de este ecosistema, ese hueco de memoria es aceptable frente
a la complejidad de una tabla de traducción de IDs.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from bm25_ranking_engine.index_reader import open_index_reader
from bm25_ranking_engine.models import DocumentRecord, Posting

_MANIFEST_FILENAME: Final[str] = "manifest.json"
_CLUSTER_MANIFEST_FILENAME: Final[str] = "cluster_manifest.json"
_SHARD_DIR_PREFIX: Final[str] = "shard-"


@dataclass(frozen=True, slots=True)
class ClusterManifest:
    """Describe el resultado de un particionado: cuántos shards hay y dónde
    vive cada uno, para que `cluster.py`/`__main__.py` puedan descubrirlos sin
    tener que volver a particionar ni adivinar nombres de carpeta."""

    num_shards: int
    shard_dir_names: tuple[str, ...]

    def to_json_dict(self) -> dict[str, Any]:
        return {"num_shards": self.num_shards, "shard_dir_names": list(self.shard_dir_names)}

    @staticmethod
    def from_json_dict(data: dict[str, Any]) -> ClusterManifest:
        return ClusterManifest(
            num_shards=int(data["num_shards"]),
            shard_dir_names=tuple(str(name) for name in data["shard_dir_names"]),
        )


def _read_vocabulary_terms(source_dir: Path) -> list[str]:
    """Enumera los nombres de todos los términos del índice de origen.

    Lector independiente y mínimo (solo el campo `term`, igual que
    `query_parser_autocomplete.index_reader` hace con `document_frequency`):
    ninguno de los dos formatos de índice de origen expone "listar todo el
    vocabulario" a través de `bm25_ranking_engine.protocols.IndexReader`
    (solo `lookup(term)` puntual), así que hace falta leer el directorio de
    términos por separado para saber qué términos preguntar.
    """
    manifest_path = source_dir / _MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"no existe {manifest_path}: ¿es un directorio de índice válido?")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    if "terms_file" in manifest:
        terms_path = source_dir / str(manifest["terms_file"])
    else:
        terms_path = source_dir / str(manifest["postings_file"])

    terms: list[str] = []
    for line in terms_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        terms.append(str(json.loads(line)["term"]))
    return terms


def _write_shard_index(
    shard_dir: Path,
    documents: list[DocumentRecord],
    postings_by_term: dict[str, list[Posting]],
) -> None:
    shard_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "format_version": 1,
        "documents_file": "documents.jsonl",
        "postings_file": "postings.jsonl",
        "stats_file": "stats.json",
    }
    (shard_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    documents_lines = [
        json.dumps({"doc_id": doc.doc_id, "url": doc.url, "title": doc.title, "length": doc.length})
        for doc in sorted(documents, key=lambda d: d.doc_id)
    ]
    (shard_dir / "documents.jsonl").write_text(
        "\n".join(documents_lines) + ("\n" if documents_lines else ""), encoding="utf-8"
    )

    postings_lines = []
    for term in sorted(postings_by_term):
        postings = postings_by_term[term]
        if not postings:
            continue
        postings_lines.append(
            json.dumps(
                {
                    "term": term,
                    "document_frequency": len(postings),
                    "postings": [
                        {
                            "doc_id": posting.doc_id,
                            "term_frequency": posting.term_frequency,
                            "positions": list(posting.positions),
                        }
                        for posting in sorted(postings, key=lambda p: p.doc_id)
                    ],
                }
            )
        )
    (shard_dir / "postings.jsonl").write_text(
        "\n".join(postings_lines) + ("\n" if postings_lines else ""), encoding="utf-8"
    )

    total_documents = len(documents)
    vocabulary_size = sum(1 for postings in postings_by_term.values() if postings)
    total_postings = sum(len(postings) for postings in postings_by_term.values())
    average_document_length = (
        statistics.fmean(doc.length for doc in documents) if documents else 0.0
    )
    stats = {
        "total_documents": total_documents,
        "vocabulary_size": vocabulary_size,
        "total_postings": total_postings,
        "average_document_length": average_document_length,
    }
    (shard_dir / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")


def partition_index(source_dir: Path, output_root: Path, num_shards: int) -> ClusterManifest:
    """Particiona `source_dir` (índice plano o comprimido) en `num_shards`
    directorios de shard, por documento (`doc_id % num_shards`), bajo
    `output_root/shard-<id>/`. Devuelve el `ClusterManifest` ya escrito en
    `output_root/cluster_manifest.json`.
    """
    if num_shards <= 0:
        raise ValueError(f"num_shards debe ser positivo, recibido {num_shards}")

    reader = open_index_reader(source_dir)
    try:
        total_documents = reader.total_documents
        documents_by_shard: list[list[DocumentRecord]] = [[] for _ in range(num_shards)]
        for doc_id in range(total_documents):
            document = reader.document(doc_id)
            if document is None:
                continue
            documents_by_shard[doc_id % num_shards].append(document)

        postings_by_shard: list[dict[str, list[Posting]]] = [{} for _ in range(num_shards)]
        for term in _read_vocabulary_terms(source_dir):
            postings_list = reader.lookup(term)
            if postings_list is None:
                continue
            buckets: list[list[Posting]] = [[] for _ in range(num_shards)]
            for posting in postings_list.postings:
                buckets[posting.doc_id % num_shards].append(posting)
            for shard_id, bucket in enumerate(buckets):
                if bucket:
                    postings_by_shard[shard_id][term] = bucket
    finally:
        reader.close()

    shard_dir_names: list[str] = []
    for shard_id in range(num_shards):
        shard_dir_name = f"{_SHARD_DIR_PREFIX}{shard_id}"
        _write_shard_index(
            output_root / shard_dir_name,
            documents_by_shard[shard_id],
            postings_by_shard[shard_id],
        )
        shard_dir_names.append(shard_dir_name)

    cluster_manifest = ClusterManifest(
        num_shards=num_shards, shard_dir_names=tuple(shard_dir_names)
    )
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / _CLUSTER_MANIFEST_FILENAME).write_text(
        json.dumps(cluster_manifest.to_json_dict(), indent=2), encoding="utf-8"
    )
    return cluster_manifest


def read_cluster_manifest(output_root: Path) -> ClusterManifest:
    manifest_path = output_root / _CLUSTER_MANIFEST_FILENAME
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"no existe {manifest_path}: ejecuta primero 'partition' sobre este directorio"
        )
    return ClusterManifest.from_json_dict(json.loads(manifest_path.read_text(encoding="utf-8")))
