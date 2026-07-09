"""Post-filtro de `excluded_terms` y `filters` sobre los candidatos que ya
puntuó `bm25-ranking-engine` en un shard.

## Por qué esto vive en el shard, no en el coordinador

`bm25_ranking_engine.pipeline.BM25RankingPipeline.search_tokens` solo acepta
una bolsa de términos plana (ver su README, sección "Troubleshooting") y
devuelve `ScoredDocument(doc_id, score)` — ni frases, ni exclusión, ni
filtros, ni el texto o metadatos necesarios para resolverlos. Ese soporte de
operadores vive exclusivamente en `query_parser_autocomplete.ParsedQuery`
(`required_terms`, `phrases`, `excluded_terms`, `filters`). La decisión de
diseño de este proyecto (documentada en `../PROGRESS.md`) es:

- `required_terms` + los términos de cada frase de `phrases` se aplanan en la
  bolsa de `query_terms` que puntúa cada shard (`query_translation.py`) — BM25
  no distingue entre "frase" y "términos sueltos" de todas formas, al no ser
  consciente de adyacencia en su resultado.
- `excluded_terms`/`filters` se aplican como post-filtro sobre los
  `ScoredDocument` ya devueltos, **en el shard que los produjo**, no en el
  coordinador: el shard es el único que tiene, en memoria, tanto las postings
  del término excluido como los metadatos (`url`) del documento — devolver
  primero el resultado sin filtrar por la red y filtrar después en el
  coordinador obligaría a mandar también esos datos por HTTP, sin ganar nada.

## Limitación v1 explícita: frase exacta

La adyacencia real de una frase exacta ("estos términos aparecen consecutivos
en el documento") requeriría volver a las posiciones del índice para
comprobar que sus offsets son consecutivos — información que
`bm25-ranking-engine` no expone en su resultado (`list[ScoredDocument]`, solo
`doc_id` + `score`). v1 de este proyecto no reconstruye esa comprobación: una
consulta con `"exact phrase"` puntúa exactamente igual que si sus dos
términos se hubieran pedido sueltos. Se documenta aquí y en el README en vez
de fingir que se resuelve gratis.

## Limitación v1 explícita: `excluded_terms` solo sobre shards en formato plano

`ExclusionIndex.build` necesita enumerar, para un término dado, todos los
`doc_id` que lo contienen — la forma más simple de conseguirlo es leer
directamente el `postings.jsonl` que `partitioning.py` siempre escribe para
cada shard. Si un shard se sirve desde un directorio *comprimido* por
`index-compression-codec` (un paso posterior y opcional, ver
`partitioning.py`), ese `postings.jsonl` no está disponible junto al binario
comprimido: `excluded_terms`/`filters` se degradan a no-op en ese shard
(se registra un aviso, la búsqueda no falla) en vez de reimplementar aquí
también el decodificador binario solo para este post-filtro secundario.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path
from urllib.parse import urlparse

from bm25_ranking_engine.models import DocumentRecord

from distributed_index_sharding.models import QueryFilter

logger = logging.getLogger(__name__)

_MANIFEST_FILENAME = "manifest.json"

# Único campo de filtro con semántica real en v1: no hay ningún almacén de
# metadatos por documento más allá de `url`/`title`/`length` (ver
# `inverted-index-builder`), así que "site:" es el único operador que se
# puede resolver sin inventar un campo que no existe en el índice.
_SITE_FILTER_FIELD = "site"


class ExclusionIndex:
    """Mapa `término -> doc_ids que lo contienen`, para resolver
    `excluded_terms` localmente en un shard sin volver a consultar postings
    término a término en cada búsqueda."""

    def __init__(self, term_to_doc_ids: dict[str, frozenset[int]]) -> None:
        self._term_to_doc_ids = term_to_doc_ids

    def contains(self, term: str, doc_id: int) -> bool:
        return doc_id in self._term_to_doc_ids.get(term, frozenset())

    @staticmethod
    def build(shard_dir: Path) -> ExclusionIndex:
        manifest_path = shard_dir / _MANIFEST_FILENAME
        if not manifest_path.exists():
            raise FileNotFoundError(
                f"no existe {manifest_path}: ¿es un directorio de shard válido?"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        if "terms_file" in manifest:
            logger.warning(
                "shard en %s es un índice comprimido: excluded_terms/filters se "
                "degradan a no-op en este shard (limitación v1, ver README, "
                "sección Troubleshooting)",
                shard_dir,
            )
            return ExclusionIndex({})

        postings_path = shard_dir / str(manifest["postings_file"])
        term_to_doc_ids: dict[str, frozenset[int]] = {}
        for line in postings_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            data = json.loads(line)
            term_to_doc_ids[str(data["term"])] = frozenset(
                int(posting["doc_id"]) for posting in data["postings"]
            )
        return ExclusionIndex(term_to_doc_ids)


def is_excluded(document_index: ExclusionIndex, doc_id: int, excluded_terms: Sequence[str]) -> bool:
    return any(document_index.contains(term, doc_id) for term in excluded_terms)


def matches_filters(document: DocumentRecord, filters: Sequence[QueryFilter]) -> bool:
    """`True` si `document` satisface todos los `filters` que este shard sabe
    resolver. Un filtro con `field` distinto de `site` se ignora (no hay
    metadato al que compararlo, ver docstring del módulo) en vez de descartar
    el documento por un campo que la query nunca podría haber comprobado."""
    for query_filter in filters:
        if query_filter.field != _SITE_FILTER_FIELD:
            continue
        host = urlparse(document.url).hostname or ""
        if host != query_filter.value and not host.endswith(f".{query_filter.value}"):
            return False
    return True
