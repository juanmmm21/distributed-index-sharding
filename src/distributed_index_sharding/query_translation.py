"""Traduce una `ParsedQuery` de `query-parser-autocomplete` (JSON) al
`ShardSearchRequest` que entiende cada shard.

El contrato de entrada es exactamente el que documenta
`query_parser_autocomplete.models.ParsedQuery.to_json_dict` (ver su README,
sección "Data formats"):

```json
{
  "required_terms": ["python", "tutorial"],
  "phrases": [["exact", "phrase"]],
  "excluded_terms": ["deprecated"],
  "filters": [{"field": "site", "value": "example.com"}],
  "corrections": [{"original": "pyhton", "corrected": "python"}]
}
```

`corrections` se ignora aquí a propósito: es información de depuración sobre
qué corrigió el spellchecker, ya aplicada dentro de `required_terms`/`phrases`
por `query-parser-autocomplete` antes de serializar — no hay nada que este
proyecto tenga que volver a hacer con ella.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from distributed_index_sharding.models import QueryFilter, ShardSearchRequest

_WORD_PATTERN = re.compile(r"\w+", re.UNICODE)


def from_parsed_query_dict(data: dict[str, Any], *, top_k: int) -> ShardSearchRequest:
    """Aplana `required_terms` + los términos de cada frase de `phrases` en
    `query_terms` (BM25 no distingue frase de términos sueltos, ver
    `post_filtering.py` para la limitación de adyacencia que esto implica) y
    conserva `excluded_terms`/`filters` para que cada shard los resuelva
    localmente."""
    required_terms = [str(term) for term in data.get("required_terms", [])]
    phrase_terms = [str(term) for phrase in data.get("phrases", []) for term in phrase]
    excluded_terms = tuple(str(term) for term in data.get("excluded_terms", []))
    filters = tuple(QueryFilter.from_json_dict(f) for f in data.get("filters", []))

    return ShardSearchRequest(
        query_terms=tuple(required_terms + phrase_terms),
        excluded_terms=excluded_terms,
        filters=filters,
        top_k=top_k,
    )


def from_raw_text(text: str, *, top_k: int) -> ShardSearchRequest:
    """Conveniencia de CLI para probar sin pasar por `query-parser-autocomplete`:
    tokeniza con NFC + `casefold()` + runs de letras/dígitos, la misma
    normalización que `inverted-index-builder` aplica al indexar (ver su
    README), pero sin ningún soporte de operadores — para eso está
    `from_parsed_query_dict`."""
    normalized = unicodedata.normalize("NFC", text).casefold()
    query_terms = tuple(_WORD_PATTERN.findall(normalized))
    return ShardSearchRequest(query_terms=query_terms, top_k=top_k)
