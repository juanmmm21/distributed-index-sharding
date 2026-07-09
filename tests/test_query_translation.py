from __future__ import annotations

from distributed_index_sharding.models import QueryFilter
from distributed_index_sharding.query_translation import from_parsed_query_dict, from_raw_text


def test_from_parsed_query_dict_flattens_required_terms_and_phrases() -> None:
    parsed_query = {
        "required_terms": ["python", "tutorial"],
        "phrases": [["exact", "phrase"]],
        "excluded_terms": ["deprecated"],
        "filters": [{"field": "site", "value": "example.com"}],
        "corrections": [{"original": "pyhton", "corrected": "python"}],
    }

    request = from_parsed_query_dict(parsed_query, top_k=7)

    assert request.query_terms == ("python", "tutorial", "exact", "phrase")
    assert request.excluded_terms == ("deprecated",)
    assert request.filters == (QueryFilter(field="site", value="example.com"),)
    assert request.top_k == 7


def test_from_parsed_query_dict_handles_missing_optional_fields() -> None:
    request = from_parsed_query_dict({"required_terms": ["python"]}, top_k=3)

    assert request.query_terms == ("python",)
    assert request.excluded_terms == ()
    assert request.filters == ()


def test_from_raw_text_applies_nfc_and_casefold() -> None:
    request = from_raw_text("Búsqueda EN Español", top_k=5)
    assert request.query_terms == ("búsqueda", "en", "español")


def test_from_raw_text_ignores_punctuation() -> None:
    request = from_raw_text("search-engine, ranking!", top_k=5)
    assert request.query_terms == ("search", "engine", "ranking")
