from __future__ import annotations

from pathlib import Path

from bm25_ranking_engine.models import DocumentRecord

from distributed_index_sharding.models import QueryFilter
from distributed_index_sharding.post_filtering import ExclusionIndex, is_excluded, matches_filters


def test_exclusion_index_contains_known_postings(tiny_index_dir: Path) -> None:
    exclusion_index = ExclusionIndex.build(tiny_index_dir)

    assert exclusion_index.contains("python", 0) is True
    assert exclusion_index.contains("python", 1) is True
    assert exclusion_index.contains("python", 2) is False
    assert exclusion_index.contains("nonexistent-term", 0) is False


def test_is_excluded_true_when_any_term_matches(tiny_index_dir: Path) -> None:
    exclusion_index = ExclusionIndex.build(tiny_index_dir)

    assert is_excluded(exclusion_index, doc_id=0, excluded_terms=["python"]) is True
    assert is_excluded(exclusion_index, doc_id=0, excluded_terms=["javascript"]) is False
    assert is_excluded(exclusion_index, doc_id=0, excluded_terms=["javascript", "python"]) is True


def test_exclusion_index_degrades_to_empty_for_compressed_shard(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(
        '{"format_version": 1, "terms_file": "terms.jsonl", "postings_file": "postings.bin", '
        '"documents_file": "documents.jsonl", "stats_file": "stats.json", "block_size": 128, '
        '"bitpack_threshold": 64}',
        encoding="utf-8",
    )

    exclusion_index = ExclusionIndex.build(tmp_path)

    assert is_excluded(exclusion_index, doc_id=0, excluded_terms=["python"]) is False


def test_matches_filters_resolves_site_field() -> None:
    document = DocumentRecord(doc_id=0, url="https://example.com/article", title="t", length=10)

    assert matches_filters(document, [QueryFilter(field="site", value="example.com")]) is True
    assert matches_filters(document, [QueryFilter(field="site", value="other.com")]) is False


def test_matches_filters_accepts_subdomains() -> None:
    document = DocumentRecord(
        doc_id=0, url="https://docs.example.com/article", title="t", length=10
    )

    assert matches_filters(document, [QueryFilter(field="site", value="example.com")]) is True


def test_matches_filters_ignores_unknown_field() -> None:
    document = DocumentRecord(doc_id=0, url="https://example.com/article", title="t", length=10)

    assert matches_filters(document, [QueryFilter(field="filetype", value="pdf")]) is True


def test_matches_filters_empty_list_always_matches() -> None:
    document = DocumentRecord(doc_id=0, url="https://example.com/article", title="t", length=10)
    assert matches_filters(document, []) is True
