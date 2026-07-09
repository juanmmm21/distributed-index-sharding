from __future__ import annotations

import json
from pathlib import Path

import pytest
from bm25_ranking_engine.pipeline import BM25RankingPipeline

from distributed_index_sharding.partitioning import partition_index, read_cluster_manifest


def test_partition_assigns_documents_by_doc_id_modulo(tiny_index_dir: Path, tmp_path: Path) -> None:
    partition_index(tiny_index_dir, tmp_path, num_shards=2)

    shard_0_docs = [
        json.loads(line)["doc_id"]
        for line in (tmp_path / "shard-0" / "documents.jsonl").read_text().splitlines()
    ]
    shard_1_docs = [
        json.loads(line)["doc_id"]
        for line in (tmp_path / "shard-1" / "documents.jsonl").read_text().splitlines()
    ]

    assert shard_0_docs == [0, 2, 4]
    assert shard_1_docs == [1, 3]


def test_partition_recomputes_local_document_frequency(
    tiny_index_dir: Path, tmp_path: Path
) -> None:
    partition_index(tiny_index_dir, tmp_path, num_shards=2)

    def _document_frequency(shard_dir_name: str, term: str) -> int:
        for line in (tmp_path / shard_dir_name / "postings.jsonl").read_text().splitlines():
            data = json.loads(line)
            if data["term"] == term:
                return int(data["document_frequency"])
        raise AssertionError(f"término '{term}' no encontrado en {shard_dir_name}")

    # "python" aparece globalmente en doc_ids 0, 1, 3 (ver samples/tiny_example).
    assert _document_frequency("shard-0", "python") == 1  # solo doc 0
    assert _document_frequency("shard-1", "python") == 2  # docs 1 y 3


def test_partition_omits_terms_absent_from_a_shard(tiny_index_dir: Path, tmp_path: Path) -> None:
    partition_index(tiny_index_dir, tmp_path, num_shards=2)

    shard_0_terms = {
        json.loads(line)["term"]
        for line in (tmp_path / "shard-0" / "postings.jsonl").read_text().splitlines()
    }
    # "javascript" solo aparece en doc_id=2, que cae en shard-0 (2 % 2 == 0).
    assert "javascript" in shard_0_terms
    # "programming" solo aparece en doc_id=3, que cae en shard-1.
    assert "programming" not in shard_0_terms


def test_partition_writes_cluster_manifest(tiny_index_dir: Path, tmp_path: Path) -> None:
    written_manifest = partition_index(tiny_index_dir, tmp_path, num_shards=3)

    read_back_manifest = read_cluster_manifest(tmp_path)

    assert read_back_manifest == written_manifest
    assert read_back_manifest.num_shards == 3
    assert read_back_manifest.shard_dir_names == ("shard-0", "shard-1", "shard-2")


def test_partition_rejects_non_positive_num_shards(tiny_index_dir: Path, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="num_shards"):
        partition_index(tiny_index_dir, tmp_path, num_shards=0)


def test_partitioned_shards_are_independently_queryable(
    tiny_index_dir: Path, tmp_path: Path
) -> None:
    partition_index(tiny_index_dir, tmp_path, num_shards=2)

    with BM25RankingPipeline(tmp_path / "shard-0") as shard_0_pipeline:
        results = shard_0_pipeline.search_tokens(["python"], top_k=10)
        assert {scored.doc_id for scored in results} == {0}

    with BM25RankingPipeline(tmp_path / "shard-1") as shard_1_pipeline:
        results = shard_1_pipeline.search_tokens(["python"], top_k=10)
        assert {scored.doc_id for scored in results} == {1, 3}
