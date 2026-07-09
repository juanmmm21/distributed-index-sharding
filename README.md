# distributed-index-sharding

**Project 9/10 of the [`beacon-search-engine`](https://github.com/juanmmm21/beacon-search-engine) ecosystem** — *Query Service* category.
Repository: [`github.com/juanmmm21/distributed-index-sharding`](https://github.com/juanmmm21/distributed-index-sharding)

A sharded inverted index with distributed query fan-out and ranked result
merging across nodes. Partitions an inverted index by document across
independent shard processes, fans a parsed query out to every shard
concurrently over HTTP with a per-shard timeout and explicit tolerance to
shard failure, and fuses the already-ranked partial results of every healthy
shard into a single final list with an efficient k-way merge — no
Elasticsearch, no Solr, no distributed-systems framework: the partitioning
strategy, the async fan-out, and the merge are all implemented here.

## What problem it solves

No inverted index of real size fits on one machine, and no single process
can answer thousands of queries per second against a corpus of millions of
documents. Real search engines split the index across many machines (shards),
send every query to all of them at once, and combine whatever comes back —
accepting that some shards will occasionally be slow or unreachable, and that
the whole system must keep answering anyway. This project is exactly that
layer: given an index already built by
[`inverted-index-builder`](https://github.com/juanmmm21/inverted-index-builder)
(optionally compressed by
[`index-compression-codec`](https://github.com/juanmmm21/index-compression-codec)),
it partitions it into independent shards, each backed by its own
[`bm25-ranking-engine`](https://github.com/juanmmm21/bm25-ranking-engine)
instance, and answers a query by asking every shard at once and merging their
answers — the same shape of problem every large-scale search engine (and
every sharded database) has to solve, reduced to a small, fully working,
locally-runnable system.

## Role in `beacon-search-engine`

```text
inverted-index-builder ── index-compression-codec
                                  │ documents.jsonl, postings, stats.json
                                  ▼
                    ┌─────────────────────────────┐
                    │  distributed-index-sharding   │   (this project)
                    │  partition -> shard-0 .. N-1   │
                    └───────────────┬─────────────────┘
                                    │ N self-contained index dirs
                                    ▼
                      ┌───────────────────────────┐
   ParsedQuery ──────▶│  fan-out (asyncio + HTTP)  │◀── bm25-ranking-engine
 (query-parser-        │  one BM25RankingPipeline    │    (one instance per shard,
  autocomplete)        │  per shard, in its own       │     used as a package)
                        │  process                      │
                        └───────────────┬───────────────┘
                                        │ k-way merge of per-shard
                                        │ already-ranked results
                                        ▼
                             (converges in beacon-search-console)
```

This is the last piece of query-serving infrastructure before the flagship
app. It consumes the index directory produced by `inverted-index-builder` (or
`index-compression-codec`) and the structured query produced by
[`query-parser-autocomplete`](https://github.com/juanmmm21/query-parser-autocomplete)
(`ParsedQuery`); it delegates all scoring to `bm25-ranking-engine`, used as a
real Python package dependency inside every shard process — the one
documented exception to this ecosystem's "no cross-repo imports" rule,
because `bm25-ranking-engine` produces no on-disk result format of its own
(see *Data formats* below). Its own output — a single merged, ranked list —
is what
[`beacon-search-console`](https://github.com/juanmmm21/beacon-search-console)
will present to the user, optionally after
[`learning-to-rank-reranker`](https://github.com/juanmmm21/learning-to-rank-reranker)
reorders the top candidates.

## Goal and skills demonstrated

- **A partitioning strategy chosen and justified, not assumed**: document-based
  sharding (`doc_id % num_shards`) over term-range sharding, with the
  trade-off spelled out — a shard's BM25 score uses shard-local statistics
  (`document_frequency`, `average_document_length`), the same local-IDF
  approximation real distributed search engines (Elasticsearch, SolrCloud)
  accept by default in exchange for every shard's result already being
  final and independently rankable, which is exactly the precondition an
  efficient merge needs. See `partitioning.py`'s module docstring for the
  full reasoning, including why term-range partitioning was rejected.
- **Async fan-out with real, tested failure handling**: every shard is asked
  concurrently (`asyncio.gather`) with a per-shard timeout, and a shard that
  times out, refuses the connection, or returns a malformed response never
  raises — it degrades to a `ShardOutcome` the coordinator folds into a
  partial result, verified by an end-to-end test that kills a real shard
  process mid-session (`tests/test_cluster_end_to_end.py`) and checks the
  search still returns the other shards' results.
- **An actual k-way merge, not "sort everything again"**: `merge.py` uses
  `heapq.merge` with a `(score, -doc_id)` key over already-sorted per-shard
  lists — `O(kn log k)` instead of `O(kn log(kn))` — and stays lazy through
  `top_k` truncation via `itertools.islice`.
- **Reusing a sibling project as a real dependency, deliberately, once**: every
  other integration in this ecosystem is file-format-only, but
  `bm25-ranking-engine` documents that its result is purely in-memory — so
  this project depends on it as an actual installable package (a direct git
  URL dependency, `tool.hatch.metadata.allow-direct-references`), the same
  exception `learning-to-rank-reranker`'s README already documents for any
  future caller of `bm25-ranking-engine`.
- **Local multi-process simulation used as the real test harness, not a mock**:
  `cluster.py` spawns actual `serve-shard` OS subprocesses on real ports and
  waits on real `/health` HTTP checks — `tests/test_cluster_end_to_end.py`
  runs a real query over real sockets and kills a real process, instead of
  only testing the merge logic against canned data.
- **Honest scope on query operators over a scorer that doesn't understand
  them**: `bm25-ranking-engine.search_tokens` only accepts a flat bag of
  terms — no phrases, no exclusion, no filters. This project flattens
  `required_terms` + phrase terms into that bag and resolves
  `excluded_terms`/`site:` filters as a shard-local post-filter over already
  BM25-ranked candidates, documenting explicitly (not hiding) that v1 cannot
  reconstruct true phrase adjacency, because `bm25-ranking-engine`'s result
  carries no term positions. See *Troubleshooting*.

## How it works

1. **Partitioning** (`partitioning.py`): `partition_index` opens the source
   index — plain (`inverted-index-builder`) or compressed
   (`index-compression-codec`) — through `bm25_ranking_engine.index_reader`
   (reused rather than reimplemented a second time, see the module's
   docstring), assigns every `doc_id` to shard `doc_id % num_shards`, and
   writes each shard as a self-contained, uncompressed
   `inverted-index-builder`-format directory: only the postings whose
   `doc_id` falls in that shard, with `document_frequency` and
   `average_document_length` recomputed from that shard's own subset of
   documents. A `cluster_manifest.json` at the output root records how many
   shards were written and their directory names.
2. **Serving a shard** (`shard_server.py`): each shard is an independent
   `aiohttp` process that opens one `BM25RankingPipeline` over its own
   directory at startup (auto-detecting plain or compressed format, exactly
   like `bm25-ranking-engine`'s CLI does) and exposes `POST /search` +
   `GET /health`. A search over-fetches (`top_k * 3` by default) from BM25,
   then applies `excluded_terms`/`site:` filters locally (`post_filtering.py`)
   before truncating to the requested `top_k` — the shard is the only place
   that has both the postings needed to check exclusion and the document
   metadata needed to check a filter, so resolving both there avoids a second
   network round trip.
3. **Translating the query** (`query_translation.py`): `from_parsed_query_dict`
   takes `ParsedQuery.to_json_dict()` from `query-parser-autocomplete` and
   flattens `required_terms` plus every phrase's terms into the flat
   `query_terms` bag BM25 understands, carrying `excluded_terms`/`filters`
   through unchanged for the shard to resolve.
4. **Fan-out and merge** (`coordinator.py`, `shard_client.py`, `merge.py`):
   `SearchCoordinator.search` issues one HTTP request per shard concurrently
   via `asyncio.gather`, each wrapped in its own timeout; `HttpShardTransport`
   guarantees it never raises — timeouts, connection failures and malformed
   responses all become a `ShardOutcome` with a non-`"ok"` status instead.
   `merge_shard_outcomes` then k-way-merges only the healthy shards' already
   score-sorted results.
5. **Local cluster simulation** (`cluster.py`): `LocalShardCluster.start`
   reads a `cluster_manifest.json`, launches one `serve-shard` subprocess per
   shard on consecutive ports, and polls each `/health` endpoint before
   returning — the "several processes or ports" local simulation this
   project's Definition of Done requires, with no real deployment
   infrastructure involved. `LocalShardCluster.kill(shard_id)` lets tests (and
   manual smoke-testing) simulate a shard going down mid-session.

## Architecture

```text
src/distributed_index_sharding/
├── models.py                # ShardTarget, ShardSearchRequest/Response, ShardHit, ShardOutcome, ...
├── protocols.py               # ShardTransport interface
├── partitioning.py             # document-based partitioner (doc_id % num_shards)
├── post_filtering.py            # excluded_terms / site: filter, applied shard-side
├── query_translation.py          # ParsedQuery JSON -> ShardSearchRequest
├── merge.py                       # k-way merge of already-ranked shard results
├── shard_server.py                 # aiohttp server for one shard (POST /search, GET /health)
├── shard_client.py                  # HTTP transport: never raises, degrades to ShardOutcome
├── coordinator.py                    # async fan-out to all shards + merge
├── cluster.py                         # local multi-process shard simulation
├── pipeline.py                         # ties cluster + coordinator + translation together
└── __main__.py                          # CLI (partition, serve-shard, simulate, search)
```

## Requirements and installation

- Python `>=3.11`
- Runtime dependencies: `aiohttp` (async HTTP client/server for the fan-out and
  the shard servers) and `bm25-ranking-engine`, installed directly from its
  GitHub repository — it is not published on PyPI, and this is the one
  documented exception to this ecosystem's "no cross-repo Python imports"
  rule (see *Role in `beacon-search-engine`* above).

```bash
git clone https://github.com/juanmmm21/distributed-index-sharding.git
cd distributed-index-sharding
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Usage (CLI)

```bash
# 1. Partition an index (from inverted-index-builder or index-compression-codec)
#    into N document-based shards.
distributed-index-sharding partition ./index ./partitioned --num-shards 3

# 2. Try it end-to-end against the tiny fixture checked into samples/:
#    launches 3 real shard processes, runs one query, prints the merged
#    result, then shuts every shard down.
distributed-index-sharding partition samples/tiny_example/index /tmp/shards --num-shards 3
distributed-index-sharding simulate /tmp/shards --text "python" --top-k 5

# 3. Same, but with a real ParsedQuery from query-parser-autocomplete
#    (operators: exact phrase, exclusion, filter):
distributed-index-sharding simulate /tmp/shards \
  --parsed-query-file ./parsed-query.json --top-k 5

# 4. Keep a local cluster running to test 'search' against it manually, or to
#    kill one of its shard processes and watch the next search degrade:
distributed-index-sharding simulate /tmp/shards --keep-running &
distributed-index-sharding search \
  --shard 0:127.0.0.1:9300 --shard 1:127.0.0.1:9301 --shard 2:127.0.0.1:9302 \
  --text "python" --top-k 5

# 5. Serve a single shard directly (what 'simulate'/'cluster.py' launch as a
#    subprocess under the hood) — useful to run shards on separate machines:
distributed-index-sharding serve-shard /tmp/shards/shard-0 \
  --shard-id 0 --host 0.0.0.0 --port 9300
```

Real transcript against the fixture in `samples/tiny_example/` (3 shards,
query `python`, excluding `tutorial`, filtered to `site:example.com`):

```text
$ distributed-index-sharding simulate /tmp/shards \
    --parsed-query-file /tmp/parsed-query.json --top-k 10
Clúster local levantado: http://127.0.0.1:9300, http://127.0.0.1:9301, http://127.0.0.1:9302
  shard 0: ok [1.4 ms, 1 hits]
  shard 1: ok [1.3 ms, 1 hits]
  shard 2: ok [1.1 ms, 0 hits]
1. doc_id=1 score=0.6659
2. doc_id=3 score=0.1823
```

It can also be invoked as a module: `python -m distributed_index_sharding partition ...`.

## Data formats

**Input — source index**: either the uncompressed format of
[`inverted-index-builder`](https://github.com/juanmmm21/inverted-index-builder#data-formats)
or the compressed one of
[`index-compression-codec`](https://github.com/juanmmm21/index-compression-codec#data-formats)
— `partition` detects which via `manifest.json`, the same way
`bm25-ranking-engine` does.

**Input — parsed query**: `ParsedQuery.to_json_dict()` from
[`query-parser-autocomplete`](https://github.com/juanmmm21/query-parser-autocomplete#data-formats):

```json
{
  "required_terms": ["python", "tutorial"],
  "phrases": [["exact", "phrase"]],
  "excluded_terms": ["deprecated"],
  "filters": [{"field": "site", "value": "example.com"}],
  "corrections": [{"original": "pyhton", "corrected": "python"}]
}
```

**Output — `partition`**: `output_root/cluster_manifest.json` plus one
`output_root/shard-<id>/` directory per shard, each a complete, independently
valid `inverted-index-builder`-format index (`manifest.json`,
`documents.jsonl`, `postings.jsonl`, `stats.json` — see that project's
README for the exact per-file schema):

```json
{"num_shards": 3, "shard_dir_names": ["shard-0", "shard-1", "shard-2"]}
```

**Shard HTTP contract** — `POST /search`, request body:

```json
{"query_terms": ["python"], "excluded_terms": ["tutorial"], "filters": [{"field": "site", "value": "example.com"}], "top_k": 10}
```

response body:

```json
{"shard_id": 0, "hits": [{"doc_id": 3, "score": 0.1823}]}
```

`GET /health` responds `{"shard_id": 0, "status": "ok"}` once the shard has
finished loading its index.

**In-memory only — `bm25-ranking-engine`**: this project depends on it as a
real package (`BM25RankingPipeline`), not through a file format — its own
README documents that its result (`list[ScoredDocument]`) is never serialized
to disk, so there is nothing to document here beyond "installed and imported
directly" (see *Requirements and installation*).

## Programmatic usage

```python
import asyncio
from pathlib import Path

from distributed_index_sharding.partitioning import partition_index
from distributed_index_sharding.pipeline import DistributedSearchPipeline


async def main() -> None:
    partition_index(Path("./index"), Path("./partitioned"), num_shards=3)

    async with await DistributedSearchPipeline.start(Path("./partitioned")) as pipeline:
        # Primary contract: a ParsedQuery from query-parser-autocomplete
        result = await pipeline.search_parsed_query(
            {"required_terms": ["python"], "excluded_terms": ["tutorial"]}, top_k=10
        )
        for hit in result.merged:
            print(hit.doc_id, hit.score)
        if result.failed_shard_ids:
            print("degraded, missing shards:", result.failed_shard_ids)


asyncio.run(main())
```

## Development

```bash
pytest
ruff check .
ruff format --check .
mypy --strict src/
```

Tests cover the domain models (JSON round-trips, `top_k` validation), the
merge (deterministic fusion of disjoint and tied score lists, truncation,
all-shards-failed), query translation (flattening `required_terms` +
`phrases`, raw-text NFC/casefold tokenization), the post-filter
(`excluded_terms` membership from a real fixture, the `site:` filter
including subdomains, graceful degradation to a no-op over a compressed
shard), the partitioner (`doc_id % num_shards` assignment, local
`document_frequency` recomputation, terms absent from a shard being omitted,
and every resulting shard being independently queryable through a real
`BM25RankingPipeline`), the coordinator (using an in-memory
`ShardTransport` double: healthy merge, degradation on a timed-out shard
without raising, and the all-shards-down case), the shard's HTTP contract
(`aiohttp.test_utils`, including a malformed request returning `400`), and,
critically, an end-to-end test that partitions the fixture into 3 shards,
launches 3 real `serve-shard` OS subprocesses, runs a real query over real
sockets, kills one process mid-session, and asserts the next query still
returns the surviving shards' results instead of raising
(`tests/test_cluster_end_to_end.py`).

## Adding or removing a shard without downtime

Because each shard is a fully self-contained index directory and the
coordinator only needs a list of `ShardTarget`s to fan a query out to, the
cluster can be resized without stopping the shards that are already serving
traffic:

- **Adding a shard**: re-run `partition` with a larger `--num-shards` against
  the *original* source index into a new output directory, start
  `serve-shard` for the newly created shard directories, and update the
  `ShardTarget` list any caller passes to `SearchCoordinator`/
  `DistributedSearchPipeline` to include the new target — existing shards
  keep answering queries throughout, since nothing about them changes.
  Note this reshuffles which shard owns which `doc_id` (`doc_id % num_shards`
  changes when `num_shards` does), so the already-running old shards would
  now hold a different partition than a fresh `num_shards`-way split — in
  production this step is paired with retiring the old shard set once the new
  one is confirmed healthy, the same rolling-cutover pattern real sharded
  systems use, not a live re-partition of data already sitting on a shard.
- **Removing a shard**: stop sending it new queries by dropping its
  `ShardTarget` from the list passed to `SearchCoordinator` (the coordinator
  never discovers shards on its own — the caller always hands it an explicit
  list), then shut its process down. Because a search coordinator with N-1
  targets simply search N-1 shards and merges what comes back, the cluster
  keeps answering immediately — the fan-out/merge path is exactly the code
  path already exercised when a shard fails a health check or times out (see
  `tests/test_cluster_end_to_end.py`), not a separate mechanism. The
  documents that only lived on the removed shard become unsearchable until a
  re-partition restores them somewhere else — an explicit, documented
  consequence of resizing rather than a hidden data-loss bug.
- `LocalShardCluster` (the local simulation used for development and tests)
  intentionally does not implement live resizing itself — it always starts
  from a fixed `cluster_manifest.json` — because the mechanics above (which
  targets a coordinator is given) are the actual answer to "add/remove a
  shard," independent of whether the shards are local subprocesses or real
  machines; building a second, parallel resizing API on `LocalShardCluster`
  would duplicate that answer instead of demonstrating it.

## Troubleshooting

- **A phrase query (`"exact phrase"`) scores identically to its terms
  requested separately**: expected in v1. `required_terms` and every
  phrase's terms are flattened into the same flat bag before scoring
  (`query_translation.py`), because `bm25-ranking-engine.search_tokens` has
  no concept of phrase adjacency and its result (`ScoredDocument`) carries no
  term positions to reconstruct it downstream either. True phrase matching
  would require a scorer that exposes matched positions — out of scope here,
  documented rather than faked. See `post_filtering.py`'s module docstring.
- **`excluded_terms`/`filters` silently stop working after I compressed a
  shard with `index-compression-codec`**: also expected. `ExclusionIndex`
  reads a shard's own `postings.jsonl` directly; a shard served from a
  compressed `index-compression-codec` directory (an optional step this
  project doesn't perform itself, see `partitioning.py`) has no
  `postings.jsonl` next to its binary blob, so that shard's exclusion/filter
  post-filtering degrades to a no-op with a logged warning instead of
  crashing the shard. Scoring itself is unaffected either way, since
  `BM25RankingPipeline` auto-detects both formats.
- **A shard returns fewer than `top_k` results even though more candidates
  exist**: `shard_server.py` over-fetches `top_k * 3` candidates from BM25
  before applying `excluded_terms`/`filters`, to absorb the common case where
  a handful of top candidates get filtered out. If more than two thirds of a
  shard's top candidates are excluded, this mitigation isn't enough — a known,
  documented v1 trade-off (see `shard_server.py`'s module docstring), not a
  silent bug.
- **A search returns fewer documents than expected and one `ShardOutcome` has
  `status: "timeout"` or `"error"`**: this is the tolerance to shard failure
  working as designed, not a bug — check `error_message` on that outcome
  (connection refused almost always means the shard process isn't running or
  is bound to a different port/host than the `ShardTarget` you passed).
- **`FileNotFoundError: ... cluster_manifest.json`** when calling `simulate`
  or `LocalShardCluster.start`: that directory was never partitioned — run
  `partition` against it first.
- **`ValueError: no existe shard_id=... en este clúster`** from
  `LocalShardCluster.kill`: the `shard_id` doesn't match any target this
  cluster started; shard IDs are always `0..num_shards-1`, in the order
  `partition` created them.
- **`mypy` fails under `tests/`**: expected — only `src/` is type-checked in
  `--strict` mode; test fixtures use deliberately looser typing.

## License

MIT — see [`LICENSE`](./LICENSE).
