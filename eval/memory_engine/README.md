# Memory Engine Badcase Probes

This directory is for memory-engine-level evaluation. It does not measure final
agent QA quality. It focuses on whether `memory2` writes, retrieves, ranks, and
injects the right memory items.

## Isolation

Daily workspace extraction is read-only:

- SQLite databases are opened with read-only URI modes.
- `recall_inspector.jsonl` is read as an append-only log.
- Generated daily cases and reports are written under this directory and are
  ignored by git because they can contain private conversation snippets.
- No script writes back to the source workspace.

## Outputs

`badcases/daily/*.json` contains reusable probe cases. A case may include the
minimal source message window needed to recreate the memory context in a
sandbox.

`reports/*.jsonl` contains one JSON object per extracted finding or probe. It is
an audit/index stream: case id, failure type, query, expected ids, hit ids,
rank, paths, and extraction metadata. It is not the primary replay artifact; use
the corresponding case JSON for replay.

## Daily Extraction

```bash
python3 -m eval.memory_engine.daily_badcase_extract \
  --workspace ~/.akashic/workspace \
  --badcase-dir eval/memory_engine/badcases/daily \
  --report eval/memory_engine/reports/daily_extract.jsonl
```

The extractor currently emits:

- `explicit_empty_recall`: explicit `recall_memory` returned no items.
- `explicit_recall_review`: explicit recall calls with observed ranked hits.
- `short_query_over_recall`: very short user turns that still injected memories.
- `sticky_memory`: memory items injected unusually often.
- `memory_item_probe`: positive-control probes derived from active memory items.

## Daily Sandbox Eval

```bash
python3 -m eval.memory_engine.daily_badcase_eval \
  --workspace ~/.akashic/workspace \
  --badcase-dir eval/memory_engine/badcases/daily \
  --report eval/memory_engine/reports/daily_eval.jsonl
```

The eval runner creates an isolated copy of `memory/memory2.db` under
`eval/memory_engine/sandbox/` and runs every case against that copy. Source
workspace data remains read-only.

The runner reports:

- `memory_item_probe` / `positive_control`: `recall@1/3/5/8`, `rank`, `MRR`.
- `short_query_over_recall`: retrieved hit count and injected memory count.
- `explicit_empty_recall`: whether the replay is still empty.
- `explicit_recall_review`: overlap with the originally observed hit ids.
- `sticky_memory`: whether the sticky item appears in sampled replay turns.
- Lane trace for retrieval cases:
  - `dense`: current `memory2` vector cosine + hotness lane.
  - `keyword`: current `memory2.keyword_search_summary` summary LIKE lane.
  - `bm25_summary`: eval-only BM25 over `memory_items.summary` for comparison.
  - `fusion`: current `memory2` RRF fusion of dense + keyword.

Use `--embedding-mode keyword-only` for a no-network smoke run. The default
`auto` mode uses the configured embedding service and records the effective
mode in the summary.

## Dense And BM25 Top-N Eval

```bash
python3 -m eval.memory_engine.daily_top10_eval \
  --badcase-dir eval/memory_engine/badcases/daily_deduped \
  --report eval/memory_engine/reports/daily_top10_eval.jsonl
```

This script copies `memory/memory2.db` into a sandbox workspace, requires live
embedding, and reports dense-only and eval-only BM25 summary retrieval with a
maximum of 10 hits per lane. Metrics use each lane's actual returned `N`, so
the JSONL and generated Markdown report expose `top_n`, matched hit count, and
`precision@topn`, `recall@topn`, `f1@topn`. It exits with an error instead of
silently falling back when embedding is unavailable.

For an empty-gold case, an empty result is recorded as `empty_gold_correct`
instead of inventing a `precision@0` value. Use these cases to measure
over-recall separately from precision and recall.

## Badcase Lane Eval

`badcases/badcase_with_expectedid/` is the private, appendable input directory
for badcases that have been manually labeled with `expected.memory_ids`.
Generated reports stay under `reports/`, and sandbox DB copies stay under
`sandbox/`; none of these paths are tracked by Git.

Initialize the current labeled review set:

```bash
python3 -m eval.memory_engine.badcase_lane_eval \
  --init-from eval/memory_engine/badcases/high_review_20260602 \
  --badcase-dir eval/memory_engine/badcases/badcase_with_expectedid
```

Run lane diagnostics:

```bash
python3 -m eval.memory_engine.badcase_lane_eval \
  --badcase-dir eval/memory_engine/badcases/badcase_with_expectedid
```

The runner groups cases by `source_workspace` by default and copies each
source `memory/memory2.db` into an isolated sandbox before replay. It reports
per-case dense, keyword summary LIKE, eval-only BM25 summary, and RRF fusion
`precision@topn`, `recall@topn`, and `f1@topn`, where `N` is each lane's actual
returned count. It also tags chain categories such as dense-hit/keyword-miss,
dense-miss/keyword-hit, both-miss-with-different-results, and both-hit-but-RRF
missed. Sticky-memory cases are kept as target-presence diagnostics and are not
included in precision/recall aggregates.

## Pull Remote Workspace Snapshots

Copy the tracked example config and fill in your SSH alias and paths:

```bash
cp eval/memory_engine/remote_sync.example.toml \
  eval/memory_engine/remote_sync.local.toml
```

The local config, pulled snapshots, generated badcases, and reports are ignored
by Git. Prefer an SSH alias in `~/.ssh/config` so credentials never appear in
the TOML file. SSH runs in batch mode, so scheduled runs fail instead of
waiting for an interactive password prompt.

Pull a consistent remote snapshot and optionally extract badcases:

```bash
python3 -m eval.memory_engine.pull_remote_snapshot
```

The pull runner sends `snapshot_workspace.py` to the remote host over SSH. The
remote script uses SQLite's backup API for `sessions.db`, `memory/memory2.db`,
and optional `observe/observe.db`; it also copies complete lines from
`observe/recall_inspector.jsonl` and copies `memory/*.json`. It never writes to
the live workspace. The downloaded snapshot is checksum-verified before the
local `latest` pointer is updated.

For local scheduling, run the pull command from cron or a Windows scheduled
task. To generate remote snapshots independently, deploy the repository on the
server and run:

```bash
python3 -m eval.memory_engine.snapshot_workspace \
  --workspace ~/.akashic/workspace \
  --output-root ~/.akashic/memory-engine-snapshots \
  --retention 7
```
