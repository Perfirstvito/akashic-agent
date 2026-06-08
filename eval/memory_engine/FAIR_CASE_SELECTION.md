# Fair Memory Engine Case Selection

This note defines the fair core set used to compare `memory2` and `akasha`.
The concrete case JSON files stay under `eval/memory_engine/badcases/`, which is
ignored because cases may contain private conversation snippets.

## Fairness Standard

A case is fair only if both engines can be judged against the same underlying
evidence:

- It is a real daily recall scenario, not only a synthetic positive control.
- The query is understandable from the case itself, or has enough concrete
  anchors to evaluate without replaying omitted previous turns.
- `expected.memory_ids` exist and are active in the current `memory2.db`.
- `expected.memory_items[].source_ref` resolves to original messages in the
  current `sessions.db`.
- Gold targets can be normalized into evidence groups shared by both engines.
- Near-duplicate topics are collapsed into one representative case.
- The set covers profile, operational task, project, paper, and update recall.

## Exclusions

The core set excludes:

- no-gold or no-current-active-gold cases;
- sticky-memory injection-frequency diagnostics;
- pure greetings, pronoun-only turns, or highly omitted-context short queries;
- soft or partial labels;
- very broad old-summary categories with many plausible matches, except one
  explicit broad-recall stress case;
- near duplicates of an already selected coverage bucket.

## Scoring

Use evidence groups as the shared unit. One expected old memory item becomes one
group containing its old `memory_id`, source message ids, and Akasha turn keys.

- `memory2` hits a group when it returns the exact expected memory id, or a
  returned memory whose `source_ref` overlaps the group.
- `akasha` hits a group when it returns a turn/source ref that overlaps the
  group.
- Primary metrics: Hit@1, Hit@3, Hit@8, MRR, and evidence-group coverage@N.
- Secondary metrics: normalized precision and unjudged/over-recalled result
  count.
- Broad-recall stress cases are not all-or-nothing exact-id tests; evaluate
  first useful hit and group coverage.

## Current Core Set

The selected local set is:

`eval/memory_engine/badcases/fair_core_20260608/`

The case copies in that directory point `source_workspace` to the current
migrated runtime workspace, so `memory2` and `akasha` read the same source data.
The original snapshot path is preserved in
`case.fair_selection.original_source_workspace`.

It contains 12 cases:

- command/tool usage
- API billing/account recall
- profile/location recall
- daily plan and temporal recall
- file/PDF delivery recall
- concrete update recall
- broad research-task stress
- research-paper topic recall
- project capability/TODO recall
- account conversion workflow recall

See that directory's `_manifest.json` for normalized evidence groups and
`_selection_report.json` for included/excluded decisions.
