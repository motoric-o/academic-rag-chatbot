# Data Sources (Member B)

Document the source campus regulation PDF here (title, issuing institution, year, page count, redistribution policy, SHA-256 checksums of `source_id/` and frozen `source_en/corpus.json`).

| Artifact | Path | SHA-256 | Notes |
|---|---|---|---|
| Original Indonesian PDF | `data/source_id/` | `9e058c8ae166bb32b55a42807700bf586782c7ca7bca8715242450bd5fa6fc1b` | Check redistribution rights before committing |
| Frozen English corpus | `data/source_en/corpus.json` | Pending bilingual review | The freeze command writes `corpus.sha256` (§5) |

## Week 1 extraction

The official source is [Buku Panduan Akademik Program Sarjana Teknik
Informatika, 2024](https://if.it.maranatha.edu/wp-content/uploads/2025/01/Buku-Panduan-S1-Teknik-Informatika-v2024.pdf).
It contains 74 PDF pages and 50 numbered bookmarks (10 sections, 26 subsections,
14 sub-subsections).

Use `data/source_id/` for the local PDF or pass its path. Confirm institutional
terms before sending text to a hosted service or publishing the translated corpus.
Keep source files, generated drafts and prompt logs local while that is pending.

The parser starts at the first numbered body bookmark and uses the headings to
split the text into sections. It keeps list items and table text in layout order, so
table cells and figure contents need checking against the PDF when used as evidence.
Numbered sections with no body keep an empty `text` field.

    uv run python scripts/build_corpus.py path/to/handbook.pdf

This writes `.tmp/corpus.id.json`. The command prints the SHA-256 of the output.
When the hosted translator is approved, use the team's confirmed Gemini model ID with the `gemini` extra and a local `GEMINI_API_KEY`:

    uv run --extra gemini python scripts/build_corpus.py path/to/handbook.pdf --model TEAM_MODEL_ID --output .tmp/corpus.en.draft.json --prompt-log .tmp/translation-prompts.jsonl

The English output is for bilingual review. The prompt log contains the original
section text. Pick new filenames for reruns. After source and license checks,
review at least 10 to 15 percent of the sections including every section used by
eval questions before treating the English corpus as frozen.

## Week 2 review and retrieval

1. After the source terms and team model are confirmed, build the English draft
   with the command above. Add `--priority` and use a separate draft and
   prompt log to draft the 16 priority sections for early review. Member A reviews
   the sections in [the bilingual handoff](../docs/handoffs/a_to_b.md) plus every
   section cited by eval questions. Record one final review row per section in
   [the review log](review_log.csv). Apply corrections to
   `.tmp/corpus.en.reviewed.json`. Record unresolved items as `flagged`.
2. All four members write approximately ten dev questions each. Member B
   validates their combined set with [the annotation guide](eval/ANNOTATION_GUIDE.md).
   Run the freeze command after the 40 dev questions and the review are ready.

       uv run python -m regqa.translate.review_log --eval data/eval/eval_set.jsonl --reviewed .tmp/corpus.en.reviewed.json --freeze data/source_en/corpus.json

   Without `--freeze` the command reports review coverage, pending sections, and
   the review error rate. Freeze writes the English corpus and checksum without
   replacing existing files. The command expects `.tmp/corpus.id.json` from
   extraction, or an explicit `--source` path.
3. Confirm the pinned embedding model in `configs/retrieval.yaml`. Run
   `make calibrate` (or `uv run python -m regqa.retrieval.calibrate_chunks` on
   Windows). It writes `data/cache/chunk_calibration.json` using dev only.
   Copy `chosen_max_chunk_tokens` into `chunking.max_chunk_tokens`. Then use
   `make cache` or `uv run python scripts/build_cache.py`. The cache preserves
   five ranked passages per question. The prompt uses the first four.
4. The later 110-question test split can be locked with
   `uv run python -m regqa.evaluation.validate_eval data/eval/eval_set.jsonl --corpus data/source_en/corpus.json --lock-test`.
   Review any newly cited sections before locking the test split. If a dev
   cache already exists, use `--output-dir` to build the combined
   dev/test cache as a fresh artifact with the original calibration. Compare
   the dev rows and coordinate the new cache path with the runners.

Both retrieval commands accept `--synthetic` for the Week 1 fixture and keep
synthetic outputs under `.tmp/`. Calibration uses the Qwen3 chat template to
count the assembled prompt, including system instructions. It rejects candidates
over the p95 budget and uses a paired dev bootstrap to choose the smaller size
when the Recall@4 gap includes zero.
