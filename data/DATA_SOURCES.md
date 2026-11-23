# Data Sources (Member B)

Document the source campus regulation PDF here (title, issuing institution, year, page count, redistribution policy, SHA-256 checksums of `source_id/` and frozen `source_en/corpus.json`).

| Artifact | Path | SHA-256 | Notes |
|---|---|---|---|
| Original Indonesian PDF | `data/source_id/` | `9e058c8ae166bb32b55a42807700bf586782c7ca7bca8715242450bd5fa6fc1b` | Check redistribution rights before committing |
| Frozen English corpus | `data/source_en/corpus.json` | _TBD_ | Frozen after translation checks & human review (§5) |

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
