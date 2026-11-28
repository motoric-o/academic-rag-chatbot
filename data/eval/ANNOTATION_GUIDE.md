# Eval Set Annotation Guide (Member B coordinates; all four contribute)

See [docs/project_spec.md](../../docs/project_spec.md) §9 for schema and target question mix.

- **Authoring order:** Write and validate the 40 `dev` questions first (required for §5.1 chunk calibration and Week 3 hosted dev runs), then lock the 110 `test` questions.
- **Overlap:** 20% double-annotated by a different member for Cohen's kappa.
- **No fabricated human data:** Any placeholder fixture files must be clearly marked `SYNTHETIC_FIXTURE`.

## Dev authoring

Each member writes roughly 10 English questions from the English corpus and checks
numbers, dates and conditions against the Indonesian guide. Member B coordinates
IDs and validation. Member A checks the original sections. For the 40-question
dev split, aim for 14 factual/numeric, 8 procedural, 6 multi-section, 4 conditional,
6 to 8 unanswerable, and any remaining items ambiguous.

Use one JSON object per line. Copy section IDs from the corpus and add a
paragraph reference only if a specific numbered item supports the answer.
Each inner list in `key_facts` contains interchangeable wordings of one required
fact. For unanswerable rows use `NOT_FOUND`, an empty `key_facts` and empty
`gold_spans`. Another member checks the double-annotated sample independently.

### Source discrepancy in §9.4

The PDF prints "Seminar Tugas Akhir (IN270)" in §9.4, item 1. Section 9.1
identifies IN270 as Kerja Praktik and IN280 as Seminar Tugas Akhir. IN281 is
Tugas Akhir. Preserve the printed code in the corpus and use §9.1 as evidence
for questions about the canonical course codes.

Validate the accumulated dev questions with the local corpus:

    uv run python -m regqa.evaluation.validate_eval data/eval/eval_set.jsonl --corpus data/source_en/corpus.json --expected-dev 40

For integration testing, use the fictional campus corpus and ten labeled rows:

    uv run python -m regqa.evaluation.validate_eval tests/fixtures/mini_eval.jsonl --corpus tests/fixtures/mini_corpus.json --synthetic --expected-dev 10
