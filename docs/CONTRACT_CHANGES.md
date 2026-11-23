# Cross-Scope Contract Changes

Per [docs/project_spec.md](project_spec.md) §0.1 & §0.3, cross-scope contracts must not be changed unilaterally. Record any proposed or agreed contract change below, naming the **owner** and every **consumer**.

## Template

### [YYYY-MM-DD] Short title
- **Contract:** (e.g., `eval_set.jsonl`, `ModelAdapter`, `predictions.jsonl`, `openapi.json`)
- **Owner:** Member X
- **Consumers:** Members Y, Z
- **Proposed by:** Member ?
- **Change:** Describe the exact field/interface change and migration impact.
- **Status:** Proposed | Approved | Applied

### 2026-11-22 Numbered handbook section IDs

- **Contract:** `corpus.json` sections, `eval_set.jsonl` gold spans, and retrieval cache `section_ids`.
- **Owner:** Member B. **Consumers:** Members A, C, and D.
- **Proposed by:** Member B.
- **Change:** The 2024 handbook uses numbered headings such as `8.1` and `6.1.10`. Store the exact number as a string in `sections[].section_id` and copy it across languages, gold spans, and retrieval metadata. Keep optional numbered items in `paragraph`. Use physical PDF page numbers in `page_start` and `page_end`.
- **Status:** Proposed. Git contains no formal eval set or corpus to migrate.
