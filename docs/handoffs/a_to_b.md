# Handoff: Member A -> Member B (Bilingual Glossary Review & Early Translation)

- **From:** Member A (Bilingual Translation Review Lead)
- **To:** Member B (Corpus, Translation & Retrieval Owner)
- **Date:** 2026-11-24
- **Scope Reference:** `docs/project_spec.md` §0.1, §0.2, §0.5, §5

---

## 1. Audit of `data/glossary.yaml` (`v0.1.0`) against `.tmp/corpus.id.json`

Member A audited the current `data/glossary.yaml` against the 50 numbered sections extracted from `Buku-Panduan-S1-Teknik-Informatika-v2024.pdf` (`.tmp/corpus.id.json`) and against `src/regqa/translate/checks.py`.

### 1.1 Blocking Defect in `checks.py` + `glossary.yaml`

`src/regqa/translate/checks.py` enforces an exact case-insensitive word-boundary match for every glossary term in both `title` and `text`:
```python
for term, equivalent in glossary.items():
    if contains_term(original, term) and not contains_term(english, equivalent):
        issues.append(f"glossary term missing: {term} -> {equivalent}")
```

**Collision found:**
- `wajib -> must` matches **8 sections** (`4.2`, `4.2.1`, `6.1.9`, `6.1.10`, `7.1`, `8.4`, `8.5`, `9.4`), including the section title **`7.1: Mata Kuliah Wajib`**.
- Translating `"Mata Kuliah Wajib"` as `"Required Courses"` will **immediately fail** `translate_sections.py` at section `7.1` (and `4.2`, `6.1.10`) with `glossary term missing: wajib -> must`.

**Proposed fix in `src/regqa/translate/checks.py`:**
- Match longer multi-word glossary terms first (sorted by descending token/character length) and mask matched spans in `original` before checking shorter single-word terms (so `"mata kuliah wajib"` matches `"required courses"` without triggering the standalone modal rule `"wajib" -> "must"`).

### 1.2 Audit of Existing `data/glossary.yaml` Entries

| Indonesian Term | Current English | Sections Matched | Member A Review Recommendation |
|---|---|---:|---|
| `BAB` | `Chapter` | 2 (`8.6`, `9.4`) | Keep (used for report/thesis chapters, e.g., `bab 1, 2, 3, 4, 5 dan 6`). |
| `Bagian` | `Part` | 4 (`3.4`, `4.2`, `4.2.2`, `7.1`) | Review carefully — appears as general prose `"bagian"` ("part/section"), not structural `Bagian`. |
| `Pasal` | `Article` | 0 | Unused in 2024 handbook; safe to remove or archive. |
| `Ayat` | `Paragraph` | 0 | Unused in 2024 handbook; safe to remove or archive. |
| `SKS` | `credit units` | 10 | Keep. |
| `IPK` | `cumulative GPA` | 2 (`8.1`, `9.1`) | Keep. |
| `cuti akademik` | `academic leave` | 0 | Unused in 2024 handbook; safe to remove or archive. |
| `skripsi` | `undergraduate thesis` | 2 (`3.2`, `7.1`) | Keep (matches `Tugas Akhir` below). |
| `dosen pembimbing` | `supervisor` | 5 (`8.3`–`8.6`, `9.4`) | Keep; also add `dosen pembimbing utama` / `pembimbing lapangan` if needed. |
| `sanksi` | `sanction` | 0 | Unused in 2024 handbook; safe to remove or archive. |
| `wajib` (modal) | `must` | 8 | Keep **only after** multi-word masking fix in `checks.py` (otherwise collides with `Mata Kuliah Wajib`). |
| `dapat` (modal) | `may` | 22 | Keep. |
| `dilarang` (modal) | `prohibited` | 0 | Unused in 2024 handbook; safe to keep or remove. |

### 1.3 Proposed Handbook-Specific Additions to `data/glossary.yaml`

Approved by Member A (bilingual lead) for the 2024 Maranatha Informatics handbook:

```yaml
terms:
  Kerja Praktik: "Practical Work"
  Tugas Akhir: "Undergraduate Thesis"
  Seminar Tugas Akhir: "Undergraduate Thesis Seminar"
  Mata Kuliah Wajib: "Required Courses"
  Mata Kuliah Pilihan: "Elective Courses"
  Ujian Sidang Kerja Praktik: "Practical Work Defense Exam"
  # Additional frequent handbook terms (confirm plural/singular compatibility with checks.py):
  Mata Kuliah: "course"
  Program Studi: "study program"
  Prasyarat: "prerequisite"
  Konversi Nilai: "grade conversion"
  Program Pengayaan: "enrichment program"
  Topik Lanjut: "advanced topics"
  magang: "internship"
  bimbingan: "supervision"
  koordinator: "coordinator"
  semester reguler: "regular semester"
  semester antara: "intersession semester"
modal_verbs:
  wajib: "must"
  harus: "must"
  dapat: "may"
  diizinkan: "permitted"
```

> Note: Before freezing `data/glossary.yaml`, verify singular vs. plural surface forms (e.g., `"Required Course"` vs. `"Required Courses"`) against `checks.py`'s exact substring/word-boundary check.

---

## 2. Proposed `data/review_log.csv` Schema & Early Translated Sections Request

`src/regqa/translate/review_log.py` is currently a stub and no English draft exists yet in `.tmp/` or `data/source_en/`. To unblock Member A's human bilingual review (§5 step 6):

1. **Proposed `data/review_log.csv` columns** (for `src/regqa/translate/review_log.py`):
   ```csv
   section_id,reviewer,review_date,status,error_type,source_excerpt,draft_en_excerpt,corrected_en_excerpt,notes
   ```
   - `status`: `approved | corrected | flagged`
   - `error_type`: `none | numeric_or_date | modal_verb | glossary_mismatch | omission | hallucination | table_layout | mistranslation`
2. **Priority batch for early translated sections:**
   Once the glossary and `checks.py` are updated and `scripts/build_corpus.py --model ...` produces `.tmp/corpus.en.draft.json`, Member A will begin human bilingual review with the **16 high-density regulatory sections** (`4.2`, `6.1.9`, `6.1.10`, `6.2.1`, `6.2.2`, `8.1`–`8.7`, `9.1`–`9.4` — 32% of all 50 sections, exceeding the 10–15% minimum and covering the core numeric, procedural, and conditional rules).
