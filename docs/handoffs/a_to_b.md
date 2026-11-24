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

---

## 3. Pre-Review Audit of Priority Regulatory Sections (Week 2)

To unblock bilingual translation review as soon as `.tmp/corpus.en.draft.json`, `data/review_log.csv`, and the dev eval split land, Member A audited the **16 high-density regulatory sections** in `.tmp/corpus.id.json` (32% of all 50 sections; **228 numeric tokens total**):

| Section | Pages | Chars | Numeric Tokens | Course Codes | Title (ID) | Review Focus |
|---|---:|---:|---:|---|---|---|
| `4.2` | 15 | 517 | 8 | — | Pembagian Mata Kuliah | 113 / 31 SKS split; 1 SKS minute breakdown |
| `6.1.9` | 22 | 388 | 1 | `IN281` | Pemilihan Mata Kuliah | MBKM exclusion of `IN281` |
| `6.1.10` | 22–23 | 2,703 | 94 | — | Konversi Nilai | **Verify all table rows/cells against PDF** |
| `6.2.1` | 23 | 898 | 6 | — | Mata Kuliah Reguler/ Pilihan | Conversion procedure |
| `6.2.2` | 24 | 938 | 14 | — | Mata Kuliah Program Pengayaan/ Topik Lanjut | Enrichment conversion rules |
| `8.1` | 67 | 142 | 5 | — | Syarat mengambil Kerja Praktik | ≥60 SKS, IPK ≥ 2.0 |
| `8.2` | 67 | 266 | 0 | — | Jenis Kerja Praktik yang Diizinkan | Allowed RG types, regular/intersession |
| `8.3` | 67–68 | 2,963 | 11 | — | Prosedur Menempuh Kerja Praktik | Multi-step KP workflow |
| `8.4` | 68–69 | 757 | 6 | — | Pelaksanaan Bimbingan Kerja Praktik | ≥8 main supervisor, ≥3 field supervisor; -1 grade penalty |
| `8.5` | 69 | 795 | 4 | — | Pelaksanaan Kelas Kerja Praktik | Attendance obligation (`wajib`) |
| `8.6` | 69–70 | 1,762 | 15 | — | Prosedur Ujian Sidang Kerja Praktik (USKP) | Required forms & chapters 1–6 |
| `8.7` | 70 | 95 | 0 | — | Format Penulisan Laporan Kerja Praktik | Report formatting reference |
| `9.1` | 70 | 196 | 8 | `IN270`, `IN280` | Syarat mengambil Tugas Akhir | ≥120 SKS, IPK ≥ 2.0, `IN270` ≥ C, `IN280` |
| `9.2` | 70 | 162 | 0 | — | Jenis TA yang diizinkan | Allowed RG types |
| `9.3` | 71 | 1,501 | 9 | — | Bentuk Tugas Akhir (TA) | Thesis format options |
| `9.4` | 71–74 | 5,566 | 47 | `IN270`*, `IN281` | Prosedur Pengambilan STA dan/ atau TA | **Source-PDF typo flagged below** |

### 3.1 Canonical Course Codes & Source-PDF Anomaly Log

- **Canonical mapping (locked by Member A):**
  - `IN270` = **Kerja Praktik** (Practical Work)
  - `IN280` = **Seminar Tugas Akhir (STA)** (Undergraduate Thesis Seminar)
  - `IN281` = **Tugas Akhir (TA)** (Undergraduate Thesis)
- **Source-PDF typo in §9.4 (item 1):** The original Indonesian PDF prints *"mata kuliah Seminar Tugas Akhir (IN270)"*, contradicting §9.1 (*"IN280 Seminar Tugas Akhir (STA)"* and *"Kerja Praktik (IN270)"*). Please note this in `ANNOTATION_GUIDE.md` / `DATA_SOURCES.md` so dev-question authors and translation reviewers do not treat this source typo as a translation error or model hallucination.

### 3.2 Bilingual Review Checklist & Error Rate Formula

Once `.tmp/corpus.en.draft.json` and dev question `gold_spans` are available, Member A will review every section referenced by dev questions (plus the 16 priority sections above) against the Indonesian PDF, checking:
1. All numbers, thresholds, credit units, GPAs, grades, and durations preserved verbatim.
2. Modal verbs (`wajib`/`harus` → `must`, `dapat` → `may`, `diizinkan` → `permitted`) preserve obligation vs. permission.
3. Approved glossary terms applied consistently.
4. Table rows/cells in §6.1.10 match the PDF layout.
5. Error rate reported as `(corrected + flagged) / reviewed_sections` in `data/review_log.csv`.
