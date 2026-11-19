# PROJECT SPEC v3: Model Scaling in Grounded Regulation QA
**Local Qwen3 (quantized) vs. hosted open-weight models, fixed RAG pipeline, single translated PDF**

> Audience: a coding agent. Follow literally. Anything "configurable" goes in YAML, never hard-coded.
> **[VERIFY]** marks facts from memory or third-party summaries (model IDs, sizes, rate limits, library support). Check them before relying on them and pin exact versions.
> v3 changes vs v2 are listed in the **Decision log (§17)**.

---

## 0. Team scope and ownership (read this before editing any file)

Four people work on this repo. Member letters A to D. **The person invoking you will say which member they are** (e.g., "I am Member B"). Work only inside that member's scope.

### 0.1 Rules for the agent
1. **Only create or modify files in the invoking member's scope** (§0.2). Files in other scopes are **read-only** for you.
2. If your task needs a change in another member's scope, **do not edit it**. Write a short note in `docs/handoffs/<from>_to_<to>.md` (what you need, why, proposed interface) and, if needed, build against a minimal stub so you are not blocked.
3. **Cross-scope contracts (§0.3) must not be changed unilaterally.** A contract change needs a note in `docs/CONTRACT_CHANGES.md` naming the owner and every consumer.
4. **Never fabricate human-only artifacts** (§0.5): translation reviews, eval questions, gold labels, human ratings, annotations. Produce tooling, templates, and validators instead; mark placeholder data clearly as `SYNTHETIC_FIXTURE`.
5. **No GPU assumption outside Member A's scope.** Only Member A has the RTX 3070. Shared code must run on CPU (guard CUDA use); tests use the mock adapter or a tiny CPU model (§0.4).
6. **Secrets:** API keys (`GROQ_API_KEY`, `GEMINI_API_KEY`) live in a gitignored `.env`; commit only `.env.example`. Never print or log keys, and never put them in configs, notebooks, frontend code, or run logs.
7. Each owner writes tests for their own modules, plus tiny fixtures others can use.

### 0.2 Ownership by member

| Member | Role | Owns (read/write) | Main experiments/outputs |
|---|---|---|---|
| **A** (bilingual lead, **owns the RTX 3070**) | Local models and hardware | `src/regqa/models/local_hf.py`, quantization setup, `src/regqa/evaluation/efficiency.py` (VRAM/latency), `src/regqa/analysis/tokenizer_stats.py`, `src/regqa/analysis/scaling_fit.py`, `configs/models/local_*`, `frontend/src/features/compare/` and `frontend/src/features/telemetry/` | **All local runs** (E1, E2, local half of E0/E3/E4/E6, A1), VRAM/latency measurements, scaling plots. Human: bilingual translation review lead |
| **B** | Corpus, retrieval, deterministic metrics | `src/regqa/ingest/`, `src/regqa/translate/`, `src/regqa/retrieval/`, `configs/retrieval.yaml`, `data/` (source files, `glossary.yaml`, `eval/`, `cache/`, `DATA_SOURCES.md`, `review_log.csv`), `ANNOTATION_GUIDE.md`, eval-set schema validator and split-lock/checksum tooling, `src/regqa/evaluation/` files `keyfact.py`, `citation.py`, `abstention.py`, `retrieval_metrics.py`, `stats.py`, `report.py`, `scripts/make_tables.py` | Frozen translated corpus, **chunk-size calibration (§5.1)**, eval set v1, retrieval cache + oracle cache, correctness/citation/abstention metrics, CIs and tables |
| **C** | Hosted models, pipeline, backend API | `src/regqa/models/base.py` (adapter interface + `MockAdapter`), `models/hosted_openai_compat.py`, optional `hosted_gemini.py`, `models/rate_limiter.py`, `src/regqa/generation/` (prompt builder, **`parser.py`**), `src/regqa/pipeline.py`, `src/regqa/api/` (FastAPI, OpenAPI schema, replay mode), `scripts/run_experiment.py`, `configs/` (base, experiments, prompts, hosted configs; **except** `configs/retrieval.yaml`), `requirements.txt`, `Makefile`, `src/regqa/utils/`, `.env.example` | E3 (hosted half), hosted part of E0/E4/E6, resumable runs, run-log format, the backend API |
| **D** | Frontend and model-based evaluation | `frontend/` (all except Member A's two feature folders), `src/regqa/evaluation/` files `faithfulness.py`, `judge.py`, human-rating sheet and agreement scripts, `src/regqa/analysis/error_taxonomy.py`, `scripts/evaluate.py`, `results/` | Chat page, sources panel, app shell, faithfulness/NLI, judge validation, error taxonomy, E5 |

**Shared (anyone may append, no one may rewrite others' entries):** `README.md` (each owner maintains their own section), `docs/handoffs/`, `docs/CONTRACT_CHANGES.md`, `docs/run_requests/`, `tests/` (each owner in their own subfolder: `tests/local/`, `tests/corpus/`, `tests/hosted/`, `tests/eval/`, `tests/frontend/`).

**Frontend split (inside `frontend/`)**
- **D (lead):** app shell, routing, state management, styling/design system, Chat page, Sources panel, disclaimer, error and empty states, build tooling.
- **A:** Compare page (2 to 3 models side by side) and Telemetry panel (latency, tokens, VRAM).
- **C:** provides the API and `openapi.json`; `frontend/src/api/` types are **generated** from it and never hand-edited.
- Until the backend is ready, the frontend develops against canned JSON or a mock server.

### 0.3 Cross-scope contracts

| Contract | Producer | Consumers | Spec |
|---|---|---|---|
| `corpus.json` structured, checksum-frozen corpus (section IDs) | B | A, C, D | §5 |
| `eval_set.jsonl` schema (`key_facts`, `gold_spans`, `answerable`, `split`) | B | A, C, D | §9.1 |
| `contexts.jsonl` / `contexts_oracle.jsonl` retrieval cache (5 ranked passages; `top_k_context` of them are fed to models) | B | A, C, D | §6 |
| `ModelAdapter.generate(messages, gen_config) -> GenerationResult` (+ `MockAdapter`) | C (defines) | A (local), C (hosted) | §7 |
| Output contract: answer + `Sources: [n]` + `NOT_FOUND`; `parser.py` output fields | C | B, D | §8 |
| `predictions.jsonl` fields (`qid, model, precision, run_id, prompt_hash, raw_output, parsed, tokens_in, tokens_out, latency_s, hardware, timestamp`) | A and C | B, D | §13 |
| Config YAML schema (`configs/models/*.yaml`, `configs/experiments/*.yaml`) | C | all | §4, §7 |
| REST API + `openapi.json` | C | A, D | §12 |
| Run request file (`docs/run_requests/*.yaml`) | anyone | A | §0.4 |

**Unblocking rule (week 1):** B provides `tests/fixtures/mini_corpus.json` and `mini_eval.jsonl` (~10 `SYNTHETIC_FIXTURE` questions); C provides the adapter interface, `MockAdapter`, and a canned-JSON mock API. Everyone else can then build and test in parallel without the real corpus, a GPU, or API quota.

### 0.4 Local-model testing depends on Member A

Member A owns the only RTX 3070, so **every local-model run, VRAM measurement, and local latency measurement is executed by A** (on the 3070, or on Colab/Kaggle for models that don't fit).

- **Run requests:** anyone needing a local run adds a YAML file to `docs/run_requests/` (experiment config, models, precisions, split, priority). A runs `make run EXP=...` and commits results under `runs/local/<exp>/<model>/<run_id>/`. Large artifacts go to shared storage, not git.
- **Others never need a GPU:** develop against `MockAdapter`, canned outputs, or Qwen3-0.6B on CPU.
- **Hardware logging:** every local run records GPU, driver, CUDA, library versions, and quantization method. Local **latency is reported only from the 3070**; if unavailable, report "not measured" and never mix in Colab/Kaggle timings.
- **Critical path:** A is on the critical path for M4, E1, E2, A1, and the local half of E3. A verifies CUDA and `bitsandbytes` in week 1 and has a Qwen3-0.6B 4-bit smoke test passing by end of week 2. If `bitsandbytes` fails, fall back to Colab/Kaggle for quantized runs and note it.
- **Demo modes (§12):** `replay` (default; works anywhere), `live-hosted` (needs keys and quota), `live-local` (only on A's GPU machine).

### 0.5 Human-only tasks (agent builds tooling; people do the work)

| Task | Who | Notes |
|---|---|---|
| Translation review (≥10 to 15%, always sections used by eval questions) | A leads (bilingual); others help | Log in `review_log.csv` |
| Writing eval questions, `key_facts`, gold spans | All four (B coordinates) | Roughly equal shares; A checks against the Indonesian original. **Dev questions come first** (needed for §5.1) |
| Second-annotator overlap (20%) for kappa | Two members per question | Different annotator from the author |
| Human rating sample (≥60 answers per main system) + 20% double-rated overlap | Split across all four | D provides the rating sheet and agreement script |
| Account setup and keys | C (Groq); B (Gemini for translation); D (Gemini for judge) | Check each provider's data terms |
| Local GPU runs, Colab/Kaggle sessions, 3070 latency runs | **A only** | Log hardware every run |
| Waiting out hosted free-tier quota windows | C monitors | Runs are resumable; restart daily |
| Final decisions on open items (§0.6) | All four | |

### 0.6 Decisions locked vs. still open
**Locked:** single translated PDF; frozen retrieval cache; Qwen3 local ladder; one quantization library; Groq hosted models; thinking disabled for Qwen3; Argon out of scope by default; agentic RAG out of scope; local runs executed by A; cache stores 5 ranked passages and models are fed the top 4; chunk size chosen by the bounded dev-only calibration in §5.1 (method locked, value still open); hosted collection starts in week 3 on the dev split.
**Open (record here once settled):** exact Qwen3 sizes and Hugging Face IDs; hosted ~27B Qwen ID (confirm in the Groq console); current Groq limits; which optional experiments (E6, E7, E8) to run; judge model; embedding model; **chunk size value (output of §5.1)**; **frontend framework** (default React + TypeScript + Vite + Tailwind; Streamlit only as fallback prototype). If E8 is chosen, B supplies Indonesian questions and a second retrieval cache for the Indonesian corpus.

### 0.7 Milestone ownership
M1 B (A reviews translation) · M2 B (all four contribute questions) · M3 B · M4 A (local runner) + C (parser, prompt) · M5 C · M6 B (deterministic metrics) + D (faithfulness, judge) · M7 D (all four rate) · M8 C (backend API, Makefile) + D (frontend) + A (compare/telemetry) + B (`make report`)

---

## 1. Goal

Measure how the **generator model** affects answer quality in a RAG system, while everything else is held constant. The corpus is **one 74-page campus regulation PDF, translated from Indonesian to English**. Retrieval runs **once**, its output is cached, and every model receives the identical context and prompt.

Variables under study:
1. **Parameter count** within one local family (Qwen3, ~0.6B to 8B).
2. **Quantization** (fp16 vs 8-bit vs 4-bit) at fixed size.
3. **Local vs. hosted** open-weight models (GPT-OSS-20B, GPT-OSS-120B, a hosted ~27B Qwen) via Groq's free tier.

### Non-goals
- No retriever/chunking *experiments* or grids, no agentic pipeline, no fine-tuning. The only chunking work is the bounded pre-freeze calibration in §5.1, which is a design decision and not a research result.
- No claims before experiments run. All result tables are generated by scripts.
- Argon (Gemini 4) is **out of scope by default** (paid API tier). The adapter interface must make adding it trivial (§14).

---

## 2. Research questions

| ID | Question |
|----|----------|
| RQ1 | How do correctness, faithfulness, citation quality, and abstention change with parameter count in the Qwen3 family (same pipeline, same prompt)? |
| RQ2 | How much quality is lost under 8-bit and 4-bit quantization, and does it depend on model size or question type (e.g., numeric/conditional)? |
| RQ3 | How do local small models compare with hosted larger open-weight models on quality, latency, and effective cost? Does "total parameters" or "active parameters" better explain differences (GPT-OSS is mixture-of-experts [VERIFY])? |
| RQ4 | How reliably do models follow the output contract (citations, abstention string) and abstain on unanswerable questions, and how does this vary by size? |
| RQ5 (optional) | Does enabling reasoning ("thinking") change results, and at what token/latency cost? |

---

## 3. Tech stack (defaults)

- Python 3.11; `requirements.txt` with pinned versions; seeds fixed
- PDF: `pymupdf` (text-based PDF; no OCR)
- Translation: LLM-assisted (§5), with glossary + automated checks
- Retrieval (run once): `rank_bm25`, `sentence-transformers` (e.g., `BAAI/bge-m3` or an English embedding model [VERIFY]), `faiss-cpu`; hybrid via Reciprocal Rank Fusion
- Local inference: `transformers` + `accelerate` + `bitsandbytes` (**one quantization method**: LLM.int8 for 8-bit, NF4 for 4-bit) [VERIFY on your OS/GPU; fallback Colab/Kaggle]
- Hosted inference: OpenAI-compatible client against Groq (`https://api.groq.com/openai/v1`) [VERIFY], behind a provider-agnostic adapter
- NLI faithfulness checker: a local NLI cross-encoder (e.g., DeBERTa-v3 MNLI/ANLI) [VERIFY]
- Judge (secondary): a model **not under test** (e.g., Gemini Flash-class, AI Studio free tier) [VERIFY quotas]
- Backend: `fastapi` + `uvicorn` + `pydantic`; the OpenAPI schema is the frontend contract
- Frontend: React + TypeScript + Vite + Tailwind CSS [default], types generated from OpenAPI, `eslint` + `vitest`
- Stats/plots: `numpy`, `scipy`, `pandas`, `matplotlib`
- Quality: `pytest`, `ruff`, type hints

---

## 4. Repository layout

```
regqa-scaling/
├── README.md
├── requirements.txt
├── configs/
│   ├── base.yaml
│   ├── retrieval.yaml          # chunking + retrieval params (Member B)
│   ├── models/                 # one YAML per model (id, provider, precision, gen params)
│   ├── experiments/            # one YAML per experiment
│   └── prompts/                # versioned prompt templates
├── data/
│   ├── source_id/              # original Indonesian PDF (if redistribution allowed)
│   ├── source_en/              # translated structured JSON + frozen checksum
│   ├── glossary.yaml
│   ├── eval/                   # eval_set.jsonl, ANNOTATION_GUIDE.md
│   ├── cache/                  # chunks, indices, contexts*.jsonl, chunk_calibration.json
│   └── DATA_SOURCES.md
├── src/regqa/
│   ├── ingest/                 # parse_pdf.py, structure.py (BAB/Pasal -> Chapter/Article)
│   ├── translate/              # translate_sections.py, checks.py, review_log.py
│   ├── retrieval/              # chunking.py, bm25.py, dense.py, hybrid.py, calibrate_chunks.py, build_cache.py
│   ├── models/                 # base.py, local_hf.py, hosted_openai_compat.py, rate_limiter.py
│   ├── generation/             # prompt_builder.py, parser.py (answer/citations/abstain)
│   ├── evaluation/             # keyfact.py, faithfulness.py, citation.py, abstention.py,
│   │                           # judge.py, retrieval_metrics.py, efficiency.py, stats.py, report.py
│   ├── api/                    # FastAPI app: routes, schemas, replay mode, openapi.json
│   ├── analysis/               # tokenizer_stats.py, scaling_fit.py, error_taxonomy.py
│   └── utils/                  # seeds, logging, io, hardware_info.py
├── frontend/                   # React app (src/api generated; features/compare + telemetry = Member A)
├── docs/                       # handoffs/, run_requests/, CONTRACT_CHANGES.md
├── scripts/                    # build_corpus.py, build_cache.py, run_experiment.py,
│                               # evaluate.py, make_tables.py
├── runs/                       # append-only predictions per (experiment, model, run_id)
├── results/                    # generated tables, figures
└── tests/
```

---

## 5. Corpus: translation pipeline (Indonesian PDF to English)

1. **Parse** the PDF into structured JSON keeping original IDs: `BAB`/`Bagian`/`Pasal`/`Ayat` map to `chapter`/`part`/`article`/`paragraph`. Keep `pasal_12` style IDs unchanged across languages so gold labels work for both.
2. **Glossary** (`data/glossary.yaml`) fixed before translating, e.g.: Pasal to Article, Ayat to Paragraph, SKS to credit units, IPK to cumulative GPA, cuti akademik to academic leave, skripsi to undergraduate thesis, dosen pembimbing to supervisor, sanksi to sanction. Pass the glossary in every translation prompt.
3. **Translate section by section** (one Article or a few at a time), never the whole PDF at once. Instruct: stay literal; preserve every number, date, duration, and percentage; map modal wording consistently (wajib = must/shall, dapat = may, dilarang = prohibited).
4. **Use a translator model that is not evaluated** (e.g., Gemini Flash-class via AI Studio). Log every translation prompt verbatim (needed for the AI-usage appendix).
5. **Automated checks** (`translate/checks.py`): same number of chapters/articles/paragraphs; every digit sequence and date in the source appears in the translation; no section shorter than a threshold ratio of the source length; glossary terms applied.
6. **Human review**: bilingual reviewer(s) check at least 10 to 15% of sections, always including every section referenced by an eval question; record corrections and error types in `review_log.csv`. Report the error rate.
7. **Freeze**: write `source_en/corpus.json` with a SHA-256 checksum; all experiments assert the checksum. Chunking is *downstream* of this file, so §5.1 does not break the corpus freeze.
8. Tables/annexes: translate as plain text rows or exclude and document.

### 5.1 Chunking and chunk-size calibration (new in v3)

**Why this section exists.** Chunk size affects dense (FAISS) retrieval quality and every model's prompt length, so it must be justified, not assumed. "Optimal" here means *best on the dev split under the prompt-token budget*, and the choice is documented as a design decision, not a finding.

**Fixed chunking rules (all candidates share them)**
- Structure-aware: the unit is the Article; an Article that fits under `max_chunk_tokens` stays whole. Longer Articles split at **Paragraph (Ayat) boundaries first**, then at sentence boundaries; never mid-sentence.
- Every chunk carries a header `[Chapter X > Article N]` (the header counts toward the token limit and is included in the embedded text).
- When a split is forced inside a Paragraph, use `overlap_tokens` (default 10 to 15% of `max_chunk_tokens`, in YAML). No overlap across Article boundaries.
- Each chunk keeps `section_ids` (article + paragraph range). A chunk is relevant if its `section_ids` overlap gold, so **gold labels are independent of chunk size** and the same dev questions work for every candidate.
- Tokens are counted with the **embedding model's tokenizer** (`chunk_tokenizer` in YAML); the prompt budget check below additionally uses the Qwen3 tokenizer, as the more conservative of the two for English text.
- Dense index: **`faiss.IndexFlatIP` over L2-normalised embeddings** (exact search). The corpus is only hundreds of chunks, so approximate indexes (IVF/HNSW) add error and no benefit. Record embedding model ID, normalisation, and FAISS version.

**Calibration procedure (`retrieval/calibrate_chunks.py`, Member B, CPU only)**
1. Candidates: `max_chunk_tokens` in **{128, 256, 384, 512}** (list in `configs/retrieval.yaml`). 512 is the upper bound: it matches common embedding-model limits [VERIFY for the chosen model] and keeps 4 passages near 2k tokens.
2. For each candidate: rebuild chunks, BM25 + dense indices, and hybrid RRF (k=60), with all other settings identical.
3. Evaluate on the **dev split only** (40 questions, answerable ones for recall): Recall@{1,3,4,5}, MRR@10, and **mean tokens of the top-4 context**. Report the number of dev questions used.
4. **Hard constraint:** the assembled prompt (system + question + top-4 chunks) must stay **≤ `prompt_token_budget` (default 2,500)** at p95 over dev; this protects the Groq free-tier limits (§10). Candidates that violate it are rejected.
5. **Decision rule (written before looking at results):** pick the candidate with the highest Recall@4 among those meeting the constraint. If candidates are within one bootstrap standard error (or their paired bootstrap 95% CI for the difference includes 0), choose the **smaller** chunk size (cheaper prompts, more precise citations). Use MRR@10 only to break remaining ties.
6. Save `data/cache/chunk_calibration.json` (per-candidate metrics, constraint checks, chosen value, date, git commit, embedding model ID) and set `max_chunk_tokens` in `configs/retrieval.yaml`.
7. **Freeze** the chosen chunking, then build the cache (§6). The test split is never used here, and the chosen value is never revisited after the cache is built.

**Caveats to state in the report:** 40 dev questions give wide uncertainty, so a small recall difference between sizes is not conclusive; the choice also depends on the embedding model, so the embedding model must be fixed *before* calibration; this is a calibration of one pipeline component, not a study of chunking in general.

---

## 6. Retrieval (run once, then frozen)

- Retriever: hybrid (BM25 + dense, RRF k=60) [defaults; a sanity-checked component, not an experiment]
- **Cache size vs. context size:** the cache stores the **top 5 ranked passages** per question (`cache_k: 5`), so Recall@5 can be reported. Models are fed only the **top 4** (`top_k_context: 4`, YAML), keeping prompts at about 1.5k to 2.5k tokens for the hosted free-tier limits. All models get the same top 4 in the same order.
- `scripts/build_cache.py` writes `data/cache/contexts.jsonl`: `{qid, chunks:[{chunk_id, section_ids, text}], scores}` (5 entries each)
- **Oracle contexts**: also write `contexts_oracle.jsonl` (gold sections only, dev split) to separate reading ability from retrieval errors
- Report Recall@{1,3,4,5}, MRR@10 once on dev and test (dev is also the calibration set, so mark dev numbers as "used for chunk selection")
- Add an `evidence_in_context` flag per question: whether any of the **fed top-4** chunks overlaps gold (answerable questions only). It feeds the abstention analysis (§11.4).
- All models read contexts **only** from the cache. No model-specific retrieval.

---

## 7. Models

### 7.1 Local (Qwen3 instruct family) [VERIFY sizes and HF IDs]

| Name | Approx. params | Precisions to run |
|------|---------------|-------------------|
| qwen3-0.6b | 0.6B | fp16, 4-bit |
| qwen3-1.7b | 1.7B | fp16, 8-bit, 4-bit |
| qwen3-4b | 4B | fp16, 8-bit, 4-bit |
| qwen3-8b | ~8.2B | 8-bit, 4-bit (fp16 only if Kaggle dual T4 works) |

Hardware notes [VERIFY]: RTX 3070 has 8 GB, so 8B only at 4-bit, and 4B fp16 (~8 GB weights) does not fit with context. Free Colab T4 (~15 GB) handles 8B at 8-bit/4-bit. 8B fp16 (~16 GB) needs two GPUs and is **not part of the default plan**. Qwen3-8B is slightly above 7B; call it "8B" and say so.

**Two ladder conditions** (both reported):
- **L-uniform:** all four sizes at **4-bit NF4** (3070-reproducible).
- **L-best:** each size at the highest precision that fits.

### 7.2 Hosted (Groq free tier) [VERIFY IDs/limits in your console]
- `openai/gpt-oss-20b`
- `openai/gpt-oss-120b`
- a hosted ~27B Qwen (ID reported as `qwen/qwen3.8-27b` in one source and 3.6 in another; its architecture description is also unverified; **confirm both in the console** and drop it from E3 if unavailable)
- Reported free-plan limits: 30 RPM, 1,000 RPD, 8,000 TPM, 200,000 TPD, per organisation; the console is authoritative.

### 7.3 Optional tiers (adapter-ready, off by default)
- Gemini Flash/Flash-Lite (free tier) as a proprietary reference
- Gemini 4 Argon (paid API; §14)

### 7.4 Generation settings (identical wherever possible)
- Greedy / temperature 0; `max_new_tokens` 256 (hosted: the cap includes reasoning tokens, so set carefully)
- **Local Qwen3:** thinking disabled (`enable_thinking=False`)
- **GPT-OSS:** lowest reasoning effort; it cannot be fully disabled, a known confound
- Same system/user prompt text; only the chat-template wrapper differs per model
- Record: model ID/revision hash, precision, library versions, hardware, date, prompt hash

---

## 8. Prompt and output contract

System prompt (configurable, versioned in `configs/prompts/`):
1. Answer only from the numbered context passages.
2. Be concise; give the specific rule, number, or condition.
3. End with `Sources: [n], [m]` listing passages used.
4. If the context does not contain the answer, reply exactly: `NOT_FOUND`.

`generation/parser.py` (Member C) extracts: `answer_text`, `cited_ids`, `abstained` (exact `NOT_FOUND`), `format_ok`. **Malformed outputs count as incorrect; never silently dropped.** Report `format_ok` rate per model.

---

## 9. Evaluation set

### 9.1 Schema (`data/eval/eval_set.jsonl`)
```json
{
  "id": "q017",
  "question": "How many semesters at most may a student take academic leave?",
  "type": "factual | numeric | procedural | multi_section | conditional | unanswerable | ambiguous",
  "answerable": true,
  "gold_answer": "...",
  "key_facts": [["two semesters", "2 semesters"], ["consecutive"]],
  "gold_spans": [{"section_id": "pasal_12", "paragraph": "3"}],
  "split": "dev | test",
  "annotator": "A",
  "notes": ""
}
```
`key_facts` = required facts, each a list of acceptable surface variants (enables **automatic strict scoring** without an LLM judge). The schema is a contract (§0.3); do not add fields without a contract-change note.

### 9.2 Size and mix
- **~150 questions**: dev 40 (prompt/format checks, chunk calibration, E0, E4), test 110 (final reporting; lock with checksum)
- Mix: ~35% factual/numeric, ~20% procedural, ~15% multi-section, ~15 to 20% unanswerable, ~10% conditional, rest ambiguous
- Written in English from the translated text, with the Indonesian original consulted for precision-critical rules
- 20% overlap annotated by two members; report Cohen's kappa; write `ANNOTATION_GUIDE.md`
- **Authoring order:** write and validate the 40 dev questions first, because §5.1 and the week-3 hosted dev runs depend on them
- Small n means wide confidence intervals; plan analysis accordingly (§11.6)

---

## 10. Experiments

All use the frozen contexts. Each model/config runs **3 repeats** (hosted at least; local too if cheap), because decoding may not be perfectly deterministic.

| Exp | Description |
|-----|-------------|
| **E0** | Closed-book anchor on **dev answerable** questions: no context, same question. Scored by answer content (key facts), not citations. Shows how much retrieval matters and checks contamination. Exploratory (small n) |
| **E1** | Parameter ladder: Qwen3 0.6B/1.7B/4B/8B under L-uniform and L-best (RQ1) |
| **E2** | Quantization: 0.6B (fp16/4-bit), 1.7B and 4B (fp16/8-bit/4-bit where feasible), 8B (8-bit/4-bit) (RQ2) |
| **E3** | Local vs hosted: local 8B (best precision) vs GPT-OSS-20B, GPT-OSS-120B, hosted ~27B Qwen if available (RQ3); also plot against **total and active** parameters. Descriptive: model, precision, service, and reasoning all change together |
| **E4** | Oracle vs retrieved context on **dev answerable** questions, subset of models (RQ4). Exploratory (small n) |
| **E5** | Reliability breakdown by question type, format compliance, abstention (RQ4) |
| **E6 (optional)** | Thinking on/off for Qwen3; reasoning-effort levels for GPT-OSS (RQ5) |
| **E7 (optional)** | Position effect: gold passage first/middle/last in context |
| **E8 (optional)** | Indonesian vs English version of the same questions/corpus, by size |
| **A1** | Tokenizer analysis (no GPU): tokens per passage for ID vs EN across model tokenizers |

### Hosted-API budget design
Free-tier limits (30 RPM, 8K TPM, 200K TPD, 1K RPD per model, per organisation [VERIFY]) imply roughly **60 to 80 calls/day per hosted model** at ~2.5k tokens per call. Therefore:
- Keep prompts small (`top_k_context`=4, chunk size from §5.1 under the `prompt_token_budget`, `max_new_tokens` ≤ 256)
- Quotas are **per model**, so run the hosted models in parallel
- `rate_limiter.py`: token bucket for RPM/TPM + persistent daily counter for RPD/TPD, honoring `retry-after` and 429s
- Runs are **append-only and resumable**: skip any `(qid, run_id)` already in `predictions.jsonl`
- Estimated collection: 150 q x 3 runs ≈ 450 calls ≈ 6 to 8 days per hosted model

### Collection schedule (new in v3)
- **Week 3:** start hosted runs on the **dev split** (40 q x 3 = 120 calls per model). This tests the rate limiter and format handling and uses quota that would otherwise be idle. It requires the frozen cache, so §5.1 must finish first.
- **Week 4, after the test split is locked:** hosted test collection (110 q x 3 = 330 calls per model).
- Do not push hosted collection later than this. Optional: a developer-tier upgrade with a spend cap raises limits [VERIFY].

---

## 11. Metrics

### 11.1 Answer correctness (primary)
- **Strict key-fact correctness**: all required `key_facts` present (normalized string match, number normalization) **and** `format_ok`. Contradictions with the gold answer are not auto-detected; they are caught in human rating and the error taxonomy.
- **Soft key-fact recall**: fraction of key facts present
- **Human rating** (0/1/2) on a stratified sample: **≥60 answers per main system**, split across the 4 members, plus a **20% double-rated overlap** for agreement (Cohen's kappa). Optionally 20 dev responses per reference configuration, if workload allows.
- **LLM judge** (secondary, model not under test; batch several answers per call); report judge-human agreement
- Token F1/EM only as secondary

### 11.2 Faithfulness / hallucination
- Sentence-level (or atomic-claim) NLI entailment against the provided context; **faithfulness** = fraction of claims supported
- **Hallucination rate** = fraction of answers with ≥1 unsupported claim
- Validate the NLI checker on a blind human-labeled subset

### 11.3 Citation quality
- **Citation precision**: cited passages that overlap gold sections
- **Citation recall**: gold sections cited
- **Invalid citations**: references to passage numbers not supplied

### 11.4 Abstention (on `answerable` flag)
- Confusion matrix; abstention precision/recall/F1
- **False-answer rate** on unanswerable questions (safety-critical)
- **Over-abstention rate** on answerable questions
- **Split by evidence sufficiency:** report over-abstention separately for answerable questions with `evidence_in_context = true` (the model failed to use evidence) and `false` (retrieval missed it, so abstaining is arguably correct). Computed from the cache, no extra runs.

### 11.5 Efficiency (local on the RTX 3070 only for comparability)
- Peak VRAM (`torch.cuda.max_memory_allocated`), model load time
- Latency p50/p95, tokens/sec, input/output tokens (batch size 1, warm-up excluded)
- Hosted: latency (note network effects), tokens, effective cost at paid rates (computed, not paid)

### 11.6 Statistics
- 95% bootstrap CIs (≥1,000 resamples, fixed seed) on every headline metric; report n and response coverage
- Paired bootstrap per question between systems; **exact McNemar with Holm correction** across the family of pairwise comparisons
- **Flip analysis:** per question, report correct→wrong and wrong→correct changes between systems (especially across precisions), compared with run-to-run variation across repeats
- **Scaling fit**: correctness vs log10(parameters), slope with bootstrap CI; do **not** call it a "scaling law" with only 4 points
- Differences within CI are labeled inconclusive

### 11.7 Success indicators (demo-suitability targets, new in v3)
Reported for the **best configuration** and read exploratorily, not as pass/fail for every model. A small model missing them is a valid finding.

| Metric | Target |
|---|---|
| Strict correctness | ≥ 80% |
| Faithfulness | ≥ 90% |
| Format valid | ≥ 95% |
| Abstention recall (unanswerable) | ≥ 80% |
| Over-abstention (answerable) | ≤ 10% |

Report each with its CI and coverage. Thresholds are operational choices, not results.

---

## 12. Application (backend API + frontend)

### 12.1 Backend (`src/regqa/api/`, Member C)
- FastAPI app with a committed OpenAPI schema (`openapi.json`); CORS configured for the frontend dev origin
- Endpoints (under `/api`): `GET /health`, `GET /models` (id, family, params, active params, precision, provider, available modes), `POST /ask` (question, model_id, precision, top_k, mode), `POST /compare` (question, list of model_ids, 2 to 3), `GET /sections/{section_id}` (article text for the sources panel)
- Responses include: answer text, parsed citations mapped to section IDs, `abstained`, tokens in/out, latency, peak VRAM if local, and a structured `error` for quota exhaustion, model unavailable, timeout, or format failure
- **Modes:** `replay` (serves saved predictions from `runs/`; **default**, no GPU or quota), `live-hosted` (keys), `live-local` (A's GPU machine)
- Keys stay server-side

### 12.2 Frontend (`frontend/`, Member D lead, Member A for two features)
- **Chat page (D):** question box, model/precision selector, mode indicator, answer with numbered citations, `NOT_FOUND` styling, retry and error states
- **Sources panel (D):** expandable article text with the cited passage highlighted, chapter/article/paragraph labels
- **Compare page (A):** same question and identical context to 2 to 3 models side by side; differences in answer, citations, and abstention visible at a glance
- **Telemetry panel (A):** latency, tokens in/out, peak VRAM (local), and a clear label when a number comes from replay data
- States must distinguish abstention, format failure, and service failure
- Disclaimer on every page: informational only; consult the official regulation
- Responsive, keyboard-accessible, loading skeletons, no secrets in the bundle
- Runs with `npm run dev` and `uvicorn regqa.api.main:app`; `make app` starts both

### 12.3 Acceptance
- A reviewer can ask a question in `replay` mode with no GPU and no keys, see the answer with highlighted sources, and compare two models
- `openapi.json` and generated frontend types stay in sync (`make check-api` or CI)

---

## 13. Engineering requirements
- Deterministic seeds; log hardware, library versions, git commit in every results file
- Every hyperparameter in YAML (including `max_chunk_tokens`, `overlap_tokens`, `cache_k`, `top_k_context`, `prompt_token_budget`); no magic numbers
- Unit tests for chunker (boundaries, header, overlap, section-ID mapping), parser, key-fact scorer, metrics (hand-computed fixtures), rate limiter (simulated clock), checksum assertion
- `Makefile`: `make corpus`, `make calibrate`, `make cache`, `make run EXP=...`, `make eval`, `make report`, `make app`
- Predictions JSONL fields: `qid, model, precision, run_id, prompt_hash, raw_output, parsed, tokens_in, tokens_out, latency_s, hardware, timestamp`
- Graceful failure handling (OOM, 429, timeouts), always logged
- Quality results may be generated on Colab/Kaggle, but **log the hardware**; never compare latency across machines

---

## 14. Optional extensions (only if ahead of schedule)
- **Argon / Gemini tiers:** add `hosted_gemini.py`. Argon reportedly needs the paid Gemini API tier [VERIFY]. Pin the exact model ID and fix the reasoning level. Do not use the Gemini app for controlled results.
- Agentic RAG, reranking, broader hybrid-retrieval grids (out of current scope)

---

## 15. Milestones and acceptance criteria

| Milestone | Deliverable | Acceptance |
|-----------|-------------|------------|
| M1 | Parsed + translated + frozen corpus, review log | Checks pass; ≥10 to 15% reviewed; checksum set |
| M2 | Eval set v1 (dev first, then test), annotation guide, kappa | Schema validates; key_facts present; split locked |
| M3 | **Chunk-size calibration**, retrieval cache (5 passages) + oracle cache, sanity metrics | `chunk_calibration.json` committed with decision rule applied; prompt budget respected; Recall@{1,3,4,5} reported; cache reused by all models |
| M4 | Local runner (all sizes/precisions), parser, key-fact scorer | 20-question smoke test per model passes; VRAM logged |
| M5 | Hosted runner with rate limiter; **dev-split collection started in week 3** | Runs resume after interruption; daily quota respected |
| M6 | Full metrics (faithfulness, citation, abstention, efficiency) + stats | Tables with CIs, Holm-corrected tests auto-generated |
| M7 | Human rating sample + judge validation | Agreement computed |
| M8 | Backend API + frontend (chat, sources, compare, telemetry, replay mode) + `make report` | One-command reproduction; demo works in replay mode without GPU or keys |

---

## 16. Risks the code should guard against
- **Test-set leakage:** tune prompts and chunk size only on dev; lock the test split
- **Chunk-size choice overfitting 40 dev questions:** state the decision rule beforehand, prefer the smaller size on ties, report uncertainty
- **Translation errors mistaken for model errors:** keep the review log; analyze errors by whether the source section was reviewed
- **Format failures in small models:** count as incorrect, also report separately
- **Precision/hardware confounds:** log everything; use the L-uniform condition
- **Reasoning asymmetry:** GPT-OSS reasoning tokens vs Qwen3 non-thinking; document and, if possible, run E6
- **Hosted model drift/deprecation:** save raw outputs; pin IDs; collect early
- **Quota exhaustion mid-run:** persistent counters, resumable runs, retry-after handling
- **Judge bias:** never judge a model with itself; validate against humans
- **Privacy/terms:** send only public text and your own questions; check each provider's data terms

---

## 17. Decision log (v2 to v3)

| # | Change | Reason |
|---|---|---|
| 1 | Cache holds 5 ranked passages; models are fed the top 4 | Allows Recall@5 while protecting the hosted token budget |
| 2 | New §5.1: bounded dev-only chunk-size calibration (128/256/384/512), exact FAISS `IndexFlatIP`, written decision rule, prompt-token constraint | Feedback that chunk size must be defined and justified |
| 3 | `configs/retrieval.yaml` owned by Member B | Chunking params are retrieval-owned; avoids cross-scope edits |
| 4 | E2 grid: 0.6B fp16/4-bit added; 8B fp16 removed from default | Matches hardware limits |
| 5 | Hosted dev-split collection starts week 3; test after lock in week 4 | Free-tier quota makes late starts risky |
| 6 | E0 and E4 run on dev answerable questions, labeled exploratory | Closed-book on unanswerable is meaningless; protects test |
| 7 | Exact McNemar with Holm correction; flip analysis | Many pairwise comparisons; per-question change evidence |
| 8 | Abstention split by `evidence_in_context` | Separates model failure from retrieval miss |
| 9 | Success targets added as demo-suitability thresholds | Course requires success indicators |
| 10 | Human rating: ≥60 per system plus 20% double-rated overlap; "no contradiction" not in strict score | Keeps workload feasible; avoids a schema change |
| 11 | Hosted ~27B Qwen ID and description flagged unverified | Sources disagree |
| 12 | Dev questions authored first | §5.1 and week-3 hosted dev runs depend on them |
