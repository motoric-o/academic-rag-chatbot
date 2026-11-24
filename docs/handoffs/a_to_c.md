# Handoff: Member A -> Member C (Local Hardware Metadata & Precision Naming)

- **From:** Member A (Local Models & Hardware)
- **To:** Member C (Hosted Models, Pipeline & Backend API)
- **Date:** 2026-11-24
- **Scope Reference:** `docs/project_spec.md` §0.2, §0.3, §0.4, §12.1, §13

---

## 1. `hardware` Metadata Schema Returned by `LocalHFAdapter`

Because `src/regqa/utils/hardware_info.py` is currently a stub (Member C scope), `LocalHFAdapter` implements a self-contained `collect_local_hardware_info()` helper inside `src/regqa/models/local_hf.py` and attaches the resulting dict to `GenerationResult.hardware`.

Keys populated by `LocalHFAdapter`:
- `device`: `"cuda"` | `"cpu"`
- `gpu_name`: `str | None` (e.g., `"NVIDIA GeForce RTX 3070"`)
- `gpu_vram_total_mb`: `float | None`
- `cuda_version`: `str | None`
- `torch_version`: `str | None`
- `transformers_version`: `str | None`
- `accelerate_version`: `str | None`
- `bitsandbytes_version`: `str | None`
- `quantization_method`: `str | None` (e.g., `"bitsandbytes_nf4"`, `"bitsandbytes_int8"`, or `None` for `fp16`)
- `hf_model_id`: `str`
- `revision`: `str | None` (resolved commit hash after load, or configured revision)
- `target_hardware`: `str | None` (from `configs/models/local_*.yaml`, e.g., `"rtx3070_8gb"`)
- `platform`: `str` (`platform.platform()`)

When you implement `src/regqa/utils/hardware_info.py` and `src/regqa/pipeline.py`, you can either adopt these keys directly or let Member A know if you want `LocalHFAdapter` to delegate to a shared function in `hardware_info.py`.

---

## 2. Precision String Alignment (`"4bit"` vs `"4-bit"`)

- `configs/models/local_*.yaml` uses `"fp16"`, `"8bit"`, `"4bit"` (no hyphen).
- `src/regqa/api/schemas.py` (`ModelInfo`, `AskRequest`, `CompareRequest`) uses `"fp16"`, `"8-bit"`, `"4-bit"` (with hyphen).

`LocalHFConfig` normalizes `"4-bit" -> "4bit"` and `"8-bit" -> "8bit"` defensively, so `LocalHFAdapter` will accept either form. Please keep this in mind when wiring `live-local` and `replay` model lookups in `src/regqa/api/routes.py` and `scripts/run_experiment.py`.
